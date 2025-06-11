import monai
print(f"MONAI version being used by train.py: {monai.__version__}")
import sys
print(f"Python executable for train.py: {sys.executable}")

import os
import time
import csv
import argparse
from datetime import datetime
import importlib
import torch
import torch.nn as nn # For nn.CrossEntropyLoss
import torch.optim as optim
import numpy as np
from monai.losses import DiceLoss # Changed from DiceCELoss back to DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, SaveImage
from monai.utils import ensure_tuple_rep

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocess_and_dataloader import train_loader, val_loader

set_determinism(seed=42)

AVAILABLE_MODELS = {
    "unet3d": "models.unet3d",
    "resunet3d": "models.resunet3d",
    "attention_unet": "models.attention_unet",
    "hybrid_model": "models.hybrid_model"
}

def get_model_from_name(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(AVAILABLE_MODELS.keys())}")
    module_path = AVAILABLE_MODELS[model_name]
    module = importlib.import_module(module_path)
    return module.get_model

def ensure_shape_match(outputs, labels):
    if outputs.shape[2:] != labels.shape[2:]:
        print(f"[WARNING] Shape mismatch detected before loss/metric: Output {outputs.shape[2:]} vs Label {labels.shape[2:]}")
        from monai.transforms import ResizeWithPadOrCrop
        resize_transform = ResizeWithPadOrCrop(spatial_size=labels.shape[2:], mode='bilinear')
        aligned_outputs = resize_transform(outputs)
        print(f"[DEBUG] Resized outputs shape: {aligned_outputs.shape}")
        return aligned_outputs
    return outputs

def process_for_metric(outputs_batch, labels_batch, post_pred_metric, post_label_metric):
    """
    Correctly process outputs and labels for metric calculation
    Fixes the critical post-processing issue causing Dice=0
    """
    processed_outputs = []
    processed_labels = []
    
    # Process each sample in the batch individually
    for i in range(outputs_batch.shape[0]):
        # Get single sample (batch dimension already present)
        output_sample = outputs_batch[i:i+1]  # Keep batch dim: [1, C, D, H, W]
        label_sample = labels_batch[i:i+1]    # Keep batch dim: [1, 1, D, H, W]
        
        # Apply post-processing transforms
        output_processed = post_pred_metric(output_sample)  # [1, 2, D, H, W] -> onehot
        label_processed = post_label_metric(label_sample)   # [1, 1, D, H, W] -> onehot [1, 2, D, H, W]
        
        processed_outputs.append(output_processed)
        processed_labels.append(label_processed)
    
    return processed_outputs, processed_labels

def process_decollated_for_metric(outputs_decol, labels_decol, post_pred_metric, post_label_metric):
    """
    Correctly process decollated outputs and labels for metric calculation
    Fixes the critical post-processing issue causing Dice=0
    
    Args:
        outputs_decol: List of tensors [C, D, H, W] after decollate_batch
        labels_decol: List of tensors [1, D, H, W] or [D, H, W] after decollate_batch
    """
    processed_outputs = []
    processed_labels = []
    
    for output_tensor, label_tensor in zip(outputs_decol, labels_decol):
        # Add batch dimension back for post-processing transforms
        output_with_batch = output_tensor.unsqueeze(0)  # [C, D, H, W] -> [1, C, D, H, W]
        
        # Handle label tensor: AsDiscrete with to_onehot expects [1, D, H, W] format
        # It will output [2, D, H, W] for 2 classes
        if label_tensor.dim() == 4 and label_tensor.shape[0] == 1:  # [1, D, H, W] - perfect!
            label_for_transform = label_tensor  # Keep as is
        elif label_tensor.dim() == 3:  # [D, H, W] - add channel dimension
            label_for_transform = label_tensor.unsqueeze(0)  # -> [1, D, H, W]
        else:
            # For any other shape, try to get to [1, D, H, W]
            label_for_transform = label_tensor.squeeze().unsqueeze(0)
        
        # Apply post-processing transforms using the working manual method
        # For outputs: apply argmax manually, then onehot
        manual_argmax = torch.argmax(output_with_batch, dim=1, keepdim=True)  # [1, 1, D, H, W]
        argmax_no_channel = manual_argmax.squeeze(1)  # [1, D, H, W]
        output_onehot = post_label_metric(argmax_no_channel)  # [2, D, H, W] - use label transform for onehot
        output_processed = output_onehot.unsqueeze(0)  # [1, 2, D, H, W]
        
        # For labels: apply onehot directly
        label_processed = post_label_metric(label_for_transform)   # Apply onehot -> [2, D, H, W]
        label_processed = label_processed.unsqueeze(0)  # -> [1, 2, D, H, W]
        
        processed_outputs.append(output_processed)
        processed_labels.append(label_processed)
    
    return processed_outputs, processed_labels

def train(model_name="unet3d", max_epochs=100, learning_rate=1e-4, val_interval=1, checkpoint_interval=10, early_stop_patience=10, use_patch_training=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        get_model_func = get_model_from_name(model_name)
        print(f"Successfully imported {model_name} model factory")
    except ValueError as e:
        print(f"Error importing model: {e}")
        print("Falling back to UNet3D model")
        from models.unet3d import get_model as get_model_func
        model_name = "unet3d"

    model = get_model_func().to(device)
    print(f"Model architecture: {model_name}")

    # --- Advanced loss function for extreme class imbalance ---
    print("Using advanced CompoundMedicalLoss for extreme class imbalance.")
    
    # Import the new loss functions
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.loss_functions import get_loss_function
    
    # Use FocalLoss with aggressive parameters for extreme class imbalance
    # Based on our analysis: 2.3% foreground vs 97.7% background
    # More aggressive alpha to heavily weight the rare foreground class
    loss_function = get_loss_function('focal', alpha=0.05, gamma=3.0)  # Very low alpha = high foreground weight
    print("  - Using FocalLoss (α=0.05, γ=3.0) for EXTREME class imbalance")
    print("  - α=0.05: Heavy emphasis on rare foreground class (95% weight)")
    print("  - γ=3.0: Strong focus on hard examples")
    print("  - Designed for ~2% foreground vs 98% background ratio")
    # --- End of advanced loss combination ---

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, threshold=0.001
    )
    print("Learning rate scheduler (ReduceLROnPlateau) initialized.")

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    post_pred_metric = AsDiscrete(argmax=True, to_onehot=2, n_classes=2)
    post_label_metric = AsDiscrete(to_onehot=2, n_classes=2)

    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_root_dir = os.path.join(project_root_dir, "checkpoints")
    os.makedirs(checkpoint_root_dir, exist_ok=True)
    model_checkpoint_dir = os.path.join(checkpoint_root_dir, model_name)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    prediction_save_dir = os.path.join(model_checkpoint_dir, "epoch_0_predictions")

    log_filename = os.path.join(model_checkpoint_dir, f"{model_name}_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Dice", "LR"])

    best_metric_val = -1.0
    best_metric_epoch = -1
    patience_counter = 0
    
    print(f"Starting training for {max_epochs} epochs (early stopping patience: {early_stop_patience})...")
    training_start_time = time.time()

    pred_saver, label_saver, prob_saver = None, None, None

    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        print(f"-" * 10)
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0.0
        step = 0
        previous_lr = optimizer.param_groups[0]['lr']

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            if epoch == 0 and step == 1:
                print(f"[DEBUG Train First Batch] Input Shape: {inputs.shape}, dtype: {inputs.dtype}")
                print(f"[DEBUG Train First Batch] Label Shape: {labels.shape}, dtype: {labels.dtype}")
                print(f"[DEBUG Train First Batch] Unique label values in batch: {torch.unique(labels.cpu())}")
                for i_sample in range(labels.shape[0]):
                    sample_fg = torch.sum(labels[i_sample] > 0).item()
                    sample_total = labels[i_sample].numel()
                    filename = "N/A"
                    if "image_meta_dict" in batch_data and "filename_or_obj" in batch_data["image_meta_dict"]:
                         if i_sample < len(batch_data["image_meta_dict"]["filename_or_obj"]):
                            filename = os.path.basename(batch_data["image_meta_dict"]["filename_or_obj"][i_sample])
                    print(f"[DEBUG Train First Batch] Sample {i_sample} ({filename}): {sample_fg}/{sample_total} fg pixels ({sample_fg/sample_total*100:.6f}%)")
                    if sample_fg == 0 and (labels[i_sample]==0).all():
                         print(f"  [INFO] Training Sample {i_sample} ({filename}) in first batch appears to have NO foreground pixels (all zeros).")
                    elif sample_fg == 0:
                         print(f"  [WARNING] Training Sample {i_sample} ({filename}) in first batch has NO foreground pixels after processing, but label tensor not all zeros (unexpected).")

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_matched = ensure_shape_match(outputs, labels)
            # Pass labels as-is, loss function will handle dimension requirements
            loss = loss_function(outputs_matched, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"\rEpoch {epoch + 1}, Training step {step}/{len(train_loader)}, Loss: {loss.item():.4f}", end="")

        epoch_loss /= step
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != previous_lr:
            print(f"\nLearning rate changed to: {current_lr:.6f}")
        print(f"\rEpoch {epoch + 1} avg train loss: {epoch_loss:.4f}, Time: {(time.time()-epoch_start_time)/60:.2f} mins, LR: {current_lr:.6f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0.0
            val_step = 0
            save_viz_counter_this_epoch = 0

            if epoch == 0:
                os.makedirs(prediction_save_dir, exist_ok=True)
                pred_saver = SaveImage(output_dir=prediction_save_dir, output_postfix="pred", output_ext=".nii.gz", separate_folder=False, print_log=False, resample=False)
                label_saver = SaveImage(output_dir=prediction_save_dir, output_postfix="gt", output_ext=".nii.gz", separate_folder=False, print_log=False, resample=False)
                prob_saver = SaveImage(output_dir=prediction_save_dir, output_postfix="prob_fg", output_ext=".nii.gz", separate_folder=False, print_log=False, resample=False)

            with torch.no_grad():
                for val_batch_data in val_loader:
                    val_step += 1
                    val_inputs, val_labels = val_batch_data["image"].to(device), val_batch_data["label"].to(device)
                    
                    if epoch == 0 and val_step == 1:
                        print(f"[DEBUG Val First Batch] Input Shape: {val_inputs.shape}, dtype: {val_inputs.dtype}")
                        print(f"[DEBUG Val First Batch] Label Shape: {val_labels.shape}, dtype: {val_labels.dtype}")
                        print(f"[DEBUG Val First Batch] Unique label values: {torch.unique(val_labels.cpu())}")
                        for i_sample in range(val_labels.shape[0]):
                             sample_fg = torch.sum(val_labels[i_sample] > 0).item()
                             sample_total = val_labels[i_sample].numel()
                             filename = "N/A"
                             if "image_meta_dict" in val_batch_data and "filename_or_obj" in val_batch_data["image_meta_dict"]:
                                 if i_sample < len(val_batch_data["image_meta_dict"]["filename_or_obj"]):
                                    filename = os.path.basename(val_batch_data["image_meta_dict"]["filename_or_obj"][i_sample])
                             print(f"[DEBUG Val First Batch] Sample {i_sample} ({filename}): {sample_fg}/{sample_total} fg pixels ({sample_fg/sample_total*100:.6f}%)")

                    val_outputs_logits = model(val_inputs)
                    val_outputs_matched = ensure_shape_match(val_outputs_logits, val_labels)
                    
                    # Add debugging for first validation batch of first epoch
                    if epoch == 0 and val_step == 1:
                        print(f"\n[DICE DEBUG] Model output analysis:")
                        print(f"  Logits shape: {val_outputs_logits.shape}")
                        print(f"  Logits range: [{val_outputs_logits.min():.4f}, {val_outputs_logits.max():.4f}]")
                        
                        # Apply softmax to see probabilities
                        val_probs = torch.softmax(val_outputs_logits, dim=1)
                        print(f"  Background prob mean: {val_probs[:, 0].mean():.4f}")
                        print(f"  Foreground prob mean: {val_probs[:, 1].mean():.4f}")
                        print(f"  Max foreground prob: {val_probs[:, 1].max():.4f}")
                        
                        # Check predicted classes
                        predicted_classes = torch.argmax(val_outputs_logits, dim=1)
                        print(f"  Predicted classes unique: {torch.unique(predicted_classes)}")
                        
                        # Count foreground predictions
                        pred_fg_pixels = torch.sum(predicted_classes > 0).item()
                        total_pixels = predicted_classes.numel()
                        pred_fg_percentage = pred_fg_pixels / total_pixels * 100
                        print(f"  Model predicts {pred_fg_pixels}/{total_pixels} foreground pixels ({pred_fg_percentage:.4f}%)")
                        
                        # Compare with ground truth
                        gt_fg_pixels = torch.sum(val_labels > 0).item()
                        gt_total_pixels = val_labels.numel()
                        gt_fg_percentage = gt_fg_pixels / gt_total_pixels * 100
                        print(f"  Ground truth has {gt_fg_pixels}/{gt_total_pixels} foreground pixels ({gt_fg_percentage:.4f}%)")
                        
                        if pred_fg_pixels == 0:
                            print(f"  🚨 CRITICAL: Model predicts NO foreground pixels!")
                        if gt_fg_pixels == 0:
                            print(f"  ⚠️  WARNING: Ground truth has NO foreground pixels!")
                        print()
                    
                    # Pass labels as-is, loss function will handle dimension requirements  
                    loss_val_item = loss_function(val_outputs_matched, val_labels)
                    val_loss += loss_val_item.item()

                    if epoch == 0 and pred_saver and save_viz_counter_this_epoch < 3:
                        probs_sample = torch.softmax(val_outputs_matched[0], dim=0).cpu()
                        pred_map_sample = torch.argmax(probs_sample, dim=0).unsqueeze(0).float()
                        current_meta_dict = val_batch_data["image_meta_dict"]
                        pred_saver(pred_map_sample, meta_data=current_meta_dict)
                        label_saver(val_labels[0].cpu().float(), meta_data=current_meta_dict)
                        prob_map_fg_sample = probs_sample[1].unsqueeze(0)
                        prob_saver(prob_map_fg_sample, meta_data=current_meta_dict)
                        save_viz_counter_this_epoch += 1

                    val_outputs_decol = decollate_batch(val_outputs_matched)
                    val_labels_decol = decollate_batch(val_labels)
                    val_outputs_metric, val_labels_metric = process_decollated_for_metric(val_outputs_decol, val_labels_decol, post_pred_metric, post_label_metric)
                    
                    # Add debugging for post-processed data
                    if epoch == 0 and val_step == 1:
                        print(f"[DICE DEBUG] Post-processing analysis:")
                        print(f"  Decollated outputs length: {len(val_outputs_decol)}")
                        print(f"  Decollated labels length: {len(val_labels_decol)}")
                        
                        if len(val_outputs_metric) > 0 and len(val_labels_metric) > 0:
                            sample_out = val_outputs_metric[0]
                            sample_lbl = val_labels_metric[0]
                            print(f"  Processed output shape: {sample_out.shape}")
                            print(f"  Processed label shape: {sample_lbl.shape}")
                            
                            # Check channels - should be [1, 2, D, H, W] for 2-class onehot
                            if sample_out.shape[1] == 2:  # 2 classes
                                out_bg_sum = torch.sum(sample_out[0, 0]).item()  # Background channel
                                out_fg_sum = torch.sum(sample_out[0, 1]).item()  # Foreground channel
                                lbl_bg_sum = torch.sum(sample_lbl[0, 0]).item()  # Background channel
                                lbl_fg_sum = torch.sum(sample_lbl[0, 1]).item()  # Foreground channel
                                
                                print(f"  Processed output: bg_sum={out_bg_sum}, fg_sum={out_fg_sum}")
                                print(f"  Processed label: bg_sum={lbl_bg_sum}, fg_sum={lbl_fg_sum}")
                                
                                # Manual Dice calculation for debugging
                                intersection = torch.sum(sample_out[0, 1] * sample_lbl[0, 1]).item()
                                dice_manual = (2.0 * intersection) / (out_fg_sum + lbl_fg_sum + 1e-8)
                                print(f"  Manual Dice calculation: intersection={intersection}, dice={dice_manual:.6f}")
                                
                                if out_fg_sum == 0:
                                    print(f"  🚨 CRITICAL: Post-processed model output has NO foreground!")
                                if lbl_fg_sum == 0:
                                    print(f"  ⚠️  WARNING: Post-processed label has NO foreground!")
                        print()
                    
                    dice_metric(y_pred=val_outputs_metric, y=val_labels_metric)

            val_loss /= val_step
            aggregated_results = dice_metric.aggregate()
            if isinstance(aggregated_results, tuple): 
                metric_tensor = aggregated_results[0]
            else: 
                metric_tensor = aggregated_results
            metric_result_epoch = metric_tensor.item() if torch.is_tensor(metric_tensor) else float(metric_tensor)
            dice_metric.reset()

            print(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}, Dice: {metric_result_epoch:.4f}")

            with open(log_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, epoch_loss, val_loss, metric_result_epoch, current_lr])

            if metric_result_epoch > best_metric_val:
                best_metric_val = metric_result_epoch
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, f"{model_name}_best_model.pth"))
                print(f"New best model saved with Dice: {best_metric_val:.4f} at epoch {epoch + 1}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement in Dice for {patience_counter} epochs. Best Dice: {best_metric_val:.4f} at epoch {best_metric_epoch}")

            scheduler.step(metric_result_epoch)

            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                torch.save(model.state_dict(), os.path.join(model_checkpoint_dir, f"{model_name}_early_stopped_model.pth"))
                print(f"Early stopped model saved.")
                break
        
        if (epoch + 1) % checkpoint_interval == 0 and (epoch + 1) != best_metric_epoch :
            chkpt_path = os.path.join(model_checkpoint_dir, f"{model_name}_checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "loss": epoch_loss,
                "best_metric_val": best_metric_val, "best_metric_epoch": best_metric_epoch
            }, chkpt_path)
            print(f"Checkpoint saved: {chkpt_path}")
        
        if patience_counter >= early_stop_patience: break

    total_training_time = time.time() - training_start_time
    final_epochs_ran = epoch + 1
    print(f"\nTraining completed in {total_training_time/3600:.2f} hours ({total_training_time/60:.2f} minutes) over {final_epochs_ran} epochs.")
    print(f"Best Validation Dice: {best_metric_val:.4f} at epoch {best_metric_epoch}")
    
    if not (patience_counter >= early_stop_patience) and final_epochs_ran >= max_epochs:
        final_model_path = os.path.join(model_checkpoint_dir, f"{model_name}_final_model_epoch_{final_epochs_ran}.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved: {final_model_path}")

    return {
        "model_name": model_name, "best_metric": best_metric_val,
        "best_metric_epoch": best_metric_epoch, "final_epoch": final_epochs_ran,
        "training_time_seconds": total_training_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D segmentation model")
    parser.add_argument("--model", type=str, default="unet3d", choices=list(AVAILABLE_MODELS.keys()), help="Model architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval (epochs)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint saving interval (epochs)")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Patience for early stopping")
    
    args = parser.parse_args()
    
    print(f"Starting training with configuration:")
    for arg, value in sorted(vars(args).items()): print(f"- {arg}: {value}")
    
    if not torch.cuda.is_available(): print("\nWARNING: CUDA is not available. Training will run on CPU.\n")
    
    if not hasattr(train_loader, 'dataset') or not train_loader.dataset or len(train_loader.dataset) == 0 or \
       not hasattr(val_loader, 'dataset') or not val_loader.dataset or len(val_loader.dataset) == 0:
        print("CRITICAL: Training or validation dataloader's dataset is empty. Please check preprocess_and_dataloader.py and dataset paths.")
        sys.exit(1)

    train_history = train(
        model_name=args.model, max_epochs=args.epochs, learning_rate=args.lr,
        val_interval=args.val_interval, checkpoint_interval=args.checkpoint_interval,
        early_stop_patience=args.early_stop_patience
    )
    
    print("\n--- Training Summary ---")
    print(f"Model: {train_history['model_name']}")
    print(f"Completed Epochs: {train_history['final_epoch']}")
    print(f"Best Validation Dice: {train_history['best_metric']:.4f} at Epoch {train_history['best_metric_epoch']}")
    print(f"Total Training Time: {train_history['training_time_seconds']/3600:.2f} hours")