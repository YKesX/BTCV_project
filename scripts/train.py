import os
import sys
import time
import csv
import argparse
from datetime import datetime
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.transforms import ResizeWithPadOrCrop

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dataloader (model will be imported dynamically)
from scripts.preprocess_and_dataloader import train_loader, val_loader

# Set deterministic training for reproducibility
set_determinism(seed=42)

# Dictionary of available models with their module paths
AVAILABLE_MODELS = {
    "unet3d": "models.unet3d",
    "resunet3d": "models.resunet3d",
    "attention_unet": "models.attention_unet",
    "hybrid_model": "models.hybrid_model"
}

def get_model_from_name(model_name):
    """
    Dynamically import the model based on the provided name
    
    Args:
        model_name: Name of the model to import (must be in AVAILABLE_MODELS)
        
    Returns:
        Model factory function
    
    Raises:
        ValueError: If model name is not recognized
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    module_path = AVAILABLE_MODELS[model_name]
    module = importlib.import_module(module_path)
    
    return module.get_model


def ensure_shape_match(outputs, labels):
    """
    Ensure that outputs have the same spatial dimensions as the labels.
    
    Args:
        outputs: Model outputs tensor [B, C, D, H, W]
        labels: Ground truth labels tensor [B, 1, D, H, W]
    
    Returns:
        Outputs tensor with spatial dimensions matching the labels
    """
    # Check if spatial dimensions match
    if outputs.shape[2:] != labels.shape[2:]:
        print(f"[WARNING] Shape mismatch detected: Output {outputs.shape[2:]} vs Label {labels.shape[2:]}")
        # Create a resize transform to match spatial dimensions
        resize_transform = ResizeWithPadOrCrop(spatial_size=labels.shape[2:])
        
        # Process each sample in the batch
        aligned_outputs = []
        for i in range(outputs.shape[0]):
            # Process each channel separately
            aligned_channels = []
            for c in range(outputs.shape[1]):
                # Add a dummy batch dimension for the transform
                channel = outputs[i:i+1, c:c+1]
                # Apply the transform
                resized = resize_transform(channel)
                aligned_channels.append(resized[:, 0])  # Remove the channel dim that was added
            # Stack the channels
            aligned_sample = torch.stack(aligned_channels, dim=1)
            aligned_outputs.append(aligned_sample)
        
        # Stack all samples back together
        return torch.cat(aligned_outputs, dim=0)
    else:
        # No shape mismatch, return original outputs
        return outputs


def train(model_name="unet3d", max_epochs=100, learning_rate=1e-4, val_interval=1, checkpoint_interval=10, early_stop_patience=5):
    """
    Train the selected model for segmentation
    
    Args:
        model_name: Name of the model to use
        max_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        val_interval: How often to run validation (every N epochs)
        checkpoint_interval: How often to save model checkpoints (every N epochs)
        early_stop_patience: Number of epochs with no improvement after which training will be stopped
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get model factory function based on model name
    try:
        get_model = get_model_from_name(model_name)
        print(f"Successfully imported {model_name} model")
    except ValueError as e:
        print(f"Error: {e}")
        print("Falling back to UNet3D model")
        from models.unet3d import get_model
        model_name = "unet3d"
    
    # Initialize the model and move to device
    model = get_model().to(device)
    print(f"Model architecture: {model_name}")
    
    # Define loss function
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define metric
    metric = DiceMetric(include_background=False, reduction="mean")
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create model-specific subdirectory
    model_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    # Create log file with model name included
    log_filename = os.path.join(model_checkpoint_dir, f"{model_name}_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Dice"])
    
    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []
    
    # Early stopping variables
    patience_counter = 0
    early_stopped = False
    
    print(f"Starting training for {max_epochs} epochs (early stopping patience: {early_stop_patience})...")
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Set model to train mode
        model.train()
        epoch_loss = 0
        step = 0
        
        # Training loop
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            if epoch == 0 and step == 1:
                print(f"[DEBUG] Train Input Shape: {inputs.shape}")
                print(f"[DEBUG] Train Label Shape: {labels.shape}")

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Ensure outputs have the same spatial dimensions as labels before loss calculation
            outputs = ensure_shape_match(outputs, labels)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"\rEpoch {epoch + 1}/{max_epochs} - Training step {step}/{len(train_loader)}", end="")
        
        # Calculate average loss for the epoch
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
        # Print epoch summary
        print(f"\rEpoch {epoch + 1}/{max_epochs} - Training loss: {epoch_loss:.4f}" + " " * 20)
        
        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            step = 0
            
            with torch.no_grad():
                for val_batch_data in val_loader:
                    step += 1
                    val_inputs, val_labels = val_batch_data["image"].to(device), val_batch_data["label"].to(device)
                    
                    if epoch == 0 and step == 1:
                        print(f"[DEBUG] Val Input Shape: {val_inputs.shape}")
                        print(f"[DEBUG] Val Label Shape: {val_labels.shape}")


                    val_outputs = model(val_inputs)
                    
                    # Ensure outputs have the same spatial dimensions as labels before loss calculation
                    val_outputs = ensure_shape_match(val_outputs, val_labels)
                    
                    # Calculate validation loss
                    val_loss_item = loss_function(val_outputs, val_labels).item()
                    val_loss += val_loss_item
                    
                    # Compute metrics
                    # Need to decollate and extract items for metric computation
                    val_outputs_list = decollate_batch(val_outputs)
                    val_labels_list = decollate_batch(val_labels)
                    
                    # Apply softmax to get probability maps
                    val_outputs_list = [torch.softmax(i, dim=0) for i in val_outputs_list]
                    
                    # Ensure model outputs and ground truth labels have the same shape
                    # This is less critical now since we already aligned the tensors above
                    # but we keep it for extra safety at the individual sample level
                    for idx in range(len(val_outputs_list)):
                        # Get spatial dimensions of both output and label
                        output_shape = val_outputs_list[idx].shape[1:]  # Skip channel dim
                        label_shape = val_labels_list[idx].shape[1:]    # Skip channel dim
                        
                        # Check if shapes differ
                        if output_shape != label_shape:
                            print(f"[WARNING] Shape mismatch after decollate: Output {output_shape} vs Label {label_shape}")
                            # Create a resize transform to align shapes
                            resize_transform = ResizeWithPadOrCrop(spatial_size=label_shape)
                            # Apply to each channel separately and recombine
                            resized_channels = []
                            for channel_idx in range(val_outputs_list[idx].shape[0]):
                                # Extract single channel and add a dummy batch dimension
                                channel = val_outputs_list[idx][channel_idx:channel_idx+1]
                                # Apply resize transform
                                resized_channel = resize_transform(channel)
                                resized_channels.append(resized_channel[0])  # Remove dummy batch dim
                            # Stack resized channels
                            val_outputs_list[idx] = torch.stack(resized_channels)
                    
                    # Compute metrics with aligned shapes
                    metric(y_pred=val_outputs_list, y=val_labels_list)

            # Average validation loss
            val_loss /= step
            val_loss_values.append(val_loss)
            
            # Get metric result
            metric_result = metric.aggregate().item()
            metric_values.append(metric_result)
            metric.reset()
            
            # Check if this is the best metric
            if metric_result > best_metric:
                best_metric = metric_result
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(model_checkpoint_dir, f"{model_name}_best_model.pth")
                )
                print(f"New best model saved with Dice: {best_metric:.4f}")
                
                # Reset patience counter since we improved
                patience_counter = 0
            else:
                # Increment early stopping counter
                patience_counter += 1
                print(f"No improvement in Dice score for {patience_counter} epochs")
                
                # Check if we should stop training early
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                    print(f"No improvement in validation Dice score for {early_stop_patience} consecutive epochs.")
                    early_stopped = True
                    
                    # Save final model before early stopping
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_checkpoint_dir, f"{model_name}_early_stopped_model.pth")
                    )
                    print(f"Early stopped model saved as {model_name}_early_stopped_model.pth")
                    break
            
            print(f"Validation loss: {val_loss:.4f}, Dice: {metric_result:.4f}")
            print(f"Best Dice: {best_metric:.4f} at Epoch: {best_metric_epoch}")
            
            # Log to CSV file
            with open(log_filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, epoch_loss, val_loss, metric_result])
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "best_metric": best_metric,
                    "best_metric_epoch": best_metric_epoch
                },
                os.path.join(model_checkpoint_dir, f"{model_name}_checkpoint_epoch_{epoch+1}.pth")
            )
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Training completed (either fully or due to early stopping)
    final_epoch = epoch + 1
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes after {final_epoch} epochs")
    print(f"Best Dice score: {best_metric:.4f} at epoch {best_metric_epoch}")
    
    # Save the final model (if not early stopped)
    if not early_stopped:
        torch.save(
            model.state_dict(),
            os.path.join(model_checkpoint_dir, f"{model_name}_final_model.pth")
        )
        print(f"Final model saved as {model_name}_final_model.pth")
    
    # Return training history for plotting if needed
    return {
        "model_name": model_name,
        "epoch_loss": epoch_loss_values,
        "val_loss": val_loss_values,
        "val_metric": metric_values,
        "best_metric": best_metric,
        "best_metric_epoch": best_metric_epoch,
        "early_stopped": early_stopped,
        "final_epoch": final_epoch,
        "early_stop_patience": early_stop_patience,
        "training_time": total_time
    }


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Train a 3D segmentation model for BTCV dataset")
    parser.add_argument(
        "--model", 
        type=str, 
        default="unet3d",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model architecture to use for training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--val-interval", 
        type=int, 
        default=1,
        help="Validation interval (epochs)"
    )
    parser.add_argument(
        "--checkpoint-interval", 
        type=int, 
        default=10,
        help="Checkpoint saving interval (epochs)"
    )
    parser.add_argument(
        "--early-stop-patience", 
        type=int, 
        default=5,
        help="Number of epochs with no improvement after which training will be stopped"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print training configuration
    print(f"Starting training with configuration:")
    print(f"- Model: {args.model}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Validation interval: {args.val_interval}")
    print(f"- Checkpoint interval: {args.checkpoint_interval}")
    print(f"- Early stopping patience: {args.early_stop_patience}")
    
    # Start training
    train_history = train(
        model_name=args.model,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        val_interval=args.val_interval,
        checkpoint_interval=args.checkpoint_interval,
        early_stop_patience=args.early_stop_patience
    )
    
    # Print additional training summary
    if train_history["early_stopped"]:
        print(f"Training of {args.model} was early stopped at epoch {train_history['final_epoch']} due to no improvement for {args.early_stop_patience} epochs")
    else:
        print(f"Training of {args.model} completed for all {args.epochs} epochs")