import os
import sys
import time
import argparse
import csv
from datetime import datetime
import importlib
import torch
import numpy as np
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, SaveImage, ResizeWithPadOrCrop
from monai.utils import first

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dataloader
from scripts.preprocess_and_dataloader import val_loader

# Dictionary of available models with their module paths (same as in train.py)
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


def evaluate(model_name, checkpoint_path, save_predictions=False):
    """
    Evaluate a trained model on the validation dataset
    
    Args:
        model_name: Name of the model architecture
        checkpoint_path: Path to the model checkpoint file
        save_predictions: Whether to save model predictions to disk
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get model factory function
    try:
        get_model = get_model_from_name(model_name)
        print(f"Successfully imported {model_name} model")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Initialize the model
    model = get_model().to(device)
    print(f"Model initialized: {model_name}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if the checkpoint is a full dictionary or just state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get("epoch", "unknown")
            print(f"Loaded model checkpoint from epoch {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dictionary")
            
        print(f"Checkpoint loaded from: {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        sys.exit(1)
    
    # Create output directory for predictions if needed
    if save_predictions:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "results", "predictions", model_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Will save predictions to: {output_dir}")
        
        # Initialize SaveImage transform
        saver = SaveImage(
            output_dir=output_dir,
            output_postfix="pred",
            output_ext=".nii.gz",
            separate_folder=False,
            resample=False
        )
    
    # Set up metrics
    metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    
    # Post-processing transforms
    post_pred = AsDiscrete(argmax=True, to_onehot=2)  # 2 classes: background and colon cancer
    post_label = AsDiscrete(to_onehot=2)
    
    # Set model to evaluation mode
    model.eval()
    
    # Start evaluation
    print("Starting evaluation...")
    start_time = time.time()
    
    # Store case-wise metrics
    case_metrics = []
    
    with torch.no_grad():
        for step, val_data in enumerate(val_loader):
            # Get validation data
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            
            # Get file name for reporting
            file_name = val_data.get("image_meta_dict", {}).get("filename_or_obj", [f"case_{step}"])
            if isinstance(file_name, list):
                file_name = os.path.basename(file_name[0])
            
            # Forward pass
            val_outputs = model(val_inputs)
            
            # Ensure model outputs and ground truth labels have the same shape
            # Apply shape alignment at the batch level first
            val_outputs = ensure_shape_match(val_outputs, val_labels)
            
            # Apply post-processing
            val_outputs_list = decollate_batch(val_outputs)
            val_labels_list = decollate_batch(val_labels)
            
            # Apply post-processing transforms
            val_outputs_list = [post_pred(p) for p in val_outputs_list]
            val_labels_list = [post_label(l) for l in val_labels_list]
            
            # Ensure model outputs and ground truth labels have the same shape at the individual sample level
            for idx in range(len(val_outputs_list)):
                # Get spatial dimensions of both output and label
                output_shape = val_outputs_list[idx].shape[1:]  # Skip channel dim
                label_shape = val_labels_list[idx].shape[1:]    # Skip channel dim
                
                # Check if shapes differ
                if output_shape != label_shape:
                    print(f"[WARNING] Shape mismatch detected for {file_name}: Output {output_shape} vs Label {label_shape}")
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
            
            # Get metric for this case
            case_dice = metric.aggregate().item()
            metric.reset()
            
            # Save predictions if requested
            if save_predictions:
                # Convert softmax output to class indices
                pred_np = torch.argmax(val_outputs[0], dim=0).detach().cpu().numpy().astype(np.uint8)
                # Attach metadata for saving
                val_data["pred"] = pred_np
                val_data["pred_meta_dict"] = val_data["image_meta_dict"]
                saver(val_data)
            
            # Store case metrics
            case_metrics.append({
                "file": file_name,
                "dice": case_dice
            })
            
            print(f"Case {step + 1}/{len(val_loader)}: {file_name}, Dice: {case_dice:.4f}")
    
    # Calculate overall metrics
    dice_scores = [case["dice"] for case in case_metrics]
    average_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    std_dice = np.std(dice_scores) if len(dice_scores) > 1 else 0
    
    # Total evaluation time
    total_time = time.time() - start_time
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print(f"Number of cases evaluated: {len(case_metrics)}")
    print(f"Average Dice score: {average_dice:.4f} ± {std_dice:.4f}")
    print(f"Min Dice score: {min(dice_scores):.4f}")
    print(f"Max Dice score: {max(dice_scores):.4f}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed metrics to CSV
    metrics_csv_path = os.path.join(results_dir, "metrics.csv")
    
    # Check if file exists to decide whether to write header
    file_exists = os.path.isfile(metrics_csv_path)
    
    with open(metrics_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'model', 'checkpoint', 'average_dice', 'std_dice', 'min_dice', 'max_dice', 'num_cases', 'evaluation_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write metrics row
        writer.writerow({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_name,
            'checkpoint': os.path.basename(checkpoint_path),
            'average_dice': f"{average_dice:.4f}",
            'std_dice': f"{std_dice:.4f}",
            'min_dice': f"{min(dice_scores):.4f}",
            'max_dice': f"{max(dice_scores):.4f}",
            'num_cases': len(case_metrics),
            'evaluation_time': f"{total_time:.2f}"
        })
    
    # Save detailed case metrics
    case_metrics_path = os.path.join(results_dir, f"{model_name}_{os.path.basename(checkpoint_path).split('.')[0]}_case_metrics.csv")
    with open(case_metrics_path, 'w', newline='') as csvfile:
        fieldnames = ['file', 'dice']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for case in case_metrics:
            writer.writerow(case)
    
    # Save summary to text file
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write(f"Number of cases: {len(case_metrics)}\n")
        f.write(f"Average Dice score: {average_dice:.4f} ± {std_dice:.4f}\n")
        f.write(f"Min/Max Dice score: {min(dice_scores):.4f}/{max(dice_scores):.4f}\n")
        f.write(f"Evaluation time: {total_time:.2f} seconds\n")
        f.write(f"{'='*50}\n")
    
    print(f"\nResults saved to:")
    print(f"- {metrics_csv_path}")
    print(f"- {case_metrics_path}")
    print(f"- {summary_path}")
    
    if save_predictions:
        print(f"- Predictions saved to {output_dir}")
    
    return {
        "model_name": model_name,
        "checkpoint": os.path.basename(checkpoint_path),
        "average_dice": average_dice,
        "std_dice": std_dice,
        "min_dice": min(dice_scores),
        "max_dice": max(dice_scores),
        "num_cases": len(case_metrics),
        "evaluation_time": total_time,
        "case_metrics": case_metrics
    }


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the validation dataset")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model architecture to evaluate"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to the model checkpoint file (.pth)"
    )
    parser.add_argument(
        "--save-predictions", 
        action="store_true",
        help="Save model predictions to disk"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print evaluation configuration
    print(f"\nStarting evaluation with configuration:")
    print(f"- Model: {args.model}")
    print(f"- Checkpoint: {args.checkpoint}")
    print(f"- Save predictions: {'Yes' if args.save_predictions else 'No'}")
    
    # Run evaluation
    evaluation_results = evaluate(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        save_predictions=args.save_predictions
    )
    
    print(f"\nEvaluation of {args.model} completed!")