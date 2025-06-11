"""
Comprehensive Model Evaluation with Medical Metrics
Evaluates trained models using the full suite of medical validation metrics.
"""

import torch
import numpy as np
import json
import argparse
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.medical_metrics import MedicalMetrics, MetricAggregator, evaluate_model_medical_metrics
from scripts.preprocess_and_dataloader import train_loader, val_loader
from models.unet3d import get_model as get_unet3d_model
from models.resunet3d import get_model as get_resunet3d_model

def load_trained_model(model_factory, model_path, device):
    """Load a trained model from checkpoint."""
    model = model_factory().to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully")
    else:
        print(f"⚠️ No checkpoint found at {model_path}, using randomly initialized model")
    
    return model

def evaluate_model_comprehensive(model, dataloader, device, model_name, spacing=(1.0, 1.0, 1.0)):
    """
    Comprehensive evaluation of a model using medical metrics.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        model_name: Name of the model for reporting
        spacing: Physical voxel spacing in mm
    
    Returns:
        Dictionary of evaluation results
    """
    print(f"\n🔬 Evaluating {model_name} with comprehensive medical metrics...")
    
    model.eval()
    aggregator = MetricAggregator()
    calculator = MedicalMetrics(spacing=spacing)
    
    sample_count = 0
    total_samples = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(images)
            # Take channel 1 (tumor) and apply sigmoid + threshold
            predictions = torch.sigmoid(outputs[:, 1:2]) > 0.5
            # Also get tumor channel from labels (channel 1)
            tumor_labels = labels[:, 1:2]
            
            # Compute metrics for each sample in batch
            batch_size = predictions.shape[0]
            for i in range(batch_size):
                sample_count += 1
                
                try:
                    metrics = calculator.compute_all_metrics(
                        predictions[i], tumor_labels[i]
                    )
                    aggregator.add_sample(metrics)
                    
                    # Progress reporting
                    if sample_count % 10 == 0:
                        print(f"  Processed {sample_count}/{total_samples * batch_size} samples...")
                        
                except Exception as e:
                    print(f"  ⚠️ Error processing sample {sample_count}: {e}")
                    continue
    
    # Compute summary statistics
    summary = aggregator.compute_summary()
    
    print(f"✅ Completed evaluation of {model_name}")
    print(f"   Total samples processed: {sample_count}")
    
    return {
        'model_name': model_name,
        'total_samples': sample_count,
        'spacing_mm': spacing,
        'metrics_summary': summary,
        'evaluation_timestamp': datetime.now().isoformat()
    }

def print_metrics_summary(results):
    """Print a formatted summary of evaluation results."""
    model_name = results['model_name']
    metrics = results['metrics_summary']
    
    print(f"\n" + "="*60)
    print(f"📊 COMPREHENSIVE MEDICAL METRICS - {model_name}")
    print(f"="*60)
    print(f"Samples Evaluated: {results['total_samples']}")
    print(f"Voxel Spacing: {results['spacing_mm']} mm")
    print(f"Evaluation Time: {results['evaluation_timestamp']}")
    print()
    
    # Group metrics by category
    basic_metrics = ['dice_coefficient', 'jaccard_index', 'sensitivity', 'specificity', 'precision', 'accuracy']
    distance_metrics = ['hausdorff_distance', 'hausdorff_distance_95', 'average_surface_distance', 'surface_dice']
    volume_metrics = ['volume_similarity', 'relative_volume_error']
    
    def print_metric_group(title, metric_names):
        print(f"{title}:")
        print("-" * 40)
        for metric in metric_names:
            if metric in metrics:
                mean_val = metrics[metric]['mean']
                std_val = metrics[metric]['std']
                print(f"  {metric:25}: {mean_val:8.4f} ± {std_val:6.4f}")
        print()
    
    print_metric_group("🎯 Classification Metrics", basic_metrics)
    print_metric_group("📏 Distance Metrics (mm)", distance_metrics)  
    print_metric_group("📦 Volume Metrics", volume_metrics)
    
    # Highlight key metrics
    if 'dice_coefficient' in metrics:
        dice_mean = metrics['dice_coefficient']['mean']
        dice_std = metrics['dice_coefficient']['std']
        print(f"🏆 KEY RESULT - Dice Score: {dice_mean:.4f} ± {dice_std:.4f}")
    
    if 'hausdorff_distance_95' in metrics:
        hd95_mean = metrics['hausdorff_distance_95']['mean']
        hd95_std = metrics['hausdorff_distance_95']['std']
        print(f"🏆 KEY RESULT - HD95: {hd95_mean:.2f} ± {hd95_std:.2f} mm")

def save_results(results, output_dir="results/medical_metrics"):
    """Save evaluation results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = results['model_name'].lower().replace(' ', '_')
    filename = f"medical_metrics_{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {filepath}")
    return filepath

def compare_models(results_list):
    """Compare multiple model results."""
    if len(results_list) < 2:
        return
    
    print(f"\n" + "="*80)
    print(f"🏁 MODEL COMPARISON")
    print(f"="*80)
    
    # Key metrics for comparison
    key_metrics = ['dice_coefficient', 'hausdorff_distance_95', 'sensitivity', 'specificity']
    
    print(f"{'Model':15} {'Dice':>10} {'HD95(mm)':>10} {'Sensitivity':>12} {'Specificity':>12}")
    print("-" * 65)
    
    for results in results_list:
        model_name = results['model_name'][:14]  # Truncate if too long
        metrics = results['metrics_summary']
        
        dice = metrics.get('dice_coefficient', {}).get('mean', 0.0)
        hd95 = metrics.get('hausdorff_distance_95', {}).get('mean', 999.0)
        sens = metrics.get('sensitivity', {}).get('mean', 0.0)
        spec = metrics.get('specificity', {}).get('mean', 0.0)
        
        print(f"{model_name:15} {dice:10.4f} {hd95:10.2f} {sens:12.4f} {spec:12.4f}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Medical Metrics Evaluation')
    parser.add_argument('--models', nargs='+', choices=['unet3d', 'resunet3d'], 
                       default=['unet3d'], help='Models to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for evaluation')
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                       help='Voxel spacing in mm (z y x)')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory containing model checkpoints')
    parser.add_argument('--use-val-set', action='store_true', default=True,
                       help='Use validation set (default: True)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Comprehensive Medical Metrics Evaluation")
    print(f"Device: {args.device}")
    print(f"Models: {args.models}")
    print(f"Voxel Spacing: {args.spacing} mm")
    
    # Choose dataset
    dataloader = val_loader if args.use_val_set else train_loader
    dataset_name = "Validation" if args.use_val_set else "Training"
    print(f"Dataset: {dataset_name} ({len(dataloader)} batches)")
    
    results_list = []
    
    # Evaluate each model
    for model_name in args.models:
        try:
            if model_name == 'unet3d':
                model_factory = get_unet3d_model
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoints', 'unet3d', 'unet3d_best_model.pth')
                display_name = "UNet3D"
            elif model_name == 'resunet3d':
                model_factory = get_resunet3d_model
                checkpoint_path = os.path.join(args.checkpoint_dir, 'resunet3d_best_model.pth')
                display_name = "ResUNet3D"
            else:
                print(f"⚠️ Unknown model: {model_name}")
                continue
            
            # Load and evaluate model
            model = load_trained_model(model_factory, checkpoint_path, args.device)
            results = evaluate_model_comprehensive(
                model, dataloader, args.device, display_name, tuple(args.spacing)
            )
            
            # Print and save results
            print_metrics_summary(results)
            save_results(results)
            results_list.append(results)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # Compare models if multiple were evaluated
    if len(results_list) > 1:
        compare_models(results_list)
    
    print(f"\n✅ Evaluation completed! {len(results_list)} models evaluated.")

if __name__ == "__main__":
    main() 