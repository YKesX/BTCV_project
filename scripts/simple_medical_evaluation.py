"""
Simple Medical Evaluation Script
Focus on robust metrics computation without complex surface analysis.
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

from scripts.preprocess_and_dataloader import train_loader, val_loader
from models.unet3d import get_model as get_unet3d_model
from models.resunet3d import get_model as get_resunet3d_model

def compute_basic_metrics(pred, target):
    """Compute basic medical metrics safely."""
    # Ensure binary and flatten
    pred_flat = (pred > 0.5).flatten().astype(np.uint8)
    target_flat = (target > 0.5).flatten().astype(np.uint8)
    
    # Confusion matrix
    tp = np.sum((pred_flat == 1) & (target_flat == 1))
    tn = np.sum((pred_flat == 0) & (target_flat == 0))
    fp = np.sum((pred_flat == 1) & (target_flat == 0))
    fn = np.sum((pred_flat == 0) & (target_flat == 1))
    
    epsilon = 1e-8
    
    # Basic metrics
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    precision = tp / (tp + fp + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    jaccard = tp / (tp + fp + fn + epsilon)
    
    return {
        'dice': float(dice),
        'jaccard': float(jaccard),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn)
    }

def load_trained_model(model_factory, model_path, device):
    """Load a trained model from checkpoint."""
    model = model_factory().to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("Using randomly initialized model")
    else:
        print(f"⚠️ No checkpoint found at {model_path}, using randomly initialized model")
    
    return model

def evaluate_model(model, dataloader, device, model_name):
    """Evaluate model with robust metrics computation."""
    print(f"\n🔬 Evaluating {model_name}...")
    
    model.eval()
    all_metrics = []
    
    total_samples = 0
    successful_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get tumor predictions (channel 1) and labels (single channel contains tumor mask)
            pred_tumor = torch.sigmoid(outputs[:, 1:2]) > 0.5
            label_tumor = labels > 0.5  # Labels already contain tumor mask directly
            
            batch_size = pred_tumor.shape[0]
            for i in range(batch_size):
                total_samples += 1
                
                try:
                    # Convert to numpy
                    pred_np = pred_tumor[i].cpu().numpy().squeeze()
                    label_np = label_tumor[i].cpu().numpy().squeeze()
                    
                    # Compute metrics
                    metrics = compute_basic_metrics(pred_np, label_np)
                    all_metrics.append(metrics)
                    successful_samples += 1
                    
                    if total_samples % 5 == 0:
                        print(f"  Processed {total_samples} samples, Dice: {metrics['dice']:.4f}")
                        
                except Exception as e:
                    print(f"  ⚠️ Error processing sample {total_samples}: {e}")
                    continue
    
    print(f"✅ Completed {model_name}: {successful_samples}/{total_samples} samples processed")
    
    # Compute summary statistics
    if all_metrics:
        summary = {}
        metric_names = list(all_metrics[0].keys())
        
        for metric in metric_names:
            values = [m[metric] for m in all_metrics]
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return {
            'model_name': model_name,
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'metrics_summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    else:
        return {
            'model_name': model_name,
            'total_samples': total_samples,
            'successful_samples': 0,
            'metrics_summary': {},
            'timestamp': datetime.now().isoformat()
        }

def print_results(results):
    """Print formatted results."""
    print(f"\n" + "="*60)
    print(f"📊 MEDICAL EVALUATION RESULTS - {results['model_name']}")
    print(f"="*60)
    print(f"Total Samples: {results['total_samples']}")
    print(f"Successful: {results['successful_samples']}")
    print(f"Success Rate: {results['successful_samples']/results['total_samples']*100:.1f}%")
    print()
    
    if results['metrics_summary']:
        metrics = results['metrics_summary']
        
        # Key metrics
        key_metrics = ['dice', 'jaccard', 'sensitivity', 'specificity', 'precision', 'accuracy']
        
        print("🎯 Key Metrics:")
        print("-" * 40)
        for metric in key_metrics:
            if metric in metrics:
                mean_val = metrics[metric]['mean']
                std_val = metrics[metric]['std']
                print(f"  {metric.capitalize():12}: {mean_val:8.4f} ± {std_val:6.4f}")
        
        # Highlight Dice score
        if 'dice' in metrics:
            dice_mean = metrics['dice']['mean']
            print(f"\n🏆 DICE SCORE: {dice_mean:.4f}")
    else:
        print("❌ No successful metric computations")

def save_results(results, output_dir="results/simple_medical_metrics"):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = results['model_name'].lower().replace(' ', '_')
    filename = f"simple_metrics_{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {filepath}")

def compare_models(results_list):
    """Compare multiple models."""
    if len(results_list) < 2:
        return
    
    print(f"\n" + "="*70)
    print(f"🏁 MODEL COMPARISON")
    print(f"="*70)
    
    print(f"{'Model':15} {'Dice':>8} {'Jaccard':>8} {'Sensitivity':>11} {'Specificity':>11}")
    print("-" * 60)
    
    for results in results_list:
        model_name = results['model_name'][:14]
        metrics = results['metrics_summary']
        
        if metrics:
            dice = metrics.get('dice', {}).get('mean', 0.0)
            jaccard = metrics.get('jaccard', {}).get('mean', 0.0)
            sens = metrics.get('sensitivity', {}).get('mean', 0.0)
            spec = metrics.get('specificity', {}).get('mean', 0.0)
            
            print(f"{model_name:15} {dice:8.4f} {jaccard:8.4f} {sens:11.4f} {spec:11.4f}")
        else:
            print(f"{model_name:15} {'FAILED':>8} {'FAILED':>8} {'FAILED':>11} {'FAILED':>11}")

def main():
    parser = argparse.ArgumentParser(description='Simple Medical Metrics Evaluation')
    parser.add_argument('--models', nargs='+', choices=['unet3d', 'resunet3d'], 
                       default=['unet3d'], help='Models to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("🚀 Starting Simple Medical Evaluation")
    print(f"Device: {args.device}")
    print(f"Models: {args.models}")
    
    # Use validation set
    dataloader = val_loader
    print(f"Dataset: Validation ({len(dataloader)} batches)")
    
    results_list = []
    
    # Evaluate each model
    for model_name in args.models:
        try:
            if model_name == 'unet3d':
                model_factory = get_unet3d_model
                checkpoint_path = "./checkpoints/unet3d/unet3d_best_model.pth"
                display_name = "UNet3D"
            elif model_name == 'resunet3d':
                model_factory = get_resunet3d_model
                checkpoint_path = "./resunet3d_best_model.pth"
                display_name = "ResUNet3D"
            
            # Load and evaluate
            model = load_trained_model(model_factory, checkpoint_path, args.device)
            results = evaluate_model(model, dataloader, args.device, display_name)
            
            # Print and save
            print_results(results)
            save_results(results)
            results_list.append(results)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # Compare models
    if len(results_list) > 1:
        compare_models(results_list)
    
    print(f"\n✅ Evaluation completed! {len(results_list)} models evaluated.")

if __name__ == "__main__":
    main() 