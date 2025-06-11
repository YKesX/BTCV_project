"""
Comprehensive Medical Metrics Suite for 3D Medical Image Segmentation
Implements medical-specific evaluation metrics beyond Dice score for robust validation.
"""

import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion, binary_dilation
from typing import Tuple, Dict, List, Optional
import warnings

class MedicalMetrics:
    """
    Comprehensive medical metrics for 3D segmentation evaluation.
    
    Includes:
    - Hausdorff Distance (HD) and 95th percentile HD
    - Average Symmetric Surface Distance (ASSD)
    - Sensitivity (Recall) and Specificity
    - Positive/Negative Predictive Value
    - Volume Similarity
    - Surface Dice
    """
    
    def __init__(self, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize medical metrics calculator.
        
        Args:
            spacing: Physical spacing between voxels (z, y, x) in mm
        """
        self.spacing = np.array(spacing)
    
    def compute_all_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute all medical metrics for a prediction-target pair.
        
        Args:
            pred: Predicted segmentation (B, 1, D, H, W) or (D, H, W)
            target: Ground truth segmentation (B, 1, D, H, W) or (D, H, W)
            
        Returns:
            Dictionary of all computed metrics
        """
        # Convert to numpy and ensure binary
        pred_np = self._prepare_mask(pred)
        target_np = self._prepare_mask(target)
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics.update(self._compute_basic_metrics(pred_np, target_np))
            
            # Distance metrics (only if both masks have positive voxels)
            if np.any(pred_np) and np.any(target_np):
                metrics.update(self._compute_distance_metrics(pred_np, target_np))
            else:
                # Handle empty predictions/targets
                metrics.update({
                    'hausdorff_distance': float('inf') if not np.any(target_np) else 100.0,
                    'hausdorff_distance_95': float('inf') if not np.any(target_np) else 100.0,
                    'average_surface_distance': float('inf') if not np.any(target_np) else 100.0,
                    'surface_dice': 0.0
                })
            
            # Volume metrics
            metrics.update(self._compute_volume_metrics(pred_np, target_np))
            
        except Exception as e:
            warnings.warn(f"Error computing metrics: {e}")
            # Return default values
            metrics = self._get_default_metrics()
        
        return metrics
    
    def _prepare_mask(self, mask: torch.Tensor) -> np.ndarray:
        """Convert tensor to binary numpy array."""
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        # Handle different tensor shapes - be more careful with squeezing
        while len(mask.shape) > 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # Remove batch dimensions only if size 1
        while len(mask.shape) > 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # Remove trailing dimensions only if size 1
            
        # If still more than 3D, take the first 3D slice
        if len(mask.shape) > 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[0, 0] if mask.shape[1] == 1 else mask[0, 0, 0]
        
        return (mask > 0.5).astype(np.uint8)
    
    def _compute_basic_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics."""
        # Flatten for computation
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Confusion matrix components
        tp = np.sum((pred_flat == 1) & (target_flat == 1))
        tn = np.sum((pred_flat == 0) & (target_flat == 0))
        fp = np.sum((pred_flat == 1) & (target_flat == 0))
        fn = np.sum((pred_flat == 0) & (target_flat == 1))
        
        # Avoid division by zero
        epsilon = 1e-8
        
        # Basic metrics
        sensitivity = tp / (tp + fn + epsilon)  # Recall
        specificity = tn / (tn + fp + epsilon)
        precision = tp / (tp + fp + epsilon)    # PPV
        npv = tn / (tn + fn + epsilon)          # NPV
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        
        # Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn + epsilon)
        
        # Jaccard/IoU
        jaccard = tp / (tp + fp + fn + epsilon)
        
        return {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'negative_predictive_value': float(npv),
            'accuracy': float(accuracy),
            'dice_coefficient': float(dice),
            'jaccard_index': float(jaccard)
        }
    
    def _compute_distance_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute distance-based metrics."""
        # Get surface points
        pred_surface = self._get_surface_points(pred)
        target_surface = self._get_surface_points(target)
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return {
                'hausdorff_distance': 100.0,
                'hausdorff_distance_95': 100.0,
                'average_surface_distance': 100.0,
                'surface_dice': 0.0
            }
        
        # Apply physical spacing
        pred_surface_mm = pred_surface * self.spacing
        target_surface_mm = target_surface * self.spacing
        
        # Compute distances
        distances_pred_to_target = cdist(pred_surface_mm, target_surface_mm).min(axis=1)
        distances_target_to_pred = cdist(target_surface_mm, pred_surface_mm).min(axis=1)
        
        # Hausdorff Distance
        hd = max(distances_pred_to_target.max(), distances_target_to_pred.max())
        
        # 95th percentile Hausdorff Distance
        hd95 = max(np.percentile(distances_pred_to_target, 95),
                   np.percentile(distances_target_to_pred, 95))
        
        # Average Symmetric Surface Distance
        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        assd = np.mean(all_distances)
        
        # Surface Dice (tolerance = 1mm)
        tolerance = 1.0
        surface_dice = self._compute_surface_dice(pred, target, tolerance)
        
        return {
            'hausdorff_distance': float(hd),
            'hausdorff_distance_95': float(hd95),
            'average_surface_distance': float(assd),
            'surface_dice': float(surface_dice)
        }
    
    def _get_surface_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract surface/boundary points from binary mask."""
        # Erode mask to find boundary
        eroded = binary_erosion(mask)
        boundary = mask.astype(bool) & ~eroded
        
        # Get coordinates of boundary points
        surface_points = np.argwhere(boundary)
        return surface_points
    
    def _compute_surface_dice(self, pred: np.ndarray, target: np.ndarray, tolerance: float) -> float:
        """
        Compute Surface Dice coefficient within tolerance.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask  
            tolerance: Distance tolerance in mm
        """
        pred_surface = self._get_surface_points(pred)
        target_surface = self._get_surface_points(target)
        
        if len(pred_surface) == 0 and len(target_surface) == 0:
            return 1.0
        elif len(pred_surface) == 0 or len(target_surface) == 0:
            return 0.0
        
        # Apply spacing
        pred_surface_mm = pred_surface * self.spacing
        target_surface_mm = target_surface * self.spacing
        
        # Count surface points within tolerance
        distances_pred = cdist(pred_surface_mm, target_surface_mm).min(axis=1)
        distances_target = cdist(target_surface_mm, pred_surface_mm).min(axis=1)
        
        pred_within_tolerance = np.sum(distances_pred <= tolerance)
        target_within_tolerance = np.sum(distances_target <= tolerance)
        
        # Surface Dice
        surface_dice = (pred_within_tolerance + target_within_tolerance) / (len(pred_surface) + len(target_surface))
        
        return surface_dice
    
    def _compute_volume_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute volume-based metrics."""
        pred_volume = np.sum(pred) * np.prod(self.spacing)
        target_volume = np.sum(target) * np.prod(self.spacing)
        
        # Volume similarity
        if target_volume > 0:
            volume_similarity = 1 - abs(pred_volume - target_volume) / target_volume
        else:
            volume_similarity = 1.0 if pred_volume == 0 else 0.0
        
        # Relative volume error
        if target_volume > 0:
            relative_volume_error = (pred_volume - target_volume) / target_volume
        else:
            relative_volume_error = 0.0 if pred_volume == 0 else float('inf')
        
        return {
            'volume_similarity': float(volume_similarity),
            'relative_volume_error': float(relative_volume_error),
            'predicted_volume_mm3': float(pred_volume),
            'target_volume_mm3': float(target_volume)
        }
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metric values for error cases."""
        return {
            'sensitivity': 0.0,
            'specificity': 1.0,
            'precision': 0.0,
            'negative_predictive_value': 1.0,
            'accuracy': 0.0,
            'dice_coefficient': 0.0,
            'jaccard_index': 0.0,
            'hausdorff_distance': 100.0,
            'hausdorff_distance_95': 100.0,
            'average_surface_distance': 100.0,
            'surface_dice': 0.0,
            'volume_similarity': 0.0,
            'relative_volume_error': 0.0,
            'predicted_volume_mm3': 0.0,
            'target_volume_mm3': 0.0
        }


class MetricAggregator:
    """Aggregate metrics across multiple samples/batches."""
    
    def __init__(self):
        self.metrics_list = []
    
    def add_sample(self, metrics: Dict[str, float]):
        """Add metrics from a single sample."""
        self.metrics_list.append(metrics)
    
    def compute_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics across all samples.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not self.metrics_list:
            return {}
        
        # Get all metric names
        metric_names = list(self.metrics_list[0].keys())
        
        summary = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in self.metrics_list if not np.isinf(m[metric_name])]
            
            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
            else:
                summary[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 
                    'max': 0.0, 'median': 0.0, 'count': 0
                }
        
        return summary
    
    def reset(self):
        """Clear all stored metrics."""
        self.metrics_list = []


# Convenience functions for easy integration
def compute_medical_metrics(pred: torch.Tensor, 
                          target: torch.Tensor,
                          spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    Convenience function to compute all medical metrics.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        spacing: Physical voxel spacing (z, y, x) in mm
    
    Returns:
        Dictionary of computed metrics
    """
    calculator = MedicalMetrics(spacing=spacing)
    return calculator.compute_all_metrics(pred, target)


def evaluate_model_medical_metrics(model, dataloader, device, spacing=(1.0, 1.0, 1.0)):
    """
    Evaluate model using comprehensive medical metrics.
    
    Args:
        model: Trained model
        dataloader: Validation/test dataloader
        device: Device to run inference on
        spacing: Physical voxel spacing
    
    Returns:
        Summary statistics of all metrics
    """
    model.eval()
    aggregator = MetricAggregator()
    calculator = MedicalMetrics(spacing=spacing)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            
            # Compute metrics for each sample in batch
            for i in range(predictions.shape[0]):
                metrics = calculator.compute_all_metrics(
                    predictions[i], labels[i]
                )
                aggregator.add_sample(metrics)
    
    return aggregator.compute_summary()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Medical Metrics Suite...")
    
    # Create dummy data
    torch.manual_seed(42)
    pred = torch.rand(1, 1, 32, 64, 64) > 0.7  # Sparse prediction
    target = torch.rand(1, 1, 32, 64, 64) > 0.8  # Even sparser target
    
    # Test metrics computation
    spacing = (2.0, 1.0, 1.0)  # Example spacing in mm
    metrics = compute_medical_metrics(pred, target, spacing)
    
    print("\nComputed Medical Metrics:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:25}: {value:.4f}")
    
    print("\n✅ Medical Metrics Suite created successfully!")
    print("Features:")
    print("- Hausdorff Distance & 95th percentile HD")
    print("- Average Symmetric Surface Distance") 
    print("- Sensitivity, Specificity, PPV, NPV")
    print("- Surface Dice coefficient")
    print("- Volume similarity metrics")
    print("- Robust error handling")
    print("- Physical spacing awareness") 