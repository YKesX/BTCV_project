#!/usr/bin/env python3
"""
Advanced loss functions for medical image segmentation with extreme class imbalance.
Implements FocalLoss, TverskyLoss, and compound loss strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Paper: "Focal Loss for Dense Object Detection" - Lin et al.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C, H, W, D] - logits
            targets: [N, H, W, D] or [N, 1, H, W, D] - class indices
        """
        # Handle targets with channel dimension
        if targets.dim() == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # [N, 1, H, W, D] -> [N, H, W, D]
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets.long(), num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # [N, C, H, W, D]
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probability of the correct class
        p_t = probs * targets_one_hot
        p_t = p_t.sum(dim=1)  # [N, H, W, D]
        
        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting for class balance
        if self.alpha is not None:
            # For binary: alpha for class 1, (1-alpha) for class 0
            alpha_weight = torch.ones_like(targets, dtype=torch.float)
            alpha_weight[targets == 1] = self.alpha
            alpha_weight[targets == 0] = 1 - self.alpha
            focal_weight = alpha_weight * focal_weight
        
        # Apply focal weighting
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling class imbalance by emphasizing false negatives.
    Paper: "Tversky loss function for image segmentation using 3D fully convolutional deep networks"
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for False Positives
        self.beta = beta    # Weight for False Negatives (higher = more recall-focused)
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C, H, W, D] - logits or probabilities
            targets: [N, H, W, D] or [N, 1, H, W, D] - class indices
        """
        # Handle targets with channel dimension
        if targets.dim() == 5 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # [N, 1, H, W, D] -> [N, H, W, D]
        # Convert logits to probabilities if needed
        if inputs.shape[1] > 1:
            probs = F.softmax(inputs, dim=1)
            # Focus on foreground class (class 1)
            probs = probs[:, 1]  # [N, H, W, D]
        else:
            probs = torch.sigmoid(inputs.squeeze(1))
            
        # Convert targets to binary (background=0, foreground=1)
        targets = (targets > 0).float()
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Tversky components
        true_pos = (probs * targets).sum()
        false_pos = (probs * (1 - targets)).sum()
        false_neg = ((1 - probs) * targets).sum()
        
        # Tversky index
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        
        # Return Tversky loss (1 - Tversky index)
        return 1 - tversky

class FocalDiceLoss(nn.Module):
    """
    Combination of Focal Loss and Dice Loss for extreme class imbalance.
    """
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, dice_smooth=1e-6, 
                 focal_weight=0.6, dice_weight=0.4):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth_nr=dice_smooth, smooth_dr=dice_smooth, 
                                  include_background=False, sigmoid=False, softmax=True)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        # FocalLoss handles dimension conversion internally
        focal = self.focal_loss(inputs, targets)
        
        # DiceLoss expects targets with channel dimension
        dice_targets = targets  # Keep original shape for DiceLoss
        dice = self.dice_loss(inputs, dice_targets)
        return self.focal_weight * focal + self.dice_weight * dice

class CompoundMedicalLoss(nn.Module):
    """
    Compound loss function optimized for medical image segmentation with extreme class imbalance.
    Combines FocalDice + Tversky losses as specified in tasks.yml
    """
    def __init__(self, 
                 focal_alpha=0.25, focal_gamma=2.0,
                 tversky_alpha=0.3, tversky_beta=0.7,
                 focal_dice_weight=0.4, tversky_weight=0.6,
                 class_weights=None):
        super(CompoundMedicalLoss, self).__init__()
        
        self.focal_dice_loss = FocalDiceLoss(
            focal_alpha=focal_alpha, 
            focal_gamma=focal_gamma,
            focal_weight=0.6,  # Within FocalDice: 60% focal, 40% dice
            dice_weight=0.4
        )
        
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        
        self.focal_dice_weight = focal_dice_weight
        self.tversky_weight = tversky_weight
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C, H, W, D] - model logits
            targets: [N, H, W, D] - ground truth labels
        """
        focal_dice = self.focal_dice_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        
        total_loss = (self.focal_dice_weight * focal_dice + 
                     self.tversky_weight * tversky)
        
        return total_loss

class WeightedCompoundLoss(nn.Module):
    """
    Automatically weighted compound loss based on actual class distribution in the batch.
    """
    def __init__(self):
        super(WeightedCompoundLoss, self).__init__()
        self.compound_loss = CompoundMedicalLoss()
        
    def forward(self, inputs, targets):
        # Calculate actual class weights from the batch
        targets_flat = targets.view(-1)
        bg_count = (targets_flat == 0).sum().float()
        fg_count = (targets_flat == 1).sum().float()
        total_count = targets_flat.numel()
        
        # Calculate class ratios
        bg_ratio = bg_count / total_count
        fg_ratio = fg_count / total_count
        
        # Adjust Tversky parameters based on class imbalance
        # More imbalance = higher beta (emphasize recall more)
        if fg_ratio < 0.05:  # Less than 5% foreground
            tversky_alpha = 0.2
            tversky_beta = 0.8
        elif fg_ratio < 0.15:  # Less than 15% foreground
            tversky_alpha = 0.3
            tversky_beta = 0.7
        else:
            tversky_alpha = 0.4
            tversky_beta = 0.6
            
        # Create dynamic loss with adjusted parameters
        dynamic_loss = CompoundMedicalLoss(
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            focal_dice_weight=0.4,
            tversky_weight=0.6
        )
        
        return dynamic_loss(inputs, targets)

def get_loss_function(loss_type='compound', **kwargs):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        loss_type: 'focal', 'tversky', 'focal_dice', 'compound', 'weighted_compound'
        **kwargs: Parameters for the specific loss function
    """
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'focal_dice':
        return FocalDiceLoss(**kwargs)
    elif loss_type == 'compound':
        return CompoundMedicalLoss(**kwargs)
    elif loss_type == 'weighted_compound':
        return WeightedCompoundLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# Export key classes
__all__ = [
    'FocalLoss', 'TverskyLoss', 'FocalDiceLoss', 
    'CompoundMedicalLoss', 'WeightedCompoundLoss', 'get_loss_function'
] 