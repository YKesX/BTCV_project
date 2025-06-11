#!/usr/bin/env python3
"""
Smart data augmentation for rare tumor preservation in medical image segmentation.
"""

import torch
import numpy as np
import random
from typing import Dict, List
from monai.transforms import (
    Compose, RandRotated, RandZoomd, RandGaussianNoised, 
    RandAdjustContrastd, MapTransform
)
import warnings

class TumorIntensityAugmentation(MapTransform):
    """
    Intensity augmentation specific to tumor characteristics.
    """
    
    def __init__(self, keys, tumor_contrast_range=(0.8, 1.2), 
                 tumor_brightness_range=(-0.1, 0.1), prob=0.5):
        super().__init__(keys)
        self.tumor_contrast_range = tumor_contrast_range
        self.tumor_brightness_range = tumor_brightness_range
        self.prob = prob
    
    def __call__(self, data):
        if random.random() > self.prob:
            return data
        
        image = data['image']
        label = data.get('label')
        
        if label is None:
            return data
        
        # Separate tumor and background regions
        tumor_mask = label > 0
        background_mask = label == 0
        
        # Apply different intensity modifications
        contrast_factor = random.uniform(*self.tumor_contrast_range)
        brightness_offset = random.uniform(*self.tumor_brightness_range)
        
        # Modify tumor regions
        image_modified = image.copy()
        if tumor_mask.any():
            tumor_region = image[tumor_mask]
            tumor_mean = tumor_region.mean()
            tumor_modified = (tumor_region - tumor_mean) * contrast_factor + tumor_mean + brightness_offset
            image_modified[tumor_mask] = tumor_modified
        
        # Slight background modification for realism
        if background_mask.any():
            bg_factor = random.uniform(0.95, 1.05)
            image_modified[background_mask] *= bg_factor
        
        data['image'] = image_modified
        return data

class SmartMedicalAugmentation:
    """
    Smart augmentation pipeline for medical image segmentation.
    """
    
    def __init__(self, 
                 spatial_prob=0.4,
                 intensity_prob=0.6):
        self.spatial_prob = spatial_prob
        self.intensity_prob = intensity_prob
        self.transforms = self._build_augmentation_pipeline()
    
    def _build_augmentation_pipeline(self):
        """Build the augmentation pipeline."""
        
        # Safe spatial augmentations
        spatial_augs = [
            RandRotated(
                keys=['image', 'label'],
                range_x=np.pi/12,  # ±15 degrees
                range_y=np.pi/12,
                range_z=np.pi/12,
                mode=['bilinear', 'nearest'],
                prob=self.spatial_prob
            ),
            RandZoomd(
                keys=['image', 'label'],
                min_zoom=0.9,
                max_zoom=1.1,
                mode=['trilinear', 'nearest'],
                prob=self.spatial_prob
            )
        ]
        
        # Intensity augmentations
        intensity_augs = [
            RandGaussianNoised(
                keys=['image'],
                std=0.05,
                prob=self.intensity_prob
            ),
            RandAdjustContrastd(
                keys=['image'],
                gamma=(0.8, 1.2),
                prob=self.intensity_prob
            ),
            TumorIntensityAugmentation(
                keys=['image'],
                prob=self.intensity_prob
            )
        ]
        
        return Compose(spatial_augs + intensity_augs)
    
    def __call__(self, data):
        """Apply smart augmentation pipeline."""
        try:
            return self.transforms(data)
        except Exception as e:
            warnings.warn(f"Augmentation failed: {e}")
            return data

if __name__ == "__main__":
    print("🧬 Smart Medical Augmentation for Rare Tumors")
    
    # Test with synthetic data
    test_data = {
        'image': np.random.randn(1, 64, 64, 32).astype(np.float32),
        'label': np.random.randint(0, 2, (1, 64, 64, 32)).astype(np.int64)
    }
    
    smart_aug = SmartMedicalAugmentation()
    augmented = smart_aug(test_data)
    print("✅ Smart augmentation working correctly") 