#!/usr/bin/env python3
"""
Patch-based training utilities for focusing on tumor regions.
Addresses extreme class imbalance by intelligently sampling patches.
"""

import torch
import numpy as np
import random
from typing import Tuple, List, Dict, Optional
from monai.data import Dataset
from monai.transforms import Compose, RandCropByPosNegLabeld
import warnings

class TumorFocusedPatchSampler:
    """
    Intelligent patch sampler that balances tumor-positive and tumor-negative patches.
    Implements the patch-based training strategy from tasks.yml.
    """
    
    def __init__(self, 
                 patch_size: Tuple[int, int, int] = (64, 64, 32),
                 samples_per_image: int = 4,
                 pos_neg_ratio: float = 0.7,  # 70% positive patches, 30% negative
                 min_tumor_pixels: int = 50,   # Minimum pixels to consider positive patch
                 background_threshold: float = 0.95):  # Max background % for negative patches
        """
        Args:
            patch_size: Size of patches to extract (D, H, W)
            samples_per_image: Number of patches to extract per image
            pos_neg_ratio: Ratio of positive (tumor-containing) to negative patches
            min_tumor_pixels: Minimum tumor pixels required for positive patch
            background_threshold: Maximum background ratio for negative patches
        """
        self.patch_size = patch_size
        self.samples_per_image = samples_per_image
        self.pos_neg_ratio = pos_neg_ratio
        self.min_tumor_pixels = min_tumor_pixels
        self.background_threshold = background_threshold
        
        # Calculate number of positive vs negative patches
        self.n_pos_patches = int(samples_per_image * pos_neg_ratio)
        self.n_neg_patches = samples_per_image - self.n_pos_patches
        
        print(f"PatchSampler: {self.n_pos_patches} positive + {self.n_neg_patches} negative patches per image")
    
    def find_tumor_centers(self, label_array: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Find centers of tumor regions for targeted patch extraction.
        """
        tumor_coords = np.where(label_array > 0)
        if len(tumor_coords[0]) == 0:
            return []
        
        # Get unique tumor coordinates
        coords = list(zip(tumor_coords[0], tumor_coords[1], tumor_coords[2]))
        
        # If many tumor pixels, subsample to get diverse centers
        if len(coords) > 100:
            coords = random.sample(coords, 100)
        
        return coords
    
    def is_valid_positive_patch(self, label_patch: np.ndarray) -> bool:
        """Check if patch contains enough tumor pixels to be considered positive."""
        tumor_pixels = np.sum(label_patch > 0)
        return tumor_pixels >= self.min_tumor_pixels
    
    def is_valid_negative_patch(self, label_patch: np.ndarray) -> bool:
        """Check if patch is sufficiently background-heavy for negative sample."""
        total_pixels = label_patch.size
        background_pixels = np.sum(label_patch == 0)
        background_ratio = background_pixels / total_pixels
        return background_ratio >= self.background_threshold
    
    def extract_patch_around_center(self, 
                                   image_array: np.ndarray, 
                                   label_array: np.ndarray,
                                   center: Tuple[int, int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract patch centered around given coordinates.
        """
        d, h, w = image_array.shape[-3:]
        pd, ph, pw = self.patch_size
        
        # Calculate patch boundaries
        z_start = max(0, center[0] - pd // 2)
        z_end = min(d, z_start + pd)
        z_start = max(0, z_end - pd)  # Adjust if near boundary
        
        y_start = max(0, center[1] - ph // 2)
        y_end = min(h, y_start + ph)
        y_start = max(0, y_end - ph)
        
        x_start = max(0, center[2] - pw // 2)
        x_end = min(w, x_start + pw)
        x_start = max(0, x_end - pw)
        
        # Extract patches
        if image_array.ndim == 4:  # [C, D, H, W]
            image_patch = image_array[:, z_start:z_end, y_start:y_end, x_start:x_end]
        else:  # [D, H, W]
            image_patch = image_array[z_start:z_end, y_start:y_end, x_start:x_end]
            
        if label_array.ndim == 4:  # [C, D, H, W]
            label_patch = label_array[:, z_start:z_end, y_start:y_end, x_start:x_end]
        else:  # [D, H, W]
            label_patch = label_array[z_start:z_end, y_start:y_end, x_start:x_end]
        
        return image_patch, label_patch
    
    def sample_positive_patches(self, 
                               image_array: np.ndarray, 
                               label_array: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample patches that contain tumor regions.
        """
        positive_patches = []
        tumor_centers = self.find_tumor_centers(label_array.squeeze() if label_array.ndim == 4 else label_array)
        
        if not tumor_centers:
            print("Warning: No tumor pixels found for positive patch sampling")
            return positive_patches
        
        attempts = 0
        max_attempts = self.n_pos_patches * 10  # Allow multiple attempts
        
        while len(positive_patches) < self.n_pos_patches and attempts < max_attempts:
            attempts += 1
            
            # Randomly select a tumor center
            center = random.choice(tumor_centers)
            
            # Add some randomness around the center
            jitter = 10  # pixels
            center_jittered = (
                center[0] + random.randint(-jitter, jitter),
                center[1] + random.randint(-jitter, jitter),
                center[2] + random.randint(-jitter, jitter)
            )
            
            patch = self.extract_patch_around_center(image_array, label_array, center_jittered)
            if patch is None:
                continue
                
            image_patch, label_patch = patch
            
            # Validate patch quality
            label_for_validation = label_patch.squeeze() if label_patch.ndim == 4 else label_patch
            if self.is_valid_positive_patch(label_for_validation):
                positive_patches.append((image_patch, label_patch))
        
        if len(positive_patches) < self.n_pos_patches:
            print(f"Warning: Only found {len(positive_patches)}/{self.n_pos_patches} valid positive patches")
        
        return positive_patches
    
    def sample_negative_patches(self, 
                               image_array: np.ndarray, 
                               label_array: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample patches that are primarily background.
        """
        negative_patches = []
        d, h, w = image_array.shape[-3:]
        pd, ph, pw = self.patch_size
        
        attempts = 0
        max_attempts = self.n_neg_patches * 20
        
        while len(negative_patches) < self.n_neg_patches and attempts < max_attempts:
            attempts += 1
            
            # Random center for negative patch
            center = (
                random.randint(pd//2, d - pd//2),
                random.randint(ph//2, h - ph//2),
                random.randint(pw//2, w - pw//2)
            )
            
            patch = self.extract_patch_around_center(image_array, label_array, center)
            if patch is None:
                continue
                
            image_patch, label_patch = patch
            
            # Validate patch is sufficiently background
            label_for_validation = label_patch.squeeze() if label_patch.ndim == 4 else label_patch
            if self.is_valid_negative_patch(label_for_validation):
                negative_patches.append((image_patch, label_patch))
        
        if len(negative_patches) < self.n_neg_patches:
            print(f"Warning: Only found {len(negative_patches)}/{self.n_neg_patches} valid negative patches")
        
        return negative_patches
    
    def sample_patches_from_volume(self, 
                                  image_array: np.ndarray, 
                                  label_array: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Sample balanced patches from a single volume.
        
        Returns:
            List of patch dictionaries with 'image' and 'label' keys
        """
        # Sample positive and negative patches
        positive_patches = self.sample_positive_patches(image_array, label_array)
        negative_patches = self.sample_negative_patches(image_array, label_array)
        
        # Combine and format
        all_patches = []
        
        # Add positive patches
        for i, (img_patch, lbl_patch) in enumerate(positive_patches):
            all_patches.append({
                'image': img_patch,
                'label': lbl_patch,
                'patch_type': 'positive',
                'patch_id': f'pos_{i}'
            })
        
        # Add negative patches
        for i, (img_patch, lbl_patch) in enumerate(negative_patches):
            all_patches.append({
                'image': img_patch,
                'label': lbl_patch,
                'patch_type': 'negative', 
                'patch_id': f'neg_{i}'
            })
        
        # Shuffle patches to avoid pattern bias
        random.shuffle(all_patches)
        
        return all_patches

class PatchBasedDataset(Dataset):
    """
    Dataset that creates patches on-the-fly for training.
    """
    
    def __init__(self, 
                 base_dataset: Dataset,
                 patch_sampler: TumorFocusedPatchSampler,
                 transform=None):
        """
        Args:
            base_dataset: Original MONAI dataset with full volumes
            patch_sampler: PatchSampler to extract patches
            transform: Optional transforms to apply to patches
        """
        self.base_dataset = base_dataset
        self.patch_sampler = patch_sampler
        self.transform = transform
        
        # Pre-generate patch metadata
        self._generate_patch_metadata()
    
    def _generate_patch_metadata(self):
        """Pre-generate metadata about patches from each volume."""
        self.patch_metadata = []
        
        print("Generating patch metadata...")
        for vol_idx in range(len(self.base_dataset)):
            # Each volume generates multiple patches
            for patch_idx in range(self.patch_sampler.samples_per_image):
                self.patch_metadata.append({
                    'volume_idx': vol_idx,
                    'patch_idx': patch_idx
                })
        
        print(f"Generated {len(self.patch_metadata)} patch samples from {len(self.base_dataset)} volumes")
    
    def __len__(self):
        return len(self.patch_metadata)
    
    def __getitem__(self, idx):
        """Get a specific patch."""
        metadata = self.patch_metadata[idx]
        vol_idx = metadata['volume_idx']
        
        # Get the full volume
        volume_data = self.base_dataset[vol_idx]
        image_array = volume_data['image']
        label_array = volume_data['label']
        
        # Convert to numpy if needed
        if torch.is_tensor(image_array):
            image_array = image_array.numpy()
        if torch.is_tensor(label_array):
            label_array = label_array.numpy()
        
        # Sample patches from this volume
        patches = self.patch_sampler.sample_patches_from_volume(image_array, label_array)
        
        if not patches:
            # Fallback: return a random crop if patch sampling fails
            print(f"Warning: No patches extracted for volume {vol_idx}, using random crop")
            # TODO: Implement fallback random crop
            return volume_data
        
        # Select one patch (could be randomized)
        patch_data = random.choice(patches)
        
        # Convert back to tensors
        patch_sample = {
            'image': torch.from_numpy(patch_data['image']).float(),
            'label': torch.from_numpy(patch_data['label']).long(),
            'patch_type': patch_data['patch_type'],
            'volume_idx': vol_idx
        }
        
        # Apply transforms if provided
        if self.transform:
            patch_sample = self.transform(patch_sample)
        
        return patch_sample

def create_patch_based_loader(original_dataset: Dataset,
                             batch_size: int = 4,
                             patch_size: Tuple[int, int, int] = (64, 64, 32),
                             samples_per_image: int = 4,
                             pos_neg_ratio: float = 0.7,
                             num_workers: int = 0,
                             shuffle: bool = True):
    """
    Factory function to create a patch-based data loader.
    
    Args:
        original_dataset: Original MONAI dataset with full volumes
        batch_size: Batch size for training
        patch_size: Size of patches to extract
        samples_per_image: Number of patches per volume
        pos_neg_ratio: Ratio of positive to negative patches
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader with patch-based sampling
    """
    from torch.utils.data import DataLoader
    
    # Create patch sampler
    patch_sampler = TumorFocusedPatchSampler(
        patch_size=patch_size,
        samples_per_image=samples_per_image,
        pos_neg_ratio=pos_neg_ratio
    )
    
    # Create patch-based dataset
    patch_dataset = PatchBasedDataset(
        base_dataset=original_dataset,
        patch_sampler=patch_sampler
    )
    
    # Create data loader
    patch_loader = DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return patch_loader

# Export key classes
__all__ = [
    'TumorFocusedPatchSampler', 'PatchBasedDataset', 'create_patch_based_loader'
] 