
# Optimized Preprocessing Pipeline
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped, ToTensord
)

# Optimized transforms with reduced redundancy
def get_optimized_transforms():
    """Get memory and speed optimized transforms."""
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Optimized spacing - only once
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.0, 1.0), mode=("bilinear", "nearest")),
        # Efficient intensity scaling
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        # Smart cropping
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Memory-efficient random cropping
        RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label",
            spatial_size=(96, 96, 64),
            pos=2, neg=1,  # 2:1 ratio of positive to negative patches
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        # Light augmentation for speed
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        # Optimized data types
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float16),  # Half precision
        EnsureTyped(keys=["label"], data_type="tensor", dtype=torch.uint8),   # Byte labels
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Fixed crop for validation
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label", 
            spatial_size=(96, 96, 64),
            pos=1, neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        EnsureTyped(keys=["image"], data_type="tensor", dtype=torch.float16),
        EnsureTyped(keys=["label"], data_type="tensor", dtype=torch.uint8),
    ])
    
    return train_transforms, val_transforms

# Optimized DataLoader settings
def get_optimized_dataloaders(train_ds, val_ds):
    """Get optimized dataloaders with improved performance."""
    
    train_loader = DataLoader(
        train_ds,
        batch_size=1,  # Keep batch size 1 for memory efficiency
        shuffle=True,
        num_workers=4,  # Increase workers for speed
        pin_memory=True,  # For GPU transfer speed
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2,  # Prefetch batches
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,  # Fewer workers for validation
        pin_memory=True,
        persistent_workers=True,
    )
    
    return train_loader, val_loader
