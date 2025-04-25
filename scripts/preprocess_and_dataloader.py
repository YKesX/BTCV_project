import sys
import os
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, CropForegroundd, ToTensord, Compose, ResizeWithPadOrCropd, EnsureTyped
)
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.config import print_config
import torch


# Define data directory and dataset json path
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(base_dir, "BTCV")
dataset_json = os.path.join(data_dir, "dataset.json")

# Define transforms for training and validation
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=-200, a_max=250, b_min=0.0, b_max=1.0, clip=True
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 64),
        pos=1, neg=1, num_samples=4,
        image_key="image", image_threshold=0,
    ),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(96, 96, 64)),
    EnsureTyped(keys=["image", "label"]),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=-200, a_max=250, b_min=0.0, b_max=1.0, clip=True
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    ToTensord(keys=["image", "label"]),
])



# Load dataset from dataset.json
try:
    train_files = load_decathlon_datalist(dataset_json, True, "training")
    val_files = load_decathlon_datalist(dataset_json, True, "validation")
    print(f"Successfully loaded training ({len(train_files)}) and validation ({len(val_files)}) datasets")
except Exception as e:
    print(f"Error loading dataset split from {dataset_json}: {e}")
    print("Generating training/validation split manually...")
    
    # If dataset doesn't have predefined splits, create them
    all_files = load_decathlon_datalist(dataset_json, True, "training")
    
    import numpy as np
    np.random.seed(42)
    indices = np.random.permutation(len(all_files))
    val_split = int(len(all_files) * 0.2)
    
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]
    
    train_files = [all_files[i] for i in train_indices]
    val_files = [all_files[i] for i in val_indices]
    
    print(f"Created manual split: training ({len(train_files)}) and validation ({len(val_files)}) datasets")



# Create datasets
train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=0.5,  # Cache 50% of the data for memory efficiency
    num_workers=4,
    progress=True  # Show progress bar during caching
)

val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,  # Cache all validation data for consistent evaluation
    num_workers=2,
    progress=True  # Show progress bar during caching
)

# Create dataloaders with GPU optimizations
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # Speed up host to GPU transfers
    persistent_workers=True  # Keep workers alive between iterations for speed
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,  # Speed up host to GPU transfers
    persistent_workers=True  # Keep workers alive between iterations for speed
)

if __name__ == "__main__":
    
    # Debug info
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    
    # Check GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable MONAI's GPU acceleration where applicable
    torch.backends.cudnn.benchmark = True
    print("CUDNN benchmark enabled")

    # Print GPU information if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print_config()

    # Set PyTorch's number of threads for efficiency
    num_threads = 4  # Adjust based on your CPU
    torch.set_num_threads(num_threads)
    print(f"PyTorch thread count set to: {num_threads}")

    # Test the dataloaders and GPU transfer
    print("Testing dataloaders with GPU transfer...")
    

    for batch in train_loader:
        print(f"Train batch shape: {batch['image'].shape}, {batch['label'].shape}")
        break

    for batch in val_loader:
        print(f"Val batch shape: {batch['image'].shape}, {batch['label'].shape}")
        break
    
    '''
    # Get a batch from the training dataloader
    for batch_idx, batch_data in enumerate(train_loader):
        print(f"Batch {batch_idx}: {batch_data['image'].shape}, {batch_data['label'].shape}")
        
        # Transfer batch to GPU
        image = batch_data['image'].to(device)
        label = batch_data['label'].to(device)
        
        print(f"Data transferred to {device}")
        print(f"Image tensor on GPU: {image.device}, shape: {image.shape}")
        print(f"Label tensor on GPU: {label.device}, shape: {label.shape}")
        
        # Check image intensity range
        print(f"Image min: {image.min().item()}, max: {image.max().item()}")
        print(f"Label unique values: {torch.unique(label).cpu().numpy()}")
        
        # Test GPU memory usage
        if torch.cuda.is_available():
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Only process one batch for testing
        break
    '''
    print("Dataloader GPU testing complete!")