import sys
import os
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged,
    CropForegroundd, Compose, ResizeWithPadOrCropd, EnsureTyped, MapTransform,
    RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd, RandGaussianNoised,
    RandAdjustContrastd, SpatialPadd, Resized # Corrected: Changed SpatialPad to SpatialPadd
)
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.config import print_config
import torch
import numpy as np

# Define data directory and dataset json path
base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(base_dir, "BTCV")
dataset_json = os.path.join(data_dir, "dataset.json")

def is_foreground_pixel(x):
    return x > 0

class PostProcessLabeld(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            if not isinstance(label, torch.LongTensor) and not isinstance(label, torch.cuda.LongTensor) \
               and not isinstance(label, torch.IntTensor) and not isinstance(label, torch.cuda.IntTensor):
                label = torch.round(label).to(torch.int64)
            else:
                label = label.to(torch.int64)
            d[key] = label
        return d

TARGET_SPATIAL_SIZE = (96, 96, 64)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=False, reader="NibabelReader"),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image"], dtype=torch.float32),
    EnsureTyped(keys=["label"], dtype=torch.int64),

    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    PostProcessLabeld(keys="label"),  # Essential after spacing

    Orientationd(keys=["image", "label"], axcodes="RAS"),
    PostProcessLabeld(keys="label"),  # Essential after orientation

    ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=250, b_min=0.0, b_max=1.0, clip=True),

    CropForegroundd(keys=["image", "label"], source_key="label", select_fn=is_foreground_pixel,
                    margin=10, k_divisible=[16,16,16], allow_smaller=True),
    PostProcessLabeld(keys="label"),  # Essential after cropping

    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
    # Removed PostProcessLabeld here - augmentations shouldn't change label values

    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.8, 1.2)),

    Resized(keys="image", spatial_size=TARGET_SPATIAL_SIZE, mode="bilinear", anti_aliasing=True),
    Resized(keys="label", spatial_size=TARGET_SPATIAL_SIZE, mode="nearest"),
    PostProcessLabeld(keys="label"),  # Essential after resizing

    # Corrected: Use SpatialPadd (dictionary version)
    SpatialPadd(keys=["image", "label"], spatial_size=TARGET_SPATIAL_SIZE, mode='constant', constant_values=0),
    # Removed PostProcessLabeld here - padding shouldn't change existing label values

    ResizeWithPadOrCropd(keys="image", spatial_size=TARGET_SPATIAL_SIZE, mode="bilinear", constant_values=0),
    ResizeWithPadOrCropd(keys="label", spatial_size=TARGET_SPATIAL_SIZE, mode="nearest", constant_values=0),
    PostProcessLabeld(keys="label"),  # Final essential processing

    EnsureTyped(keys=["image"], dtype=torch.float32),
    EnsureTyped(keys=["label"], dtype=torch.int64),  # Final type ensure
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=False, reader="NibabelReader"),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image"], dtype=torch.float32),
    EnsureTyped(keys=["label"], dtype=torch.int64),

    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    PostProcessLabeld(keys="label"),  # Essential after spacing

    Orientationd(keys=["image", "label"], axcodes="RAS"),
    PostProcessLabeld(keys="label"),  # Essential after orientation

    ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=250, b_min=0.0, b_max=1.0, clip=True),

    CropForegroundd(keys=["image", "label"], source_key="label", select_fn=is_foreground_pixel,
                    margin=10, k_divisible=[16,16,16], allow_smaller=True),
    PostProcessLabeld(keys="label"),  # Essential after cropping
    
    Resized(keys="image", spatial_size=TARGET_SPATIAL_SIZE, mode="bilinear", anti_aliasing=True),
    Resized(keys="label", spatial_size=TARGET_SPATIAL_SIZE, mode="nearest"),
    PostProcessLabeld(keys="label"),  # Essential after resizing

    # Corrected: Use SpatialPadd (dictionary version)
    SpatialPadd(keys=["image", "label"], spatial_size=TARGET_SPATIAL_SIZE, mode='constant', constant_values=0),
    # Removed PostProcessLabeld here - padding shouldn't change existing label values

    ResizeWithPadOrCropd(keys="image", spatial_size=TARGET_SPATIAL_SIZE, mode="bilinear", constant_values=0),
    ResizeWithPadOrCropd(keys="label", spatial_size=TARGET_SPATIAL_SIZE, mode="nearest", constant_values=0),
    PostProcessLabeld(keys="label"),  # Final essential processing

    EnsureTyped(keys=["image"], dtype=torch.float32),
    EnsureTyped(keys=["label"], dtype=torch.int64),  # Final type ensure
])

# --- (Datalist loading, CacheDataset, DataLoader definitions remain the same) ---
try:
    train_files = load_decathlon_datalist(dataset_json, True, "training", base_dir=data_dir)
    val_files = load_decathlon_datalist(dataset_json, True, "validation", base_dir=data_dir)
    print(f"Successfully loaded training ({len(train_files)}) and validation ({len(val_files)}) datasets using base_dir: {data_dir}")
    if not train_files: print("Warning: Training files list is empty.")
    if not val_files: print("Warning: Validation files list is empty.")
except Exception as e:
    print(f"Error loading dataset split from {dataset_json} with base_dir {data_dir}: {e}")
    train_files, val_files = [], []

if not train_files and not val_files:
    print("No data loaded. Exiting script.")
    sys.exit(1)

cache_train_workers = 0
cache_val_workers = 0

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5,
                        num_workers=cache_train_workers, copy_cache=False, progress=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0,
                      num_workers=cache_val_workers, copy_cache=False, progress=True)

actual_train_num_workers = 0
actual_val_num_workers = 0

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                          num_workers=actual_train_num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers= (actual_train_num_workers > 0) )
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=actual_val_num_workers,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers= (actual_val_num_workers > 0) )

# --- Debugging section for __main__ ---
def print_label_stats(data_dict, stage_name=""):
    label_tensor = data_dict.get("label", torch.empty(0))
    img_tensor = data_dict.get("image", torch.empty(0))
    fg_pixels = torch.sum(label_tensor > 0).item() if label_tensor.numel() > 0 else 0
    unique_labels_str = str(torch.unique(label_tensor).cpu().numpy()) if label_tensor.numel() > 0 else "N/A"
    print(
        f"Stage: {stage_name:<35} "
        f"Label Shape: {str(label_tensor.shape):<25}, Dtype: {str(label_tensor.dtype):<15}, "
        f"Unique: {unique_labels_str:<10}, "
        f"FG Pixels: {fg_pixels:<7}"
        f" || Image Shape: {str(img_tensor.shape):<25}, Dtype: {img_tensor.dtype}"
    )

if __name__ == "__main__":
    print_config()
    print(f"Device for potential model ops (not dataloading): {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    problematic_image_file = os.path.join(data_dir, "imagesTr/colon_171.nii.gz")
    problematic_label_file = os.path.join(data_dir, "labelsTr/colon_171.nii.gz")

    if not (os.path.exists(problematic_image_file) and os.path.exists(problematic_label_file)):
        print(f"WARNING: Debug file not found: {problematic_label_file}. Skipping single file debug.")
    else:
        print(f"\n--- Debugging Single File with explicit Resized/SpatialPadd: {os.path.basename(problematic_label_file)} ---")
        
        debug_transform_list_cf = [
            LoadImaged(keys=["image", "label"], image_only=False, reader="NibabelReader"),
            (lambda data: (print_label_stats(data, "After LoadImaged"), data)[1]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.int64),
            (lambda data: (print_label_stats(data, "After Initial EnsureTyped"), data)[1]),

            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            PostProcessLabeld(keys="label"),
            (lambda data: (print_label_stats(data, "After Spacingd & PostProc"), data)[1]),

            Orientationd(keys=["image", "label"], axcodes="RAS"),
            PostProcessLabeld(keys="label"),
            (lambda data: (print_label_stats(data, "After Orientationd & PostProc"), data)[1]),
            
            CropForegroundd(keys=["image", "label"], source_key="label", select_fn=is_foreground_pixel,
                            margin=10, k_divisible=[16,16,16], allow_smaller=True),
            PostProcessLabeld(keys="label"),
            (lambda data: (print_label_stats(data, "After CropForegroundd & PostProc"), data)[1]),

            Resized(keys="image", spatial_size=TARGET_SPATIAL_SIZE, mode="bilinear", anti_aliasing=True),
            Resized(keys="label", spatial_size=TARGET_SPATIAL_SIZE, mode="nearest"),
            PostProcessLabeld(keys="label"),
            (lambda data: (print_label_stats(data, "After Explicit Resized & PostProc"), data)[1]),

            # Corrected: Use SpatialPadd (dictionary version)
            SpatialPadd(keys=["image", "label"], spatial_size=TARGET_SPATIAL_SIZE, mode='constant', constant_values=0),
            PostProcessLabeld(keys="label"),
            (lambda data: (print_label_stats(data, "After Explicit SpatialPadd & PostProc"), data)[1]),

            ResizeWithPadOrCropd(keys="image", spatial_size=TARGET_SPATIAL_SIZE, mode="bilinear", constant_values=0),
            ResizeWithPadOrCropd(keys="label", spatial_size=TARGET_SPATIAL_SIZE, mode="nearest", constant_values=0),
            PostProcessLabeld(keys="label"),
            (lambda data: (print_label_stats(data, "After Final ResizeWithPadOrCropd"), data)[1]),

            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.int64),
            (lambda data: (print_label_stats(data, "After Final EnsureTyped"), data)[1]),
        ]
        debug_pipeline_cf = Compose(debug_transform_list_cf)
        
        debug_files_list = [{"image": problematic_image_file, "label": problematic_label_file}]
        
        from monai.data import Dataset
        debug_ds_single_cf = Dataset(data=debug_files_list, transform=debug_pipeline_cf)
        debug_loader_single_cf = DataLoader(debug_ds_single_cf, batch_size=1, num_workers=0)

        try:
            for _ in debug_loader_single_cf:
                pass 
        except Exception as e:
            print(f"Error during single file debug: {e}")
            import traceback
            traceback.print_exc()

    test_num_workers = 0
    print(f"\n--- Testing Training Dataloader (using num_workers={test_num_workers} for this test block) ---")
    _test_train_ds_for_main = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0, num_workers=0)
    _test_train_loader = DataLoader(_test_train_ds_for_main, batch_size=2, shuffle=True, num_workers=test_num_workers)

    if not _test_train_loader.dataset or len(_test_train_loader.dataset) == 0:
        print("Training dataset is empty for general test.")
    else:
        print(f"Number of training samples for general test: {len(_test_train_loader.dataset)}")
        try:
            for i, batch_data in enumerate(_test_train_loader):
                print_label_stats(batch_data, f"General Train Batch {i+1} Aggregated")
                if i >= 0: break 
        except Exception as e:
            print(f"Error during general training dataloader test: {e}")
            import traceback
            traceback.print_exc()
    print("\nDataloader testing complete!")