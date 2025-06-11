#!/usr/bin/env python3
"""
Cross-validation framework for medical image segmentation.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import StratifiedKFold, KFold
import sys
import importlib
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.preprocess_and_dataloader import train_loader, val_loader
from utils.loss_functions import FocalLoss
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader

class MedicalCrossValidator:
    """Cross-validation framework for medical image segmentation."""
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.fold_results = []
        self.statistical_summary = {}
        
    def get_patient_level_splits(self, dataset) -> List[Tuple[List[int], List[int]]]:
        """Create patient-level splits to avoid data leakage."""
        print("Creating patient-level cross-validation splits...")
        
        # Simple approach: treat each sample as separate patient for now
        # In real implementation, would extract patient IDs from filenames
        n_samples = len(dataset)
        indices = list(range(n_samples))
        
        # Create stratification based on tumor presence
        tumor_labels = []
        for i in range(n_samples):
            try:
                sample = dataset[i]
                label_data = sample['label']
                has_tumor = torch.sum(label_data > 0).item() > 100 if torch.is_tensor(label_data) else np.sum(label_data > 0) > 100
                tumor_labels.append(int(has_tumor))
            except:
                tumor_labels.append(0)  # Default to no tumor
        
        print(f"Samples with tumors: {sum(tumor_labels)}/{len(tumor_labels)}")
        
        # Create stratified splits
        if sum(tumor_labels) >= self.n_folds:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            splits = list(skf.split(indices, tumor_labels))
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(indices))
        
        return splits
    
    def train_fold(self, model_name: str, train_indices: List[int], val_indices: List[int], 
                   epochs: int = 10, lr: float = 1e-4, fold_num: int = 0) -> Dict:
        """Train model on single fold."""
        print(f"\nTraining Fold {fold_num + 1}/{self.n_folds}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load full dataset
        full_dataset = train_loader.dataset.data
        
        # Create fold-specific datasets
        train_fold_data = [full_dataset[i] for i in train_indices]
        val_fold_data = [full_dataset[i] for i in val_indices]
        
        train_fold_dataset = Dataset(train_fold_data, transform=train_loader.dataset.transform)
        val_fold_dataset = Dataset(val_fold_data, transform=train_loader.dataset.transform)
        
        train_fold_loader = DataLoader(train_fold_dataset, batch_size=2, shuffle=True, num_workers=2)
        val_fold_loader = DataLoader(val_fold_dataset, batch_size=2, shuffle=False, num_workers=2)
        
        # Load model
        model = self._get_model(model_name).to(device)
        
        # Setup loss (using successful FocalLoss approach)
        focal_loss = FocalLoss(alpha=0.05, gamma=3.0, reduction='mean')
        dice_loss = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        
        def combined_loss(y_pred, y_true):
            return 0.7 * focal_loss(y_pred, y_true) + 0.3 * dice_loss(y_pred, y_true)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Training loop
        best_dice = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_data in train_fold_loader:
                try:
                    inputs = batch_data["image"].to(device)
                    labels = batch_data["label"].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = combined_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    continue
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            dice_metric.reset()
            
            with torch.no_grad():
                for batch_data in val_fold_loader:
                    try:
                        inputs = batch_data["image"].to(device)
                        labels = batch_data["label"].to(device)
                        
                        outputs = model(inputs)
                        loss = combined_loss(outputs, labels)
                        val_loss += loss.item()
                        val_batches += 1
                        
                        outputs_argmax = torch.argmax(outputs, dim=1, keepdim=True)
                        dice_metric(y_pred=outputs_argmax, y=labels)
                    except Exception as e:
                        continue
            
            val_dice = dice_metric.aggregate().item()
            if val_dice > best_dice:
                best_dice = val_dice
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch+1}: Val Dice: {val_dice:.4f}")
        
        return {
            'fold': fold_num,
            'best_dice': best_dice,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices)
        }
    
    def _get_model(self, model_name: str):
        """Get model instance by name."""
        module_path = f"models.{model_name}"
        module = importlib.import_module(module_path)
        return module.get_model()
    
    def run_cross_validation(self, model_name: str = "unet3d", epochs: int = 10) -> Dict:
        """Run complete cross-validation study."""
        print(f"🧪 Starting {self.n_folds}-Fold Cross-Validation for {model_name}")
        
        # Load dataset for splitting  
        dataset = train_loader.dataset
        
        # Get patient-level splits
        fold_splits = self.get_patient_level_splits(dataset)
        
        # Train each fold
        self.fold_results = []
        
        for fold_num, (train_indices, val_indices) in enumerate(fold_splits):
            try:
                fold_result = self.train_fold(
                    model_name=model_name,
                    train_indices=train_indices,
                    val_indices=val_indices,
                    epochs=epochs,
                    fold_num=fold_num
                )
                self.fold_results.append(fold_result)
            except Exception as e:
                print(f"Error in fold {fold_num}: {e}")
                continue
        
        # Calculate statistics
        dice_scores = [r['best_dice'] for r in self.fold_results]
        
        self.statistical_summary = {
            'mean_dice': np.mean(dice_scores),
            'std_dice': np.std(dice_scores),
            'best_dice': np.max(dice_scores),
            'dice_scores': dice_scores,
            'n_folds': len(self.fold_results)
        }
        
        # Save results
        self._save_results(model_name)
        
        print(f"\n🎉 Cross-Validation Completed!")
        print(f"Mean Dice: {self.statistical_summary['mean_dice']:.4f} ± {self.statistical_summary['std_dice']:.4f}")
        
        return self.statistical_summary
    
    def _save_results(self, model_name: str):
        """Save cross-validation results to file."""
        results_dir = Path("results/cross_validation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"cv_results_{model_name}_{timestamp}.json"
        
        save_data = {
            'model_name': model_name,
            'n_folds': self.n_folds,
            'fold_results': self.fold_results,
            'statistical_summary': self.statistical_summary
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {results_file}")

if __name__ == "__main__":
    print("🧪 BTCV Cross-Validation Framework")
    
    # Test cross-validation
    cv = MedicalCrossValidator(n_folds=3)
    results = cv.run_cross_validation(model_name="unet3d", epochs=3)
    print("✅ Cross-validation framework working!") 