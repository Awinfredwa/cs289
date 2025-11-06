"""PyTorch Dataset and DataLoader utilities for time-series data."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for windowed time-series data.
    
    Args:
        X: Windowed features of shape (num_samples, window_size, num_features)
        y: Targets of shape (num_samples,)
        task: 'classification' or 'regression'
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, task: str = 'classification'):
        self.X = torch.FloatTensor(X)
        
        if task == 'classification':
            # Use FloatTensor for BCELoss compatibility (especially with MPS)
            self.y = torch.FloatTensor(y.astype(np.float32))
        elif task == 'regression':
            self.y = torch.FloatTensor(y)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        self.task = task
        
        assert len(self.X) == len(self.y), \
            f"X and y length mismatch: {len(self.X)} vs {len(self.y)}"
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    task: str = 'classification'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size for training
        task: 'classification' or 'regression'
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = TimeSeriesDataset(X_train, y_train, task=task)
    val_dataset = TimeSeriesDataset(X_val, y_val, task=task)
    test_dataset = TimeSeriesDataset(X_test, y_test, task=task)
    
    # Disable pin_memory on MPS (not supported)
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    print(f"Created DataLoaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

