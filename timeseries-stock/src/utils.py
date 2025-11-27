"""Utility functions for metrics, scaling, splitting, and configuration."""

import numpy as np
import pandas as pd
import torch
import random
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict
import os


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def time_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split DataFrame into train/val/test.
    
    Args:
        df: DataFrame with DatetimeIndex (sorted)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test (if None, computed as 1 - train - val)
    
    Returns:
        train_df, val_df, test_df
    """
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio
    
    # Verify ratios sum to 1
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Time-based split:")
    print(f"  Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Val:   {len(val_df)} samples ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"  Test:  {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
    
    return train_df, val_df, test_df


def standardize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features using training set statistics (fit on train only).
    
    Args:
        X_train: Training features (num_samples, window_size, num_features)
        X_val: Validation features
        X_test: Test features
    
    Returns:
        Standardized X_train, X_val, X_test, and fitted scaler
    """
    num_samples_train, window_size, num_features = X_train.shape
    
    # Reshape to 2D for StandardScaler
    X_train_2d = X_train.reshape(-1, num_features)
    X_val_2d = X_val.reshape(-1, num_features)
    X_test_2d = X_test.reshape(-1, num_features)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_2d)
    
    # Transform val and test using training statistics
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(num_samples_train, window_size, num_features)
    X_val_scaled = X_val_scaled.reshape(X_val.shape[0], window_size, num_features)
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], window_size, num_features)
    
    print("✓ Features standardized (fit on train only)")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    
    # Directional hit rate (same as accuracy for binary)
    metrics['hit_rate'] = metrics['accuracy']
    
    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics for stock prediction.
    
    Includes both regression metrics (MAE, RMSE) and trading-critical
    directional metrics (accuracy, precision, recall, F1).
    
    Args:
        y_true: True values (returns)
        y_pred: Predicted values (returns)
    
    Returns:
        Dictionary of metrics
    """
    # Standard regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Correlation - how well predictions track actuals
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    else:
        correlation = np.nan
    
    # CRITICAL FOR TRADING: Directional accuracy
    # Convert returns to direction (1 = up, 0 = down)
    pred_direction = (y_pred > 0).astype(int).flatten()
    true_direction = (y_true > 0).astype(int).flatten()
    
    # Directional accuracy (most important for trading!)
    directional_accuracy = (pred_direction == true_direction).mean()
    
    # Trading metrics: treat as classification problem
    # If we get direction right, we make money!
    dir_precision = precision_score(true_direction, pred_direction, zero_division=0)
    dir_recall = recall_score(true_direction, pred_direction, zero_division=0)
    dir_f1 = f1_score(true_direction, pred_direction, zero_division=0)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'dir_accuracy': directional_accuracy,  # MOST IMPORTANT!
        'dir_precision': dir_precision,
        'dir_recall': dir_recall,
        'dir_f1': dir_f1,
    }
    
    return metrics


def get_device() -> torch.device:
    """Get the best available device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    
    return device


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check for improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False

