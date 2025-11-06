"""Sliding window construction for time-series sequences."""

import numpy as np
from typing import Tuple


def make_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from time-series data.
    
    Args:
        X: Feature array of shape (num_days, num_features)
        y: Target array of shape (num_days,)
        window_size: Number of time steps in each window
        stride: Step size between consecutive windows (default 1)
    
    Returns:
        X_win: Array of shape (num_samples, window_size, num_features)
        y_win: Array of shape (num_samples,)
    
    Example:
        If X has 100 days and window_size=7, we get 94 samples.
        Sample 0: days [0:7] -> predict day 7
        Sample 1: days [1:8] -> predict day 8
        ...
        Sample 93: days [93:100] -> predict day 100
    """
    num_days, num_features = X.shape
    
    if window_size > num_days:
        raise ValueError(f"window_size ({window_size}) cannot exceed number of days ({num_days})")
    
    # Calculate number of windows
    num_samples = (num_days - window_size) // stride + 1
    
    if num_samples <= 0:
        raise ValueError(f"Not enough data for windowing: num_days={num_days}, window_size={window_size}")
    
    # Pre-allocate arrays
    X_win = np.zeros((num_samples, window_size, num_features), dtype=np.float32)
    y_win = np.zeros((num_samples,), dtype=np.float32)
    
    # Build windows
    for i in range(num_samples):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        # Features from [start_idx:end_idx]
        X_win[i] = X[start_idx:end_idx]
        
        # Target is at end_idx (the day after the window)
        y_win[i] = y[end_idx - 1]
    
    return X_win, y_win


def validate_window_shapes(X_win: np.ndarray, y_win: np.ndarray, window_size: int, num_features: int):
    """
    Validate that window shapes are correct.
    
    Args:
        X_win: Windowed features
        y_win: Windowed targets
        window_size: Expected window size
        num_features: Expected number of features
    """
    assert X_win.ndim == 3, f"X_win should be 3D, got {X_win.ndim}D"
    assert y_win.ndim == 1, f"y_win should be 1D, got {y_win.ndim}D"
    
    assert X_win.shape[0] == y_win.shape[0], \
        f"Mismatch: X_win has {X_win.shape[0]} samples, y_win has {y_win.shape[0]}"
    
    assert X_win.shape[1] == window_size, \
        f"X_win window_size is {X_win.shape[1]}, expected {window_size}"
    
    assert X_win.shape[2] == num_features, \
        f"X_win has {X_win.shape[2]} features, expected {num_features}"
    
    print(f"✓ Window validation passed: X_win {X_win.shape}, y_win {y_win.shape}")

