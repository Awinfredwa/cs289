"""Tests for sliding window construction."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.windows import make_sliding_windows, validate_window_shapes


def test_window_shapes():
    """Test that window shapes are correct."""
    print("Testing window shapes...")
    
    # Create dummy data: 100 days, 5 features
    N = 100
    num_features = 5
    window_size = 7
    
    X = np.random.randn(N, num_features).astype(np.float32)
    y = np.random.randint(0, 2, size=N).astype(np.float32)
    
    # Create windows
    X_win, y_win = make_sliding_windows(X, y, window_size=window_size, stride=1)
    
    # Expected number of samples
    expected_samples = N - window_size + 1
    
    # Check shapes
    assert X_win.shape == (expected_samples, window_size, num_features), \
        f"X_win shape mismatch: {X_win.shape} vs expected {(expected_samples, window_size, num_features)}"
    
    assert y_win.shape == (expected_samples,), \
        f"y_win shape mismatch: {y_win.shape} vs expected {(expected_samples,)}"
    
    print(f"✓ Shape test passed: X_win {X_win.shape}, y_win {y_win.shape}")


def test_window_target_alignment():
    """Test that targets are correctly aligned with windows (shifted by +1)."""
    print("\nTesting window-target alignment...")
    
    # Create sequential data for easy verification
    N = 20
    num_features = 2
    window_size = 5
    
    # X: sequential features
    X = np.arange(N * num_features).reshape(N, num_features).astype(np.float32)
    
    # y: simple sequential targets
    y = np.arange(N).astype(np.float32)
    
    # Create windows
    X_win, y_win = make_sliding_windows(X, y, window_size=window_size, stride=1)
    
    # Check first sample
    # Window 0 should contain days [0:5], and target should be day 4 (the last day in window)
    expected_first_window = X[0:window_size]
    expected_first_target = y[window_size - 1]
    
    assert np.allclose(X_win[0], expected_first_window), \
        "First window doesn't match expected data"
    
    assert y_win[0] == expected_first_target, \
        f"First target mismatch: {y_win[0]} vs expected {expected_first_target}"
    
    # Check second sample
    # Window 1 should contain days [1:6], and target should be day 5
    expected_second_window = X[1:window_size+1]
    expected_second_target = y[window_size]
    
    assert np.allclose(X_win[1], expected_second_window), \
        "Second window doesn't match expected data"
    
    assert y_win[1] == expected_second_target, \
        f"Second target mismatch: {y_win[1]} vs expected {expected_second_target}"
    
    print("✓ Target alignment test passed")


def test_window_stride():
    """Test that stride parameter works correctly."""
    print("\nTesting window stride...")
    
    N = 50
    num_features = 3
    window_size = 7
    stride = 3
    
    X = np.random.randn(N, num_features).astype(np.float32)
    y = np.random.randint(0, 2, size=N).astype(np.float32)
    
    X_win, y_win = make_sliding_windows(X, y, window_size=window_size, stride=stride)
    
    # Expected number of samples with stride
    expected_samples = (N - window_size) // stride + 1
    
    assert len(X_win) == expected_samples, \
        f"Sample count with stride mismatch: {len(X_win)} vs expected {expected_samples}"
    
    print(f"✓ Stride test passed: {len(X_win)} samples with stride={stride}")


def test_window_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    # Case 1: window_size == num_days (should produce 1 sample)
    N = 10
    num_features = 2
    
    X = np.random.randn(N, num_features).astype(np.float32)
    y = np.random.randint(0, 2, size=N).astype(np.float32)
    
    X_win, y_win = make_sliding_windows(X, y, window_size=N, stride=1)
    
    assert len(X_win) == 1, f"Should produce 1 sample, got {len(X_win)}"
    
    # Case 2: window_size > num_days (should raise error)
    try:
        X_win, y_win = make_sliding_windows(X, y, window_size=N+1, stride=1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ Edge case tests passed")


def test_validation_function():
    """Test the validation function."""
    print("\nTesting validation function...")
    
    N = 100
    window_size = 7
    num_features = 5
    
    X = np.random.randn(N, num_features).astype(np.float32)
    y = np.random.randint(0, 2, size=N).astype(np.float32)
    
    X_win, y_win = make_sliding_windows(X, y, window_size=window_size, stride=1)
    
    # Should pass validation
    validate_window_shapes(X_win, y_win, window_size, num_features)
    
    print("✓ Validation function test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING WINDOW TESTS")
    print("=" * 60)
    
    test_window_shapes()
    test_window_target_alignment()
    test_window_stride()
    test_window_edge_cases()
    test_validation_function()
    
    print("\n" + "=" * 60)
    print("ALL WINDOW TESTS PASSED ✓")
    print("=" * 60)

