"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import build_stock_features, make_targets, drop_warmup_rows, get_feature_columns


def create_dummy_ohlcv(n_days=100):
    """Create dummy OHLCV data for testing."""
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Generate realistic-looking price data
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_days) * 2)
    open_ = close + np.random.randn(n_days) * 0.5
    high = np.maximum(open_, close) + np.abs(np.random.randn(n_days) * 0.5)
    low = np.minimum(open_, close) - np.abs(np.random.randn(n_days) * 0.5)
    volume = np.random.randint(1000000, 10000000, size=n_days)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


def test_feature_creation():
    """Test that features are created correctly."""
    print("Testing feature creation...")
    
    df = create_dummy_ohlcv(n_days=100)
    
    cfg = {
        'rsi_period': 14,
        'roll_means': [5, 10],
        'roll_stds': [5, 10],
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'volume_windows': [5, 20]
    }
    
    df_feat = build_stock_features(df, cfg)
    
    # Check that expected columns exist
    expected_cols = [
        'close_ret_1',
        'roll_mean_5', 'roll_mean_10',
        'roll_std_5', 'roll_std_10',
        'rsi_14',
        'macd', 'macd_signal', 'macd_hist',
        'vol_roll_mean_5', 'vol_zscore_5',
        'vol_roll_mean_20', 'vol_zscore_20'
    ]
    
    for col in expected_cols:
        assert col in df_feat.columns, f"Missing feature column: {col}"
    
    # Check that original OHLCV columns are preserved
    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert col in df_feat.columns, f"Missing OHLCV column: {col}"
    
    print(f"✓ Feature creation test passed ({len(df_feat.columns)} total columns)")


def test_target_creation_direction():
    """Test direction target creation."""
    print("\nTesting direction target creation...")
    
    df = create_dummy_ohlcv(n_days=50)
    df_target = make_targets(df, mode='direction')
    
    # Check target column exists
    assert 'target' in df_target.columns, "Missing target column"
    
    # Check target is binary
    assert set(df_target['target'].unique()).issubset({0, 1}), \
        "Direction target should be binary (0 or 1)"
    
    # Check length (should be original length - 1, since last day has no target)
    assert len(df_target) == len(df) - 1, \
        f"Target df should have {len(df)-1} rows, got {len(df_target)}"
    
    # Manually verify a few samples
    for i in range(min(5, len(df_target))):
        expected = 1 if df.iloc[i+1]['close'] > df.iloc[i]['close'] else 0
        actual = df_target.iloc[i]['target']
        assert actual == expected, \
            f"Target mismatch at index {i}: {actual} vs expected {expected}"
    
    print("✓ Direction target test passed")


def test_target_creation_return():
    """Test return target creation."""
    print("\nTesting return target creation...")
    
    df = create_dummy_ohlcv(n_days=50)
    df_target = make_targets(df, mode='return')
    
    # Check target column exists
    assert 'target' in df_target.columns, "Missing target column"
    
    # Check target is continuous
    assert df_target['target'].dtype in [np.float32, np.float64], \
        "Return target should be continuous"
    
    # Check length
    assert len(df_target) == len(df) - 1, \
        f"Target df should have {len(df)-1} rows, got {len(df_target)}"
    
    # Manually verify a few samples
    for i in range(min(5, len(df_target))):
        expected = (df.iloc[i+1]['close'] / df.iloc[i]['close']) - 1
        actual = df_target.iloc[i]['target']
        assert np.isclose(actual, expected, rtol=1e-5), \
            f"Return target mismatch at index {i}: {actual} vs expected {expected}"
    
    print("✓ Return target test passed")


def test_no_target_leakage():
    """Test that targets only use future (t+1) data, not current or past."""
    print("\nTesting for target leakage...")
    
    # Create simple sequential data
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    close_prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
    
    df = pd.DataFrame({
        'open': close_prices,
        'high': close_prices,
        'low': close_prices,
        'close': close_prices,
        'volume': [1000000] * 10
    }, index=dates)
    
    df_target = make_targets(df, mode='direction')
    
    # Manually check each target
    for i in range(len(df_target)):
        current_close = df.iloc[i]['close']
        next_close = df.iloc[i+1]['close']
        expected_target = 1 if next_close > current_close else 0
        actual_target = df_target.iloc[i]['target']
        
        assert actual_target == expected_target, \
            f"Leakage detected at index {i}: target {actual_target} doesn't match next-day direction"
    
    print("✓ No target leakage detected")


def test_warmup_removal():
    """Test that warm-up NaN rows are removed."""
    print("\nTesting warm-up row removal...")
    
    df = create_dummy_ohlcv(n_days=100)
    
    cfg = {
        'rsi_period': 14,
        'roll_means': [5, 10],
        'roll_stds': [5, 10],
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'volume_windows': [5, 20]
    }
    
    df_feat = build_stock_features(df, cfg)
    
    # Count NaNs before
    nan_count_before = df_feat.isnull().sum().sum()
    
    # Drop warm-up rows
    df_clean = drop_warmup_rows(df_feat)
    
    # Count NaNs after
    nan_count_after = df_clean.isnull().sum().sum()
    
    assert nan_count_after == 0, f"Still have {nan_count_after} NaN values after dropping warm-up"
    assert len(df_clean) < len(df_feat), "Should have removed some rows"
    
    print(f"✓ Warm-up removal test passed (removed {len(df_feat) - len(df_clean)} rows)")


def test_get_feature_columns():
    """Test feature column extraction."""
    print("\nTesting feature column extraction...")
    
    df = create_dummy_ohlcv(n_days=100)
    
    cfg = {
        'rsi_period': 14,
        'roll_means': [5, 10],
        'roll_stds': [5, 10],
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'volume_windows': [5, 20]
    }
    
    df_feat = build_stock_features(df, cfg)
    df_feat = make_targets(df_feat, mode='direction')
    
    feature_cols = get_feature_columns(df_feat)
    
    # Check that OHLCV and target are excluded
    excluded = ['open', 'high', 'low', 'close', 'volume', 'target']
    for col in excluded:
        assert col not in feature_cols, f"Column {col} should be excluded from features"
    
    # Check that engineered features are included
    assert 'close_ret_1' in feature_cols, "close_ret_1 should be in features"
    assert 'rsi_14' in feature_cols, "rsi_14 should be in features"
    
    print(f"✓ Feature column extraction test passed ({len(feature_cols)} features)")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING FEATURE TESTS")
    print("=" * 60)
    
    test_feature_creation()
    test_target_creation_direction()
    test_target_creation_return()
    test_no_target_leakage()
    test_warmup_removal()
    test_get_feature_columns()
    
    print("\n" + "=" * 60)
    print("ALL FEATURE TESTS PASSED ✓")
    print("=" * 60)

