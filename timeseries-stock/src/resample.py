"""
Resampling utilities for converting daily stock data to weekly frequency.
"""

import pandas as pd
import numpy as np


def resample_to_weekly(df: pd.DataFrame, method: str = 'last') -> pd.DataFrame:
    """
    Resample daily OHLCV data to weekly frequency.
    
    Args:
        df: DataFrame with DatetimeIndex and OHLCV columns
        method: Resampling method for each column
            - 'ohlc': Proper OHLC aggregation (open=first, high=max, low=min, close=last)
            - 'last': Use last value of the week (simpler, for features)
            - 'mean': Average over the week
    
    Returns:
        DataFrame resampled to weekly frequency (Friday end-of-week)
    """
    if method == 'ohlc':
        # Proper OHLC aggregation
        weekly = pd.DataFrame()
        weekly['open'] = df['open'].resample('W-FRI').first()
        weekly['high'] = df['high'].resample('W-FRI').max()
        weekly['low'] = df['low'].resample('W-FRI').min()
        weekly['close'] = df['close'].resample('W-FRI').last()
        weekly['volume'] = df['volume'].resample('W-FRI').sum()  # Sum volume over week
        
        # For other columns (features), take last value
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        for col in feature_cols:
            weekly[col] = df[col].resample('W-FRI').last()
    
    elif method == 'last':
        # Take last value of each week
        weekly = df.resample('W-FRI').last()
    
    elif method == 'mean':
        # Average over each week
        weekly = df.resample('W-FRI').mean()
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'ohlc', 'last', or 'mean'.")
    
    # Drop any rows with NaN (incomplete weeks)
    weekly = weekly.dropna()
    
    return weekly


def align_weekly_fear_greed(stock_df: pd.DataFrame, fg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align weekly Fear & Greed Index with weekly stock data.
    
    Args:
        stock_df: Weekly stock DataFrame (W-FRI frequency)
        fg_df: Fear & Greed DataFrame (weekly, various dates)
    
    Returns:
        Merged DataFrame with both stock and F&G features
    """
    # Ensure both have DatetimeIndex
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        raise ValueError("stock_df must have DatetimeIndex")
    if not isinstance(fg_df.index, pd.DatetimeIndex):
        raise ValueError("fg_df must have DatetimeIndex")
    
    # Resample F&G to same weekly frequency as stock (W-FRI)
    fg_weekly = fg_df.resample('W-FRI').last()
    
    # Forward-fill F&G to handle missing weeks
    fg_weekly = fg_weekly.fillna(method='ffill')
    
    # Merge on date index
    merged = stock_df.join(fg_weekly, how='left')
    
    # Fill any remaining NaN with forward-fill
    fg_cols = [col for col in fg_weekly.columns]
    merged[fg_cols] = merged[fg_cols].fillna(method='ffill')
    
    return merged


def create_weekly_targets(df: pd.DataFrame, mode: str = 'direction', horizon: int = 1) -> pd.DataFrame:
    """
    Create weekly prediction targets.
    
    Args:
        df: Weekly DataFrame with 'close' column
        mode: 'direction' or 'return'
        horizon: Number of weeks ahead to predict (1=next week)
    
    Returns:
        DataFrame with 'target' column added
    """
    df = df.copy()
    
    if mode == 'direction':
        # Binary: 1 if future close > current close
        df['target'] = (df['close'].shift(-horizon) > df['close']).astype(int)
    
    elif mode == 'return':
        # Continuous: future return
        df['target'] = (df['close'].shift(-horizon) / df['close']) - 1
    
    else:
        raise ValueError(f"Unknown target mode: {mode}")
    
    # Drop last 'horizon' rows (no target available)
    df = df.iloc[:-horizon]
    
    return df


def weekly_summary(daily_df: pd.DataFrame, weekly_df: pd.DataFrame):
    """Print summary of daily vs weekly data."""
    print(f"\n{'='*70}")
    print("WEEKLY RESAMPLING SUMMARY")
    print(f"{'='*70}")
    print(f"Daily data:   {len(daily_df)} days")
    print(f"Weekly data:  {len(weekly_df)} weeks")
    print(f"Date range:   {weekly_df.index.min().date()} to {weekly_df.index.max().date()}")
    print(f"Weeks/year:   ~{len(weekly_df) / ((weekly_df.index.max() - weekly_df.index.min()).days / 365.25):.1f}")
    print(f"{'='*70}\n")

