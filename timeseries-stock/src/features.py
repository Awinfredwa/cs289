"""Feature engineering module for stock technical indicators."""

import pandas as pd
import numpy as np
from typing import Optional
from ta.momentum import RSIIndicator
from ta.trend import MACD


def build_stock_features(
    df: pd.DataFrame,
    cfg: dict,
    sent_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build technical features from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        cfg: Config dict with features section
        sent_df: Optional sentiment DataFrame (for future extension)
    
    Returns:
        DataFrame with original OHLCV + engineered features
    """
    df = df.copy()
    
    # 1. Returns
    df['close_ret_1'] = df['close'].pct_change(1)
    
    # 2. Rolling statistics for close
    for window in cfg['roll_means']:
        df[f'roll_mean_{window}'] = df['close'].rolling(window=window).mean()
    
    for window in cfg['roll_stds']:
        df[f'roll_std_{window}'] = df['close'].rolling(window=window).std()
    
    # 3. RSI
    rsi_period = cfg['rsi_period']
    rsi = RSIIndicator(close=df['close'], window=rsi_period)
    df[f'rsi_{rsi_period}'] = rsi.rsi()
    
    # 4. MACD
    macd_cfg = cfg['macd']
    macd = MACD(
        close=df['close'],
        window_slow=macd_cfg['slow'],
        window_fast=macd_cfg['fast'],
        window_sign=macd_cfg['signal']
    )
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # 5. Volume features
    for window in cfg['volume_windows']:
        if window <= len(df):
            df[f'vol_roll_mean_{window}'] = df['volume'].rolling(window=window).mean()
            
            # Volume z-score
            vol_mean = df['volume'].rolling(window=window).mean()
            vol_std = df['volume'].rolling(window=window).std()
            df[f'vol_zscore_{window}'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)
    
    # 6. Optional: Merge sentiment data (for future extension)
    if sent_df is not None:
        # Ensure sent_df has DatetimeIndex
        if not isinstance(sent_df.index, pd.DatetimeIndex):
            raise ValueError("sent_df must have DatetimeIndex")
        
        # Merge on date (left join to keep all stock data)
        df = df.join(sent_df, how='left')
        
        # Fill missing sentiment with neutral values (0 or median)
        sent_cols = sent_df.columns
        df[sent_cols] = df[sent_cols].fillna(0)
    
    return df


def make_targets(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Create target variable for prediction.
    
    Args:
        df: DataFrame with 'close' column
        mode: 'direction' or 'return'
    
    Returns:
        DataFrame with target column added
    """
    df = df.copy()
    
    if mode == 'direction':
        # Binary: 1 if next-day close > today's close, else 0
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    elif mode == 'return':
        # Continuous: next-day return
        df['target'] = df['close'].pct_change(1).shift(-1)
    
    else:
        raise ValueError(f"Unknown target mode: {mode}. Choose 'direction' or 'return'.")
    
    # Drop the last row (no target for last day)
    df = df.iloc[:-1]
    
    return df


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN values from rolling window warm-up.
    
    Args:
        df: DataFrame with features
    
    Returns:
        DataFrame with NaN rows removed
    """
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    
    if n_before > n_after:
        print(f"Dropped {n_before - n_after} rows with NaN values from warm-up period")
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """
    Get list of feature columns (excluding OHLCV raw, target, etc.).
    
    Args:
        df: DataFrame with all columns
        exclude_cols: Additional columns to exclude
    
    Returns:
        List of feature column names
    """
    # Default exclusions
    default_exclude = ['open', 'high', 'low', 'close', 'volume', 'target']
    
    if exclude_cols:
        default_exclude.extend(exclude_cols)
    
    feature_cols = [col for col in df.columns if col not in default_exclude]
    
    return feature_cols

