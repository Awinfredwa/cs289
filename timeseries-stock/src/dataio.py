"""Data I/O module for loading and cleaning OHLCV CSV data."""

import pandas as pd
from typing import List


def load_ohlcv_csv(path: str, date_col: str, cols: List[str] = None) -> pd.DataFrame:
    """
    Load OHLCV CSV data with date parsing and validation.
    
    Args:
        path: Path to CSV file
        date_col: Name of the date column
        cols: Optional list of columns to load (if None, loads all)
    
    Returns:
        DataFrame with date as index, sorted chronologically
    """
    # Load CSV
    if cols is not None:
        usecols = [date_col] + cols
        df = pd.read_csv(path, usecols=usecols)
    else:
        df = pd.read_csv(path)
    
    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Set date as index and sort
    df = df.set_index(date_col).sort_index()
    
    # Basic validation
    if df.index.duplicated().any():
        print(f"Warning: Found {df.index.duplicated().sum()} duplicate dates. Keeping first occurrence.")
        df = df[~df.index.duplicated(keep='first')]
    
    # Check for required OHLCV columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"Warning: Found null values:\n{null_counts[null_counts > 0]}")
        print("Dropping rows with null values...")
        df = df.dropna()
    
    return df


def validate_trading_continuity(df: pd.DataFrame, max_gap_days: int = 10) -> pd.DataFrame:
    """
    Check for large gaps in trading days (e.g., due to holidays or data issues).
    
    Args:
        df: DataFrame with DatetimeIndex
        max_gap_days: Maximum allowed gap in calendar days
    
    Returns:
        Same DataFrame (with warnings if gaps found)
    """
    if len(df) < 2:
        return df
    
    date_diffs = df.index.to_series().diff()
    large_gaps = date_diffs[date_diffs > pd.Timedelta(days=max_gap_days)]
    
    if len(large_gaps) > 0:
        print(f"Warning: Found {len(large_gaps)} gaps > {max_gap_days} days:")
        for date, gap in large_gaps.items():
            print(f"  {date}: {gap.days} days")
    
    return df


def align_dates_multi_ticker(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Align multiple ticker DataFrames to common dates (for future multi-ticker support).
    
    Args:
        dfs: List of DataFrames with DatetimeIndex
    
    Returns:
        List of aligned DataFrames (inner join on dates)
    """
    if len(dfs) <= 1:
        return dfs
    
    # Find common dates
    common_dates = dfs[0].index
    for df in dfs[1:]:
        common_dates = common_dates.intersection(df.index)
    
    print(f"Aligned {len(dfs)} tickers to {len(common_dates)} common dates")
    
    # Filter to common dates
    aligned = [df.loc[common_dates] for df in dfs]
    
    return aligned

