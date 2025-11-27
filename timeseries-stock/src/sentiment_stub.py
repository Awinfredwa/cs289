"""
Sentiment data stub for future extension.

This module provides placeholder functions for integrating sentiment data
into the stock prediction pipeline. When sentiment data becomes available,
implement these functions to fetch/load daily sentiment scores.
"""

import pandas as pd
import numpy as np
from typing import List


def load_daily_sentiment(dates: pd.DatetimeIndex, ticker: str = None, 
                         csv_path: str = "data/raw/Daily News Sentiment Index.csv") -> pd.DataFrame:
    """
    Load daily sentiment data for given dates.
    
    Args:
        dates: DatetimeIndex of trading days to fetch sentiment for
        ticker: Optional ticker symbol (for ticker-specific sentiment)
        csv_path: Path to sentiment CSV file
    
    Returns:
        DataFrame with DatetimeIndex and sentiment columns:
            - sent_raw: Raw sentiment score from CSV
            - sent_ma_5: 5-day moving average of sentiment
            - sent_std_20: 20-day rolling std of sentiment
            - sent_change: Day-over-day sentiment change
    
    Example:
        # In features.py build_stock_features():
        sent_df = load_daily_sentiment(df.index, ticker='^GSPC')
        df_feat = build_stock_features(df, cfg, sent_df=sent_df)
    """
    import os
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Warning: Sentiment file not found at {csv_path}")
        print(f"  Returning empty sentiment DataFrame")
        return pd.DataFrame(index=dates)
    
    # Load sentiment CSV
    sent_raw = pd.read_csv(csv_path)
    
    # Parse dates (format: M/D/YY)
    sent_raw['date'] = pd.to_datetime(sent_raw['date'], format='%m/%d/%y')
    sent_raw = sent_raw.set_index('date').sort_index()
    
    # Rename column for clarity
    sent_raw = sent_raw.rename(columns={'News Sentiment': 'sent_raw'})
    
    # Create derived sentiment features (reduced redundancy)
    sent_raw['sent_ma_5'] = sent_raw['sent_raw'].rolling(window=5, min_periods=1).mean()
    sent_raw['sent_std_20'] = sent_raw['sent_raw'].rolling(window=20, min_periods=1).std()
    sent_raw['sent_change'] = sent_raw['sent_raw'].diff()
    # Dropped: sent_ma_20 (redundant with sent_ma_5), sent_positive (constant/low info)
    
    # Align with requested dates (inner join - only keep matching dates)
    sent_aligned = sent_raw.loc[sent_raw.index.isin(dates)]
    
    print(f"✓ Loaded sentiment data: {len(sent_aligned)}/{len(dates)} days matched")
    print(f"  Sentiment range: [{sent_aligned['sent_raw'].min():.3f}, {sent_aligned['sent_raw'].max():.3f}]")
    print(f"  Mean sentiment: {sent_aligned['sent_raw'].mean():.3f}")
    
    return sent_aligned


def aggregate_sentiment_to_daily(
    raw_sentiment: pd.DataFrame,
    date_col: str = 'timestamp',
    sentiment_col: str = 'score'
) -> pd.DataFrame:
    """
    Aggregate intraday sentiment data to daily frequency.
    
    Args:
        raw_sentiment: DataFrame with timestamp and sentiment scores
        date_col: Name of timestamp column
        sentiment_col: Name of sentiment score column
    
    Returns:
        DataFrame with daily aggregated sentiment
    
    TODO: Implement aggregation logic:
        - Convert timestamps to dates
        - Group by date
        - Compute daily statistics (mean, std, counts, etc.)
    """
    raise NotImplementedError("Implement sentiment aggregation logic")


def fetch_twitter_sentiment(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Twitter sentiment for a ticker in date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with daily Twitter sentiment
    
    TODO: Implement Twitter API integration:
        - Use Twitter API v2 or similar
        - Search for tweets mentioning ticker
        - Apply sentiment analysis (e.g., VADER, FinBERT)
        - Aggregate to daily frequency
    """
    raise NotImplementedError("Implement Twitter sentiment fetching")


def fetch_news_sentiment(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch news sentiment for a ticker in date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with daily news sentiment
    
    TODO: Implement news API integration:
        - Use News API, Alpha Vantage, or similar
        - Fetch news headlines/articles mentioning ticker
        - Apply sentiment analysis
        - Aggregate to daily frequency
    """
    raise NotImplementedError("Implement news sentiment fetching")


def load_fear_greed_index(dates: pd.DatetimeIndex, 
                          csv_path: str = "data/raw/Fear and Greed Index Data - Daily Interpolated.csv") -> pd.DataFrame:
    """
    Load daily interpolated Fear and Greed Index data and align with stock dates.
    
    The Fear and Greed Index is a daily interpolated market sentiment indicator (0-100 scale):
    - 0-25: Extreme Fear
    - 25-45: Fear  
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed
    
    Args:
        dates: DatetimeIndex of daily trading days
        csv_path: Path to daily interpolated Fear and Greed Index CSV file
    
    Returns:
        DataFrame with DatetimeIndex and fear/greed features:
            - fg_raw: Raw fear/greed index (0-100)
            - fg_change: Day-over-day change in index
            - fg_ma_4: 4-day moving average
            - fg_normalized: Normalized to [-1, 1] range (0 = neutral, -1 = extreme fear, 1 = extreme greed)
    
    Note: Use scripts/interpolate_fear_greed.py to generate daily interpolated data
          from the original weekly Fear & Greed Index CSV.
    """
    import os
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Warning: Fear and Greed Index file not found at {csv_path}")
        print(f"  Run: python scripts/interpolate_fear_greed.py")
        return pd.DataFrame(index=dates)
    
    # Load daily interpolated Fear and Greed Index CSV
    fg_daily = pd.read_csv(csv_path)
    
    # Parse dates (format: YYYY-MM-DD)
    fg_daily['Date'] = pd.to_datetime(fg_daily['Date'])
    fg_daily = fg_daily.set_index('Date').sort_index()
    
    # Rename column for clarity
    fg_daily = fg_daily.rename(columns={'Value': 'fg_raw'})
    
    # Create derived features
    fg_daily['fg_change'] = fg_daily['fg_raw'].diff()
    fg_daily['fg_ma_4'] = fg_daily['fg_raw'].rolling(window=4, min_periods=1).mean()
    
    # Normalize to [-1, 1] range (50 = 0, 0 = -1, 100 = 1)
    fg_daily['fg_normalized'] = (fg_daily['fg_raw'] - 50) / 50
    
    # Align with requested stock dates
    fg_aligned = fg_daily.loc[fg_daily.index.isin(dates)]
    
    print(f"✓ Loaded Fear & Greed Index: {len(fg_aligned)}/{len(dates)} days matched")
    print(f"  Daily interpolated data (smooth transitions)")
    print(f"  Index range: [{fg_aligned['fg_raw'].min():.0f}, {fg_aligned['fg_raw'].max():.0f}]")
    print(f"  Mean index: {fg_aligned['fg_raw'].mean():.1f} ({'Neutral' if 45 <= fg_aligned['fg_raw'].mean() <= 55 else 'Greed' if fg_aligned['fg_raw'].mean() > 55 else 'Fear'})")
    
    return fg_aligned


def compute_fear_greed_index(df: pd.DataFrame) -> pd.Series:
    """
    Compute a fear/greed index from sentiment data.
    
    Args:
        df: DataFrame with sentiment columns
    
    Returns:
        Series with fear/greed index (0=extreme fear, 100=extreme greed)
    
    TODO: Implement fear/greed calculation:
        - Combine multiple sentiment signals
        - Normalize to 0-100 scale
        - Consider market volatility, volume, etc.
    """
    raise NotImplementedError("Implement fear/greed index calculation")


# Extension example: how to integrate sentiment into the pipeline
"""
INTEGRATION GUIDE
=================

1. In src/train.py, after loading OHLCV data:
   
   from src.sentiment_stub import load_daily_sentiment
   
   # Load sentiment
   sent_df = load_daily_sentiment(df.index, ticker='^GSPC')
   
   # Pass to feature builder
   df_feat = build_stock_features(df, cfg['features'], sent_df=sent_df)

2. The feature builder (src/features.py) already supports sent_df:
   
   def build_stock_features(df, cfg, sent_df=None):
       # ... build stock features ...
       
       if sent_df is not None:
           df = df.join(sent_df, how='left')
           df[sent_cols] = df[sent_cols].fillna(0)
       
       return df

3. The rest of the pipeline (windowing, scaling, training) will
   automatically work with the additional sentiment features!

4. Update config.yaml to enable sentiment:
   
   data:
     use_sentiment: true
     sentiment_source: "twitter"  # or "news", "reddit", etc.
   
   features:
     sentiment_windows: [1, 3, 7]  # rolling sentiment over N days

5. Implement the actual sentiment fetching/loading logic in this file.
"""

