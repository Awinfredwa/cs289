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
                          csv_path: str = "data/raw/Fear and Greed Index Data.csv",
                          fill_method: str = 'bfill',
                          shift_days: int = 0) -> pd.DataFrame:
    """
    Load Fear and Greed Index data and align with daily stock dates.
    
    The Fear and Greed Index is weekly data (0-100 scale):
    - 0-25: Extreme Fear
    - 25-45: Fear  
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed
    
    Args:
        dates: DatetimeIndex of daily trading days
        csv_path: Path to Fear and Greed Index CSV file
        fill_method: 'bfill' (default) or 'ffill'
            - 'bfill': Week N's F&G (published on day D) applies to days BEFORE day D
                       → Jan 7 F&G (representing Jan 1-7 emotion) applies to Jan 1-7 features
                       → These features predict future targets based on that week's emotion
            - 'ffill': Week N's F&G applies to days AFTER day D
        shift_days: Number of days to shift the F&G data forward (default: 0)
            - shift_days=0: Jan 7 F&G → predicts Jan 7+horizon target
            - shift_days=N: Jan 7 F&G → predicts Jan 7-N+horizon target
            - For perfect alignment with weekly prediction: set shift_days = -prediction_horizon
              (e.g., shift_days=-5 makes Jan 7 F&G predict Jan 7-14 performance)
    
    Returns:
        DataFrame with DatetimeIndex and fear/greed features:
            - fg_raw: Raw fear/greed index (0-100)
            - fg_change: Week-over-week change in index
            - fg_ma_4: 4-week moving average (monthly trend)
            - fg_normalized: Normalized to [-1, 1] range (0 = neutral, -1 = extreme fear, 1 = extreme greed)
    
    Example with 5-day prediction horizon and backward-fill:
        Jan 7: F&G=63 published (represents Jan 1-7 emotion)
        → Jan 1-6: features use F&G=63 (backward-fill)
        → Jan 7: features use F&G=63
        → Jan 1-7 features predict Jan 6-12 targets (5 days ahead)
        → Result: Jan 1-7 emotion (published Jan 7) → predicts next week ✓
    """
    import os
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Warning: Fear and Greed Index file not found at {csv_path}")
        return pd.DataFrame(index=dates)
    
    # Load Fear and Greed Index CSV
    fg_raw = pd.read_csv(csv_path)
    
    # Parse dates (format: YYYY-MM-DD)
    fg_raw['Date'] = pd.to_datetime(fg_raw['Date'])
    fg_raw = fg_raw.set_index('Date').sort_index()
    
    # Rename column for clarity
    fg_raw = fg_raw.rename(columns={'Value': 'fg_raw'})
    
    # Create derived features BEFORE reindexing to daily
    fg_raw['fg_change'] = fg_raw['fg_raw'].diff()
    fg_raw['fg_ma_4'] = fg_raw['fg_raw'].rolling(window=4, min_periods=1).mean()
    
    # Normalize to [-1, 1] range (50 = 0, 0 = -1, 100 = 1)
    fg_raw['fg_normalized'] = (fg_raw['fg_raw'] - 50) / 50
    
    # Fill to daily frequency
    # Backward-fill: Week N's F&G (published day D) applies to days before D
    #   → Jan 7 F&G (representing Jan 1-7 emotion) applies to Jan 1-7
    # Forward-fill: Week N's F&G applies to days after D
    #   → Jan 7 F&G applies to Jan 7-13
    all_dates = pd.date_range(start=fg_raw.index.min(), end=dates.max(), freq='D')
    fg_daily = fg_raw.reindex(all_dates, method=fill_method)
    
    # Apply shift if requested
    # Negative shift = shift backward in time (earlier dates get later F&G values)
    # This aligns "week N emotion → week N+1 performance"
    if shift_days != 0:
        fg_daily = fg_daily.shift(shift_days)
    
    # Align with requested stock dates
    fg_aligned = fg_daily.loc[fg_daily.index.isin(dates)]
    
    # Determine causality direction
    if fill_method == 'bfill':
        causality = "Week N emotion (published day D) → applies to days before D"
    else:
        causality = "Week N emotion (published day D) → applies to days after D"
    
    if shift_days != 0:
        shift_info = f" | Shift: {shift_days} days ({'earlier dates get later F&G' if shift_days < 0 else 'later dates get later F&G'})"
    else:
        shift_info = ""
    
    print(f"✓ Loaded Fear & Greed Index: {len(fg_aligned)}/{len(dates)} days matched")
    print(f"  Fill method: {fill_method} ({causality}){shift_info}")
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

