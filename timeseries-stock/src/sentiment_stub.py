"""
Sentiment data stub for future extension.

This module provides placeholder functions for integrating sentiment data
into the stock prediction pipeline. When sentiment data becomes available,
implement these functions to fetch/load daily sentiment scores.
"""

import pandas as pd
from typing import List


def load_daily_sentiment(dates: pd.DatetimeIndex, ticker: str = None) -> pd.DataFrame:
    """
    Load daily sentiment data for given dates.
    
    Args:
        dates: DatetimeIndex of trading days to fetch sentiment for
        ticker: Optional ticker symbol (for ticker-specific sentiment)
    
    Returns:
        DataFrame with DatetimeIndex and sentiment columns:
            - sent_mean: Mean sentiment score (-1 to 1)
            - sent_std: Std of sentiment scores
            - pos_ratio: Ratio of positive sentiments
            - neg_ratio: Ratio of negative sentiments
            - fear: Fear index (0 to 1)
            - joy: Joy index (0 to 1)
            - volume: Number of sentiment data points
    
    Example:
        # In features.py build_stock_features():
        sent_df = load_daily_sentiment(df.index, ticker='^GSPC')
        df_feat = build_stock_features(df, cfg, sent_df=sent_df)
    
    TODO: Implement actual sentiment loading:
        - Option 1: Load from pre-computed CSV
        - Option 2: Fetch from sentiment API (Twitter, Reddit, news)
        - Option 3: Compute from raw text data
    """
    # Placeholder: return neutral sentiment for all dates
    sentiment_data = {
        'sent_mean': 0.0,
        'sent_std': 0.0,
        'pos_ratio': 0.5,
        'neg_ratio': 0.5,
        'fear': 0.0,
        'joy': 0.0,
        'volume': 0,
    }
    
    df = pd.DataFrame(sentiment_data, index=dates)
    
    print(f"Warning: Using placeholder sentiment data for {len(dates)} days")
    print("  Implement load_daily_sentiment() to use real sentiment data")
    
    return df


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

