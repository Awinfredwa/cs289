#!/usr/bin/env python3
"""
Download real stock data from Yahoo Finance and save to SAMPLE.csv.

This script downloads AAPL (Apple) stock data for the period
2020-01-01 to 2020-05-21 and saves it in the format expected by
the training pipeline.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# Configuration
TICKER = 'AAPL'
START_DATE = '2010-01-01'
END_DATE = '2020-05-21'
OUTPUT_FILE = Path(__file__).parent.parent / 'data' / 'raw' / 'SAMPLE.csv'


def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download stock data from Yahoo Finance."""
    print(f"Downloading {ticker} data from {start} to {end}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    
    if df.empty:
        raise ValueError(f"No data retrieved for {ticker}. Check date range and ticker symbol.")
    
    return df


def format_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format the dataframe to match expected CSV format."""
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Rename columns to match expected format
    df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    # Format date column as YYYY-MM-DD
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Select and reorder columns
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    return df


def main():
    """Main function to download and save stock data."""
    try:
        # Download data
        df = download_stock_data(TICKER, START_DATE, END_DATE)
        
        # Format data
        df = format_data(df)
        
        # Ensure output directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n✓ Successfully downloaded {len(df)} days of {TICKER} data")
        print(f"✓ Saved to: {OUTPUT_FILE}")
        print(f"✓ Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        print(f"\nFirst few rows:")
        print(df.head().to_string(index=False))
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

