"""
Verify that different data sources (stock, sentiment, Fear & Greed) are properly aligned by date.

This script checks:
1. Date ranges of each data source
2. Date alignment after merging
3. That features correctly match their dates
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.dataio import load_ohlcv_csv
from src.sentiment_stub import load_daily_sentiment, load_fear_greed_index
from src.features import build_stock_features
from src.utils import load_config

def verify_data_alignment():
    """Verify date alignment across all data sources."""
    
    print("=" * 80)
    print("DATA ALIGNMENT VERIFICATION")
    print("=" * 80)
    
    # Load config
    cfg = load_config('config/default.yaml')
    
    # ========== 1. Load Stock Data ==========
    print("\n[1] Loading stock data...")
    stock_df = load_ohlcv_csv(
        path=cfg['data']['input_csv'],
        date_col=cfg['data']['date_col'],
        cols=cfg['data']['cols']
    )
    print(f"✓ Stock data loaded: {len(stock_df)} days")
    print(f"  Date range: {stock_df.index[0].date()} to {stock_df.index[-1].date()}")
    print(f"  Sample dates (first 5): {[d.date() for d in stock_df.index[:5]]}")
    
    # ========== 2. Load Sentiment Data ==========
    print("\n[2] Loading sentiment data...")
    sent_csv_path = cfg['sentiment'].get('csv_path', 'data/raw/Daily News Sentiment Index.csv')
    
    if os.path.exists(sent_csv_path):
        # Load raw sentiment to check date range
        sent_raw = pd.read_csv(sent_csv_path)
        sent_raw['date'] = pd.to_datetime(sent_raw['date'], format='%m/%d/%y')
        sent_raw = sent_raw.set_index('date').sort_index()
        
        print(f"✓ Sentiment data loaded: {len(sent_raw)} days")
        print(f"  Date range: {sent_raw.index[0].date()} to {sent_raw.index[-1].date()}")
        print(f"  Sample dates (first 5): {[d.date() for d in sent_raw.index[:5]]}")
        
        # Now load aligned sentiment
        sent_aligned = load_daily_sentiment(stock_df.index, csv_path=sent_csv_path)
        print(f"  After alignment: {len(sent_aligned)} days")
        
        if len(sent_aligned) > 0:
            print(f"  Aligned date range: {sent_aligned.index[0].date()} to {sent_aligned.index[-1].date()}")
    else:
        print(f"⚠ Sentiment file not found: {sent_csv_path}")
        sent_aligned = pd.DataFrame()
    
    # ========== 3. Load Fear & Greed Index ==========
    print("\n[3] Loading Fear & Greed Index...")
    fg_csv_path = cfg['fear_greed'].get('csv_path', 'data/raw/Fear and Greed Index Data - Daily Interpolated.csv')
    
    if os.path.exists(fg_csv_path):
        # Load raw F&G to check date range
        fg_raw = pd.read_csv(fg_csv_path)
        fg_raw['Date'] = pd.to_datetime(fg_raw['Date'])
        fg_raw = fg_raw.set_index('Date').sort_index()
        
        print(f"✓ Fear & Greed data loaded: {len(fg_raw)} days")
        print(f"  Date range: {fg_raw.index[0].date()} to {fg_raw.index[-1].date()}")
        print(f"  Sample dates (first 5): {[d.date() for d in fg_raw.index[:5]]}")
        
        # Now load aligned F&G
        fg_aligned = load_fear_greed_index(stock_df.index, csv_path=fg_csv_path)
        print(f"  After alignment: {len(fg_aligned)} days")
        
        if len(fg_aligned) > 0:
            print(f"  Aligned date range: {fg_aligned.index[0].date()} to {fg_aligned.index[-1].date()}")
    else:
        print(f"⚠ Fear & Greed file not found: {fg_csv_path}")
        fg_aligned = pd.DataFrame()
    
    # ========== 4. Check Date Alignment ==========
    print("\n" + "=" * 80)
    print("DATE ALIGNMENT CHECK")
    print("=" * 80)
    
    # Check if aligned data indices match stock data indices
    if len(sent_aligned) > 0:
        print("\n📊 Sentiment Alignment Check:")
        matching_dates = sent_aligned.index.isin(stock_df.index)
        print(f"  All sentiment dates in stock dates? {matching_dates.all()}")
        
        if not matching_dates.all():
            mismatched = sent_aligned.index[~matching_dates]
            print(f"  ⚠ WARNING: {len(mismatched)} sentiment dates not in stock data:")
            print(f"    {[d.date() for d in mismatched[:5]]}")
        else:
            print(f"  ✅ All {len(sent_aligned)} sentiment dates match stock dates")
    
    if len(fg_aligned) > 0:
        print("\n📊 Fear & Greed Alignment Check:")
        matching_dates = fg_aligned.index.isin(stock_df.index)
        print(f"  All F&G dates in stock dates? {matching_dates.all()}")
        
        if not matching_dates.all():
            mismatched = fg_aligned.index[~matching_dates]
            print(f"  ⚠ WARNING: {len(mismatched)} F&G dates not in stock data:")
            print(f"    {[d.date() for d in mismatched[:5]]}")
        else:
            print(f"  ✅ All {len(fg_aligned)} F&G dates match stock dates")
    
    # ========== 5. Merge Features and Verify ==========
    print("\n" + "=" * 80)
    print("FEATURE MERGING VERIFICATION")
    print("=" * 80)
    
    print("\n[5] Building features with merged data...")
    
    # Merge sentiment and F&G
    sent_df = None
    if len(sent_aligned) > 0 and len(fg_aligned) > 0:
        sent_df = sent_aligned.join(fg_aligned, how='outer')
        print(f"  Combined sentiment + F&G: {len(sent_df)} days")
    elif len(fg_aligned) > 0:
        sent_df = fg_aligned
        print(f"  Using F&G only: {len(sent_df)} days")
    elif len(sent_aligned) > 0:
        sent_df = sent_aligned
        print(f"  Using sentiment only: {len(sent_aligned)} days")
    
    # Build features
    df_feat = build_stock_features(stock_df.copy(), cfg['features'], sent_df=sent_df)
    
    print(f"\n✓ Features built: {len(df_feat)} days")
    print(f"  Columns: {list(df_feat.columns)}")
    
    # ========== 6. Verify Row-by-Row Alignment ==========
    print("\n" + "=" * 80)
    print("ROW-BY-ROW ALIGNMENT VERIFICATION")
    print("=" * 80)
    
    print("\nChecking first 5 rows to verify dates match across all features...")
    
    for i in range(min(5, len(df_feat))):
        date = df_feat.index[i]
        print(f"\n{'─' * 60}")
        print(f"Row {i}: Date = {date.date()}")
        print(f"{'─' * 60}")
        
        # Get stock values
        close_price = df_feat.loc[date, 'close']
        print(f"  Stock close: ${close_price:.2f}")
        
        # Verify this matches original stock data
        if date in stock_df.index:
            original_close = stock_df.loc[date, 'close']
            if abs(close_price - original_close) < 1e-6:
                print(f"  ✅ Matches original stock data")
            else:
                print(f"  ❌ MISMATCH! Original close: ${original_close:.2f}")
        
        # Check sentiment if available
        if sent_df is not None and 'sent_raw' in df_feat.columns:
            if pd.notna(df_feat.loc[date, 'sent_raw']):
                sent_val = df_feat.loc[date, 'sent_raw']
                print(f"  Sentiment: {sent_val:.3f}")
                
                # Verify against original sentiment data
                if date in sent_aligned.index:
                    original_sent = sent_aligned.loc[date, 'sent_raw']
                    if abs(sent_val - original_sent) < 1e-6:
                        print(f"  ✅ Matches original sentiment data")
                    else:
                        print(f"  ❌ MISMATCH! Original sentiment: {original_sent:.3f}")
            else:
                print(f"  Sentiment: NaN (filled with 0 by default)")
        
        # Check Fear & Greed if available
        if sent_df is not None and 'fg_raw' in df_feat.columns:
            if pd.notna(df_feat.loc[date, 'fg_raw']):
                fg_val = df_feat.loc[date, 'fg_raw']
                print(f"  Fear & Greed: {fg_val:.1f}")
                
                # Verify against original F&G data
                if date in fg_aligned.index:
                    original_fg = fg_aligned.loc[date, 'fg_raw']
                    if abs(fg_val - original_fg) < 1e-6:
                        print(f"  ✅ Matches original F&G data")
                    else:
                        print(f"  ❌ MISMATCH! Original F&G: {original_fg:.1f}")
            else:
                print(f"  Fear & Greed: NaN (filled with 0 by default)")
    
    # ========== 7. Check for Missing Data Patterns ==========
    print("\n" + "=" * 80)
    print("MISSING DATA ANALYSIS")
    print("=" * 80)
    
    if sent_df is not None:
        print("\nChecking for missing data after merge...")
        
        # Count NaNs before fillna
        stock_df_temp = stock_df.copy()
        if 'sent_raw' in df_feat.columns or 'fg_raw' in df_feat.columns:
            merged_before_fill = stock_df_temp.join(sent_df, how='left')
            
            if 'sent_raw' in merged_before_fill.columns:
                sent_nans = merged_before_fill['sent_raw'].isna().sum()
                print(f"  Sentiment NaNs: {sent_nans}/{len(merged_before_fill)} ({sent_nans/len(merged_before_fill)*100:.1f}%)")
                
                if sent_nans > 0:
                    print(f"    → These will be filled with 0 (neutral)")
                    # Show sample dates with missing sentiment
                    missing_dates = merged_before_fill[merged_before_fill['sent_raw'].isna()].index[:5]
                    print(f"    Sample missing dates: {[d.date() for d in missing_dates]}")
            
            if 'fg_raw' in merged_before_fill.columns:
                fg_nans = merged_before_fill['fg_raw'].isna().sum()
                print(f"  F&G NaNs: {fg_nans}/{len(merged_before_fill)} ({fg_nans/len(merged_before_fill)*100:.1f}%)")
                
                if fg_nans > 0:
                    print(f"    → These will be filled with 0 (neutral)")
                    missing_dates = merged_before_fill[merged_before_fill['fg_raw'].isna()].index[:5]
                    print(f"    Sample missing dates: {[d.date() for d in missing_dates]}")
    
    # ========== 8. Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n✅ Data Source Summary:")
    print(f"  Stock data:     {len(stock_df)} days ({stock_df.index[0].date()} to {stock_df.index[-1].date()})")
    
    if len(sent_aligned) > 0:
        overlap_sent = len(sent_aligned.index.intersection(stock_df.index))
        print(f"  Sentiment:      {len(sent_aligned)} days, {overlap_sent} overlap with stock")
    else:
        print(f"  Sentiment:      Not loaded")
    
    if len(fg_aligned) > 0:
        overlap_fg = len(fg_aligned.index.intersection(stock_df.index))
        print(f"  Fear & Greed:   {len(fg_aligned)} days, {overlap_fg} overlap with stock")
    else:
        print(f"  Fear & Greed:   Not loaded")
    
    print(f"\n  Final features: {len(df_feat)} days")
    
    print("\n💡 Key Points:")
    print("  1. All data sources are aligned by DatetimeIndex")
    print("  2. Missing dates in sentiment/F&G are filled with 0 (neutral)")
    print("  3. Only stock dates with all required technical features are kept")
    print("  4. The .join(how='left') ensures we keep all stock dates")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    verify_data_alignment()

