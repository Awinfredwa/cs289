#!/usr/bin/env python3
"""
Interpolate weekly Fear & Greed Index to daily frequency using linear interpolation.

This script reads the weekly Fear & Greed Index data and creates a daily version
by linearly interpolating between weekly data points.

Example:
    Jan 7:  F&G = 63
    Jan 14: F&G = 70
    
    Linear interpolation:
    Jan 7:  63.0
    Jan 8:  64.0
    Jan 9:  65.0
    Jan 10: 66.0
    Jan 11: 67.0
    Jan 12: 68.0
    Jan 13: 69.0
    Jan 14: 70.0
"""

import pandas as pd
import numpy as np
from pathlib import Path


def interpolate_fear_greed(
    input_csv: str = "data/raw/Fear and Greed Index Data.csv",
    output_csv: str = "data/raw/Fear and Greed Index Data - Daily Interpolated.csv",
    method: str = 'linear'
):
    """
    Interpolate weekly Fear & Greed Index to daily frequency.
    
    Args:
        input_csv: Path to weekly Fear & Greed Index CSV
        output_csv: Path to save daily interpolated CSV
        method: Interpolation method ('linear', 'cubic', 'quadratic')
                'linear' is recommended for smoothness without overfitting
    
    Returns:
        DataFrame with daily interpolated Fear & Greed Index
    """
    
    print("=" * 80)
    print("FEAR & GREED INDEX - DAILY INTERPOLATION")
    print("=" * 80)
    
    # Load weekly data
    print(f"\n[1/4] Loading weekly Fear & Greed Index from: {input_csv}")
    fg_weekly = pd.read_csv(input_csv)
    
    # Parse dates
    fg_weekly['Date'] = pd.to_datetime(fg_weekly['Date'])
    fg_weekly = fg_weekly.set_index('Date').sort_index()
    
    # Remove any rows with missing values
    fg_weekly = fg_weekly.dropna()
    
    print(f"✓ Loaded {len(fg_weekly)} weekly data points")
    print(f"  Date range: {fg_weekly.index.min().date()} to {fg_weekly.index.max().date()}")
    print(f"  Value range: [{fg_weekly['Value'].min():.0f}, {fg_weekly['Value'].max():.0f}]")
    print(f"  Mean value: {fg_weekly['Value'].mean():.1f}")
    
    # Create daily date range
    print(f"\n[2/4] Creating daily date range...")
    daily_dates = pd.date_range(
        start=fg_weekly.index.min(),
        end=fg_weekly.index.max(),
        freq='D'
    )
    print(f"✓ Created {len(daily_dates)} daily dates")
    
    # Reindex to daily and interpolate
    print(f"\n[3/4] Interpolating with method: '{method}'...")
    fg_daily = fg_weekly.reindex(daily_dates)
    
    # Interpolate missing values
    fg_daily['Value'] = fg_daily['Value'].interpolate(method=method)
    
    # Handle any remaining NaN (at edges)
    fg_daily['Value'] = fg_daily['Value'].fillna(method='bfill').fillna(method='ffill')
    
    print(f"✓ Interpolation complete")
    print(f"  Daily values range: [{fg_daily['Value'].min():.2f}, {fg_daily['Value'].max():.2f}]")
    print(f"  Daily values mean: {fg_daily['Value'].mean():.2f}")
    
    # Show example of interpolation
    print(f"\n📊 Example: First week interpolation")
    print("=" * 60)
    first_week = fg_daily.iloc[:14]
    for i, (date, row) in enumerate(first_week.iterrows()):
        is_weekly = date in fg_weekly.index
        marker = "📍 WEEKLY" if is_weekly else "  (interp)"
        print(f"{date.date()} | Value: {row['Value']:6.2f} {marker}")
    
    # Save to CSV
    print(f"\n[4/4] Saving daily interpolated data to: {output_csv}")
    fg_daily_save = fg_daily.reset_index()
    fg_daily_save.columns = ['Date', 'Value']
    
    # Create output directory if needed
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    fg_daily_save.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(fg_daily_save)} daily data points")
    
    # Statistics
    print(f"\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Weekly data points:  {len(fg_weekly)}")
    print(f"Daily data points:   {len(fg_daily)}")
    print(f"Interpolation ratio: {len(fg_daily) / len(fg_weekly):.1f}x")
    print(f"\nWeekly mean:  {fg_weekly['Value'].mean():.2f}")
    print(f"Daily mean:   {fg_daily['Value'].mean():.2f}")
    print(f"Difference:   {abs(fg_weekly['Value'].mean() - fg_daily['Value'].mean()):.4f}")
    
    print(f"\n✅ Daily interpolated Fear & Greed Index saved successfully!")
    print(f"   You can now use: {output_csv}")
    print("=" * 80)
    
    return fg_daily


def compare_methods(input_csv: str = "data/raw/Fear and Greed Index Data.csv"):
    """
    Compare different interpolation methods visually.
    
    Shows the difference between linear, cubic, and quadratic interpolation
    for a sample week.
    """
    
    print("\n" + "=" * 80)
    print("COMPARING INTERPOLATION METHODS")
    print("=" * 80)
    
    # Load data
    fg_weekly = pd.read_csv(input_csv)
    fg_weekly['Date'] = pd.to_datetime(fg_weekly['Date'])
    fg_weekly = fg_weekly.set_index('Date').sort_index()
    fg_weekly = fg_weekly.dropna()
    
    # Take first 4 weeks for comparison
    sample_weekly = fg_weekly.iloc[:4]
    daily_dates = pd.date_range(
        start=sample_weekly.index.min(),
        end=sample_weekly.index.max(),
        freq='D'
    )
    
    methods = ['linear', 'cubic', 'quadratic']
    results = {}
    
    for method in methods:
        fg_daily = sample_weekly.reindex(daily_dates)
        fg_daily['Value'] = fg_daily['Value'].interpolate(method=method)
        results[method] = fg_daily
    
    # Display comparison
    print(f"\nSample period: {daily_dates[0].date()} to {daily_dates[-1].date()}")
    print("\n" + "-" * 80)
    print(f"{'Date':<12} | {'Linear':<8} | {'Cubic':<8} | {'Quadratic':<10} | {'Note'}")
    print("-" * 80)
    
    for date in daily_dates:
        is_weekly = date in sample_weekly.index
        linear_val = results['linear'].loc[date, 'Value']
        cubic_val = results['cubic'].loc[date, 'Value']
        quad_val = results['quadratic'].loc[date, 'Value']
        
        note = "📍 WEEKLY" if is_weekly else ""
        
        print(f"{date.date()} | {linear_val:7.2f}  | {cubic_val:7.2f}  | {quad_val:8.2f}  | {note}")
    
    print("-" * 80)
    print("\n💡 Recommendation: Use 'linear' for simplicity and to avoid overshooting")
    print("   Linear interpolation provides smooth, monotonic transitions between weeks.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interpolate weekly Fear & Greed Index to daily frequency"
    )
    parser.add_argument(
        '--input',
        default='data/raw/Fear and Greed Index Data.csv',
        help='Path to weekly Fear & Greed CSV'
    )
    parser.add_argument(
        '--output',
        default='data/raw/Fear and Greed Index Data - Daily Interpolated.csv',
        help='Path to save daily interpolated CSV'
    )
    parser.add_argument(
        '--method',
        default='linear',
        choices=['linear', 'cubic', 'quadratic'],
        help='Interpolation method (default: linear)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare different interpolation methods'
    )
    
    args = parser.parse_args()
    
    # Run interpolation
    fg_daily = interpolate_fear_greed(
        input_csv=args.input,
        output_csv=args.output,
        method=args.method
    )
    
    # Compare methods if requested
    if args.compare:
        compare_methods(input_csv=args.input)

