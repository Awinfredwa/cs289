#!/usr/bin/env python3
"""
Analyze and visualize features for debugging and exploration.

Usage:
    python scripts/analyze_features.py --config config/default.yaml
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import yaml

from src.dataio import load_ohlcv_csv
from src.sentiment_stub import load_daily_sentiment
from src.features import build_stock_features, make_targets, drop_warmup_rows, get_feature_columns


def print_feature_correlations(df, feature_cols, target_col='target'):
    """Print correlations between features and target."""
    print(f"\n{'='*70}")
    print("FEATURE-TARGET CORRELATIONS")
    print(f"{'='*70}")
    
    correlations = []
    for feat in feature_cols:
        corr = df[feat].corr(df[target_col])
        correlations.append((feat, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n📊 Top Features by Correlation with Target:\n")
    for i, (feat, corr) in enumerate(correlations[:15], 1):
        bar = '█' * int(abs(corr) * 50)
        sign = '+' if corr > 0 else '-'
        print(f"  {i:2d}. {feat:20s} | {sign}{abs(corr):.4f} {bar}")


def print_feature_distributions(df, feature_cols):
    """Print distribution statistics for each feature."""
    print(f"\n{'='*70}")
    print("FEATURE DISTRIBUTIONS")
    print(f"{'='*70}\n")
    
    stats = []
    for feat in feature_cols:
        values = df[feat]
        stats.append({
            'Feature': feat,
            'Mean': values.mean(),
            'Std': values.std(),
            'Min': values.min(),
            'Q25': values.quantile(0.25),
            'Median': values.median(),
            'Q75': values.quantile(0.75),
            'Max': values.max(),
            'Nulls': values.isnull().sum()
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False, float_format='%.4f'))


def print_feature_importance_proxy(df, feature_cols, target_col='target'):
    """Simple variance-based feature importance proxy."""
    print(f"\n{'='*70}")
    print("FEATURE VARIANCE (Higher = More Information)")
    print(f"{'='*70}\n")
    
    variances = []
    for feat in feature_cols:
        var = df[feat].var()
        variances.append((feat, var))
    
    variances.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feat, var) in enumerate(variances[:15], 1):
        normalized_var = var / max(v[1] for v in variances)
        bar = '█' * int(normalized_var * 40)
        print(f"  {i:2d}. {feat:20s} | {var:10.6f} {bar}")


def print_sentiment_analysis(df):
    """Analyze sentiment features specifically."""
    sent_cols = [c for c in df.columns if c.startswith('sent_')]
    
    if not sent_cols:
        print("\n⚠️  No sentiment features found")
        return
    
    print(f"\n{'='*70}")
    print(f"SENTIMENT ANALYSIS ({len(sent_cols)} features)")
    print(f"{'='*70}")
    
    for col in sent_cols:
        values = df[col]
        print(f"\n{col}:")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Std:  {values.std():.4f}")
        print(f"  Min:  {values.min():.4f}")
        print(f"  Max:  {values.max():.4f}")
        
        # Show sentiment regime
        if 'sent_raw' in col or 'sent_ma' in col:
            positive_pct = (values > 0).sum() / len(values) * 100
            negative_pct = (values < 0).sum() / len(values) * 100
            neutral_pct = (values == 0).sum() / len(values) * 100
            print(f"  Positive days: {positive_pct:.1f}%")
            print(f"  Negative days: {negative_pct:.1f}%")
            print(f"  Neutral days:  {neutral_pct:.1f}%")


def print_sample_window(df, feature_cols, window_size=7):
    """Show a sample window of features."""
    print(f"\n{'='*70}")
    print(f"SAMPLE {window_size}-DAY WINDOW")
    print(f"{'='*70}\n")
    
    start_idx = len(df) // 2  # Middle of dataset
    window_data = df[feature_cols].iloc[start_idx:start_idx+window_size]
    
    print(f"Dates: {df.index[start_idx]} to {df.index[start_idx+window_size-1]}")
    print(f"\nFeature values (transposed for readability):\n")
    print(window_data.T.to_string(float_format='%.4f'))


def main():
    parser = argparse.ArgumentParser(description="Analyze features")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    print("="*70)
    print("FEATURE ANALYSIS TOOL")
    print("="*70)
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_ohlcv_csv(
        path=cfg['data']['input_csv'],
        date_col=cfg['data']['date_col'],
        cols=cfg['data']['cols']
    )
    print(f"✓ Loaded {len(df)} days")
    
    # Load sentiment if enabled
    print("\n[2/5] Loading sentiment...")
    sent_df = None
    if cfg.get('sentiment', {}).get('enabled', False):
        sent_csv_path = cfg['sentiment'].get('csv_path')
        sent_df = load_daily_sentiment(df.index, csv_path=sent_csv_path)
        print(f"✓ Sentiment loaded: {len(sent_df)} days")
    else:
        print("✓ Sentiment disabled")
    
    # Build features
    print("\n[3/5] Building features...")
    df_feat = build_stock_features(df, cfg['features'], sent_df=sent_df)
    print(f"✓ Built {len(df_feat.columns)} columns")
    
    # Create targets
    print("\n[4/5] Creating targets...")
    horizon = cfg['data'].get('prediction_horizon', 1)
    df_feat = make_targets(df_feat, mode=cfg['data']['target'], horizon=horizon)
    df_feat = drop_warmup_rows(df_feat)
    print(f"✓ {len(df_feat)} samples ready")
    
    # Get features
    print("\n[5/5] Analyzing features...")
    feature_cols = get_feature_columns(df_feat)
    print(f"✓ {len(feature_cols)} features to analyze")
    
    # Run analyses
    print_feature_distributions(df_feat, feature_cols)
    print_feature_correlations(df_feat, feature_cols)
    print_feature_importance_proxy(df_feat, feature_cols)
    print_sentiment_analysis(df_feat)
    print_sample_window(df_feat, feature_cols, window_size=cfg['window']['size'])
    
    # Target distribution
    print(f"\n{'='*70}")
    print("TARGET DISTRIBUTION")
    print(f"{'='*70}\n")
    target_counts = df_feat['target'].value_counts()
    print(target_counts)
    print(f"\nBalance: {target_counts.min() / target_counts.max():.2%}")
    
    print(f"\n{'='*70}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

