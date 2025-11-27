"""
Analyze correlation between features and future returns.

This script checks if Fear & Greed Index, sentiment, and technical indicators
actually have predictive power for future returns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.dataio import load_ohlcv_csv
from src.features import build_stock_features, get_feature_columns
from src.sentiment_stub import load_daily_sentiment, load_fear_greed_index
from src.utils import load_config

print("=" * 80)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 80)

# Load config
cfg = load_config('config/aggressive_learning.yaml')

# Load stock data
print("\n[1/4] Loading data...")
df = load_ohlcv_csv(
    path=cfg['data']['input_csv'],
    date_col=cfg['data']['date_col'],
    cols=cfg['data']['cols']
)
print(f"✓ Loaded {len(df)} days of stock data")

# Load sentiment if enabled
sent_df = None
if cfg.get('sentiment', {}).get('enabled', False):
    sent_csv_path = cfg['sentiment'].get('csv_path')
    sent_df = load_daily_sentiment(df.index, csv_path=sent_csv_path)
    if len(sent_df) > 0:
        print(f"✓ Loaded sentiment data")

# Load Fear & Greed if enabled
fg_df = None
if cfg.get('fear_greed', {}).get('enabled', False):
    fg_csv_path = cfg['fear_greed'].get('csv_path')
    fg_df = load_fear_greed_index(df.index, csv_path=fg_csv_path)
    if len(fg_df) > 0:
        print(f"✓ Loaded Fear & Greed Index")

# Merge sentiment and F&G
if sent_df is not None and fg_df is not None:
    sent_df = sent_df.join(fg_df, how='outer')
elif fg_df is not None:
    sent_df = fg_df

# Build features
print("\n[2/4] Building features...")
df_feat = build_stock_features(df, cfg['features'], sent_df=sent_df)

# Create future returns at different horizons
print("\n[3/4] Computing future returns...")
horizons = [1, 5, 10, 20]  # 1-day, 1-week, 2-week, 1-month
for h in horizons:
    df_feat[f'return_{h}d'] = (df_feat['close'].shift(-h) / df_feat['close']) - 1

# Drop rows with NaN
df_feat = df_feat.dropna()
print(f"✓ Final dataset: {len(df_feat)} days")

# Get feature columns (exclude OHLCV and future returns)
exclude_cols = ['open', 'high', 'low', 'close', 'volume'] + [f'return_{h}d' for h in horizons]
feature_cols = [col for col in df_feat.columns if col not in exclude_cols]

print(f"✓ Analyzing {len(feature_cols)} features")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("CORRELATION WITH FUTURE RETURNS")
print("=" * 80)

# Analyze correlations for each horizon
results = []

for horizon in horizons:
    future_return_col = f'return_{h}d'
    
    print(f"\n{'─' * 80}")
    print(f"Predicting {horizon}-Day Ahead Returns")
    print(f"{'─' * 80}")
    
    for feature in feature_cols:
        # Remove NaN pairs
        mask = ~(df_feat[feature].isna() | df_feat[future_return_col].isna())
        feat_values = df_feat.loc[mask, feature].values
        return_values = df_feat.loc[mask, future_return_col].values
        
        if len(feat_values) < 30:  # Need minimum samples
            continue
        
        # Pearson correlation (linear)
        try:
            corr_pearson, p_pearson = pearsonr(feat_values, return_values)
        except:
            corr_pearson, p_pearson = np.nan, np.nan
        
        # Spearman correlation (monotonic)
        try:
            corr_spearman, p_spearman = spearmanr(feat_values, return_values)
        except:
            corr_spearman, p_spearman = np.nan, np.nan
        
        results.append({
            'horizon': horizon,
            'feature': feature,
            'pearson_corr': corr_pearson,
            'pearson_p': p_pearson,
            'spearman_corr': corr_spearman,
            'spearman_p': p_spearman,
            'n_samples': len(feat_values)
        })

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# ============================================================================
# DETAILED RESULTS FOR EACH HORIZON
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED RESULTS")
print("=" * 80)

for horizon in horizons:
    print(f"\n{'═' * 80}")
    print(f"📊 {horizon}-DAY AHEAD PREDICTION")
    print(f"{'═' * 80}")
    
    horizon_results = results_df[results_df['horizon'] == horizon].copy()
    
    # Sort by absolute Pearson correlation
    horizon_results['abs_pearson'] = horizon_results['pearson_corr'].abs()
    horizon_results = horizon_results.sort_values('abs_pearson', ascending=False)
    
    print(f"\n{'Feature':<25} | {'Pearson':<12} | {'p-value':<10} | {'Spearman':<12} | {'Interpretation'}")
    print("─" * 100)
    
    for _, row in horizon_results.iterrows():
        feat = row['feature']
        pearson = row['pearson_corr']
        p_pear = row['pearson_p']
        spearman = row['spearman_corr']
        
        # Interpretation
        abs_corr = abs(pearson)
        if abs_corr < 0.05:
            interp = "No relationship"
        elif abs_corr < 0.10:
            interp = "Very weak"
        elif abs_corr < 0.15:
            interp = "Weak"
        elif abs_corr < 0.25:
            interp = "Moderate"
        else:
            interp = "Strong"
        
        # Significance
        sig = "***" if p_pear < 0.001 else "**" if p_pear < 0.01 else "*" if p_pear < 0.05 else ""
        
        print(f"{feat:<25} | {pearson:>6.4f} {sig:<5} | {p_pear:>8.6f} | {spearman:>6.4f}      | {interp}")

# ============================================================================
# FEAR & GREED SPECIFIC ANALYSIS
# ============================================================================
if 'fg_raw' in feature_cols:
    print("\n" + "=" * 80)
    print("🔍 FEAR & GREED INDEX DETAILED ANALYSIS")
    print("=" * 80)
    
    fg_results = results_df[results_df['feature'] == 'fg_raw']
    
    for _, row in fg_results.iterrows():
        horizon = row['horizon']
        corr_p = row['pearson_corr']
        p_p = row['pearson_p']
        corr_s = row['spearman_corr']
        p_s = row['spearman_p']
        
        print(f"\n{horizon}-Day Ahead:")
        print(f"  Pearson correlation:  {corr_p:>7.4f} (p={p_p:.6f})")
        print(f"  Spearman correlation: {corr_s:>7.4f} (p={p_s:.6f})")
        
        # Interpretation
        if abs(corr_p) < 0.05:
            print(f"  ❌ No meaningful relationship")
            print(f"     Fear & Greed Index is NOT predictive of {horizon}-day returns")
        elif abs(corr_p) < 0.10:
            print(f"  ⚠️  Very weak relationship")
            print(f"     Fear & Greed may have minimal predictive value")
        elif abs(corr_p) < 0.15:
            print(f"  ⚠️  Weak relationship")
            print(f"     Fear & Greed has some signal but limited")
        else:
            print(f"  ✅ Moderate relationship")
            print(f"     Fear & Greed has useful predictive signal!")
        
        # Statistical significance
        if p_p < 0.001:
            print(f"  ✅ Highly statistically significant (p < 0.001)")
        elif p_p < 0.01:
            print(f"  ✅ Very statistically significant (p < 0.01)")
        elif p_p < 0.05:
            print(f"  ✅ Statistically significant (p < 0.05)")
        else:
            print(f"  ❌ NOT statistically significant (p = {p_p:.4f})")
            print(f"     Could be due to random chance")
    
    # Analyze F&G in different regimes
    print(f"\n{'─' * 80}")
    print(f"Fear & Greed in Different Market Regimes")
    print(f"{'─' * 80}")
    
    # Use 5-day returns for this analysis
    fg_values = df_feat['fg_raw'].values
    returns_5d = df_feat['return_5d'].values
    
    # Define F&G regimes
    extreme_fear = fg_values < 25
    fear = (fg_values >= 25) & (fg_values < 45)
    neutral = (fg_values >= 45) & (fg_values < 55)
    greed = (fg_values >= 55) & (fg_values < 75)
    extreme_greed = fg_values >= 75
    
    print(f"\nAverage 5-day returns by Fear & Greed regime:")
    print(f"  Extreme Fear (<25):   {returns_5d[extreme_fear].mean():>7.4f} ({extreme_fear.sum()} days)")
    print(f"  Fear (25-45):         {returns_5d[fear].mean():>7.4f} ({fear.sum()} days)")
    print(f"  Neutral (45-55):      {returns_5d[neutral].mean():>7.4f} ({neutral.sum()} days)")
    print(f"  Greed (55-75):        {returns_5d[greed].mean():>7.4f} ({greed.sum()} days)")
    print(f"  Extreme Greed (>75):  {returns_5d[extreme_greed].mean():>7.4f} ({extreme_greed.sum()} days)")
    
    # Check if there's a clear pattern
    regime_returns = [
        returns_5d[extreme_fear].mean(),
        returns_5d[fear].mean(),
        returns_5d[neutral].mean(),
        returns_5d[greed].mean(),
        returns_5d[extreme_greed].mean()
    ]
    
    if np.all(np.diff(regime_returns) > 0):
        print(f"\n  📈 Clear positive pattern: Higher F&G → Higher returns")
        print(f"     Consider keeping F&G features")
    elif np.all(np.diff(regime_returns) < 0):
        print(f"\n  📉 Clear negative pattern: Higher F&G → Lower returns (contrarian)")
        print(f"     Consider keeping F&G features")
    else:
        print(f"\n  ⚠️  No clear monotonic pattern")
        print(f"     F&G may have limited predictive value")

# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================
if 'sent_raw' in feature_cols:
    print("\n" + "=" * 80)
    print("📰 SENTIMENT INDEX DETAILED ANALYSIS")
    print("=" * 80)
    
    sent_results = results_df[results_df['feature'] == 'sent_raw']
    
    for _, row in sent_results.iterrows():
        horizon = row['horizon']
        corr_p = row['pearson_corr']
        p_p = row['pearson_p']
        
        print(f"\n{horizon}-Day Ahead:")
        print(f"  Pearson correlation:  {corr_p:>7.4f} (p={p_p:.6f})")
        
        if p_p < 0.05 and abs(corr_p) > 0.10:
            print(f"  ✅ Sentiment has predictive signal")
        else:
            print(f"  ⚠️  Sentiment has limited predictive value")

# ============================================================================
# TOP PREDICTIVE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("🏆 TOP PREDICTIVE FEATURES (5-DAY AHEAD)")
print("=" * 80)

horizon_5d = results_df[results_df['horizon'] == 5].copy()
horizon_5d['abs_corr'] = horizon_5d['pearson_corr'].abs()
top_features = horizon_5d.nlargest(10, 'abs_corr')

print(f"\nTop 10 features most correlated with 5-day returns:\n")
print(f"{'Rank':<6} {'Feature':<25} {'Correlation':<12} {'p-value':<10} {'Usefulness'}")
print("─" * 80)

for rank, (_, row) in enumerate(top_features.iterrows(), 1):
    feat = row['feature']
    corr = row['pearson_corr']
    p_val = row['pearson_p']
    
    if p_val < 0.05 and abs(corr) > 0.10:
        usefulness = "✅ Useful"
    elif p_val < 0.05:
        usefulness = "⚠️  Weak signal"
    else:
        usefulness = "❌ Not significant"
    
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{rank:<6} {feat:<25} {corr:>6.4f} {sig:<5} {p_val:>8.6f} {usefulness}")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)

# Count useful features
useful_features = results_df[(results_df['horizon'] == 5) & 
                              (results_df['pearson_p'] < 0.05) & 
                              (results_df['pearson_corr'].abs() > 0.10)]

print(f"\n📊 Summary:")
print(f"  Total features analyzed: {len(feature_cols)}")
print(f"  Statistically significant (p<0.05): {len(results_df[(results_df['horizon']==5) & (results_df['pearson_p']<0.05)])}")
print(f"  Useful (p<0.05 & |r|>0.10): {len(useful_features)}")

if len(useful_features) == 0:
    print(f"\n⚠️  WARNING: No features show strong correlation with future returns!")
    print(f"\n   This explains why your model struggles. Possible reasons:")
    print(f"   1. Stock returns are highly unpredictable (markets are efficient)")
    print(f"   2. Need more sophisticated features (non-linear relationships)")
    print(f"   3. Need shorter prediction horizon (1-day instead of 5-day)")
    print(f"   4. External factors (news, macro events) dominate technical patterns")
else:
    print(f"\n✅ Found {len(useful_features)} useful features")
    print(f"\n   Focus model training on these features for best results")

# F&G specific recommendation
if 'fg_raw' in feature_cols:
    fg_5d = results_df[(results_df['feature'] == 'fg_raw') & (results_df['horizon'] == 5)].iloc[0]
    
    if fg_5d['pearson_p'] < 0.05 and abs(fg_5d['pearson_corr']) > 0.10:
        print(f"\n✅ KEEP Fear & Greed Index - it has predictive value")
    else:
        print(f"\n❌ CONSIDER REMOVING Fear & Greed Index - minimal predictive value")
        print(f"   Correlation: {fg_5d['pearson_corr']:.4f} (p={fg_5d['pearson_p']:.4f})")

# Sentiment specific recommendation
if 'sent_raw' in feature_cols:
    sent_5d = results_df[(results_df['feature'] == 'sent_raw') & (results_df['horizon'] == 5)].iloc[0]
    
    if sent_5d['pearson_p'] < 0.05 and abs(sent_5d['pearson_corr']) > 0.10:
        print(f"✅ KEEP Sentiment - it has predictive value")
    else:
        print(f"❌ CONSIDER REMOVING Sentiment - minimal predictive value")
        print(f"   Correlation: {sent_5d['pearson_corr']:.4f} (p={sent_5d['pearson_p']:.4f})")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nNote: Even with weak correlations, neural networks might find")
print("non-linear patterns. But strong correlations indicate easier learning!")
print()

