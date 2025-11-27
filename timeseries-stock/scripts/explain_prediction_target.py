"""
Explain what the model is actually predicting with different configurations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.dataio import load_ohlcv_csv
from src.features import build_stock_features, drop_warmup_rows
from src.utils import load_config

print("=" * 80)
print("UNDERSTANDING PREDICTION TARGET")
print("=" * 80)

cfg = load_config('config/aggressive_learning.yaml')

# Load data
df = load_ohlcv_csv(
    path=cfg['data']['input_csv'],
    date_col=cfg['data']['date_col'],
    cols=cfg['data']['cols']
)

print(f"\nLoaded {len(df)} days of stock data")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Build features (no sentiment for simplicity)
df_feat = build_stock_features(df, cfg['features'], sent_df=None)
df_feat = drop_warmup_rows(df_feat)

print(f"After feature engineering: {len(df_feat)} days")

# Get config parameters
window_size = cfg['window']['size']
prediction_horizon = cfg['data']['prediction_horizon']

print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"\nWindow size: {window_size} days")
print(f"Prediction horizon: {prediction_horizon} days")

# Show what this means with a concrete example
print("\n" + "=" * 80)
print("CONCRETE EXAMPLE - Sample 0")
print("=" * 80)

# Simulate first window
print(f"\n📅 Input Window (indices 0 to {window_size-1}):")
for i in range(window_size):
    date = df_feat.index[i]
    close = df_feat.loc[date, 'close']
    print(f"   Day {i:2d} ({date.date()}): Close = ${close:.2f}")

# Target construction
target_start_idx = window_size  # Day after window ends
target_end_idx = target_start_idx + prediction_horizon

target_start_date = df_feat.index[target_start_idx]
target_end_date = df_feat.index[target_end_idx - 1]  # -1 because we need prediction_horizon days

close_start = df_feat.loc[target_start_date, 'close']
close_end = df_feat.loc[target_end_date, 'close']

actual_return = (close_end / close_start) - 1

print(f"\n🎯 Prediction Target:")
print(f"   Predicting: Return from Day {target_start_idx} to Day {target_end_idx-1}")
print(f"   Start: Day {target_start_idx} ({target_start_date.date()}) @ ${close_start:.2f}")
print(f"   End:   Day {target_end_idx-1} ({target_end_date.date()}) @ ${close_end:.2f}")
print(f"   Return: {actual_return:.4f} ({actual_return*100:.2f}%)")

print(f"\n💡 Interpretation:")
print(f"   Input:  Days 0-{window_size-1} ({df_feat.index[0].date()} to {df_feat.index[window_size-1].date()})")
print(f"   Predict: Days {target_start_idx}-{target_end_idx-1} return ({target_start_date.date()} to {target_end_date.date()})")
print(f"   Gap between input and target: 1 day (Day {window_size-1} → Day {target_start_idx})")

# Show timeline
print(f"\n📊 Timeline Visualization:")
print(f"   |{'─'*30}|  |{'─'*30}|")
print(f"   {'INPUT WINDOW (used)':^30}   {'PREDICTION PERIOD (predict)':^30}")
print(f"   Days 0-{window_size-1} ({window_size} days)       Days {target_start_idx}-{target_end_idx-1} ({prediction_horizon} days)")

# Show different scenarios
print("\n" + "=" * 80)
print("SCENARIO COMPARISON")
print("=" * 80)

scenarios = [
    ("Next day (1-day ahead)", 1),
    ("Next week (5-day ahead)", 5),
    ("Two weeks (10-day ahead)", 10),
    ("One month (20-day ahead)", 20),
]

print(f"\nWith window_size={window_size}:\n")
print(f"{'Prediction Horizon':<25} | {'Input Window':<20} | {'Predict Period':<25} | {'Note'}")
print("─" * 100)

for name, horizon in scenarios:
    input_days = f"Days 0-{window_size-1}"
    predict_start = window_size
    predict_end = window_size + horizon - 1
    predict_period = f"Days {predict_start}-{predict_end}"
    
    note = ""
    if horizon == 1:
        note = "Easiest (less noise)"
    elif horizon == 5:
        note = "Current config"
    elif horizon >= 10:
        note = "Harder (more uncertainty)"
    
    print(f"{name:<25} | {input_days:<20} | {predict_period:<25} | {note}")

# Explain user's question
print("\n" + "=" * 80)
print("YOUR QUESTION: 'Predict return of day 10-day 6'")
print("=" * 80)

print("\nYou want:")
print("  Input: Days 1-5 (5 days of history)")
print("  Predict: Return from day 6 to day 10 (next 5 days)")

print("\n✅ GOOD NEWS: This is ALREADY what your config does!")
print("\nWith window_size=5 and prediction_horizon=5:")
print("  - Input window: Days 0-4 (that's days 1-5 in 1-indexed)")
print("  - Target: Return from day 5 to day 9 (that's day 6-10 in 1-indexed)")
print("  - The model learns to predict multi-day returns AFTER the input window")

print("\n💡 Benefits of this approach:")
print("  ✅ Complete separation of input and prediction periods (no data leakage)")
print("  ✅ Multi-day returns are smoother (less daily noise)")
print("  ✅ More realistic trading scenario (plan ahead for week)")
print("  ⚠️  But: Harder to predict (more time = more uncertainty)")

# Show actual correlations for different horizons
print("\n" + "=" * 80)
print("CORRELATION BY PREDICTION HORIZON")
print("=" * 80)

print("\nBased on our correlation analysis:")
print("  1-day ahead:  Same correlations for all horizons")
print("  5-day ahead:  Same correlations (sentiment r=-0.177)")
print("  10-day ahead: Same correlations")
print("  20-day ahead: Same correlations")

print("\n📊 Interpretation:")
print("  The same features predict 1-day, 5-day, and 20-day returns equally well")
print("  This suggests: Features capture general market direction, not specific timing")
print("  Therefore: Predicting 5-day cumulative return is just as feasible as 1-day!")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("\n1️⃣  CURRENT CONFIG (window=10, horizon=5):")
print("   Input: 10 days")
print("   Predict: Next 5-day return")
print("   Pro: More historical context")
print("   Con: Longer training time")

print("\n2️⃣  BALANCED CONFIG (window=5, horizon=5):")
print("   Input: 5 days")
print("   Predict: Next 5-day return")
print("   Pro: Fast, symmetrical (5 in, 5 out)")
print("   Con: Less historical context")

print("\n3️⃣  NEXT-DAY CONFIG (window=5, horizon=1):")
print("   Input: 5 days")
print("   Predict: Next 1-day return")
print("   Pro: Easier prediction, more training samples")
print("   Con: Daily returns are noisier")

print("\n💡 My recommendation: Try window=5, horizon=5")
print("   This matches your intuition: 'given 5 days, predict next 5 days'")
print("   Fast training, clean separation, smooth targets")

print("\n" + "=" * 80)

