"""Analyze regression predictions for trading performance."""

import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def analyze_predictions(pred_file):
    """
    Analyze regression predictions with focus on directional accuracy.
    
    Args:
        pred_file: Path to .npz file with predictions
    """
    # Load predictions
    data = np.load(pred_file)
    y_true = data['test_targets']
    y_pred = data['test_preds']
    dates = data.get('test_dates', None)
    
    print("=" * 80)
    print("REGRESSION PREDICTION ANALYSIS")
    print("=" * 80)
    print(f"\nLoaded {len(y_true)} predictions")
    
    # 1. Regression metrics
    print("\n" + "=" * 80)
    print("📊 REGRESSION METRICS")
    print("=" * 80)
    
    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    print(f"MAE:         {mae:.4f} ({mae*100:.2f}%)")
    print(f"RMSE:        {rmse:.4f} ({rmse*100:.2f}%)")
    print(f"Correlation: {corr:.4f}")
    
    # Baselines
    baseline_zero = np.abs(y_true).mean()
    baseline_mean = np.abs(y_true - y_true.mean()).mean()
    
    print(f"\nBaseline (predict 0):    MAE = {baseline_zero:.4f}")
    print(f"Baseline (predict mean): MAE = {baseline_mean:.4f}")
    print(f"Your model:              MAE = {mae:.4f}")
    
    improvement = (baseline_zero - mae) / baseline_zero * 100
    print(f"\nImprovement vs baseline: {improvement:.1f}%")
    
    # 2. Directional Analysis - THE MOST IMPORTANT FOR TRADING!
    print("\n" + "=" * 80)
    print("🎯 DIRECTIONAL ANALYSIS (Trading Performance)")
    print("=" * 80)
    
    # Convert to direction (1 = up, 0 = down)
    pred_dir = (y_pred > 0).astype(int).flatten()
    true_dir = (y_true > 0).astype(int).flatten()
    
    dir_acc = (pred_dir == true_dir).mean()
    print(f"\n⭐ Directional Accuracy: {dir_acc:.4f} ({dir_acc*100:.2f}%)")
    
    if dir_acc > 0.55:
        print("   ✓✓✓ EXCELLENT - Model is useful for trading!")
    elif dir_acc > 0.52:
        print("   ✓ GOOD - Some predictive signal")
    elif dir_acc > 0.48:
        print("   ~ MEDIOCRE - Barely better than random")
    else:
        print("   ⚠ POOR - Not better than random guess")
    
    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(true_dir, pred_dir)
    print("\n                 Predicted")
    print("                DOWN    UP")
    print(f"Actual  DOWN  | {cm[0,0]:4d} | {cm[0,1]:4d} |")
    print(f"        UP    | {cm[1,0]:4d} | {cm[1,1]:4d} |")
    
    # Calculate rates
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📈 Detailed Metrics:")
    print(f"True Positives  (predicted UP, was UP):     {tp:4d} ({tp/len(y_true)*100:.1f}%)")
    print(f"True Negatives  (predicted DOWN, was DOWN): {tn:4d} ({tn/len(y_true)*100:.1f}%)")
    print(f"False Positives (predicted UP, was DOWN):   {fp:4d} ({fp/len(y_true)*100:.1f}%)")
    print(f"False Negatives (predicted DOWN, was UP):   {fn:4d} ({fn/len(y_true)*100:.1f}%)")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(true_dir, pred_dir, 
                                target_names=['DOWN', 'UP'],
                                digits=4))
    
    # 3. Distribution Analysis
    print("=" * 80)
    print("📊 PREDICTION DISTRIBUTION")
    print("=" * 80)
    
    print(f"\nActual returns:")
    print(f"  Mean:   {y_true.mean():.4f} ({y_true.mean()*100:.2f}%)")
    print(f"  Std:    {y_true.std():.4f} ({y_true.std()*100:.2f}%)")
    print(f"  Min:    {y_true.min():.4f} ({y_true.min()*100:.2f}%)")
    print(f"  Max:    {y_true.max():.4f} ({y_true.max()*100:.2f}%)")
    print(f"  UP days:   {(y_true > 0).sum()} ({(y_true > 0).mean()*100:.1f}%)")
    print(f"  DOWN days: {(y_true < 0).sum()} ({(y_true < 0).mean()*100:.1f}%)")
    
    print(f"\nPredicted returns:")
    print(f"  Mean:   {y_pred.mean():.4f} ({y_pred.mean()*100:.2f}%)")
    print(f"  Std:    {y_pred.std():.4f} ({y_pred.std()*100:.2f}%)")
    print(f"  Min:    {y_pred.min():.4f} ({y_pred.min()*100:.2f}%)")
    print(f"  Max:    {y_pred.max():.4f} ({y_pred.max()*100:.2f}%)")
    print(f"  UP predictions:   {(y_pred > 0).sum()} ({(y_pred > 0).mean()*100:.1f}%)")
    print(f"  DOWN predictions: {(y_pred < 0).sum()} ({(y_pred < 0).mean()*100:.1f}%)")
    
    # 4. Trading Strategy Simulation
    print("\n" + "=" * 80)
    print("💰 SIMPLE TRADING STRATEGY SIMULATION")
    print("=" * 80)
    
    # Strategy: Buy when predict UP, Sell/Short when predict DOWN
    strategy_returns = []
    for pred, actual in zip(y_pred.flatten(), y_true.flatten()):
        if pred > 0:
            # Predicted UP - go long
            strategy_returns.append(actual)
        else:
            # Predicted DOWN - go short (profit from down moves)
            strategy_returns.append(-actual)
    
    strategy_returns = np.array(strategy_returns)
    buy_hold_returns = y_true.flatten()
    
    total_strategy = strategy_returns.sum()
    total_buy_hold = buy_hold_returns.sum()
    
    print(f"\nStrategy Return (cumulative): {total_strategy:.4f} ({total_strategy*100:.2f}%)")
    print(f"Buy & Hold Return:            {total_buy_hold:.4f} ({total_buy_hold*100:.2f}%)")
    
    if total_strategy > total_buy_hold:
        print(f"✓ Strategy beats buy & hold by {(total_strategy - total_buy_hold)*100:.2f}%")
    else:
        print(f"⚠ Strategy underperforms buy & hold by {(total_buy_hold - total_strategy)*100:.2f}%")
    
    # Win rate
    profitable_trades = (strategy_returns > 0).sum()
    win_rate = profitable_trades / len(strategy_returns)
    print(f"\nWin Rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
    
    # Average profit per trade
    avg_profit = strategy_returns.mean()
    print(f"Average Profit per Trade: {avg_profit:.4f} ({avg_profit*100:.2f}%)")
    
    # 5. Worst predictions
    print("\n" + "=" * 80)
    print("❌ WORST PREDICTIONS (Largest Errors)")
    print("=" * 80)
    
    errors = np.abs(y_true - y_pred).flatten()
    worst_indices = np.argsort(errors)[-10:][::-1]
    
    print("\nTop 10 Worst Predictions:")
    for i, idx in enumerate(worst_indices, 1):
        actual = y_true.flatten()[idx]
        pred = y_pred.flatten()[idx]
        error = errors[idx]
        date_str = f" ({dates[idx]})" if dates is not None else ""
        print(f"{i:2d}. Actual: {actual:+.4f} | Pred: {pred:+.4f} | Error: {error:.4f}{date_str}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze regression predictions")
    parser.add_argument(
        '--pred_file',
        type=str,
        required=True,
        help='Path to predictions .npz file'
    )
    
    args = parser.parse_args()
    analyze_predictions(args.pred_file)

