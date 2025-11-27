"""Main training script for stock prediction model."""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages

from src.dataio import load_ohlcv_csv
from src.features import build_stock_features, make_targets, drop_warmup_rows, get_feature_columns
from src.windows import make_sliding_windows, validate_window_shapes
from src.dataset import create_dataloaders
from src.models import create_model
from src.losses import create_loss_function
from src.sentiment_stub import load_daily_sentiment, load_fear_greed_index
from src.utils import (
    set_seed, load_config, save_config, time_split,
    standardize_features, compute_classification_metrics,
    compute_regression_metrics, get_device, EarlyStopping
)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, data_loader, criterion, device, task):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            # Collect predictions
            if task == 'classification':
                preds = (outputs > 0.5).cpu().numpy().astype(int)
            else:
                preds = outputs.cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    if task == 'classification':
        metrics = compute_classification_metrics(all_targets, all_preds)
    else:
        metrics = compute_regression_metrics(all_targets, all_preds)
    
    metrics['loss'] = avg_loss
    
    return metrics, all_preds, all_targets


def plot_loss_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    fig = plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Loss curves saved: {save_path}")
    
    return fig


def plot_confusion_matrix(y_true, y_pred, save_path=None, title='Confusion Matrix'):
    """Plot confusion matrix for classification."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1, linecolor='black')
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    
    return fig


def plot_regression_results(y_true, y_pred, save_path=None):
    """Plot predicted vs actual values for regression."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter plot: Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Add diagonal line (perfect predictions)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Values', fontsize=11)
    ax1.set_ylabel('Predicted Values', fontsize=11)
    ax1.set_title('Predicted vs Actual Values', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Predicted Values', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual distribution
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    
    ax3.set_xlabel('Residual Value', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Time series of predictions
    ax4 = axes[1, 1]
    indices = range(len(y_true))
    ax4.plot(indices, y_true, 'b-', label='Actual', linewidth=1.5, alpha=0.7)
    ax4.plot(indices, y_pred, 'r-', label='Predicted', linewidth=1.5, alpha=0.7)
    
    ax4.set_xlabel('Sample Index', fontsize=11)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Predictions Over Time', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Regression results saved: {save_path}")
    
    return fig


def plot_direction_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix for directional predictions (up/down)."""
    # Convert to directional (sign)
    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    
    cm = confusion_matrix(y_true_dir, y_pred_dir)
    
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=True,
                square=True, linewidths=1, linecolor='black',
                xticklabels=['Down (↓)', 'Up (↑)'],
                yticklabels=['Down (↓)', 'Up (↑)'])
    
    plt.xlabel('Predicted Direction', fontsize=12)
    plt.ylabel('Actual Direction', fontsize=12)
    plt.title('Direction Prediction Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add accuracy annotation
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(0.5, -0.15, f'Direction Accuracy: {accuracy:.2%}', 
             ha='center', transform=plt.gca().transAxes, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Direction confusion matrix saved: {save_path}")
    
    return fig


def plot_metrics_over_epochs(train_losses, val_losses, val_metrics_history, task, save_path=None):
    """Plot training metrics over epochs."""
    epochs = range(1, len(train_losses) + 1)
    
    if task == 'classification':
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2 = axes[0, 1]
        ax2.plot(epochs, [m['accuracy'] for m in val_metrics_history], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.grid(True, alpha=0.3)
        
        # F1 Score
        ax3 = axes[1, 0]
        ax3.plot(epochs, [m['f1'] for m in val_metrics_history], 'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Validation F1 Score')
        ax3.grid(True, alpha=0.3)
        
        # Precision & Recall
        ax4 = axes[1, 1]
        ax4.plot(epochs, [m['precision'] for m in val_metrics_history], 'orange', label='Precision', linewidth=2)
        ax4.plot(epochs, [m['recall'] for m in val_metrics_history], 'cyan', label='Recall', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.set_title('Precision & Recall')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    else:  # regression
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2 = axes[0, 1]
        ax2.plot(epochs, [m['mae'] for m in val_metrics_history], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Validation MAE')
        ax2.grid(True, alpha=0.3)
        
        # Direction Accuracy
        ax3 = axes[1, 0]
        ax3.plot(epochs, [m['dir_accuracy'] for m in val_metrics_history], 'purple', linewidth=2)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Direction Accuracy')
        ax3.set_title('Validation Direction Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Correlation
        ax4 = axes[1, 1]
        ax4.plot(epochs, [m['correlation'] for m in val_metrics_history], 'orange', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Correlation')
        ax4.set_title('Validation Correlation')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics over epochs saved: {save_path}")
    
    return fig


def create_summary_page(cfg, test_metrics, task, timestamp):
    """Create a summary page with key information."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('Training Summary Report', fontsize=20, fontweight='bold', y=0.98)
    
    # Remove axes
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Build summary text
    summary_text = f"""
Training Timestamp: {timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Type:          {cfg['model']['type'].upper()}
Task:                {task.upper()}
Hidden Size:         {cfg['model']['hidden_size']}
Num Layers:          {cfg['model']['num_layers']}
Dropout:             {cfg['model']['dropout']}

Window Size:         {cfg['window']['size']}
Window Stride:       {cfg['window']['stride']}

Learning Rate:       {cfg['train']['lr']}
Batch Size:          {cfg['train']['batch_size']}
Weight Decay:        {cfg['train']['weight_decay']}
Epochs Trained:      {cfg['train']['epochs']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEST PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    if task == 'classification':
        summary_text += f"""
Accuracy:            {test_metrics.get('accuracy', 0):.4f}
Precision:           {test_metrics.get('precision', 0):.4f}
Recall:              {test_metrics.get('recall', 0):.4f}
F1 Score:            {test_metrics.get('f1', 0):.4f}
Test Loss:           {test_metrics.get('loss', 0):.4f}
"""
    else:
        summary_text += f"""
MAE:                 {test_metrics.get('mae', 0):.4f}
RMSE:                {test_metrics.get('rmse', 0):.4f}
Correlation:         {test_metrics.get('correlation', 0):.4f}

Direction Accuracy:  {test_metrics.get('dir_accuracy', 0):.4f} ({test_metrics.get('dir_accuracy', 0)*100:.2f}%)
Direction Precision: {test_metrics.get('dir_precision', 0):.4f}
Direction Recall:    {test_metrics.get('dir_recall', 0):.4f}
Direction F1:        {test_metrics.get('dir_f1', 0):.4f}

Test Loss:           {test_metrics.get('loss', 0):.4f}
"""
    
    summary_text += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input CSV:           {cfg['data']['input_csv']}
Target Type:         {cfg['data']['target']}
Prediction Horizon:  {cfg['data'].get('prediction_horizon', 1)} days

Train Ratio:         {cfg['split']['train_ratio']}
Val Ratio:           {cfg['split']['val_ratio']}
Test Ratio:          {1 - cfg['split']['train_ratio'] - cfg['split']['val_ratio']:.2f}

Sentiment Enabled:   {cfg.get('sentiment', {}).get('enabled', False)}
Fear & Greed:        {cfg.get('fear_greed', {}).get('enabled', False)}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    return fig


def save_visualizations_to_pdf(figures, pdf_path, cfg, test_metrics, task, timestamp):
    """Save all visualizations to a single PDF file."""
    with PdfPages(pdf_path) as pdf:
        # Add summary page first
        summary_fig = create_summary_page(cfg, test_metrics, task, timestamp)
        pdf.savefig(summary_fig, bbox_inches='tight')
        plt.close(summary_fig)
        
        # Add all visualization figures
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Stock Prediction Model Training Report'
        d['Author'] = 'Stock Prediction Training Script'
        d['Subject'] = f'{task.upper()} Model Training Results'
        d['Keywords'] = f'Stock Prediction, {task}, Deep Learning, Time Series'
        d['CreationDate'] = datetime.now()
    
    print(f"✓ All visualizations saved to PDF: {pdf_path}")


def main(config_path, overrides=None):
    """Main training pipeline."""
    
    # Load config
    cfg = load_config(config_path)
    
    # Apply command-line overrides
    if overrides:
        for override in overrides:
            keys, value = override.split('=')
            keys = keys.split('.')
            
            # Navigate to the right nested dict
            current = cfg
            for key in keys[:-1]:
                current = current[key]
            
            # Set value (try to infer type)
            try:
                current[keys[-1]] = eval(value)
            except:
                current[keys[-1]] = value
    
    print("=" * 80)
    print("STOCK PREDICTION TRAINING")
    print("=" * 80)
    
    # Set seed
    set_seed(cfg['train']['seed'])
    
    # Get device
    device = get_device()
    
    # ========== 1. Load Data ==========
    print("\n[1/9] Loading data...")
    df = load_ohlcv_csv(
        path=cfg['data']['input_csv'],
        date_col=cfg['data']['date_col'],
        cols=cfg['data']['cols']
    )
    print(f"Loaded {len(df)} days of data")
    
    # ========== 2. Build Features ==========
    print("\n[2/9] Building features...")
    
    # Load sentiment data if enabled
    sent_df = None
    if cfg.get('sentiment', {}).get('enabled', False):
        print("Loading sentiment data...")
        sent_csv_path = cfg['sentiment'].get('csv_path', 'data/raw/Daily News Sentiment Index.csv')
        sent_df = load_daily_sentiment(df.index, csv_path=sent_csv_path)
        if len(sent_df) > 0:
            print(f"✓ Sentiment features will be added ({len(sent_df.columns)} features)")
        else:
            print("⚠ Warning: No sentiment data matched stock dates")
            sent_df = None
    
    # Load Fear and Greed Index if enabled
    fg_df = None
    if cfg.get('fear_greed', {}).get('enabled', False):
        print("Loading Fear and Greed Index...")
        fg_csv_path = cfg['fear_greed'].get('csv_path', 'data/raw/Fear and Greed Index Data - Daily Interpolated.csv')
        
        fg_df = load_fear_greed_index(df.index, csv_path=fg_csv_path)
        if len(fg_df) > 0:
            print(f"✓ Fear & Greed features will be added ({len(fg_df.columns)} features)")
        else:
            print("⚠ Warning: No Fear & Greed data matched stock dates")
            fg_df = None
    
    # Merge sentiment and Fear & Greed into single sentiment DataFrame
    if sent_df is not None and fg_df is not None:
        sent_df = sent_df.join(fg_df, how='outer')
    elif fg_df is not None:
        sent_df = fg_df
    
    df_feat = build_stock_features(df, cfg['features'], sent_df=sent_df)
    
    # ========== 3. Create Targets ==========
    print("\n[3/9] Creating targets...")
    horizon = cfg['data'].get('prediction_horizon', 1)
    print(f"Prediction horizon: {horizon} days ({'next day' if horizon == 1 else f'next week' if horizon == 5 else f'{horizon} days ahead'})")
    df_feat = make_targets(df_feat, mode=cfg['data']['target'], horizon=horizon)
    
    # ========== 4. Drop Warm-up Rows ==========
    print("\n[4/9] Dropping warm-up rows...")
    df_feat = drop_warmup_rows(df_feat)
    print(f"Remaining: {len(df_feat)} days")
    
    # ========== 5. Train/Val/Test Split ==========
    print("\n[5/9] Splitting data...")
    train_df, val_df, test_df = time_split(
        df_feat,
        train_ratio=cfg['split']['train_ratio'],
        val_ratio=cfg['split']['val_ratio']
    )
    
    # Get feature columns
    feature_cols = get_feature_columns(df_feat)
    
    # ========== Feature Summary ==========
    print(f"\n{'='*70}")
    print(f"FEATURE SUMMARY ({len(feature_cols)} total features)")
    print(f"{'='*70}")
    
    # Categorize features
    tech_features = [f for f in feature_cols if not f.startswith('sent_') and not f.startswith('fg_')]
    sent_features = [f for f in feature_cols if f.startswith('sent_')]
    fg_features = [f for f in feature_cols if f.startswith('fg_')]
    
    print(f"\n📊 Technical Features ({len(tech_features)}):")
    for i, feat in enumerate(tech_features, 1):
        values = df_feat[feat]
        print(f"  {i:2d}. {feat:20s} | Mean: {values.mean():8.4f} | Std: {values.std():7.4f} | Range: [{values.min():7.3f}, {values.max():7.3f}]")
    
    if sent_features:
        print(f"\n🎭 Sentiment Features ({len(sent_features)}):")
        for i, feat in enumerate(sent_features, 1):
            values = df_feat[feat]
            print(f"  {i:2d}. {feat:20s} | Mean: {values.mean():8.4f} | Std: {values.std():7.4f} | Range: [{values.min():7.3f}, {values.max():7.3f}]")
    else:
        print(f"\n🎭 Sentiment Features: None (disabled)")
    
    if fg_features:
        print(f"\n😨😁 Fear & Greed Features ({len(fg_features)}):")
        for i, feat in enumerate(fg_features, 1):
            values = df_feat[feat]
            print(f"  {i:2d}. {feat:20s} | Mean: {values.mean():8.4f} | Std: {values.std():7.4f} | Range: [{values.min():7.3f}, {values.max():7.3f}]")
    else:
        print(f"\n😨😁 Fear & Greed Features: None (disabled)")
    
    # Show sample data
    print(f"\n📋 Sample Feature Values (first 5 days after split):")
    print(f"{'='*70}")
    sample_data = train_df[feature_cols].head(5)
    # Transpose for better readability
    print(sample_data.T.to_string())
    
    print(f"\n{'='*70}\n")
    
    # Extract features and targets
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # ========== 6. Build Sliding Windows ==========
    print("\n[6/9] Building sliding windows...")
    window_size = cfg['window']['size']
    stride = cfg['window']['stride']
    
    X_train_win, y_train_win = make_sliding_windows(X_train, y_train, window_size, stride)
    X_val_win, y_val_win = make_sliding_windows(X_val, y_val, window_size, stride)
    X_test_win, y_test_win = make_sliding_windows(X_test, y_test, window_size, stride)
    
    validate_window_shapes(X_train_win, y_train_win, window_size, len(feature_cols))
    
    # ========== 7. Standardize Features ==========
    print("\n[7/9] Standardizing features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = standardize_features(
        X_train_win, X_val_win, X_test_win
    )
    
    # ========== 8. Create DataLoaders ==========
    print("\n[8/9] Creating DataLoaders...")
    task = cfg['train']['task']
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_scaled, y_train_win,
        X_val_scaled, y_val_win,
        X_test_scaled, y_test_win,
        batch_size=cfg['train']['batch_size'],
        task=task
    )
    
    # ========== 9. Build Model ==========
    print("\n[9/9] Building model...")
    model = create_model(
        model_type=cfg['model']['type'],
        input_size=len(feature_cols),
        hidden_size=cfg['model']['hidden_size'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model']['dropout'],
        task=task
    )
    model = model.to(device)
    
    print(f"Model: {cfg['model']['type'].upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== Loss & Optimizer ==========
    if task == 'classification':
        criterion = nn.BCELoss()
    else:
        # Use configurable loss function for regression
        loss_type = cfg['train'].get('loss_function', 'mse')
        direction_weight = cfg['train'].get('direction_weight', 0.5)
        
        if loss_type == 'direction_mse':
            criterion = create_loss_function('direction_mse', direction_weight=direction_weight)
            print(f"Using DirectionAwareMSE (direction_weight={direction_weight})")
        elif loss_type == 'weighted_direction_mse':
            criterion = create_loss_function('weighted_direction_mse', direction_weight=direction_weight)
            print(f"Using WeightedDirectionMSE (direction_weight={direction_weight})")
        elif loss_type == 'huber_direction':
            delta = cfg['train'].get('huber_delta', 0.1)
            criterion = create_loss_function('huber_direction', delta=delta, direction_weight=direction_weight)
            print(f"Using HuberDirectionLoss (delta={delta}, direction_weight={direction_weight})")
        else:
            criterion = nn.MSELoss()
            print(f"Using standard MSELoss")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    
    # ========== Training Loop ==========
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    early_stopping = EarlyStopping(
        patience=cfg['train']['early_stop_patience'],
        mode='min'
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    val_metrics_history = []
    
    for epoch in range(cfg['train']['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, task)
        val_loss = val_metrics['loss']
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics.copy())
        
        # Print progress
        if task == 'classification':
            print(f"Epoch {epoch+1:3d}/{cfg['train']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")
        else:
            print(f"Epoch {epoch+1:3d}/{cfg['train']['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_metrics['mae']:.4f} | "
                  f"Val Dir Acc: {val_metrics['dir_accuracy']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # ========== Test Evaluation ==========
    print("\n" + "=" * 80)
    print("TEST EVALUATION")
    print("=" * 80)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from validation")
    
    test_metrics, test_preds, test_targets = evaluate(model, test_loader, criterion, device, task)
    
    if task == 'classification':
        print("\nTest Metrics:")
        for metric_name, metric_value in test_metrics.items():
            if not np.isnan(metric_value):
                print(f"  {metric_name}: {metric_value:.4f}")
    else:
        # For regression, show detailed metrics with baselines
        print("\n📊 REGRESSION METRICS:")
        print(f"  MAE:           {test_metrics['mae']:.4f}")
        print(f"  RMSE:          {test_metrics['rmse']:.4f}")
        print(f"  Correlation:   {test_metrics['correlation']:.4f}")
        
        print("\n🎯 TRADING METRICS (Direction):")
        print(f"  Accuracy:      {test_metrics['dir_accuracy']:.4f} ({test_metrics['dir_accuracy']*100:.2f}%)")
        print(f"  Precision:     {test_metrics['dir_precision']:.4f}")
        print(f"  Recall:        {test_metrics['dir_recall']:.4f}")
        print(f"  F1 Score:      {test_metrics['dir_f1']:.4f}")
        
        # Show baselines for comparison
        print("\n📏 BASELINES (for comparison):")
        baseline_zero_mae = np.abs(test_targets).mean()
        baseline_mean_mae = np.abs(test_targets - y_train.mean()).mean()
        print(f"  Always predict 0:    MAE = {baseline_zero_mae:.4f}")
        print(f"  Always predict mean: MAE = {baseline_mean_mae:.4f}")
        print(f"  Your model:          MAE = {test_metrics['mae']:.4f}")
        
        if test_metrics['mae'] < baseline_zero_mae:
            print("  ✓ Model beats naive baseline!")
        else:
            print("  ⚠ Model worse than predicting zero")
        
        # Trading interpretation
        print("\n💡 TRADING INTERPRETATION:")
        if test_metrics['dir_accuracy'] > 0.55:
            print(f"  ✓ Strong directional accuracy ({test_metrics['dir_accuracy']*100:.1f}%) - Model is useful for trading!")
        elif test_metrics['dir_accuracy'] > 0.52:
            print(f"  ~ Modest directional accuracy ({test_metrics['dir_accuracy']*100:.1f}%) - Some signal detected")
        else:
            print(f"  ⚠ Weak directional accuracy ({test_metrics['dir_accuracy']*100:.1f}%) - Not better than random")
    
    # ========== Generate Visualizations ==========
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    outdir = cfg['paths']['outdir']
    os.makedirs(outdir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create visualizations subfolder
    vis_dir = os.path.join(outdir, f"visualizations_{timestamp}")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Saving visualizations to: {vis_dir}")
    
    # Collect all figures for PDF
    all_figures = []
    
    # 1. Loss curves
    loss_curve_path = os.path.join(vis_dir, "1_loss_curves.png")
    fig1 = plot_loss_curves(train_losses, val_losses, loss_curve_path)
    all_figures.append(fig1)
    
    # 2. Metrics over epochs
    metrics_path = os.path.join(vis_dir, "2_metrics_over_epochs.png")
    fig2 = plot_metrics_over_epochs(train_losses, val_losses, val_metrics_history, task, metrics_path)
    all_figures.append(fig2)
    
    # 3. Task-specific plots
    if task == 'classification':
        # Confusion matrix on test set
        cm_path = os.path.join(vis_dir, "3_confusion_matrix.png")
        fig3 = plot_confusion_matrix(test_targets.flatten(), test_preds.flatten(), cm_path, 
                            title='Test Set Confusion Matrix')
        all_figures.append(fig3)
    else:
        # Regression visualizations
        regression_path = os.path.join(vis_dir, "3_regression_results.png")
        fig3 = plot_regression_results(test_targets.flatten(), test_preds.flatten(), regression_path)
        all_figures.append(fig3)
        
        # Direction confusion matrix
        dir_cm_path = os.path.join(vis_dir, "4_direction_confusion_matrix.png")
        fig4 = plot_direction_confusion_matrix(test_targets.flatten(), test_preds.flatten(), dir_cm_path)
        all_figures.append(fig4)
    
    # 4. Generate comprehensive PDF report
    pdf_path = os.path.join(vis_dir, f"training_report_{timestamp}.pdf")
    print("\nGenerating PDF report...")
    save_visualizations_to_pdf(all_figures, pdf_path, cfg, test_metrics, task, timestamp)
    
    # ========== Save Artifacts ==========
    print("\n" + "=" * 80)
    print("SAVING ARTIFACTS")
    print("=" * 80)
    
    # Save model
    model_path = os.path.join(outdir, f"model_{timestamp}.pt")
    torch.save({
        'model_state_dict': best_model_state if best_model_state else model.state_dict(),
        'config': cfg,
        'test_metrics': test_metrics,
        'feature_cols': feature_cols,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics_history': val_metrics_history,
    }, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save config
    config_save_path = os.path.join(outdir, f"config_{timestamp}.yaml")
    save_config(cfg, config_save_path)
    print(f"✓ Config saved: {config_save_path}")
    
    # Save predictions
    pred_path = os.path.join(outdir, f"predictions_{timestamp}.npz")
    np.savez(
        pred_path,
        test_preds=test_preds,
        test_targets=test_targets,
        test_dates=test_df.index[window_size-1:].values  # Align with windowed data
    )
    print(f"✓ Predictions saved: {pred_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides (e.g., window.size=20 train.task=regression)'
    )
    
    args = parser.parse_args()
    
    main(args.config, args.overrides)

