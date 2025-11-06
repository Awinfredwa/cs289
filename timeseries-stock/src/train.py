"""Main training script for stock prediction model."""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from src.dataio import load_ohlcv_csv
from src.features import build_stock_features, make_targets, drop_warmup_rows, get_feature_columns
from src.windows import make_sliding_windows, validate_window_shapes
from src.dataset import create_dataloaders
from src.models import create_model
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
    df_feat = build_stock_features(df, cfg['features'])
    
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
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
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
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    
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
    
    for epoch in range(cfg['train']['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, task)
        val_loss = val_metrics['loss']
        
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
                  f"Val RMSE: {val_metrics['rmse']:.4f}")
        
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
    
    print("\nTest Metrics:")
    for metric_name, metric_value in test_metrics.items():
        if not np.isnan(metric_value):
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # ========== Save Artifacts ==========
    print("\n" + "=" * 80)
    print("SAVING ARTIFACTS")
    print("=" * 80)
    
    outdir = cfg['paths']['outdir']
    os.makedirs(outdir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join(outdir, f"model_{timestamp}.pt")
    torch.save({
        'model_state_dict': best_model_state if best_model_state else model.state_dict(),
        'config': cfg,
        'test_metrics': test_metrics,
        'feature_cols': feature_cols,
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

