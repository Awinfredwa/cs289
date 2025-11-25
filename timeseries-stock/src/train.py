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
from src.sentiment_stub import load_daily_sentiment, load_fear_greed_index
from src.resample import resample_to_weekly, align_weekly_fear_greed, create_weekly_targets, weekly_summary
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
        batch_y = batch_y.to(device).long()  # CrossEntropyLoss expects LongTensor
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)  # (batch, 2) for classification
        
        # Calculate loss (CrossEntropyLoss handles everything)
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
            
            # For classification, convert targets to long for CrossEntropyLoss
            if task == 'classification':
                batch_y_loss = batch_y.long()
                loss = criterion(outputs, batch_y_loss)
            else:
                loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            
            # Collect predictions
            if task == 'classification':
                # For CrossEntropyLoss, take argmax of the 2 outputs
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
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
    
    # Check if weekly resampling is enabled
    use_weekly = cfg['data'].get('use_weekly', False)
    if use_weekly:
        print(f"\n📅 Resampling to weekly frequency...")
        daily_df = df.copy()  # Keep copy for reference
        df = resample_to_weekly(df, method='ohlc')
        weekly_summary(daily_df, df)
    else:
        print(f"Using daily frequency")
    
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
        fg_csv_path = cfg['fear_greed'].get('csv_path', 'data/raw/Fear and Greed Index Data.csv')
        
        if use_weekly:
            # Weekly mode: load F&G and align directly (no fill needed!)
            import pandas as pd
            fg_raw = pd.read_csv(fg_csv_path)
            fg_raw['Date'] = pd.to_datetime(fg_raw['Date'])
            fg_raw = fg_raw.set_index('Date').sort_index()
            fg_raw = fg_raw.rename(columns={'Value': 'fg_raw'})
            
            # Create derived features
            fg_raw['fg_change'] = fg_raw['fg_raw'].diff()
            fg_raw['fg_ma_4'] = fg_raw['fg_raw'].rolling(window=4, min_periods=1).mean()
            fg_raw['fg_normalized'] = (fg_raw['fg_raw'] - 50) / 50
            
            fg_df = fg_raw
            print(f"✓ Fear & Greed features (weekly, no filling needed)")
            print(f"  → Clean alignment: Week N emotion → Week N+1 performance")
        else:
            # Daily mode: use fill method
            shift_days = cfg['fear_greed'].get('shift_days', 0)
            fg_df = load_fear_greed_index(df.index, csv_path=fg_csv_path, shift_days=shift_days)
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
    
    # Build features with optional warm-up filling
    fill_warmup = cfg['features'].get('fill_warmup', True)
    df_feat = build_stock_features(df, cfg['features'], sent_df=sent_df, fill_warmup=fill_warmup)
    
    # ========== 3. Create Targets ==========
    print("\n[3/9] Creating targets...")
    horizon = cfg['data'].get('prediction_horizon', 1)
    
    if use_weekly:
        freq_str = f"{'next week' if horizon == 1 else f'{horizon} weeks ahead'}"
        print(f"Prediction horizon: {horizon} weeks ({freq_str})")
        df_feat = create_weekly_targets(df_feat, mode=cfg['data']['target'], horizon=horizon)
    else:
        freq_str = f"{'next day' if horizon == 1 else f'{horizon} days ahead'}"
        print(f"Prediction horizon: {horizon} days ({freq_str})")
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
    
    # Get task before class distribution analysis
    task = cfg['train']['task']
    
    # ========== CLASS DISTRIBUTION ANALYSIS ==========
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    
    if task == 'classification':
        train_class_0 = (y_train == 0).sum()
        train_class_1 = (y_train == 1).sum()
        val_class_0 = (y_val == 0).sum()
        val_class_1 = (y_val == 1).sum()
        test_class_0 = (y_test == 0).sum()
        test_class_1 = (y_test == 1).sum()
        
        print(f"\n📊 Train Set:")
        print(f"  Class 0 (Down): {train_class_0:4d} ({train_class_0/len(y_train)*100:5.1f}%)")
        print(f"  Class 1 (Up):   {train_class_1:4d} ({train_class_1/len(y_train)*100:5.1f}%)")
        print(f"  Ratio (1:0):    {train_class_1/max(train_class_0,1):.2f}")
        
        print(f"\n📊 Validation Set:")
        print(f"  Class 0 (Down): {val_class_0:4d} ({val_class_0/len(y_val)*100:5.1f}%)")
        print(f"  Class 1 (Up):   {val_class_1:4d} ({val_class_1/len(y_val)*100:5.1f}%)")
        print(f"  Ratio (1:0):    {val_class_1/max(val_class_0,1):.2f}")
        
        print(f"\n📊 Test Set:")
        print(f"  Class 0 (Down): {test_class_0:4d} ({test_class_0/len(y_test)*100:5.1f}%)")
        print(f"  Class 1 (Up):   {test_class_1:4d} ({test_class_1/len(y_test)*100:5.1f}%)")
        print(f"  Ratio (1:0):    {test_class_1/max(test_class_0,1):.2f}")
        
        # Calculate class weights for balancing
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_array = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}
        
        print(f"\n⚖️ Suggested Class Weights (for balanced training):")
        print(f"  Class 0: {class_weights[0]:.3f}")
        print(f"  Class 1: {class_weights[1]:.3f}")
        
    print(f"\n{'='*70}\n")
    
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
    # task already defined above
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
    if task == 'classification':
        print(f"Output: 2 neurons (for CrossEntropyLoss)")
    
    # ========== Loss & Optimizer ==========
    if task == 'classification':
        # Use CrossEntropyLoss with class weights to handle imbalance
        if 'class_weights' in locals():
            weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"Using CrossEntropyLoss with class weights: [{weight_tensor[0].item():.3f}, {weight_tensor[1].item():.3f}]")
            print(f"  (Class 0 weight={class_weights[0]:.3f}, Class 1 weight={class_weights[1]:.3f})")
        else:
            criterion = nn.CrossEntropyLoss()
            print(f"Using CrossEntropyLoss (no class weights)")
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

