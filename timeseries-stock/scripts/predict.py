"""
Example script for loading a trained model and making predictions.

Usage:
    python scripts/predict.py --model artifacts/model_20250105_123456.pt --data data/raw/SP500.csv
"""

import argparse
import torch
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataio import load_ohlcv_csv
from src.features import build_stock_features, make_targets, drop_warmup_rows, get_feature_columns
from src.windows import make_sliding_windows
from src.models import create_model
from src.utils import standardize_features, get_device


def load_trained_model(model_path):
    """Load a trained model checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    config = checkpoint['config']
    feature_cols = checkpoint['feature_cols']
    test_metrics = checkpoint.get('test_metrics', {})
    
    # Create model
    model = create_model(
        model_type=config['model']['type'],
        input_size=len(feature_cols),
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        task=config['train']['task']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, feature_cols, test_metrics


def prepare_data(csv_path, config, feature_cols):
    """Prepare data for prediction."""
    # Load data
    df = load_ohlcv_csv(
        path=csv_path,
        date_col=config['data']['date_col'],
        cols=config['data']['cols']
    )
    
    # Build features
    df_feat = build_stock_features(df, config['features'])
    
    # Create targets (needed for consistency, but we only care about features)
    df_feat = make_targets(df_feat, mode=config['data']['target'])
    
    # Drop warm-up rows
    df_feat = drop_warmup_rows(df_feat)
    
    # Extract features
    X = df_feat[feature_cols].values
    y = df_feat['target'].values
    dates = df_feat.index
    
    # Create windows
    window_size = config['window']['size']
    stride = config['window']['stride']
    X_win, y_win = make_sliding_windows(X, y, window_size, stride)
    
    return X_win, y_win, dates[window_size-1:]


def predict(model, X, device, task='classification'):
    """Make predictions with the model."""
    model.to(device)
    model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        
        if task == 'classification':
            predictions = (outputs > 0.5).cpu().numpy().astype(int)
            probabilities = outputs.cpu().numpy()
        else:
            predictions = outputs.cpu().numpy()
            probabilities = None
    
    return predictions, probabilities


def main():
    parser = argparse.ArgumentParser(description="Make predictions with a trained model")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--output', type=str, default=None, help='Path to save predictions (CSV)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STOCK PREDICTION INFERENCE")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    model, config, feature_cols, test_metrics = load_trained_model(args.model)
    
    print(f"Model type: {config['model']['type']}")
    print(f"Task: {config['train']['task']}")
    print(f"Window size: {config['window']['size']}")
    print(f"Features: {len(feature_cols)}")
    
    if test_metrics:
        print("\nModel performance on test set:")
        for metric, value in test_metrics.items():
            if not np.isnan(value):
                print(f"  {metric}: {value:.4f}")
    
    # Get device
    device = get_device()
    
    # Prepare data
    print(f"\nLoading data from: {args.data}")
    X_win, y_true, dates = prepare_data(args.data, config, feature_cols)
    
    print(f"Data shape: {X_win.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions, probabilities = predict(
        model, X_win, device, task=config['train']['task']
    )
    
    # Create results DataFrame
    results = pd.DataFrame({
        'date': dates,
        'prediction': predictions,
        'actual': y_true
    })
    
    if probabilities is not None:
        results['probability'] = probabilities
    
    # Calculate accuracy (if we have actual labels)
    if config['train']['task'] == 'classification':
        accuracy = (predictions == y_true).mean()
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Directional accuracy
        print(f"Predicted UP: {(predictions == 1).sum()} / {len(predictions)}")
        print(f"Predicted DOWN: {(predictions == 0).sum()} / {len(predictions)}")
        print(f"Actual UP: {(y_true == 1).sum()} / {len(y_true)}")
        print(f"Actual DOWN: {(y_true == 0).sum()} / {len(y_true)}")
    else:
        mae = np.abs(predictions - y_true).mean()
        rmse = np.sqrt(((predictions - y_true) ** 2).mean())
        print(f"\nMAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
    
    # Show sample predictions
    print("\nSample predictions (first 10):")
    print(results.head(10).to_string(index=False))
    
    print("\nSample predictions (last 10):")
    print(results.tail(10).to_string(index=False))
    
    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n✓ Predictions saved to: {args.output}")
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

