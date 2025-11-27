"""
Comprehensive debugging script for LSTM model producing constant predictions.

This script implements a systematic debugging protocol to identify why the model
is predicting constant values instead of learning patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from src.dataio import load_ohlcv_csv
from src.features import build_stock_features, make_targets, drop_warmup_rows, get_feature_columns
from src.windows import make_sliding_windows
from src.models import create_model
from src.utils import load_config, set_seed, time_split, standardize_features, get_device
from src.sentiment_stub import load_fear_greed_index

print("=" * 80)
print("LSTM MODEL DEBUGGING PROTOCOL")
print("=" * 80)

# Load config and data
cfg = load_config('config/default.yaml')
set_seed(cfg['train']['seed'])
device = get_device()

# Load and prepare data
print("\n[Loading Data]")
df = load_ohlcv_csv(
    path=cfg['data']['input_csv'],
    date_col=cfg['data']['date_col'],
    cols=cfg['data']['cols']
)

# Load Fear & Greed if enabled
fg_df = None
if cfg.get('fear_greed', {}).get('enabled', False):
    fg_csv_path = cfg['fear_greed'].get('csv_path')
    fg_df = load_fear_greed_index(df.index, csv_path=fg_csv_path)

df_feat = build_stock_features(df, cfg['features'], sent_df=fg_df)
df_feat = make_targets(df_feat, mode=cfg['data']['target'], horizon=cfg['data'].get('prediction_horizon', 1))
df_feat = drop_warmup_rows(df_feat)

train_df, val_df, test_df = time_split(
    df_feat,
    train_ratio=cfg['split']['train_ratio'],
    val_ratio=cfg['split']['val_ratio']
)

feature_cols = get_feature_columns(df_feat)

X_train = train_df[feature_cols].values
y_train = train_df['target'].values

# Build windows
window_size = cfg['window']['size']
X_train_win, y_train_win = make_sliding_windows(X_train, y_train, window_size, 1)

# Standardize
X_train_scaled, _, _, scaler = standardize_features(X_train_win, X_train_win, X_train_win)

print(f"✓ Data loaded: {len(X_train_scaled)} training samples")
print(f"  X shape: {X_train_scaled.shape}")
print(f"  y shape: {y_train_win.shape}")

# ============================================================================
# STEP 3: VERIFY DATA HAS VARIANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: VERIFY DATA HAS VARIANCE")
print("=" * 80)

print("\n📊 Target Statistics:")
print(f"  Mean:   {y_train_win.mean():.6f}")
print(f"  Std:    {y_train_win.std():.6f}")
print(f"  Min:    {y_train_win.min():.6f}")
print(f"  Max:    {y_train_win.max():.6f}")
print(f"  Unique: {len(np.unique(y_train_win))}")

positive_pct = (y_train_win > 0).mean() * 100
print(f"  Positive: {positive_pct:.1f}%")
print(f"  Negative: {100-positive_pct:.1f}%")

print("\n📊 Feature Statistics:")
print(f"  Shape:  {X_train_scaled.shape}")
print(f"  Mean:   {X_train_scaled.mean():.6f}")
print(f"  Std:    {X_train_scaled.std():.6f}")
print(f"  Min:    {X_train_scaled.min():.6f}")
print(f"  Max:    {X_train_scaled.max():.6f}")

if y_train_win.std() < 1e-6:
    print("\n❌ CRITICAL: Targets have NO variance!")
    print("   Problem: All targets are the same value")
    sys.exit(1)

if X_train_scaled.std() < 1e-6:
    print("\n❌ CRITICAL: Features have NO variance!")
    print("   Problem: Feature standardization broke the data")
    sys.exit(1)

print("\n✅ Data has variance - proceeding to model checks...")

# ============================================================================
# STEP 1: CHECK MODEL OUTPUT DIRECTLY (NO TRAINING)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: CHECK MODEL OUTPUT DIRECTLY (NO TRAINING)")
print("=" * 80)

# Create model
model = create_model(
    model_type=cfg['model']['type'],
    input_size=len(feature_cols),
    hidden_size=cfg['model']['hidden_size'],
    num_layers=cfg['model']['num_layers'],
    dropout=cfg['model']['dropout'],
    task=cfg['train']['task']
)
model = model.to(device)
model.eval()

print(f"\n🔍 Testing untrained model on 10 samples...")
print("(If all outputs are identical → architecture is broken)")

outputs = []
with torch.no_grad():
    for i in range(min(10, len(X_train_scaled))):
        X_sample = torch.FloatTensor(X_train_scaled[i:i+1]).to(device)
        output = model(X_sample)
        output_val = output.item() if output.numel() == 1 else output[0].item()
        outputs.append(output_val)
        print(f"  Sample {i}: {output_val:.8f} (target: {y_train_win[i]:.6f})")

# Check if all identical
outputs_array = np.array(outputs)
output_std = outputs_array.std()
print(f"\n📊 Untrained output statistics:")
print(f"  Mean: {outputs_array.mean():.8f}")
print(f"  Std:  {output_std:.8f}")

if output_std < 1e-8:
    print("  ⚠️  WARNING: All outputs are IDENTICAL before training!")
    print("     This suggests a problem with model architecture or initialization")
else:
    print("  ✅ Outputs have variance - model can produce different values")

# ============================================================================
# STEP 4: CHECK MODEL ARCHITECTURE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: CHECK MODEL ARCHITECTURE")
print("=" * 80)

print(f"\n🏗️  Model Architecture:")
print(model)

print(f"\n📋 Model Parameters:")
total_params = 0
for name, param in model.named_parameters():
    print(f"  {name:30s} | Shape: {str(param.shape):20s} | Requires Grad: {param.requires_grad}")
    total_params += param.numel()

print(f"\n  Total parameters: {total_params:,}")

# Check output layer
print(f"\n🔍 Checking Output Layer:")
if hasattr(model, 'fc'):
    print(f"  Output layer: {model.fc}")
    
    # Check if there's an activation function
    if isinstance(model.fc, nn.Sequential):
        print("  ⚠️  Output layer is Sequential - checking for activation functions...")
        for i, layer in enumerate(model.fc):
            print(f"    Layer {i}: {layer}")
            if isinstance(layer, (nn.Sigmoid, nn.Tanh, nn.ReLU)):
                print(f"    ❌ FOUND ACTIVATION IN OUTPUT! This will limit range!")
    else:
        print("  ✅ Output layer is plain Linear (no activation)")

# Check if optimizer will work
print(f"\n🔍 Checking Parameter Gradients:")
grad_enabled_count = sum(1 for p in model.parameters() if p.requires_grad)
total_param_count = sum(1 for p in model.parameters())
print(f"  {grad_enabled_count}/{total_param_count} parameters have gradients enabled")

if grad_enabled_count == 0:
    print("  ❌ CRITICAL: No parameters have gradients!")
    print("     Model cannot learn!")
else:
    print("  ✅ Parameters can be optimized")

# ============================================================================
# STEP 2: CHECK IF MODEL CAN OVERFIT SINGLE BATCH
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: OVERFIT SINGLE BATCH TEST")
print("=" * 80)

print("\n🎯 Testing if model can overfit 32 samples...")
print("(If it can't → architecture is fundamentally broken)")

# Create fresh model
model = create_model(
    model_type=cfg['model']['type'],
    input_size=len(feature_cols),
    hidden_size=cfg['model']['hidden_size'],
    num_layers=cfg['model']['num_layers'],
    dropout=0.0,  # No dropout for overfitting test
    task=cfg['train']['task']
)
model = model.to(device)

# Take one batch
batch_size = min(32, len(X_train_scaled))
X_batch = torch.FloatTensor(X_train_scaled[:batch_size]).to(device)
y_batch = torch.FloatTensor(y_train_win[:batch_size]).to(device)

print(f"\n  Batch size: {batch_size}")
print(f"  X_batch shape: {X_batch.shape}")
print(f"  y_batch shape: {y_batch.shape}")
print(f"  y_batch mean: {y_batch.mean().item():.6f}")
print(f"  y_batch std:  {y_batch.std().item():.6f}")

# Train on same batch
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # High learning rate
criterion = nn.MSELoss()

print(f"\n  Training on same {batch_size} samples for 500 epochs...")
model.train()

for epoch in range(500):
    optimizer.zero_grad()
    pred = model(X_batch)
    
    # Ensure shapes match
    pred = pred.view(-1)
    y_batch_flat = y_batch.view(-1)
    
    loss = criterion(pred, y_batch_flat)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == 499:
        with torch.no_grad():
            pred_std = pred.std().item()
            pred_mean = pred.mean().item()
            mae = torch.abs(pred - y_batch_flat).mean().item()
            
            print(f"    Epoch {epoch:3d}: Loss={loss.item():.8f} | "
                  f"Pred Mean={pred_mean:.6f} | Pred Std={pred_std:.6f} | MAE={mae:.6f}")

# Final check
model.eval()
with torch.no_grad():
    final_pred = model(X_batch).view(-1)
    final_loss = criterion(final_pred, y_batch.view(-1))
    final_mae = torch.abs(final_pred - y_batch.view(-1)).mean().item()
    final_std = final_pred.std().item()

print(f"\n📊 Overfitting Test Results:")
print(f"  Final Loss:      {final_loss.item():.8f}")
print(f"  Final MAE:       {final_mae:.6f}")
print(f"  Prediction Std:  {final_std:.6f}")
print(f"  Target Std:      {y_batch.std().item():.6f}")

if final_loss.item() > 0.001:
    print(f"\n  ⚠️  WARNING: Cannot overfit! Loss={final_loss.item():.8f} is too high")
    print("     Expected: Loss < 0.001 after 500 epochs on same batch")
    print("     → Model architecture likely has a problem")
elif final_std < 1e-6:
    print(f"\n  ❌ CRITICAL: Predictions have no variance!")
    print("     Even after training, model outputs constant values")
    print("     → Architecture is broken")
else:
    print(f"\n  ✅ Model CAN overfit! Architecture works correctly")

# Show sample predictions vs targets
print(f"\n  Sample predictions vs targets (first 5):")
for i in range(min(5, batch_size)):
    pred_val = final_pred[i].item()
    target_val = y_batch[i].item()
    error = abs(pred_val - target_val)
    print(f"    Sample {i}: Pred={pred_val:8.6f} | Target={target_val:8.6f} | Error={error:.6f}")

# ============================================================================
# STEP 5: TEST WITH DUMMY LINEAR MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DUMMY LINEAR MODEL TEST (NUCLEAR OPTION)")
print("=" * 80)

print("\n🧪 Testing simplest possible model...")

class DummyModel(nn.Module):
    """Simplest possible model - flattens input and passes through linear layer."""
    def __init__(self, input_size, seq_len):
        super().__init__()
        self.fc = nn.Linear(input_size * seq_len, 1)
    
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        return self.fc(x_flat).squeeze()

dummy = DummyModel(len(feature_cols), window_size).to(device)
dummy_optimizer = torch.optim.Adam(dummy.parameters(), lr=0.01)

print(f"  Training dummy model on same {batch_size} samples...")
dummy.train()

for epoch in range(500):
    dummy_optimizer.zero_grad()
    pred = dummy(X_batch)
    loss = criterion(pred, y_batch.view(-1))
    loss.backward()
    dummy_optimizer.step()
    
    if epoch % 100 == 0 or epoch == 499:
        pred_std = pred.std().item()
        print(f"    Epoch {epoch:3d}: Loss={loss.item():.8f} | Pred Std={pred_std:.6f}")

dummy.eval()
with torch.no_grad():
    dummy_pred = dummy(X_batch)
    dummy_loss = criterion(dummy_pred, y_batch.view(-1))
    dummy_std = dummy_pred.std().item()

print(f"\n📊 Dummy Model Results:")
print(f"  Final Loss:     {dummy_loss.item():.8f}")
print(f"  Prediction Std: {dummy_std:.6f}")

if dummy_loss.item() < 0.001:
    print(f"  ✅ Even dummy linear model can overfit")
    print(f"     → Data and training loop are fine")
    print(f"     → Problem is likely in LSTM architecture")
else:
    print(f"  ❌ CRITICAL: Even dummy model cannot fit!")
    print(f"     → Problem is in data or training loop, not architecture")

# ============================================================================
# DIAGNOSIS AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSIS AND RECOMMENDATIONS")
print("=" * 80)

# Analyze results
issues = []

if output_std < 1e-8:
    issues.append("Untrained model outputs constant values")

if final_loss.item() > 0.001:
    issues.append("Model cannot overfit single batch")

if final_std < 1e-6:
    issues.append("Trained predictions have no variance")

if dummy_loss.item() > 0.001:
    issues.append("Even dummy linear model fails")

if not issues:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nYour model architecture appears correct. The issue might be:")
    print("  1. Learning rate too low (increase to 0.001 or 0.01)")
    print("  2. Regularization too strong (reduce weight_decay)")
    print("  3. Gradient clipping too aggressive (increase or remove)")
    print("  4. Need more epochs to see learning")
else:
    print("\n❌ ISSUES DETECTED:\n")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n🔧 RECOMMENDED FIXES:\n")
    
    if "constant values" in str(issues) or "no variance" in str(issues):
        print("  1. CHECK OUTPUT LAYER:")
        print("     - Remove any activation functions from final layer")
        print("     - Ensure forward() returns squeezed tensor: pred.squeeze()")
        print("     - Verify shape: pred.shape should be (batch_size,) not (batch_size, 1)")
    
    if "cannot overfit" in str(issues):
        print("  2. CHECK MODEL ARCHITECTURE:")
        print("     - Verify LSTM hidden state is used correctly")
        print("     - Try simpler architecture first (1 layer, 32 hidden units)")
        print("     - Remove dropout during debugging")
    
    if "dummy" in str(issues):
        print("  3. CHECK DATA AND TRAINING:")
        print("     - Verify targets weren't accidentally normalized")
        print("     - Check for NaN or Inf in data")
        print("     - Ensure optimizer is updating parameters")

print("\n" + "=" * 80)
print("END OF DEBUGGING REPORT")
print("=" * 80)

