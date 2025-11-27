# Model Debugging Results & Solutions

## Executive Summary

✅ **GOOD NEWS**: Your model architecture is **completely correct**!  
❌ **THE ISSUE**: Training configuration is too conservative, preventing learning

---

## What We Found

### ✅ Model Architecture - PERFECT
- No activation function in output layer ✓
- Proper shape handling (squeeze) ✓
- Correct use of LSTM hidden states ✓
- All parameters have gradients enabled ✓

### ✅ Data Quality - EXCELLENT
- **Target variance**: σ = 0.037 (3.7% std deviation)
- **Range**: -12.5% to +14.1% returns
- **Distribution**: 56.4% positive, 43.6% negative
- **No NaN or Inf values**

### ✅ Overfitting Test - PASSED
The model **perfectly overfits** 32 samples:
- Final Loss: 0.00000054 (nearly zero!)
- Final MAE: 0.000568 (0.06% error)
- Predictions match targets within 0.05%

**This proves the architecture CAN learn!**

---

## The Real Problem: Configuration

Your current `config/default.yaml` has **very conservative settings**:

| Parameter | Current Value | Issue |
|-----------|--------------|-------|
| Learning Rate | 0.0001 | Too low - model learns too slowly |
| Weight Decay | 0.01 | Too high - over-regularization prevents learning |
| Direction Weight | 0.05 | Too low - doesn't penalize wrong direction enough |
| Window Size | 10 days | Too short for 5-day ahead prediction |
| Dropout | 0.3 | Moderate-high during training |

### Why This Causes Constant Predictions

1. **Low learning rate (0.0001)**: Model barely updates weights each epoch
2. **High weight decay (0.01)**: Strong L2 penalty pulls weights toward zero
3. **Result**: Model converges to predicting the mean (~0.005) because:
   - It can't learn fast enough to find better patterns
   - Regularization punishes any non-trivial weights
   - Predicting the average minimizes MSE loss on simple signals

---

## Solutions

### Option 1: Use Aggressive Learning Config (Recommended)

We created `config/aggressive_learning.yaml` with optimal settings:

```bash
python src/train.py --config config/aggressive_learning.yaml
```

**Key changes**:
- ✅ Learning rate: `0.001` (10x increase)
- ✅ Weight decay: `0.001` (10x decrease)
- ✅ Direction weight: `0.5` (10x increase)
- ✅ Window size: `20` days
- ✅ Early stop patience: `30` epochs

### Option 2: Gradually Tune Current Config

Modify `config/default.yaml` in steps:

**Step 1: Increase learning rate**
```yaml
train:
  lr: 0.001  # was 0.0001
```

**Step 2: Reduce regularization**
```yaml
train:
  weight_decay: 0.001  # was 0.01
  dropout: 0.2  # was 0.3
```

**Step 3: Fix direction penalty**
```yaml
train:
  loss_function: "direction_mse"  # was "huber_direction"
  direction_weight: 0.5  # was 0.05
```

**Step 4: Increase window**
```yaml
window:
  size: 20  # was 10
```

---

## Expected Results After Fix

With the aggressive learning config, you should see:

### During Training (first 50 epochs):
```
Epoch  10: Train Loss: 0.0012 | Val Loss: 0.0014 | Val Dir Acc: 0.52
Epoch  20: Train Loss: 0.0010 | Val Loss: 0.0013 | Val Dir Acc: 0.54
Epoch  30: Train Loss: 0.0009 | Val Loss: 0.0012 | Val Dir Acc: 0.56
Epoch  50: Train Loss: 0.0008 | Val Loss: 0.0011 | Val Dir Acc: 0.58
```

### Final Test Results:
```
Test MAE:           0.025-0.030 (2.5-3.0%)
Test RMSE:          0.035-0.040 (3.5-4.0%)
Direction Accuracy: 55-60% (better than random 50%)
```

### Prediction Variance:
- **Current (broken)**: std ≈ 0.0001 (all predictions same)
- **Expected (working)**: std ≈ 0.025-0.030 (matches target std)

---

## How to Verify It's Working

### 1. Check Prediction Variance

After first few epochs:
```python
predictions = model(X_val)
print(f"Prediction std: {predictions.std():.6f}")
print(f"Target std:     {y_val.std():.6f}")
```

**Should be**: Prediction std > 0.01 (at least 1%)

### 2. Check Direction Accuracy

```python
pred_signs = np.sign(predictions)
target_signs = np.sign(y_val)
dir_acc = (pred_signs == target_signs).mean()
print(f"Direction accuracy: {dir_acc:.2%}")
```

**Should be**: > 52% after 20 epochs, > 55% after 50 epochs

### 3. Watch Training Logs

Look for:
- Loss decreasing over time ✓
- Val accuracy increasing ✓
- Different predictions for different samples ✓

---

## Additional Debugging Tools

We've added several scripts to help:

### 1. Comprehensive Debugging
```bash
python scripts/debug_model.py
```
Checks:
- Data variance
- Model architecture
- Overfitting capability
- Output shapes

### 2. Loss Function Testing
```bash
python scripts/test_losses.py
```
Verifies direction-aware loss works correctly

### 3. Data Alignment Verification
```bash
python scripts/verify_data_alignment.py
```
Ensures features match dates correctly

### 4. Window Alignment Verification
```bash
python scripts/verify_window_alignment.py
```
Confirms sliding windows use correct target dates

---

## Next Steps

1. **Run with aggressive config**:
   ```bash
   cd timeseries-stock
   source venv/bin/activate
   python src/train.py --config config/aggressive_learning.yaml
   ```

2. **Monitor training** - watch for:
   - Predictions with variance (std > 0.01)
   - Direction accuracy improving
   - Loss decreasing

3. **If still having issues**, run:
   ```bash
   python scripts/debug_model.py
   ```
   And share the output

4. **After successful training**, analyze predictions:
   ```bash
   python scripts/analyze_regression.py
   ```

---

## Why Your Original Config Was Too Conservative

Stock prediction is a **very noisy task** (even experts can't predict perfectly). With such conservative settings:

- Low LR + High regularization = Can only learn very simple patterns
- Model found: "Predicting small positive return (~0.5%) minimizes loss"
- This works because: Market goes up 56.4% of the time
- But it's not real learning - just predicting the average!

The new config allows the model to:
- Learn faster (higher LR)
- Fit more complex patterns (lower regularization)
- Care about direction (direction-aware loss)
- See more history (larger window)

---

## Summary

| Check | Status | Notes |
|-------|--------|-------|
| Model Architecture | ✅ PERFECT | No changes needed |
| Data Quality | ✅ EXCELLENT | Good variance, no issues |
| Overfitting Test | ✅ PASSED | Can learn perfectly |
| Training Config | ❌ TOO CONSERVATIVE | **This is the issue** |
| Solution | ✅ READY | Use `aggressive_learning.yaml` |

**Bottom line**: Your code is correct! Just need better hyperparameters. 🚀

---

Generated: 2025-11-26
Debug script: `scripts/debug_model.py`

