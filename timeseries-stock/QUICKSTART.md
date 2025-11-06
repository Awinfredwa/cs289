# Quick Start Guide

## 5-Minute Setup

```bash
# 1. Navigate to project
cd timeseries-stock

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train on sample data (100 days of synthetic data)
python -m src.train --config config/default.yaml
```

## Expected Output

```
================================================================================
STOCK PREDICTION TRAINING
================================================================================

[1/9] Loading data...
Loaded 100 days of data

[2/9] Building features...
[3/9] Creating targets...
[4/9] Dropping warm-up rows...
Remaining: 74 days

[5/9] Splitting data...
Time-based split:
  Train: 51 samples
  Val:   11 samples
  Test:  11 samples

[6/9] Building sliding windows...
[7/9] Standardizing features...
[8/9] Creating DataLoaders...
[9/9] Building model...

Model: LSTM
Parameters: 13,633

================================================================================
TRAINING
================================================================================

Epoch   1/30 | Train Loss: 0.6521 | Val Loss: 0.6234 | Val Acc: 0.6364 | Val F1: 0.6000
Epoch   2/30 | Train Loss: 0.6312 | Val Loss: 0.6187 | Val Acc: 0.6364 | Val F1: 0.6000
...

================================================================================
TEST EVALUATION
================================================================================

Test Metrics:
  loss: 0.6123
  accuracy: 0.6364
  f1: 0.6154
  precision: 0.6667
  recall: 0.5714

================================================================================
TRAINING COMPLETE
================================================================================
```

## Try Different Configurations

### 1. Regression (Predict Returns)
```bash
python -m src.train --config config/default.yaml \
  train.task=regression \
  data.target=return
```

### 2. Longer Window (20 days)
```bash
python -m src.train --config config/default.yaml \
  window.size=20
```

### 3. CNN Model
```bash
python -m src.train --config config/default.yaml \
  model.type=cnn1d
```

### 4. Larger Model
```bash
python -m src.train --config config/default.yaml \
  model.hidden_size=128 \
  model.num_layers=2 \
  train.epochs=50
```

## Use Your Own Data

### Step 1: Get Stock Data

**Option A: Yahoo Finance (easiest)**
```bash
pip install yfinance
python -c "import yfinance as yf; yf.download('^GSPC', start='2020-01-01').to_csv('data/raw/SP500.csv')"
```

**Option B: Manual CSV**

Create `data/raw/SP500.csv`:
```csv
date,open,high,low,close,volume
2020-01-01,100.5,102.3,99.8,101.2,1500000
2020-01-02,101.3,103.1,100.9,102.5,1600000
...
```

### Step 2: Update Config

Edit `config/default.yaml`:
```yaml
data:
  input_csv: "data/raw/SP500.csv"  # Your file
```

### Step 3: Train
```bash
python -m src.train --config config/default.yaml
```

## Output Files

After training, check `artifacts/`:
- `model_YYYYMMDD_HHMMSS.pt` - Trained model
- `config_YYYYMMDD_HHMMSS.yaml` - Config used
- `predictions_YYYYMMDD_HHMMSS.npz` - Test predictions

## Load and Use Model

```python
import torch
import numpy as np

# Load model
checkpoint = torch.load('artifacts/model_20250105_123456.pt')
model_state = checkpoint['model_state_dict']
config = checkpoint['config']
test_metrics = checkpoint['test_metrics']

print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

# Load predictions
preds = np.load('artifacts/predictions_20250105_123456.npz')
print(preds['test_preds'][:10])  # First 10 predictions
print(preds['test_targets'][:10])  # First 10 true values
print(preds['test_dates'][:10])  # Corresponding dates
```

## Common Issues

### "No module named 'src'"
Run from project root: `cd timeseries-stock`

### "File not found: data/raw/SAMPLE.csv"
Check you're in the project directory: `pwd` should show `.../timeseries-stock`

### Out of memory
Reduce batch size: `python -m src.train --config config/default.yaml train.batch_size=16`

### Poor accuracy
- Try longer window: `window.size=14`
- More epochs: `train.epochs=50`
- Use real data (sample data is synthetic)

## Next Steps

1. ✅ Train on real stock data
2. ✅ Experiment with hyperparameters
3. ✅ Try different stocks (MSFT, GOOGL, TSLA, etc.)
4. ✅ Compare LSTM vs CNN models
5. ✅ Test on different time periods

## Resources

- Full documentation: `README.md`
- Setup troubleshooting: `SETUP.md`
- Test your setup: `python tests/test_windows.py`
- Example scripts: `scripts/run_train.sh`

---

**Ready to predict the market? Start training! 🚀📈**

