# Next-Day Stock Prediction (Time-Series ML)

A complete machine learning pipeline for predicting next-day stock movements using LSTM and CNN models with technical indicators.

## Overview

Given daily OHLCV (Open, High, Low, Close, Volume) historical data, this system:
- Engineers technical features (RSI, MACD, rolling statistics, volume indicators)
- Creates sliding windows over time-series data (default: 7-day look-back)
- Trains deep learning models (LSTM or 1D-CNN) for next-day prediction
- Supports both classification (direction) and regression (return) tasks
- Provides clean extension points for sentiment analysis integration

## Features

### Current (v0)
- ✅ OHLCV data ingestion from CSV
- ✅ Technical feature engineering (returns, RSI, MACD, rolling stats)
- ✅ **Optimized feature set** with redundancy reduction (correlation-based pruning)
- ✅ Sliding window construction with configurable window size
- ✅ LSTM and 1D-CNN model implementations
- ✅ Time-aware train/val/test splitting (no data leakage)
- ✅ Standardization (fit on train only)
- ✅ Classification and regression tasks
- ✅ Early stopping and model checkpointing
- ✅ Comprehensive metrics (accuracy, F1, MAE, RMSE, etc.)
- ✅ Configurable via YAML
- ✅ CLI with parameter overrides
- ✅ Optional sentiment data integration

### Future Extensions
- 📊 Advanced sentiment data sources (Twitter, Reddit, real-time news)
- 📈 Multi-ticker support
- 🎯 Advanced models (Transformers, GRU)
- 💹 Backtesting framework
- 📉 Feature importance analysis (SHAP, permutation importance)

## Project Structure

```
timeseries-stock/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml          # Hyperparameters and paths
├── data/
│   ├── raw/                  # Input CSV files
│   ├── interim/              # Cleaned data cache
│   └── processed/            # Preprocessed tensors
├── src/
│   ├── dataio.py            # CSV loading and validation
│   ├── features.py          # Technical feature engineering
│   ├── windows.py           # Sliding window construction
│   ├── dataset.py           # PyTorch Dataset/DataLoader
│   ├── models.py            # LSTM and CNN1D models
│   ├── train.py             # Training loop and evaluation
│   ├── utils.py             # Metrics, scaling, splitting
│   └── sentiment_stub.py    # Future sentiment integration
├── scripts/
│   └── run_train.sh         # Example training commands
├── tests/
│   ├── test_windows.py      # Window construction tests
│   └── test_features.py     # Feature engineering tests
└── artifacts/               # Saved models and configs
```

## Installation

```bash
# Clone or navigate to the project
cd timeseries-stock

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, scikit-learn
- ta (technical analysis library)
- pyyaml

## Quick Start

### 1. Prepare Your Data

Place your OHLCV CSV file in `data/raw/`. The CSV should have these columns:
```
date,open,high,low,close,volume
2020-01-01,100.5,102.3,99.8,101.2,1500000
2020-01-02,101.3,103.1,100.9,102.5,1600000
...
```

### 2. Train a Model

```bash
# Train with default config (LSTM, 7-day window, classification)
python -m src.train --config config/default.yaml
```

### 3. Customize Training

```bash
# Try different configurations
python -m src.train --config config/default.yaml window.size=14
python -m src.train --config config/default.yaml model.type=cnn1d
python -m src.train --config config/default.yaml train.task=regression
python -m src.train --config config/default.yaml window.size=20 model.hidden_size=128
```

## Configuration

Edit `config/default.yaml` to customize:

### Data Settings
```yaml
data:
  input_csv: "data/raw/SAMPLE.csv"
  target: "direction"  # or "return"
```

### Model Settings
```yaml
model:
  type: "lstm"        # or "cnn1d"
  hidden_size: 64
  num_layers: 1
  dropout: 0.1
```

### Window Settings
```yaml
window:
  size: 7             # Look-back period in days
  stride: 1           # Step between windows
```

### Training Settings
```yaml
train:
  epochs: 30
  batch_size: 64
  lr: 0.001
  task: "classification"  # or "regression"
  early_stop_patience: 5
```

## Usage Examples

### Example 1: Classification (Direction Prediction)
```bash
python -m src.train --config config/default.yaml \
  data.input_csv=data/raw/SAMPLE.csv \
  train.task=classification \
  data.target=direction
```

**Output:** Predicts whether next-day close > today's close (binary: 0 or 1)

### Example 2: Regression (Return Prediction)
```bash
python -m src.train --config config/default.yaml \
  data.input_csv=data/raw/SAMPLE.csv \
  train.task=regression \
  data.target=return
```

**Output:** Predicts next-day return (continuous value)

### Example 3: Long Look-Back Window
```bash
python -m src.train --config config/default.yaml \
  window.size=20 \
  model.hidden_size=128
```

**Use case:** Capture longer-term patterns (e.g., monthly trends)

### Example 4: CNN Model
```bash
python -m src.train --config config/default.yaml \
  model.type=cnn1d \
  model.hidden_size=64
```

**Use case:** Faster training, good for pattern recognition

## Output

After training, the following artifacts are saved in `artifacts/`:

1. **Model checkpoint** (`model_YYYYMMDD_HHMMSS.pt`)
   - Model weights
   - Configuration
   - Test metrics
   - Feature column names

2. **Configuration snapshot** (`config_YYYYMMDD_HHMMSS.yaml`)
   - Complete config used for this run

3. **Predictions** (`predictions_YYYYMMDD_HHMMSS.npz`)
   - Test set predictions
   - True targets
   - Corresponding dates

## Evaluation Metrics

### Classification
- **Accuracy:** Fraction of correct predictions
- **F1 Score:** Harmonic mean of precision and recall
- **Precision:** True positives / (true positives + false positives)
- **Recall:** True positives / (true positives + false negatives)
- **Hit Rate:** Directional accuracy (same as accuracy for binary)

### Regression
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error

## Testing

Run unit tests to verify correctness:

```bash
# Test sliding window construction
python tests/test_windows.py

# Test feature engineering
python tests/test_features.py
```

**Tests verify:**
- Window shapes are correct
- Targets are properly aligned (no leakage)
- Features are computed correctly
- Warm-up rows are handled properly

## Technical Details

### Feature Engineering

**Returns:**
- `close_ret_1`: 1-day close return

**Rolling Statistics (Reduced Redundancy):**
- `roll_mean_5`: Short-term moving average (5-day)
- `roll_mean_20`: Medium-term moving average (20-day)
- `roll_std_10`: Mid-range volatility (10-day)

**Momentum:**
- `rsi_14`: Relative Strength Index (14-day)

**MACD (Reduced Redundancy):**
- `macd`: MACD line (12-26 EMA difference)
- `macd_hist`: MACD histogram (divergence from signal)

**Volume (Reduced Redundancy):**
- `vol_roll_mean_20`: Volume moving average (20-day)
- `vol_zscore_20`: Volume z-score (20-day)

**Sentiment (Optional, Reduced Redundancy):**
- `sent_raw`: Raw daily sentiment score
- `sent_ma_5`: 5-day sentiment moving average
- `sent_std_20`: 20-day sentiment volatility
- `sent_change`: Day-over-day sentiment change

> **Note:** Features have been optimized to reduce multicollinearity. Highly correlated features (correlation ≥ 0.99) were removed to improve model generalization and reduce overfitting.

### Time-Series Windowing

Each training sample consists of:
- **Input:** Features over `[t - window_size + 1, ..., t]`
- **Target:** Outcome at `t+1`

Example with `window_size=7`:
```
Days 0-6   → Predict day 7
Days 1-7   → Predict day 8
Days 2-8   → Predict day 9
...
```

### Data Splitting

**Chronological split** (no shuffle, no leakage):
- Train: 70% (earliest data)
- Validation: 15% (middle data)
- Test: 15% (most recent data)

**Standardization:**
- Fit `StandardScaler` on training data only
- Transform val/test using training statistics

## Sentiment Data Integration

The pipeline includes built-in support for sentiment data with an optimized feature set:

### Enabling Sentiment Features

```yaml
# In config/default.yaml
sentiment:
  enabled: true
  csv_path: "data/raw/Daily News Sentiment Index.csv"
```

### Using Sentiment in Training

```python
# In src/train.py (automatic when sentiment.enabled=true)
from src.sentiment_stub import load_daily_sentiment

sent_df = load_daily_sentiment(df.index, ticker='^GSPC')
df_feat = build_stock_features(df, cfg['features'], sent_df=sent_df)
```

### Sentiment Features (Optimized)
- `sent_raw`: Daily sentiment score
- `sent_ma_5`: 5-day moving average (captures short-term trends)
- `sent_std_20`: 20-day volatility (measures sentiment stability)
- `sent_change`: Day-over-day change (momentum signal)

See `src/sentiment_stub.py` for integration guide and extension functions.

## Model Architecture

### LSTM Model
```
Input: (batch_size, window_size, num_features)
  ↓
LSTM layers (with dropout)
  ↓
Take last time step
  ↓
Fully connected layer
  ↓
Sigmoid (classification) or Linear (regression)
  ↓
Output: (batch_size,)
```

### CNN1D Model
```
Input: (batch_size, window_size, num_features)
  ↓
Conv1D (kernel_size=3)
  ↓
ReLU + Dropout
  ↓
Conv1D (kernel_size=3)
  ↓
ReLU + Dropout
  ↓
Global Max Pooling
  ↓
Fully connected layer
  ↓
Sigmoid (classification) or Linear (regression)
  ↓
Output: (batch_size,)
```

## Feature Redundancy Reduction

The feature set has been optimized to eliminate highly correlated features (correlation ≥ 0.99) that provide redundant information:

| **Feature Group** | **Original** | **Optimized** | **Rationale** |
|-------------------|--------------|---------------|---------------|
| Rolling means | 5, 10, 20 | 5, 20 | Dropped 10 (mid-point offers no unique signal) |
| Rolling stds | 5, 10, 20 | 10 | Kept mid-range; 5/20 are highly correlated |
| MACD | macd, macd_signal, macd_hist | macd, macd_hist | Signal is smoothed macd (redundant) |
| Volume means | 5, 20, 50 | 20 | Dropped 5 (too noisy) and 50 (too smooth) |
| Volume z-scores | 20, 50 | 20 | 50 is redundant with 20 |
| Sentiment MA | 5, 20 | 5 | Dropped 20 (highly correlated with 5) |
| Sentiment binary | sent_positive | ❌ Dropped | Constant feature (no information gain) |

**Benefits:**
- ✅ Reduced multicollinearity improves model stability
- ✅ Faster training (fewer features)
- ✅ Better generalization (less overfitting)
- ✅ Easier interpretation (each feature adds unique information)

## Tips and Best Practices

1. **Start with classification** (easier to interpret and often more robust)
2. **Try different window sizes** (7, 14, 20 days) to find what works best
3. **Use early stopping** to prevent overfitting
4. **Check for data leakage** (tests verify no leakage in features/targets)
5. **Monitor validation metrics** during training
6. **Compare LSTM vs CNN** (CNN is often faster, LSTM captures sequential patterns better)
7. **Scale up gradually** (start with small `hidden_size`, increase if underfitting)
8. **Use optimized feature set** (reduced redundancy improves performance)

## Troubleshooting

**Issue:** Model always predicts the same class
- **Solution:** Check class imbalance, try different learning rate, add more features

**Issue:** Validation loss increases while training loss decreases
- **Solution:** Overfitting - increase dropout, reduce model size, use early stopping

**Issue:** NaN losses during training
- **Solution:** Lower learning rate, check for extreme values in data

**Issue:** Poor performance
- **Solution:** Try longer window size, more features, different model architecture

## Citation

If you use this code for research, please cite:

```
@software{timeseries_stock_prediction,
  title = {Next-Day Stock Prediction with Time-Series ML},
  year = {2025},
  author = {CS289 Project}
}
```

## License

MIT License - feel free to use and modify for your projects.

## Contributing

Contributions welcome! Areas for improvement:
- Additional technical indicators
- Transformer-based models
- Multi-ticker training
- Backtesting framework
- Sentiment integration
- Real-time prediction API

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Note:** This is a research/educational tool. Past performance does not guarantee future results. Always do your own research before making investment decisions.

