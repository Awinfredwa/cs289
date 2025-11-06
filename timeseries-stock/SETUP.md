# Setup Guide

## Installation Steps

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd timeseries-stock

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Or use pyproject.toml
pip install -e .
```

### 3. Verify Installation

```bash
python scripts/verify_setup.py
```

You should see all packages installed successfully:
```
✓ numpy: X.X.X
✓ pandas: X.X.X
✓ scikit-learn: X.X.X
✓ pyyaml: OK
✓ torch: X.X.X
✓ ta: OK
```

### 4. Run Tests

```bash
# Test window construction
python tests/test_windows.py

# Test feature engineering
python tests/test_features.py
```

Both test suites should pass with `ALL TESTS PASSED ✓`.

### 5. Prepare Your Data

Place your OHLCV CSV file in `data/raw/`:

```bash
# Example: download stock data
# (You'll need to get this from Yahoo Finance, Alpha Vantage, etc.)
cp /path/to/your/SP500.csv data/raw/SP500.csv
```

Update `config/default.yaml`:
```yaml
data:
  input_csv: "data/raw/SP500.csv"  # Update this path
```

### 6. Train Your First Model

```bash
# Train with default settings
python -m src.train --config config/default.yaml
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues with PyTorch, install it separately first:

**For CPU only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**For CUDA (GPU):**
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**For Apple Silicon (M1/M2/M3):**
```bash
pip install torch
```

PyTorch on Apple Silicon will use MPS (Metal Performance Shaders) automatically.

### "ta" Library Issues

If the `ta` (technical analysis) library causes issues:

```bash
pip install --upgrade ta
```

Or install from source:
```bash
pip install git+https://github.com/bukosabino/ta.git
```

### ImportError: No module named 'src'

Make sure you're running commands from the project root directory:

```bash
cd /path/to/timeseries-stock
python -m src.train --config config/default.yaml
```

### Memory Issues

If you run out of memory during training:

1. Reduce batch size in `config/default.yaml`:
   ```yaml
   train:
     batch_size: 32  # or 16
   ```

2. Reduce model size:
   ```yaml
   model:
     hidden_size: 32  # instead of 64
     num_layers: 1
   ```

3. Use fewer features or shorter window:
   ```yaml
   window:
     size: 5  # instead of 7
   ```

### CUDA Out of Memory

If using GPU and getting CUDA OOM errors:

```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python -m src.train --config config/default.yaml
```

## Environment Variables

### Optional Settings

```bash
# Disable MPS (Apple Silicon) and use CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Set number of threads
export OMP_NUM_THREADS=4

# Disable CUDA
export CUDA_VISIBLE_DEVICES=""
```

## Development Setup

For development with additional tools:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Run tests with pytest
pytest tests/
```

## Docker Setup (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.train", "--config", "config/default.yaml"]
```

Build and run:

```bash
docker build -t timeseries-stock .
docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts timeseries-stock
```

## Data Sources

### Where to Get Stock Data

1. **Yahoo Finance** (free)
   ```bash
   pip install yfinance
   ```
   ```python
   import yfinance as yf
   df = yf.download('^GSPC', start='2020-01-01', end='2023-12-31')
   df.to_csv('data/raw/SP500.csv')
   ```

2. **Alpha Vantage** (free API key required)
   - Sign up: https://www.alphavantage.co/
   - Get daily data via their API

3. **Quandl/Nasdaq Data Link**
   - Various financial data sources

4. **Kaggle Datasets**
   - Pre-downloaded stock data

## Next Steps

After successful setup:

1. ✅ Run tests to verify everything works
2. ✅ Train on sample data
3. ✅ Get real stock data
4. ✅ Experiment with hyperparameters
5. ✅ Try different stocks and time periods
6. ✅ Extend with sentiment data (see `src/sentiment_stub.py`)

## Support

For issues or questions:
- Check the README.md for usage examples
- Review the troubleshooting section above
- Examine test files for usage patterns
- Read inline code documentation

Happy training! 🚀📈

