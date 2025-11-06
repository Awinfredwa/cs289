"""Simple verification script to check setup."""

import sys
print("Python version:", sys.version)

# Test imports
print("\n=== Testing imports ===")

try:
    import numpy as np
    print("✓ numpy:", np.__version__)
except Exception as e:
    print("✗ numpy:", e)

try:
    import pandas as pd
    print("✓ pandas:", pd.__version__)
except Exception as e:
    print("✗ pandas:", e)

try:
    import sklearn
    print("✓ scikit-learn:", sklearn.__version__)
except Exception as e:
    print("✗ scikit-learn:", e)

try:
    import yaml
    print("✓ pyyaml: OK")
except Exception as e:
    print("✗ pyyaml:", e)

try:
    import torch
    print("✓ torch:", torch.__version__)
    if torch.cuda.is_available():
        print("  - CUDA available")
    elif torch.backends.mps.is_available():
        print("  - MPS (Apple Silicon) available")
    else:
        print("  - CPU only")
except Exception as e:
    print("✗ torch:", e)

try:
    import ta
    print("✓ ta: OK")
except Exception as e:
    print("✗ ta:", e)

print("\n=== Testing basic module imports ===")

try:
    from src import dataio, features, windows, dataset, models, utils
    print("✓ All src modules imported successfully")
except Exception as e:
    print("✗ Module import failed:", e)
    import traceback
    traceback.print_exc()

print("\n=== Setup verification complete ===")

