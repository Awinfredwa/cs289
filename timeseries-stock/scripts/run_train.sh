#!/bin/bash
# Example training commands for stock prediction model

echo "==================================="
echo "Stock Prediction Training Examples"
echo "==================================="

# Example 1: Default training (LSTM, 7-day window, classification)
echo ""
echo "Example 1: Default LSTM classification"
echo "Command: python -m src.train --config config/default.yaml"
# python -m src.train --config config/default.yaml

# Example 2: Try CNN model
echo ""
echo "Example 2: CNN model"
echo "Command: python -m src.train --config config/default.yaml model.type=cnn1d"
# python -m src.train --config config/default.yaml model.type=cnn1d

# Example 3: Regression task (predict returns)
echo ""
echo "Example 3: Regression task"
echo "Command: python -m src.train --config config/default.yaml train.task=regression data.target=return"
# python -m src.train --config config/default.yaml train.task=regression data.target=return

# Example 4: Longer look-back window
echo ""
echo "Example 4: 20-day window"
echo "Command: python -m src.train --config config/default.yaml window.size=20"
# python -m src.train --config config/default.yaml window.size=20

# Example 5: Larger model
echo ""
echo "Example 5: Larger LSTM model"
echo "Command: python -m src.train --config config/default.yaml model.hidden_size=128 model.num_layers=2"
# python -m src.train --config config/default.yaml model.hidden_size=128 model.num_layers=2

# Example 6: Different hyperparameters
echo ""
echo "Example 6: Custom hyperparameters"
echo "Command: python -m src.train --config config/default.yaml train.lr=0.0001 train.batch_size=32 train.epochs=50"
# python -m src.train --config config/default.yaml train.lr=0.0001 train.batch_size=32 train.epochs=50

# Example 7: Multiple overrides
echo ""
echo "Example 7: Multiple custom settings"
echo "Command: python -m src.train --config config/default.yaml window.size=14 model.type=cnn1d model.hidden_size=128 train.epochs=40"
# python -m src.train --config config/default.yaml window.size=14 model.type=cnn1d model.hidden_size=128 train.epochs=40

echo ""
echo "==================================="
echo "To run any example, uncomment the corresponding python command line above"
echo "==================================="

