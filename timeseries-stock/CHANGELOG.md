# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-01-05

### Added - Initial Release (v0)

#### Core Features
- OHLCV data loading and validation from CSV files
- Technical feature engineering:
  - Price returns
  - Rolling means and standard deviations
  - RSI (Relative Strength Index)
  - MACD indicators
  - Volume features and z-scores
- Sliding window construction for time-series sequences
- Time-aware train/val/test splitting (no data leakage)
- Feature standardization (fit on train only)

#### Models
- LSTM model for sequence prediction
- 1D-CNN model for pattern recognition
- Support for both classification and regression tasks

#### Training & Evaluation
- Training loop with early stopping
- Model checkpointing (saves best model)
- Comprehensive metrics:
  - Classification: accuracy, F1, precision, recall
  - Regression: MAE, MSE, RMSE, MAPE
- Configuration via YAML files
- CLI with parameter override support

#### Infrastructure
- Modular codebase structure
- Unit tests for windows and features
- Sample data for testing
- Docker support (optional)
- Comprehensive documentation

#### Extension Points
- Sentiment data stub for future integration
- Multi-ticker support framework
- Configurable feature pipeline

### Documentation
- Detailed README with usage examples
- Setup guide with troubleshooting
- Inline code documentation
- Integration guide for sentiment data

### Testing
- Window construction tests
- Feature engineering tests (no leakage verification)
- Target alignment tests

## [Future]

### Planned Features
- [ ] Sentiment data integration (Twitter, Reddit, news)
- [ ] Multi-ticker training support
- [ ] Transformer-based models
- [ ] Attention mechanisms
- [ ] Backtesting framework with trading strategies
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning with Optuna
- [ ] Real-time prediction API
- [ ] Web dashboard for visualization
- [ ] More technical indicators (Bollinger Bands, Stochastic, etc.)
- [ ] Alternative data sources (options flow, insider trading, etc.)
- [ ] Ensemble models
- [ ] Transfer learning across stocks
- [ ] Incremental learning support

### Known Limitations
- Single ticker only (v0)
- No sentiment data integration yet
- Basic feature set (can be extended)
- No backtesting framework yet
- CPU/GPU training only (no distributed training)

### Contributing
Contributions welcome! See README for areas of improvement.

