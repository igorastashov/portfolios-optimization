# Development of a System for Monitoring and Optimization of Investment Portfolios

This application provides tools for monitoring and optimizing investment portfolios using various algorithms and techniques from traditional finance and machine learning.

## Features

- **Dashboard**: Overview of portfolio performance and asset allocation
- **Portfolio Optimization**: Optimize your portfolio using the Markowitz model
- **Model Comparison**: Compare performance of different portfolio optimization models
- **Backtest Results**: Analyze historical performance of different strategies

## Optimization Models

1. **Markowitz Model**: Traditional mean-variance optimization
2. **Reinforcement Learning Models**:
   - A2C (Advantage Actor-Critic)
   - PPO (Proximal Policy Optimization)
   - DDPG (Deep Deterministic Policy Gradient)
   - SAC (Soft Actor-Critic)

## Data

The system uses historical cryptocurrency price data from Binance, including:
- BNBUSDT, BTCUSDT, CAKEUSDT, ETHUSDT, LTCUSDT, SOLUSDT, STRKUSDT, TONUSDT
- USDCUSDT, XRPUSDT, PEPEUSDT, HBARUSDT, APTUSDT, LDOUSDT, JUPUSDT

## Installation

### Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- SciPy

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/portfolios-optimization.git
   cd portfolios-optimization
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Or with Poetry:
   ```
   poetry install
   ```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

### Notebooks

- `1_parse_data_binance.ipynb`: Data collection and preprocessing
- `2_markowitz-rebalanve.ipynb`: Implementation of Markowitz portfolio optimization
- `3_finrl_02_tests.ipynb`: Testing of reinforcement learning models
- `4_finrl_mvp_results.ipynb`: Analysis and comparison of model results

## Project Structure

```
.
├── app.py                          # Main Streamlit application
├── data/                           # Data directory
│   ├── data_compare_eda.csv        # Combined asset price data
│   ├── *_hourly_data.csv           # Individual asset price data
│   └── models_predictions/         # Model prediction results
├── notebooks/                      # Jupyter notebooks
├── portfolios_optimization/        # Core package
│   ├── __init__.py
│   ├── data_loader.py              # Data loading utilities
│   ├── portfolio_optimizer.py      # Portfolio optimization algorithms
│   ├── portfolio_analysis.py       # Portfolio performance analysis
│   └── visualization.py            # Data visualization utilities
└── README.md                       # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
