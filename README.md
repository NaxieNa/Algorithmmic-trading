# Algorithmmic-trading
# Algorithmic Trading: Enhanced Mean Reversion and Trend Following Strategies

This repository contains a complete algorithmic trading research project developed as part of my Certificate in Quantitative Finance (CQF) coursework and portfolio submission for MSc Finance (Part-Time) at LSE. The project integrates multiple strategy types, data pipelines, and backtesting modules, with compatibility for both historical data and real-time IBKR deployment.

## Project Overview

The goal of this project is to systematically design, optimize, and validate a set of algorithmic trading strategies across different market regimes and asset classes. The core focus is on:

- Mean reversion models (Z-score, Ornstein–Uhlenbeck process)
- Trend-following strategies (EMA, ADX, MACD)
- High-frequency modules (momentum scalping, breakout detection, volume-driven entry)
- IBKR live paper trading integration
- Cross-validation on SPY, QQQ, GLD, and TSLA

## Key Features

- End-to-end pipeline including data ingestion, feature engineering, signal generation, and execution
- Parameter optimization with in-sample and out-of-sample evaluation
- Strategy-specific performance metrics: Sharpe ratio, win rate, max drawdown, trade frequency
- Modular Python design, compatible with Jupyter Notebook and standalone `.py` execution
- Visual and statistical analysis included in final report

## Structure

```
.
├── brokers/               # Broker API integration (IBKR)
├── config/                # Global configuration and constants
├── core/                  # Strategy core engine and execution loop
├── data/                  # Data ingestion, feature engineering
├── examples/              # Example scripts for single strategy testing
├── monitoring/            # Real-time logs, error handling
├── strategies/            # Strategy definitions (mean reversion, trend following, etc.)
├── tests/                 # Unit and functional testing
├── utils/                 # Helper functions, plotting tools
├── main.py                # Main entry point for pipeline execution
├── FINAL_REPORT.ipynb     # Final project report (Jupyter Notebook)
├── docs/                  # PDF reports and visual outputs
└── requirements.txt       # Python dependencies
```

## Installation

Clone the repository:

```bash
git clone git@github.com:NaxieNa/Algorithmmic-trading.git
cd Algorithmmic-trading
```

Create a virtual environment and install dependencies:

```bash
python -m venv env
source env/bin/activate  # For macOS/Linux
pip install -r requirements.txt
```

Or use `conda`:

```bash
conda create -n trading_env python=3.11
conda activate trading_env
pip install -r requirements.txt
```

## How to Run

- For historical backtesting via notebook: open `FINAL_REPORT.ipynb`
- For real-time execution with IBKR API: run `main.py` after configuring `config/` and `brokers/ibkr_broker.py`
- Strategy-specific scripts are available in `examples/`



## Author

Yaming Xie  
London, UK   
Contact: [GitHub Profile](https://github.com/NaxieNa)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
