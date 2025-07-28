# Algorithmic Trading: Enhanced Mean Reversion and Trend Following Strategies

This repository contains a full-cycle algorithmic trading system integrating classical financial theories, quantitative factor modeling, and real-time deployment capabilities. The project was developed as part of the Certificate in Quantitative Finance (CQF) and is submitted as a technical showcase for the MSc Finance (Part-Time) program at the London School of Economics and Political Science (LSE).

## Project Objectives

The system aims to:

- Combine mean reversion and trend-following logics within a unified execution pipeline
- Design strategies that are robust across asset classes (e.g., SPY, QQQ, TSLA, GLD)
- Validate performance through both historical backtesting and real-time paper trading
- Integrate broker connectivity via Interactive Brokers (IBKR) API
- Provide modular, testable, and extensible infrastructure for live market operations

## Strategy Components

The implemented strategies include:

- **Mean Reversion Models**: Z-score based reversal detection; optional OU-process extensions
- **Trend-Following Modules**: Exponential Moving Average (EMA), Average Directional Index (ADX), MACD crossover detection
- **High-Frequency Add-ons**: Momentum scalping, breakout confirmation, volume spikes
- **Parameter Optimization**: In-sample grid search with validation on out-of-sample performance
- **Risk Control**: Drawdown monitoring, volatility filters, trade frequency limits

## Folder Structure

```
.
├── brokers/               # IBKR API wrapper and broker management
├── config/                # Config files (symbols, thresholds, trading calendar)
├── core/                  # Main pipeline orchestration logic
├── data/                  # Raw and processed data handlers
├── docs/                  # Exported visual reports and PDFs
├── examples/              # Sample scripts for running individual strategies
├── monitoring/            # Logging, error tracking, and execution monitoring
├── strategies/            # Strategy logic: reversion, trend, high-frequency modules
├── tests/                 # Unit tests and validation checks
├── utils/                 # Helper functions, plotting, and tools
├── AL_trading.ipynb     # Main notebook report: full implementation and results
├── requirements.txt       # Dependency list for reproducible environment
└── README.md              # Project documentation (this file)
```

## Key Deliverables

- **Notebook Report**: `AL_trading.ipynb` contains full workflow, backtesting visuals, and performance metrics.
- **Source Modules**: Each component is separately callable for flexible testing or integration.

## How to Use

1. Clone the repository:

```bash
git clone git@github.com:NaxieNa/Algorithmmic-trading.git
cd Algorithmmic-trading
```

2. Set up environment:

```bash
python -m venv env
source env/bin/activate   # On macOS/Linux
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook FINAL_REPORT.ipynb
```

4. Alternatively, run pipeline via:

```bash
python main.py
```

Ensure broker credentials and trading settings are configured under `config/` and `brokers/`.

## Report Highlights

The research notebook includes:

- Multi-strategy logic design
- Mathematical derivations (e.g., Lagrangian optimization, Sharpe ratio)
- In-sample and out-of-sample performance validation
- Feature engineering for predictive modeling
- Real-time market execution examples

## Author

Yaming Xie  
London, UK  
[GitHub Profile](https://github.com/NaxieNa)

## License

This project is released under the MIT License.
