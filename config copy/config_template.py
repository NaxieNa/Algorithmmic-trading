"""
System Configuration Template
"""

import os
from typing import Dict

# Environment-specific settings
ENVIRONMENT = os.getenv('TRADING_ENV', 'development')

# Base configuration
BASE_CONFIG = {
    "environment": ENVIRONMENT,
    "log_level": "INFO" if ENVIRONMENT == "production" else "DEBUG",
    "data_dir": "data",
    "cache_dir": "data/cache",
    "log_dir": "logs",
}

# Broker configurations
BROKER_CONFIGS = {
    "ibkr": {
        "host": os.getenv("IBKR_HOST", "127.0.0.1"),
        "port": int(os.getenv("IBKR_PORT", 7497)),  # 7497 for paper, 7496 for live
        "client_id": int(os.getenv("IBKR_CLIENT_ID", 1)),
        "account": os.getenv("IBKR_ACCOUNT", "DUK362248"),
        "paper": os.getenv("IBKR_PAPER", "true").lower() == "true",
        "rate_limits": {
            "historical_data": {"calls": 6, "period": 60},
            "positions": {"calls": 1, "period": 5},
            "orders": {"calls": 50, "period": 60}
        }
    },
    "alpaca": {
        "api_key": os.getenv("ALPACA_API_KEY"),
        "api_secret": os.getenv("ALPACA_API_SECRET"),
        "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        "data_url": os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets"),
        "feed": "iex",  # or "sip" for paid data
    },
    "backtest": {
        "commission": 0.001,  # 0.1%
        "slippage": 0.0005,   # 0.05%
        "initial_capital": 100000
    }
}

# Trading configurations
TRADING_CONFIG = {
    "symbols": ["SPY", "QQQ", "IWM", "TLT", "GLD"],
    "initial_capital": float(os.getenv("INITIAL_CAPITAL", 100000)),
    "loop_interval": 60,  # seconds
    "market_hours_only": True,
    "timezone": "US/Eastern",
    
    "data_config": {
        "lookback_period": "30 D",
        "bar_size": "15 mins",
        "update_interval": 300,  # 5 minutes
        "cache_expiry": 86400,   # 24 hours
    },
    
    "execution_config": {
        "max_spread_pct": 0.001,
        "urgency_threshold": 0.8,
        "slice_size": 0.1,
        "vwap_window": 5,
        "order_timeout": 300,  # 5 minutes
        "retry_attempts": 3,
    }
}

# Risk management configurations
RISK_CONFIG = {
    "base_risk_pct": 0.02,      # 2% base risk per trade
    "max_risk_pct": 0.05,       # 5% maximum risk per trade
    "max_drawdown": 0.15,       # 15% maximum drawdown
    "max_concentration": 0.40,   # 40% maximum position concentration
    
    "volatility_config": {
        "vol_lookback": 20,
        "vol_target": 0.20,      # 20% target volatility
        "vol_scalar_min": 0.5,
        "vol_scalar_max": 2.5,
    },
    
    "kelly_config": {
        "kelly_lookback": 30,
        "kelly_fraction": 0.25,  # Use 25% of Kelly
        "kelly_min_trades": 10,
    },
    
    "regime_config": {
        "regime_vol_threshold": 0.025,
        "regime_trend_threshold": 0.002,
    },
    
    "position_limits": {
        "max_positions": 10,
        "min_position_value": 1000,
        "max_position_value": 50000,
        "integer_shares_only": True,
    }
}

# Strategy configurations
STRATEGY_CONFIGS = {
    "mean_reversion": {
        "enabled": True,
        "allocation": 0.3,  # 30% of capital
        "mr_entry_z": 2.0,
        "mr_exit_z": 0.5,
        "lookback": 20,
        "min_volume": 1000000,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_periods": 20,
        "bb_stdev": 2,
    },
    
    "trend_following": {
        "enabled": True,
        "allocation": 0.3,  # 30% of capital
        "tf_fast": 10,
        "tf_slow": 30,
        "atr_multiplier": 2.0,
        "adx_threshold": 25,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
    },
    
    "combined": {
        "enabled": True,
        "allocation": 0.4,  # 40% of capital
        "use_regime_detection": True,
        "regime_lookback": 60,
        "confidence_threshold": 0.7,
    },
    
    "momentum": {
        "enabled": False,
        "allocation": 0.0,
        "lookback": 20,
        "rank_threshold": 0.8,
        "holding_period": 5,
    }
}

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "trading"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "pool_size": 5,
    "max_overflow": 10,
}

# Monitoring configuration
MONITORING_CONFIG = {
    "dashboard": {
        "enabled": True,
        "port": 8501,
        "refresh_interval": 5,  # seconds
        "theme": "dark",
    },
    
    "alerts": {
        "enabled": True,
        "email": os.getenv("ALERT_EMAIL"),
        "sms": os.getenv("ALERT_PHONE"),
        "webhook": os.getenv("ALERT_WEBHOOK"),
        
        "thresholds": {
            "max_drawdown": 0.10,      # Alert at 10% drawdown
            "daily_loss": 0.05,        # Alert at 5% daily loss
            "position_concentration": 0.35,  # Alert at 35% concentration
            "error_rate": 0.1,         # Alert if 10% of operations fail
        }
    },
    
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_rotation": "daily",
        "max_files": 7,
        "console_output": True,
    },
    
    "performance_reporting": {
        "enabled": True,
        "frequency": "daily",
        "report_time": "16:30",  # After market close
        "recipients": [os.getenv("REPORT_EMAIL")],
        "include_charts": True,
    }
}

# Market calendar configuration
MARKET_CALENDAR_CONFIG = {
    "timezone": "US/Eastern",
    "market_open": "09:30",
    "market_close": "16:00",
    "pre_market_open": "04:00",
    "after_hours_close": "20:00",
    
    # Simplified holiday list - should use pandas_market_calendars in production
    "holidays_2024": [
        "2024-01-01",  # New Year's Day
        "2024-01-15",  # Martin Luther King Jr. Day
        "2024-02-19",  # Presidents Day
        "2024-03-29",  # Good Friday
        "2024-05-27",  # Memorial Day
        "2024-06-19",  # Juneteenth
        "2024-07-04",  # Independence Day
        "2024-09-02",  # Labor Day
        "2024-11-28",  # Thanksgiving
        "2024-12-25",  # Christmas
    ]
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "technical_indicators": {
        "enabled": True,
        "indicators": [
            "RSI", "MACD", "BB", "ATR", "ADX", "CCI", 
            "STOCH", "OBV", "MFI", "ROC", "WILLR"
        ],
        "custom_periods": {
            "RSI": [7, 14, 21],
            "SMA": [5, 10, 20, 50, 200],
            "EMA": [5, 10, 20, 50],
        }
    },
    
    "statistical_features": {
        "enabled": True,
        "rolling_windows": [5, 10, 20, 60],
        "calculations": [
            "returns", "volatility", "skewness", "kurtosis",
            "autocorrelation", "hurst_exponent"
        ]
    },
    
    "microstructure_features": {
        "enabled": True,
        "include_spread": True,
        "include_imbalance": True,
        "include_toxicity": False,  # Requires L2 data
    },
    
    "regime_detection": {
        "enabled": True,
        "n_states": 3,
        "method": "hmm",  # or "threshold", "clustering"
    }
}

def load_config(override: Dict = None) -> Dict:
    """Load complete configuration with optional overrides"""
    
    config = {
        "base": BASE_CONFIG,
        "broker": BROKER_CONFIGS,
        "trading": TRADING_CONFIG,
        "risk": RISK_CONFIG,
        "strategies": STRATEGY_CONFIGS,
        "database": DATABASE_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "calendar": MARKET_CALENDAR_CONFIG,
        "features": FEATURE_CONFIG,
    }
    
    # Apply overrides if provided
    if override:
        _deep_update(config, override)
    
    # Validate configuration
    _validate_config(config)
    
    return config

def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively update nested dictionary"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def _validate_config(config: Dict):
    """Validate configuration values"""
    
    # Check required environment variables
    if config["trading"]["symbols"] is None or len(config["trading"]["symbols"]) == 0:
        raise ValueError("No trading symbols configured")
    
    # Validate risk parameters
    if config["risk"]["max_drawdown"] > 0.5:
        raise ValueError("Maximum drawdown too high (>50%)")
    
    if config["risk"]["base_risk_pct"] > config["risk"]["max_risk_pct"]:
        raise ValueError("Base risk cannot exceed maximum risk")
    
    # Validate strategy allocations
    total_allocation = sum(
        s["allocation"] for s in config["strategies"].values() 
        if s.get("enabled", False)
    )
    
    if abs(total_allocation - 1.0) > 0.01:
        raise ValueError(f"Strategy allocations must sum to 1.0, got {total_allocation}")
    
    # Validate broker configuration
    broker_type = config.get("broker_type", "ibkr")
    if broker_type == "alpaca" and not config["broker"]["alpaca"]["api_key"]:
        raise ValueError("Alpaca API key not configured")
    
    print("✅ Configuration validated successfully")

# Export functions
__all__ = ['load_config', 'ENVIRONMENT']