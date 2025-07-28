"""
Data Management Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import yfinance as yf
import os
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """Centralized data management system"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_sources = {
            'yahoo': self._fetch_yahoo_data,
            'alpaca': self._fetch_alpaca_data,
            'ibkr': self._fetch_ibkr_data
        }
        self.cache_expiry_hours = 24
        
    def get_historical_data(self, 
                          symbol: str, 
                          start_date: str,
                          end_date: str,
                          interval: str = '1d',
                          source: str = 'yahoo',
                          use_cache: bool = True) -> pd.DataFrame:
        """Get historical data with caching"""
        
        # Generate cache key
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}_{source}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache
        if use_cache and cache_file.exists():
            cache_data = self._load_cache(cache_file)
            if cache_data is not None:
                logger.info(f"Loaded {symbol} data from cache")
                return cache_data
        
        # Fetch fresh data
        logger.info(f"Fetching {symbol} data from {source}")
        
        if source in self.data_sources:
            data = self.data_sources[source](symbol, start_date, end_date, interval)
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        # Save to cache
        if use_cache and not data.empty:
            self._save_cache(data, cache_file)
        
        return data
    
    def _fetch_yahoo_data(self, symbol: str, start_date: str, 
                         end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            # Convert interval format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '60m', '1d': '1d', '1w': '1wk', '1M': '1mo'
            }
            yf_interval = interval_map.get(interval, '1d')
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Remove timezone if present
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_alpaca_data(self, symbol: str, start_date: str,
                          end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Alpaca (placeholder)"""
        logger.warning("Alpaca data fetching not implemented")
        return pd.DataFrame()
    
    def _fetch_ibkr_data(self, symbol: str, start_date: str,
                        end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from IBKR (placeholder)"""
        logger.warning("IBKR data fetching not implemented")
        return pd.DataFrame()
    
    def get_realtime_data(self, symbols: List[str], 
                         fields: List[str] = None) -> Dict[str, Dict]:
        """Get real-time data for multiple symbols"""
        
        if fields is None:
            fields = ['regularMarketPrice', 'bid', 'ask', 'volume']
        
        realtime_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                realtime_data[symbol] = {
                    'price': info.get('regularMarketPrice', 0),
                    'bid': info.get('bid', 0),
                    'ask': info.get('ask', 0),
                    'volume': info.get('volume', 0),
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"Error getting realtime data for {symbol}: {e}")
                realtime_data[symbol] = None
        
        return realtime_data
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """Get fundamental data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            fundamentals = {
                'market_cap': ticker.info.get('marketCap', 0),
                'pe_ratio': ticker.info.get('trailingPE', 0),
                'dividend_yield': ticker.info.get('dividendYield', 0),
                'beta': ticker.info.get('beta', 0),
                'earnings_growth': ticker.info.get('earningsGrowth', 0),
                'revenue_growth': ticker.info.get('revenueGrowth', 0),
                'profit_margin': ticker.info.get('profitMargins', 0),
                'debt_to_equity': ticker.info.get('debtToEquity', 0),
                'sector': ticker.info.get('sector', 'Unknown'),
                'industry': ticker.info.get('industry', 'Unknown')
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error getting fundamental data for {symbol}: {e}")
            return {}
    
    def get_options_chain(self, symbol: str, 
                         expiration_date: str = None) -> pd.DataFrame:
        """Get options chain data"""
        try:
            ticker = yf.Ticker(symbol)
            
            if expiration_date is None:
                # Get next expiration
                expirations = ticker.options
                if expirations:
                    expiration_date = expirations[0]
                else:
                    return pd.DataFrame()
            
            # Get options data
            opt_chain = ticker.option_chain(expiration_date)
            
            # Combine calls and puts
            calls = opt_chain.calls
            calls['type'] = 'call'
            
            puts = opt_chain.puts
            puts['type'] = 'put'
            
            options_data = pd.concat([calls, puts], ignore_index=True)
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return pd.DataFrame()
    
    def update_market_calendar(self, year: int = None):
        """Update market calendar with holidays"""
        if year is None:
            year = datetime.now().year
        
        # US market holidays (simplified)
        holidays = [
            f"{year}-01-01",  # New Year's Day
            f"{year}-01-18",  # MLK Day (3rd Monday of January)
            f"{year}-02-15",  # Presidents Day (3rd Monday of February)
            f"{year}-04-02",  # Good Friday (varies)
            f"{year}-05-31",  # Memorial Day (last Monday of May)
            f"{year}-07-05",  # Independence Day (observed)
            f"{year}-09-06",  # Labor Day (1st Monday of September)
            f"{year}-11-25",  # Thanksgiving (4th Thursday of November)
            f"{year}-12-24",  # Christmas (observed)
        ]
        
        calendar_file = self.cache_dir / f"market_calendar_{year}.pkl"
        
        with open(calendar_file, 'wb') as f:
            pickle.dump(holidays, f)
        
        return holidays
    
    def is_market_open(self, timestamp: datetime = None) -> bool:
        """Check if market is open"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check if weekend
        if timestamp.weekday() >= 5:
            return False
        
        # Check if holiday
        year = timestamp.year
        calendar_file = self.cache_dir / f"market_calendar_{year}.pkl"
        
        if calendar_file.exists():
            with open(calendar_file, 'rb') as f:
                holidays = pickle.load(f)
            
            if timestamp.strftime('%Y-%m-%d') in holidays:
                return False
        
        # Check market hours (9:30 AM - 4:00 PM ET)
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= timestamp <= market_close
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data"""
        
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by index
        data = data.sort_index()
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (simple method)
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                data = data[abs(data[col] - mean) <= 3 * std]
        
        # Ensure OHLC consistency
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            data = data[
                (data['high'] >= data['low']) &
                (data['high'] >= data['open']) &
                (data['high'] >= data['close']) &
                (data['low'] <= data['open']) &
                (data['low'] <= data['close'])
            ]
        
        return data
    
    def _load_cache(self, cache_file: Path) -> Optional[pd.DataFrame]:
        """Load data from cache if not expired"""
        try:
            # Check file age
            file_age_hours = (datetime.now() - datetime.fromtimestamp(
                cache_file.stat().st_mtime
            )).total_seconds() / 3600
            
            if file_age_hours > self.cache_expiry_hours:
                logger.info(f"Cache expired for {cache_file.name}")
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None
    
    def _save_cache(self, data: pd.DataFrame, cache_file: Path):
        """Save data to cache"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved to cache: {cache_file.name}")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def export_data(self, data: pd.DataFrame, filename: str, 
                   format: str = 'csv'):
        """Export data to file"""
        
        export_path = self.cache_dir.parent / 'exports'
        export_path.mkdir(exist_ok=True)
        
        file_path = export_path / filename
        
        if format == 'csv':
            data.to_csv(file_path)
        elif format == 'excel':
            data.to_excel(file_path)
        elif format == 'parquet':
            data.to_parquet(file_path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Data exported to {file_path}")
        
    def get_market_snapshot(self, indices: List[str] = None) -> Dict:
        """Get market snapshot"""
        
        if indices is None:
            indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P500, Dow, Nasdaq, VIX
        
        snapshot = {}
        
        for index in indices:
            try:
                ticker = yf.Ticker(index)
                info = ticker.info
                
                # Get today's data
                today_data = ticker.history(period='1d')
                
                if not today_data.empty:
                    snapshot[index] = {
                        'name': info.get('shortName', index),
                        'price': today_data['Close'].iloc[-1],
                        'change': today_data['Close'].iloc[-1] - today_data['Open'].iloc[0],
                        'change_pct': (today_data['Close'].iloc[-1] / today_data['Open'].iloc[0] - 1) * 100,
                        'volume': today_data['Volume'].iloc[-1]
                    }
                    
            except Exception as e:
                logger.error(f"Error getting snapshot for {index}: {e}")
        
        return snapshot