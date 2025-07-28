"""
Trend Following Strategy Implementation
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, Optional
from .base_strategy import BaseStrategy
from core.data_structures import TradeSignal

class TrendFollowingStrategy(BaseStrategy):
    """Trend following trading strategy"""
    
    def __init__(self, config: Dict):
        super().__init__('trend_following', config)
        self.tf_fast = config.get('tf_fast', 8)
        self.tf_slow = config.get('tf_slow', 21)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend following signals"""
        
        features = self._calculate_indicators(df)
        
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['position_size'] = 0.0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        signals['risk_score'] = 0.0
        
        for i in range(max(50, self.tf_slow), len(df)):
            if self._should_generate_signal(i, signals):
                signal = self._generate_tf_signal(df, features, i)
                
                if signal and signal.direction != 0:
                    signals.loc[df.index[i], 'signal'] = signal.direction
                    signals.loc[df.index[i], 'strength'] = signal.strength
                    
            self._check_exit_conditions(df, signals, i, features)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for trend following"""
        
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Moving averages
        features['sma_fast'] = talib.SMA(close, self.tf_fast)
        features['sma_slow'] = talib.SMA(close, self.tf_slow)
        features['ema_fast'] = talib.EMA(close, self.tf_fast)
        features['ema_slow'] = talib.EMA(close, self.tf_slow)
        
        # MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(close)
        
        # ADX for trend strength
        features['adx'] = talib.ADX(high, low, close, 14)
        
        # ATR for volatility
        features['atr'] = talib.ATR(high, low, close, 14)
        
        # Volume indicators
        features['volume_ratio'] = volume / volume.rolling(20).mean()
        features['volume_sma'] = talib.SMA(volume, 20)
        
        # Price momentum
        features['roc'] = talib.ROC(close, 10)
        features['mom'] = talib.MOM(close, 10)
        
        # Returns
        features['returns'] = close.pct_change()
        
        return features
    
    def _generate_tf_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> Optional[TradeSignal]:
        """Generate trend following signal"""
        
        sma_fast = features['sma_fast'].iloc[i]
        sma_slow = features['sma_slow'].iloc[i]
        ema_fast = features['ema_fast'].iloc[i]
        ema_slow = features['ema_slow'].iloc[i]
        macd = features['macd'].iloc[i]
        macd_signal = features['macd_signal'].iloc[i]
        macd_hist = features['macd_hist'].iloc[i]
        adx = features['adx'].iloc[i]
        volume_ratio = features['volume_ratio'].iloc[i]
        
        price = df['close'].iloc[i]
        
        # Trend strength
        trend_strength = abs(sma_fast - sma_slow) / sma_slow if sma_slow > 0 else 0
        
        # Price momentum
        price_momentum = features['returns'].iloc[max(0, i-10):i].mean() if i >= 10 else 0
        
        # Buy signal conditions
        sma_bull = sma_fast > sma_slow * 1.001
        ema_bull = ema_fast > ema_slow * 1.001
        macd_bull = macd > macd_signal and macd > 0
        
        if (sma_bull and ema_bull and macd_bull and 
            trend_strength > 0.005 and
            price_momentum > 0.001 and
            macd_hist > 0 and
            adx > 25):
            
            strength = min(1.0,
                         min(trend_strength * 50, 1.0) * 0.3 +
                         min(abs(macd - macd_signal) / (abs(macd_signal) + 1e-8), 1.0) * 0.2 +
                         min(price_momentum * 100, 1.0) * 0.3 +
                         min(volume_ratio / 1.5, 1.0) * 0.2)
            
            # Adjust for ADX
            if adx > 40:
                strength *= 1.2
            elif adx < 20:
                strength *= 0.8
            
            strength = min(1.0, strength)
            
            if strength > 0.65:
                return TradeSignal(
                    timestamp=df.index[i],
                    direction=1,
                    strength=strength,
                    strategy_type='trend_following',
                    indicators={'trend_strength': trend_strength, 'macd': macd, 'momentum': price_momentum, 'adx': adx}
                )
        
        # Sell signal conditions
        elif (not sma_bull and not ema_bull and not macd_bull and
              trend_strength > 0.005 and
              price_momentum < -0.001 and
              macd_hist < 0 and
              adx > 25):
            
            strength = min(1.0,
                         min(trend_strength * 50, 1.0) * 0.3 +
                         min(abs(macd_signal - macd) / (abs(macd_signal) + 1e-8), 1.0) * 0.2 +
                         min(abs(price_momentum) * 100, 1.0) * 0.3 +
                         min(volume_ratio / 1.5, 1.0) * 0.2)
            
            # Adjust for ADX
            if adx > 40:
                strength *= 1.2
            elif adx < 20:
                strength *= 0.8
            
            strength = min(1.0, strength)
            
            if strength > 0.65:
                return TradeSignal(
                    timestamp=df.index[i],
                    direction=-1,
                    strength=strength,
                    strategy_type='trend_following',
                    indicators={'trend_strength': -trend_strength, 'macd': macd, 'momentum': price_momentum, 'adx': adx}
                )
        
        return None
    
    def _check_exit_conditions(self, df: pd.DataFrame, signals: pd.DataFrame, i: int, features: pd.DataFrame):
        """Check exit conditions for trend following"""
        
        if i == 0:
            return
        
        current_position = signals['signal'].iloc[i-1]
        if current_position == 0:
            return
        
        sma_fast = features['sma_fast'].iloc[i]
        sma_slow = features['sma_slow'].iloc[i]
        macd = features['macd'].iloc[i]
        macd_signal = features['macd_signal'].iloc[i]
        
        # Exit conditions
        if current_position == 1:  # Long position
            if sma_fast < sma_slow or macd < macd_signal:
                signals.loc[signals.index[i], 'signal'] = 0
        else:  # Short position
            if sma_fast > sma_slow or macd > macd_signal:
                signals.loc[signals.index[i], 'signal'] = 0