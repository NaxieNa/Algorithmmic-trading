"""
Combined Strategy Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict
from .base_strategy import BaseStrategy
from .mean_reversion import MeanReversionStrategy
from .trend_following import TrendFollowingStrategy
from core.data_structures import TradeSignal

class CombinedStrategy(BaseStrategy):
    """Combined strategy using both mean reversion and trend following"""
    
    def __init__(self, config: Dict):
        super().__init__('combined', config)
        self.use_regime_detection = config.get('use_regime_detection', True)
        
        # Initialize sub-strategies
        mr_config = config.copy()
        mr_config.update({'mr_entry_z': 2.0, 'mr_exit_z': 0.5})
        self.mr_strategy = MeanReversionStrategy(mr_config)
        
        tf_config = config.copy()
        tf_config.update({'tf_fast': 10, 'tf_slow': 30})
        self.tf_strategy = TrendFollowingStrategy(tf_config)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate combined signals"""
        
        # Generate signals from both strategies
        mr_signals = self.mr_strategy.generate_signals(df)
        tf_signals = self.tf_strategy.generate_signals(df)
        
        # Detect market regime if enabled
        if self.use_regime_detection:
            regime = self._detect_market_regime(df)
        else:
            regime = pd.Series('normal', index=df.index)
        
        # Combine signals based on regime
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['position_size'] = 0.0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        signals['risk_score'] = 0.0
        
        for i in range(50, len(df)):
            mr_signal = mr_signals['signal'].iloc[i]
            tf_signal = tf_signals['signal'].iloc[i]
            current_regime = regime.iloc[i]
            
            # Regime-based signal selection
            if current_regime == 'high_volatility':
                # Prefer mean reversion in high volatility
                if mr_signal != 0:
                    signals.loc[df.index[i], 'signal'] = mr_signal
                    signals.loc[df.index[i], 'strength'] = mr_signals.get('strength', pd.Series(0.7)).iloc[i]
                elif tf_signal != 0:
                    signals.loc[df.index[i], 'signal'] = tf_signal
                    signals.loc[df.index[i], 'strength'] = tf_signals.get('strength', pd.Series(0.5)).iloc[i] * 0.7
                    
            elif current_regime == 'trending':
                # Prefer trend following in trending markets
                if tf_signal != 0:
                    signals.loc[df.index[i], 'signal'] = tf_signal
                    signals.loc[df.index[i], 'strength'] = tf_signals.get('strength', pd.Series(0.7)).iloc[i]
                elif mr_signal != 0:
                    signals.loc[df.index[i], 'signal'] = mr_signal
                    signals.loc[df.index[i], 'strength'] = mr_signals.get('strength', pd.Series(0.5)).iloc[i] * 0.7
                    
            else:  # normal regime
                # Use both strategies with equal weight
                if mr_signal != 0 and tf_signal != 0 and mr_signal == tf_signal:
                    # Both strategies agree
                    signals.loc[df.index[i], 'signal'] = mr_signal
                    signals.loc[df.index[i], 'strength'] = max(
                        mr_signals.get('strength', pd.Series(0.5)).iloc[i],
                        tf_signals.get('strength', pd.Series(0.5)).iloc[i]
                    )
                elif mr_signal != 0:
                    signals.loc[df.index[i], 'signal'] = mr_signal
                    signals.loc[df.index[i], 'strength'] = mr_signals.get('strength', pd.Series(0.6)).iloc[i]
                elif tf_signal != 0:
                    signals.loc[df.index[i], 'signal'] = tf_signal
                    signals.loc[df.index[i], 'strength'] = tf_signals.get('strength', pd.Series(0.6)).iloc[i]
        
        return signals
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect current market regime"""
        
        regime = pd.Series('normal', index=df.index)
        
        if len(df) < 60:
            return regime
        
        returns = df['close'].pct_change()
        
        for i in range(60, len(df)):
            # Calculate regime indicators
            recent_returns = returns.iloc[i-20:i]
            longer_returns = returns.iloc[i-60:i]
            
            short_vol = recent_returns.std()
            long_vol = longer_returns.std()
            vol_ratio = short_vol / (long_vol + 1e-8)
            
            # Trend strength
            prices = df['close'].iloc[i-20:i].values
            if len(prices) > 1:
                x = np.arange(len(prices))
                slope = np.polyfit(x, prices, 1)[0]
                trend_strength = abs(slope) / (prices.mean() + 1e-8)
                
                # Determine regime
                if vol_ratio > 1.5 and short_vol > 0.025:
                    regime.iloc[i] = 'high_volatility'
                elif trend_strength > 0.002:
                    # Check R-squared for trend quality
                    y_pred = np.polyval(np.polyfit(x, prices, 1), x)
                    ss_res = np.sum((prices - y_pred) ** 2)
                    ss_tot = np.sum((prices - prices.mean()) ** 2)
                    r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                    
                    if r_squared > 0.7:
                        regime.iloc[i] = 'trending'
        
        return regime