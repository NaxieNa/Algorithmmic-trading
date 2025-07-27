import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import talib
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


@dataclass
class TradeSignal:
    timestamp: pd.Timestamp
    direction: int
    strength: float
    strategy_type: str
    indicators: Dict[str, float]


@dataclass
class RiskMetrics:
    position_size: float
    stop_loss: float
    take_profit: float
    risk_score: float
    volatility: float
    max_risk: float


class PracticalRiskManager:
    def __init__(self,
                 base_risk_pct=0.015,
                 max_risk_pct=0.03,
                 max_position_pct=0.30,
                 vol_lookback=20,
                 vol_target=0.20,
                 vol_scalar_min=0.7,
                 vol_scalar_max=2.0,
                 kelly_lookback=30,
                 kelly_fraction=0.40,
                 kelly_min_trades=10,
                 regime_vol_threshold=0.025,
                 regime_trend_threshold=0.002,
                 max_drawdown=0.20,
                 drawdown_reduce_factor=0.8,
                 correlation_window=100,
                 max_correlation=0.7):

        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.max_position_pct = max_position_pct
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target
        self.vol_scalar_min = vol_scalar_min
        self.vol_scalar_max = vol_scalar_max
        self.kelly_lookback = kelly_lookback
        self.kelly_fraction = kelly_fraction
        self.kelly_min_trades = kelly_min_trades
        self.regime_vol_threshold = regime_vol_threshold
        self.regime_trend_threshold = regime_trend_threshold
        self.max_drawdown = max_drawdown
        self.drawdown_reduce_factor = drawdown_reduce_factor
        self.correlation_window = correlation_window
        self.max_correlation = max_correlation

        self.trade_history = deque(maxlen=kelly_lookback)
        self.current_positions = {}
        self.equity_curve = [100000]
        self.peak_equity = 100000
        self.current_drawdown = 0
        self.market_regime = 'normal'

    def calculate_position_size(self,
                                signal: TradeSignal,
                                current_price: float,
                                account_equity: float,
                                recent_data: pd.DataFrame) -> RiskMetrics:

        base_risk_amount = account_equity * self.base_risk_pct

        volatility = self._calculate_volatility(recent_data)
        atr = self._calculate_atr(recent_data)
        atr_pct = atr / current_price

        vol_scalar = self._calculate_volatility_scalar(volatility)
        kelly_scalar = self._calculate_kelly_scalar()
        regime_scalar = self._calculate_regime_scalar(recent_data, signal.strategy_type)
        drawdown_scalar = self._calculate_drawdown_scalar()
        signal_scalar = 0.5 + 0.5 * signal.strength

        total_scalar = vol_scalar * kelly_scalar * regime_scalar * drawdown_scalar * signal_scalar
        total_scalar = np.clip(total_scalar, 0.1, 2.0)

        if atr > 0:
            shares_by_risk = (base_risk_amount * total_scalar) / atr
            position_value = shares_by_risk * current_price
            position_size = position_value / account_equity
        else:
            position_size = self.base_risk_pct * total_scalar

        position_size = min(position_size, self.max_position_pct)
        position_size = max(position_size, 0.01)

        stop_loss, take_profit = self._calculate_stops(
            current_price, atr, signal.direction, signal.strategy_type
        )

        risk_metrics = RiskMetrics(
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_score=self._calculate_risk_score(volatility, self.current_drawdown),
            volatility=volatility,
            max_risk=position_size * atr_pct
        )

        return risk_metrics

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        if len(data) < self.vol_lookback:
            return 0.02

        returns = data['close'].pct_change().dropna()
        ewm_returns = returns.ewm(span=self.vol_lookback, adjust=False).std()
        current_vol = ewm_returns.iloc[-1] * np.sqrt(252 * 96)

        return current_vol

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        if len(data) < period:
            return data['close'].iloc[-1] * 0.02

        high = data['high'].iloc[-period:]
        low = data['low'].iloc[-period:]
        close = data['close'].iloc[-period:]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.mean()

        return atr

    def _calculate_volatility_scalar(self, current_vol: float) -> float:
        if current_vol <= 0:
            return 1.0

        scalar = self.vol_target / current_vol
        scalar = np.clip(scalar, self.vol_scalar_min, self.vol_scalar_max)

        return scalar

    def _calculate_kelly_scalar(self) -> float:
        if len(self.trade_history) < self.kelly_min_trades:
            return 0.5

        wins = [t['return'] for t in self.trade_history if t['return'] > 0]
        losses = [abs(t['return']) for t in self.trade_history if t['return'] <= 0]

        if not wins or not losses:
            return 0.5

        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        if avg_loss > 0:
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly = kelly * self.kelly_fraction
            kelly = np.clip(kelly, 0.1, 1.0)
        else:
            kelly = 0.5

        recent_trades = list(self.trade_history)[-10:]
        if len(recent_trades) >= 5:
            recent_wins = sum(1 for t in recent_trades if t['return'] > 0)
            recent_performance = recent_wins / len(recent_trades)
            kelly = kelly * (0.5 + recent_performance)

        return kelly

    def _calculate_regime_scalar(self, data: pd.DataFrame, strategy_type: str) -> float:
        if len(data) < 50:
            return 1.0

        returns = data['close'].pct_change().dropna()
        recent_returns = returns.iloc[-20:]

        short_vol = recent_returns.std()
        long_vol = returns.iloc[-60:].std()
        vol_ratio = short_vol / (long_vol + 1e-8)

        prices = data['close'].iloc[-20:].values
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        trend_strength = abs(slope) / (prices.mean() + 1e-8)
        r_squared = r_value ** 2

        if vol_ratio > 1.5 and short_vol > self.regime_vol_threshold:
            self.market_regime = 'high_volatility'
        elif r_squared > 0.7 and trend_strength > self.regime_trend_threshold:
            self.market_regime = 'trending'
        else:
            self.market_regime = 'normal'

        if strategy_type == 'mean_reversion':
            if self.market_regime == 'high_volatility':
                return 1.3
            elif self.market_regime == 'trending':
                return 0.6
            else:
                return 1.0
        elif strategy_type == 'trend_following':
            if self.market_regime == 'trending':
                return 1.5
            elif self.market_regime == 'high_volatility':
                return 0.7
            else:
                return 1.0
        else:
            if self.market_regime == 'trending':
                return 1.2
            elif self.market_regime == 'high_volatility':
                return 1.1
            else:
                return 1.0

    def _calculate_drawdown_scalar(self) -> float:
        if len(self.equity_curve) < 2:
            return 1.0

        current_equity = self.equity_curve[-1]
        self.peak_equity = max(self.peak_equity, current_equity)
        self.current_drawdown = (current_equity - self.peak_equity) / self.peak_equity

        if self.current_drawdown < -0.03:
            drawdown_ratio = abs(self.current_drawdown) / self.max_drawdown
            adjustment = np.sqrt(drawdown_ratio) * (1 - self.drawdown_reduce_factor)
            scalar = 1 - adjustment * min(drawdown_ratio, 1)
            scalar = max(scalar, 0.5)
        else:
            scalar = 1.0

        return scalar

    def _calculate_stops(self, current_price: float, atr: float,
                         direction: int, strategy_type: str) -> Tuple[float, float]:

        if strategy_type == 'mean_reversion':
            stop_atr_mult = 2.0
            profit_atr_mult = 3.0
        else:
            stop_atr_mult = 3.0
            profit_atr_mult = 5.0

        if self.market_regime == 'high_volatility':
            stop_atr_mult *= 1.3
            profit_atr_mult *= 1.5
        elif self.market_regime == 'trending':
            stop_atr_mult *= 1.1
            profit_atr_mult *= 1.2

        if direction == 1:
            stop_loss = current_price - atr * stop_atr_mult
            take_profit = current_price + atr * profit_atr_mult
        else:
            stop_loss = current_price + atr * stop_atr_mult
            take_profit = current_price - atr * profit_atr_mult

        return stop_loss, take_profit

    def _calculate_risk_score(self, volatility: float, drawdown: float) -> float:
        vol_risk = min(volatility / 0.5, 1.0) * 0.5
        dd_risk = min(abs(drawdown) / self.max_drawdown, 1.0) * 0.5
        return vol_risk + dd_risk

    def update_trade_result(self, trade_result: Dict):
        self.trade_history.append(trade_result)

    def update_equity(self, new_equity: float):
        self.equity_curve.append(new_equity)

    def check_portfolio_risk(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        if not positions:
            return {'total_risk': 0, 'correlation_risk': 0, 'concentration_risk': 0}

        total_exposure = sum(p['size'] for p in positions.values())
        position_sizes = [p['size'] for p in positions.values()]
        concentration_risk = max(position_sizes) / total_exposure if total_exposure > 0 else 0
        correlation_risk = 0.5 if len(positions) > 3 else 0

        return {
            'total_risk': total_exposure,
            'correlation_risk': correlation_risk,
            'concentration_risk': concentration_risk,
            'num_positions': len(positions)
        }


class SmartExecutionEngine:
    def __init__(self,
                 max_spread_pct=0.001,
                 urgency_threshold=0.8,
                 slice_size=0.1,
                 vwap_window=5):

        self.max_spread_pct = max_spread_pct
        self.urgency_threshold = urgency_threshold
        self.slice_size = slice_size
        self.vwap_window = vwap_window

    def calculate_execution_price(self,
                                  signal: TradeSignal,
                                  current_bid: float,
                                  current_ask: float,
                                  recent_data: pd.DataFrame) -> Tuple[float, str]:

        spread = current_ask - current_bid
        spread_pct = spread / ((current_ask + current_bid) / 2)

        if spread_pct > self.max_spread_pct:
            return None, "Spread too wide"

        if signal.strength > self.urgency_threshold:
            execution_price = current_ask if signal.direction == 1 else current_bid
            order_type = "market"
        else:
            mid_price = (current_bid + current_ask) / 2
            vwap = self._calculate_vwap(recent_data)

            if signal.direction == 1:
                execution_price = current_bid + spread * 0.3
                execution_price = min(execution_price, vwap)
            else:
                execution_price = current_ask - spread * 0.3
                execution_price = max(execution_price, vwap)

            order_type = "limit"

        return execution_price, order_type

    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        if len(data) < self.vwap_window:
            return data['close'].iloc[-1]

        recent = data.iloc[-self.vwap_window:]
        typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
        vwap = (typical_price * recent['volume']).sum() / recent['volume'].sum()

        return vwap


class PartITradingStrategy:
    """
    Part I 需求的策略实现
    包含：
    1. 趋势跟踪策略（EMA、ADX、MACD）
    2. 外汇趋势指标（重采样设计）
    3. 均值回归策略（只用Z-score，不用OU和ML）
    """

    def __init__(self,
                 # 趋势跟踪参数
                 ema_fast=20,
                 ema_slow=50,
                 adx_period=14,
                 adx_threshold=25,
                 macd_fast=12,
                 macd_slow=26,
                 macd_signal=9,
                 # 均值回归参数
                 z_lookback=20,
                 z_entry_threshold=2.0,
                 z_exit_threshold=0.5,
                 # 外汇趋势参数
                 fx_ratio_bull=1.01,
                 fx_ratio_bear=0.99,
                 # 风险管理
                 risk_manager=None,
                 execution_engine=None):

        # 趋势跟踪参数
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        # 均值回归参数
        self.z_lookback = z_lookback
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold

        # 外汇趋势参数
        self.fx_ratio_bull = fx_ratio_bull
        self.fx_ratio_bear = fx_ratio_bear

        self.risk_manager = risk_manager or PracticalRiskManager()
        self.execution_engine = execution_engine or SmartExecutionEngine()

        self.positions = {}
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        # 计算所有指标
        features = self._calculate_all_indicators(df)

        # 初始化信号DataFrame
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['strategy_type'] = ''
        signals['position_size'] = 0.0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        signals['risk_score'] = 0.0

        account_equity = 100000  # 默认账户资金

        # 从有足够数据的位置开始
        start_idx = max(self.ema_slow, self.z_lookback, 100)

        for i in range(start_idx, len(df)):
            if self._should_generate_signal(i, signals):
                # 检查趋势跟踪信号
                trend_signal = self._check_trend_following_signal(df, features, i)

                # 检查均值回归信号
                mr_signal = self._check_mean_reversion_signal(df, features, i)

                # 检查外汇趋势信号
                fx_signal = self._check_fx_trend_signal(df, features, i)

                # 选择最强的信号
                best_signal = None
                if trend_signal and trend_signal.strength > 0.6:
                    best_signal = trend_signal
                elif mr_signal and mr_signal.strength > 0.6:
                    best_signal = mr_signal
                elif fx_signal and fx_signal.strength > 0.6:
                    best_signal = fx_signal

                if best_signal and best_signal.direction != 0:
                    recent_data = df.iloc[max(0, i - 100):i + 1]

                    risk_metrics = self.risk_manager.calculate_position_size(
                        signal=best_signal,
                        current_price=df['close'].iloc[i],
                        account_equity=account_equity,
                        recent_data=recent_data
                    )

                    if risk_metrics.risk_score < 0.8:
                        signals.loc[df.index[i], 'signal'] = best_signal.direction
                        signals.loc[df.index[i], 'strategy_type'] = best_signal.strategy_type
                        signals.loc[df.index[i], 'position_size'] = risk_metrics.position_size
                        signals.loc[df.index[i], 'stop_loss'] = risk_metrics.stop_loss
                        signals.loc[df.index[i], 'take_profit'] = risk_metrics.take_profit
                        signals.loc[df.index[i], 'risk_score'] = risk_metrics.risk_score

            # 检查退出条件
            self._check_exit_conditions(df, signals, i)

        return signals

    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 基础指标
        features['returns'] = close.pct_change()

        # Part I 需求1：趋势跟踪指标
        # EMA
        features['ema_fast'] = talib.EMA(close, timeperiod=self.ema_fast)
        features['ema_slow'] = talib.EMA(close, timeperiod=self.ema_slow)
        features['ema_signal'] = talib.EMA(features['ema_fast'] - features['ema_slow'], timeperiod=9)

        # ADX
        features['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)
        features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
        features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)

        # MACD
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Part I 需求3：均值回归指标（只用Z-score）
        ma = close.rolling(self.z_lookback).mean()
        std = close.rolling(self.z_lookback).std()
        features['z_score'] = (close - ma) / (std + 1e-8)

        # 额外的均值回归辅助指标
        features['rsi'] = talib.RSI(close, 14)
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(close, 20, 2, 2)

        # Part I 需求2：外汇趋势指标
        # 步骤1：短期价格（模拟30秒采样，这里用当前价格）
        features['fx_short'] = close

        # 步骤2：长期平均（5分钟，在15分钟数据上约等于1个bar的移动平均）
        features['fx_long'] = close.rolling(1).mean()

        # 步骤3：计算比率
        features['fx_ratio'] = features['fx_short'] / (features['fx_long'] + 1e-8)

        # 额外指标
        features['atr'] = talib.ATR(high, low, close, 14)
        features['volume_ratio'] = volume / volume.rolling(20).mean()

        return features

    def _check_trend_following_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        """Part I 趋势跟踪策略信号"""
        # EMA交叉
        ema_fast = features['ema_fast'].iloc[i]
        ema_slow = features['ema_slow'].iloc[i]
        ema_cross = 1 if ema_fast > ema_slow else -1 if ema_fast < ema_slow else 0

        # 检查是否刚发生交叉
        if i > 0:
            prev_ema_fast = features['ema_fast'].iloc[i - 1]
            prev_ema_slow = features['ema_slow'].iloc[i - 1]
            prev_cross = 1 if prev_ema_fast > prev_ema_slow else -1

            if ema_cross == prev_cross:  # 没有新的交叉
                ema_cross = 0

        # ADX确认趋势强度
        adx = features['adx'].iloc[i]
        adx_strong = adx > self.adx_threshold

        # MACD确认
        macd = features['macd'].iloc[i]
        macd_signal = features['macd_signal'].iloc[i]
        macd_cross = 1 if macd > macd_signal else -1 if macd < macd_signal else 0

        # 综合判断
        direction = 0
        strength = 0

        if ema_cross != 0 and adx_strong and macd_cross == ema_cross:
            direction = ema_cross

            # 计算信号强度
            ema_diff = abs(ema_fast - ema_slow) / ema_slow
            adx_strength = min(adx / 50, 1.0)
            macd_strength = min(abs(macd - macd_signal) / (abs(macd_signal) + 1e-8), 1.0)

            strength = (ema_diff * 10 + adx_strength + macd_strength) / 3
            strength = min(strength, 1.0)

        return TradeSignal(
            timestamp=df.index[i],
            direction=direction,
            strength=strength,
            strategy_type='trend_following',
            indicators={
                'ema_fast': ema_fast,
                'ema_slow': ema_slow,
                'adx': adx,
                'macd': macd
            }
        )

    def _check_mean_reversion_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        """Part I 均值回归策略信号（只用Z-score）"""
        z_score = features['z_score'].iloc[i]
        rsi = features['rsi'].iloc[i]
        close = df['close'].iloc[i]
        bb_upper = features['bb_upper'].iloc[i]
        bb_lower = features['bb_lower'].iloc[i]

        direction = 0
        strength = 0

        # 超卖信号
        if z_score < -self.z_entry_threshold and rsi < 30 and close < bb_lower:
            direction = 1  # 买入
            z_strength = min((abs(z_score) - self.z_entry_threshold) / self.z_entry_threshold, 1.0)
            rsi_strength = (30 - rsi) / 30
            bb_strength = (bb_lower - close) / (bb_upper - bb_lower)

            strength = (z_strength * 0.5 + rsi_strength * 0.3 + bb_strength * 0.2)

        # 超买信号
        elif z_score > self.z_entry_threshold and rsi > 70 and close > bb_upper:
            direction = -1  # 卖出
            z_strength = min((z_score - self.z_entry_threshold) / self.z_entry_threshold, 1.0)
            rsi_strength = (rsi - 70) / 30
            bb_strength = (close - bb_upper) / (bb_upper - bb_lower)

            strength = (z_strength * 0.5 + rsi_strength * 0.3 + bb_strength * 0.2)

        return TradeSignal(
            timestamp=df.index[i],
            direction=direction,
            strength=strength,
            strategy_type='mean_reversion',
            indicators={
                'z_score': z_score,
                'rsi': rsi,
                'bb_position': (close - bb_lower) / (bb_upper - bb_lower)
            }
        )

    def _check_fx_trend_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        """Part I 外汇趋势指标信号"""
        fx_ratio = features['fx_ratio'].iloc[i]
        volume_ratio = features['volume_ratio'].iloc[i]

        direction = 0
        strength = 0

        # 上升趋势
        if fx_ratio > self.fx_ratio_bull and volume_ratio > 1.2:
            direction = 1
            ratio_strength = min((fx_ratio - 1.0) * 100, 1.0)
            volume_strength = min((volume_ratio - 1.0), 1.0)
            strength = ratio_strength * 0.7 + volume_strength * 0.3

        # 下降趋势
        elif fx_ratio < self.fx_ratio_bear and volume_ratio > 1.2:
            direction = -1
            ratio_strength = min((1.0 - fx_ratio) * 100, 1.0)
            volume_strength = min((volume_ratio - 1.0), 1.0)
            strength = ratio_strength * 0.7 + volume_strength * 0.3

        return TradeSignal(
            timestamp=df.index[i],
            direction=direction,
            strength=strength,
            strategy_type='fx_trend',
            indicators={
                'fx_ratio': fx_ratio,
                'volume_ratio': volume_ratio
            }
        )

    def _should_generate_signal(self, i: int, signals: pd.DataFrame) -> bool:
        """检查是否应该生成新信号"""
        # 如果已有持仓，不生成新信号
        if i > 0 and signals['signal'].iloc[i - 1] != 0:
            return False

        # 检查最近的信号时间
        last_signal_idx = signals[signals['signal'] != 0].index
        if len(last_signal_idx) > 0:
            last_signal_time = last_signal_idx[-1]
            current_time = signals.index[i]
            time_diff = (current_time - last_signal_time).total_seconds() / 60

            # 至少间隔60分钟
            if time_diff < 60:
                return False

        return True

    def _check_exit_conditions(self, df: pd.DataFrame, signals: pd.DataFrame, i: int):
        """检查退出条件"""
        if i == 0:
            return

        current_position = signals['signal'].iloc[i - 1]
        if current_position == 0:
            return

        current_price = df['close'].iloc[i]

        # 找到入场点
        entry_idx = None
        for j in range(i - 1, -1, -1):
            if j == 0 or signals['signal'].iloc[j - 1] == 0:
                entry_idx = j
                break

        if entry_idx is None:
            return

        entry_price = df['close'].iloc[entry_idx]
        stop_loss = signals['stop_loss'].iloc[entry_idx]
        take_profit = signals['take_profit'].iloc[entry_idx]
        strategy_type = signals['strategy_type'].iloc[entry_idx]

        # 计算盈亏
        if current_position == 1:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        exit_signal = False

        # 止损止盈检查
        if current_position == 1:
            if current_price <= stop_loss or current_price >= take_profit:
                exit_signal = True
        else:
            if current_price >= stop_loss or current_price <= take_profit:
                exit_signal = True

        # 均值回归策略的Z-score退出
        if strategy_type == 'mean_reversion' and 'z_score' in df.columns:
            z_score = df['z_score'].iloc[i]
            if abs(z_score) < self.z_exit_threshold:
                exit_signal = True

        # 时间退出
        holding_periods = i - entry_idx
        if holding_periods > 200:  # 超过200个15分钟（50小时）
            exit_signal = True

        if exit_signal:
            signals.loc[signals.index[i], 'signal'] = 0


class EnhancedPartIStrategy(PartITradingStrategy):
    """增强的Part I策略 - 增加交易频率同时保持盈利"""

    def __init__(self, symbol: str, data_characteristics: Dict, **kwargs):
        self.symbol = symbol
        self.data_chars = data_characteristics

        if symbol == 'TSLA' or data_characteristics.get('daily_volatility', 0) > 0.04:
            # TSLA - 调整参数增加交易机会
            super().__init__(
                # EMA参数 - 更敏感
                ema_fast=8,  # 从5改为8，减少噪音但保持敏感
                ema_slow=21,  # 从20改为21，费波那契数

                # ADX参数 - 降低门槛
                adx_period=14,
                adx_threshold=20,  # 从25降到20，增加信号

                # MACD参数 - 标准设置
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,

                # 均值回归参数 - 更灵活
                z_lookback=20,  # 从30缩短到20
                z_entry_threshold=1.8,  # 从2.5降到1.8
                z_exit_threshold=0.5,  # 更快退出

                # 外汇趋势参数 - 更敏感
                fx_ratio_bull=1.008,  # 从1.015降到1.008
                fx_ratio_bear=0.992,  # 从0.985升到0.992

                risk_manager=kwargs.get('risk_manager')
            )

            # 额外参数
            self.min_holding_periods = 4  # 最少持有4个bar（1小时）
            self.max_holding_periods = 96  # 最多持有96个bar（24小时）
            self.signal_cooldown = 8  # 信号冷却期8个bar（2小时）
            self.confidence_threshold = 0.4  # 降低置信度门槛

        else:  # GLD
            super().__init__(
                ema_fast=15,  # 更敏感
                ema_slow=40,  # 更敏感
                adx_threshold=18,  # 稍微降低
                z_lookback=20,
                z_entry_threshold=1.5,  # 降低门槛
                z_exit_threshold=0.5,
                fx_ratio_bull=1.003,
                fx_ratio_bear=0.997,
                risk_manager=kwargs.get('risk_manager')
            )

            self.min_holding_periods = 8
            self.max_holding_periods = 192
            self.signal_cooldown = 16
            self.confidence_threshold = 0.45

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成更多交易信号"""

        # 计算增强指标
        features = self._calculate_enhanced_indicators(df)

        # 初始化信号
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['strategy_type'] = ''
        signals['position_size'] = 0.0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        signals['risk_score'] = 0.0
        signals['confidence'] = 0.0

        account_equity = 100000

        # 跟踪状态
        in_position = False
        entry_idx = None
        last_signal_idx = None

        # 从有足够数据的位置开始
        start_idx = max(50, self.ema_slow)

        for i in range(start_idx, len(df)):
            # 检查是否可以生成新信号
            can_trade = True

            # 冷却期检查
            if last_signal_idx is not None:
                if i - last_signal_idx < self.signal_cooldown:
                    can_trade = False

            # 持仓检查
            if in_position:
                # 检查退出条件
                exit_signal = self._check_enhanced_exit_conditions(
                    df, features, i, entry_idx, signals
                )

                if exit_signal:
                    signals.loc[df.index[i], 'signal'] = 0
                    in_position = False
                    last_signal_idx = i

                can_trade = False

            if can_trade:
                # 生成入场信号
                best_signal = self._generate_enhanced_entry_signal(df, features, i)

                if best_signal and best_signal.direction != 0 and best_signal.strength > self.confidence_threshold:
                    # 风险检查
                    recent_data = df.iloc[max(0, i - 100):i + 1]

                    risk_metrics = self.risk_manager.calculate_position_size(
                        signal=best_signal,
                        current_price=df['close'].iloc[i],
                        account_equity=account_equity,
                        recent_data=recent_data
                    )

                    if risk_metrics.risk_score < 0.85:  # 稍微放宽风险限制
                        signals.loc[df.index[i], 'signal'] = best_signal.direction
                        signals.loc[df.index[i], 'strategy_type'] = best_signal.strategy_type
                        signals.loc[df.index[i], 'position_size'] = risk_metrics.position_size
                        signals.loc[df.index[i], 'stop_loss'] = risk_metrics.stop_loss
                        signals.loc[df.index[i], 'take_profit'] = risk_metrics.take_profit
                        signals.loc[df.index[i], 'risk_score'] = risk_metrics.risk_score
                        signals.loc[df.index[i], 'confidence'] = best_signal.strength

                        in_position = True
                        entry_idx = i
                        last_signal_idx = i

        return signals

    def _calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算增强的技术指标"""

        # 先计算基础指标
        features = super()._calculate_all_indicators(df)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 添加额外指标

        # 多周期RSI
        features['rsi_7'] = talib.RSI(close, 7)
        features['rsi_21'] = talib.RSI(close, 21)

        # 布林带宽度和位置
        bb_width = features['bb_upper'] - features['bb_lower']
        features['bb_width'] = bb_width / features['bb_middle']
        features['bb_position'] = (close - features['bb_lower']) / (bb_width + 1e-8)

        # 动量指标
        features['roc_5'] = talib.ROC(close, 5)
        features['roc_10'] = talib.ROC(close, 10)
        features['mom_5'] = close - close.shift(5)

        # 价格通道
        features['high_20'] = high.rolling(20).max()
        features['low_20'] = low.rolling(20).min()
        features['price_position'] = (close - features['low_20']) / (features['high_20'] - features['low_20'] + 1e-8)

        # 成交量确认
        features['obv'] = talib.OBV(close, volume)
        features['volume_sma'] = volume.rolling(20).mean()

        # ATR百分比
        features['atr_pct'] = features['atr'] / close

        # 第三条EMA作为过滤器
        features['ema_filter'] = talib.EMA(close, timeperiod=50)

        # MACD柱状图变化
        features['macd_hist_change'] = features['macd_hist'].diff()

        return features

    def _generate_enhanced_entry_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        """生成增强的入场信号"""

        # 收集所有可能的信号
        signals = []

        # 1. 原始趋势跟踪信号
        trend_signal = super()._check_trend_following_signal(df, features, i)
        if trend_signal.direction != 0:
            signals.append(trend_signal)

        # 2. 原始均值回归信号
        mr_signal = super()._check_mean_reversion_signal(df, features, i)
        if mr_signal.direction != 0:
            signals.append(mr_signal)

        # 3. 动量突破信号
        momentum_signal = self._check_momentum_breakout_signal(df, features, i)
        if momentum_signal and momentum_signal.direction != 0:
            signals.append(momentum_signal)

        # 4. 布林带压缩突破
        bb_signal = self._check_bollinger_squeeze_signal(df, features, i)
        if bb_signal and bb_signal.direction != 0:
            signals.append(bb_signal)

        # 5. 双RSI确认信号
        rsi_signal = self._check_dual_rsi_signal(df, features, i)
        if rsi_signal and rsi_signal.direction != 0:
            signals.append(rsi_signal)

        if not signals:
            return TradeSignal(df.index[i], 0, 0, '', {})

        # 如果多个信号一致，增加置信度
        if len(signals) > 1:
            directions = [s.direction for s in signals]
            if all(d == directions[0] for d in directions):
                # 所有信号方向一致
                avg_strength = sum(s.strength for s in signals) / len(signals)
                boost = 1 + 0.1 * (len(signals) - 1)
                combined_strength = min(avg_strength * boost, 1.0)

                return TradeSignal(
                    timestamp=df.index[i],
                    direction=directions[0],
                    strength=combined_strength,
                    strategy_type='combined_enhanced',
                    indicators={'signal_count': len(signals)}
                )

        # 返回最强信号
        return max(signals, key=lambda x: x.strength)

    def _check_momentum_breakout_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        """动量突破信号"""

        close = df['close'].iloc[i]
        high_20 = features['high_20'].iloc[i]
        low_20 = features['low_20'].iloc[i]
        volume_ratio = features['volume_ratio'].iloc[i]
        roc_5 = features['roc_5'].iloc[i]
        roc_10 = features['roc_10'].iloc[i]

        direction = 0
        strength = 0

        # 向上突破
        if (close > high_20 * 0.998 and  # 接近或突破20日高点
                volume_ratio > 1.1 and  # 成交量温和放大
                roc_5 > 1 and  # 短期动量为正
                roc_10 > 0.5):  # 中期动量为正

            direction = 1
            strength = min(
                0.4 +
                (close / high_20 - 1) * 20 +
                (volume_ratio - 1) * 0.5 +
                roc_5 / 10,
                1.0
            )

        # 向下突破
        elif (close < low_20 * 1.002 and
              volume_ratio > 1.1 and
              roc_5 < -1 and
              roc_10 < -0.5):

            direction = -1
            strength = min(
                0.4 +
                (1 - close / low_20) * 20 +
                (volume_ratio - 1) * 0.5 +
                abs(roc_5) / 10,
                1.0
            )

        return TradeSignal(
            timestamp=df.index[i],
            direction=direction,
            strength=strength,
            strategy_type='momentum_breakout',
            indicators={
                'price_position': features['price_position'].iloc[i],
                'volume_ratio': volume_ratio,
                'roc_5': roc_5
            }
        )

    def _check_bollinger_squeeze_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        """布林带压缩突破信号"""

        close = df['close'].iloc[i]
        bb_upper = features['bb_upper'].iloc[i]
        bb_lower = features['bb_lower'].iloc[i]
        bb_width = features['bb_width'].iloc[i]
        bb_position = features['bb_position'].iloc[i]
        rsi_7 = features['rsi_7'].iloc[i]

        direction = 0
        strength = 0

        # 检查布林带是否在压缩后扩张
        if i > 5:
            bb_width_ma = features['bb_width'].iloc[i - 5:i].mean()
            bb_expanding = bb_width > bb_width_ma * 1.2

            # 突破上轨
            if (close > bb_upper * 0.998 and
                    bb_expanding and
                    rsi_7 > 60):

                direction = 1
                strength = min(
                    0.4 +
                    (bb_position - 0.8) * 2 +
                    (rsi_7 - 60) / 40,
                    0.9
                )

            # 突破下轨
            elif (close < bb_lower * 1.002 and
                  bb_expanding and
                  rsi_7 < 40):

                direction = -1
                strength = min(
                    0.4 +
                    (0.2 - bb_position) * 2 +
                    (40 - rsi_7) / 40,
                    0.9
                )

        return TradeSignal(
            timestamp=df.index[i],
            direction=direction,
            strength=strength,
            strategy_type='bollinger_squeeze',
            indicators={
                'bb_width': bb_width,
                'bb_position': bb_position,
                'rsi_7': rsi_7
            }
        )

    def _check_dual_rsi_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        """双RSI确认信号"""

        rsi_7 = features['rsi_7'].iloc[i]
        rsi_14 = features['rsi'].iloc[i]
        rsi_21 = features['rsi_21'].iloc[i]
        volume_ratio = features['volume_ratio'].iloc[i]

        direction = 0
        strength = 0

        # 超卖反弹
        if (rsi_7 < 25 and
                rsi_14 < 35 and
                rsi_21 < 40 and
                volume_ratio > 0.8):

            direction = 1
            strength = min(
                0.4 +
                (25 - rsi_7) / 25 * 0.3 +
                (35 - rsi_14) / 35 * 0.2 +
                (40 - rsi_21) / 40 * 0.1,
                0.85
            )

        # 超买回落
        elif (rsi_7 > 75 and
              rsi_14 > 65 and
              rsi_21 > 60 and
              volume_ratio > 0.8):

            direction = -1
            strength = min(
                0.4 +
                (rsi_7 - 75) / 25 * 0.3 +
                (rsi_14 - 65) / 35 * 0.2 +
                (rsi_21 - 60) / 40 * 0.1,
                0.85
            )

        return TradeSignal(
            timestamp=df.index[i],
            direction=direction,
            strength=strength,
            strategy_type='dual_rsi',
            indicators={
                'rsi_7': rsi_7,
                'rsi_14': rsi_14,
                'rsi_21': rsi_21
            }
        )

    def _check_enhanced_exit_conditions(self, df: pd.DataFrame, features: pd.DataFrame,
                                        i: int, entry_idx: int, signals: pd.DataFrame) -> bool:
        """增强的退出条件检查"""

        holding_periods = i - entry_idx
        entry_price = df['close'].iloc[entry_idx]
        current_price = df['close'].iloc[i]
        position = signals['signal'].iloc[entry_idx]
        strategy_type = signals['strategy_type'].iloc[entry_idx]

        # 计算收益
        if position == 1:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # 1. 基础止损止盈
        if pnl_pct < -0.015:  # 1.5%止损（更紧）
            return True

        if pnl_pct > 0.04:  # 4%止盈
            return True

        # 2. 时间退出
        if holding_periods < self.min_holding_periods:
            return False  # 最少持有时间

        if holding_periods >= self.max_holding_periods:
            return True

        # 3. 策略特定退出
        if 'mr_' in strategy_type or 'mean_reversion' in strategy_type:
            # 均值回归退出
            z_score = features['z_score'].iloc[i]
            if abs(z_score) < self.z_exit_threshold:
                return True

        elif 'trend_' in strategy_type:
            # 趋势退出 - EMA交叉反转
            ema_fast = features['ema_fast'].iloc[i]
            ema_slow = features['ema_slow'].iloc[i]

            if position == 1 and ema_fast < ema_slow:
                return True
            elif position == -1 and ema_fast > ema_slow:
                return True

        elif 'momentum_' in strategy_type:
            # 动量退出
            roc_5 = features['roc_5'].iloc[i]

            if position == 1 and roc_5 < -0.5:
                return True
            elif position == -1 and roc_5 > 0.5:
                return True

        # 4. 动态追踪止损
        if pnl_pct > 0.015:  # 盈利1.5%后启动
            if position == 1:
                # 保护50%的利润
                trailing_stop = entry_price * (1 + pnl_pct * 0.5)
                if current_price < trailing_stop:
                    return True
            else:
                trailing_stop = entry_price * (1 - pnl_pct * 0.5)
                if current_price > trailing_stop:
                    return True

        # 5. 波动率退出
        if self.symbol == 'TSLA':
            atr_pct = features['atr_pct'].iloc[i]
            if atr_pct > 0.05:  # 波动率过高
                if pnl_pct > 0.01:  # 有小幅盈利就退出
                    return True

        return False


# 保留原有的OptimizedTradingStrategy类
class OptimizedTradingStrategy:
    def __init__(self,
                 strategy_type='combined',
                 mr_entry_z=1.5,
                 mr_exit_z=0.3,
                 tf_fast=10,
                 tf_slow=30,
                 risk_manager=None,
                 execution_engine=None):

        self.strategy_type = strategy_type
        self.mr_entry_z = mr_entry_z
        self.mr_exit_z = mr_exit_z
        self.tf_fast = tf_fast
        self.tf_slow = tf_slow

        self.risk_manager = risk_manager or PracticalRiskManager()
        self.execution_engine = execution_engine or SmartExecutionEngine()

        self.positions = {}
        self.pending_orders = {}
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_losses': 0
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._calculate_indicators(df)

        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['position_size'] = 0.0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        signals['risk_score'] = 0.0

        account_equity = 100000

        for i in range(max(50, self.tf_slow), len(df)):
            if self._should_generate_signal(i, signals):
                trade_signal = self._generate_trade_signal(df, features, i)

                if trade_signal.direction != 0:
                    recent_data = df.iloc[max(0, i - 100):i + 1]

                    risk_metrics = self.risk_manager.calculate_position_size(
                        signal=trade_signal,
                        current_price=df['close'].iloc[i],
                        account_equity=account_equity,
                        recent_data=recent_data
                    )

                    if risk_metrics.risk_score < 0.8:
                        signals.loc[df.index[i], 'signal'] = trade_signal.direction
                        signals.loc[df.index[i], 'position_size'] = risk_metrics.position_size
                        signals.loc[df.index[i], 'stop_loss'] = risk_metrics.stop_loss
                        signals.loc[df.index[i], 'take_profit'] = risk_metrics.take_profit
                        signals.loc[df.index[i], 'risk_score'] = risk_metrics.risk_score

            self._check_exit_conditions(df, signals, i)

        return signals

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        features['returns'] = close.pct_change()

        features['sma_fast'] = close.rolling(self.tf_fast).mean()
        features['sma_slow'] = close.rolling(self.tf_slow).mean()
        features['ema_fast'] = close.ewm(span=self.tf_fast).mean()
        features['ema_slow'] = close.ewm(span=self.tf_slow).mean()

        features['rsi'] = talib.RSI(close, 14)
        features['rsi_fast'] = talib.RSI(close, 7)

        features['atr'] = talib.ATR(high, low, close, 14)
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(close, 20, 2, 2)

        for period in [10, 20, 30]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'z_score_{period}'] = (close - ma) / (std + 1e-8)

        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(close)

        features['volume_ratio'] = volume / volume.rolling(20).mean()
        features['obv'] = talib.OBV(close, volume)

        features['roc'] = talib.ROC(close, 10)
        features['cci'] = talib.CCI(high, low, close, 14)
        features['williams_r'] = talib.WILLR(high, low, close, 14)

        features['spread'] = (df['ask'] - df['bid']) / close if 'ask' in df.columns else 0
        features['mid_price'] = (df['ask'] + df['bid']) / 2 if 'ask' in df.columns else close

        return features

    def _generate_trade_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        if self.strategy_type == 'mean_reversion':
            return self._generate_mr_signal(df, features, i)
        elif self.strategy_type == 'trend_following':
            return self._generate_tf_signal(df, features, i)
        else:
            mr_signal = self._generate_mr_signal(df, features, i)
            tf_signal = self._generate_tf_signal(df, features, i)

            if mr_signal.strength > tf_signal.strength:
                return mr_signal
            else:
                return tf_signal

    def _generate_mr_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        z_score = features['z_score_20'].iloc[i]
        rsi = features['rsi'].iloc[i]
        rsi_fast = features['rsi_fast'].iloc[i]
        bb_position = (df['close'].iloc[i] - features['bb_lower'].iloc[i]) / \
                      (features['bb_upper'].iloc[i] - features['bb_lower'].iloc[i] + 1e-8)
        volume_ratio = features['volume_ratio'].iloc[i]

        returns = features['returns'].iloc[max(0, i - 5):i + 1]
        short_term_momentum = returns.mean() if len(returns) > 0 else 0

        if (z_score < -self.mr_entry_z and
                rsi < 25 and
                rsi_fast < 20 and
                bb_position < 0.15 and
                short_term_momentum < -0.001):

            strength = min(1.0,
                           (abs(z_score) - self.mr_entry_z) / self.mr_entry_z * 0.3 +
                           (25 - rsi) / 25 * 0.3 +
                           (20 - rsi_fast) / 20 * 0.2 +
                           (0.15 - bb_position) / 0.15 * 0.2)

            if volume_ratio > 1.5:
                strength = min(1.0, strength * 1.3)
            elif volume_ratio < 0.8:
                strength *= 0.7

            if strength > 0.6:
                return TradeSignal(
                    timestamp=df.index[i],
                    direction=1,
                    strength=strength,
                    strategy_type='mean_reversion',
                    indicators={'z_score': z_score, 'rsi': rsi, 'bb_position': bb_position}
                )

        elif (z_score > self.mr_entry_z and
              rsi > 75 and
              rsi_fast > 80 and
              bb_position > 0.85 and
              short_term_momentum > 0.001):

            strength = min(1.0,
                           (z_score - self.mr_entry_z) / self.mr_entry_z * 0.3 +
                           (rsi - 75) / 25 * 0.3 +
                           (rsi_fast - 80) / 20 * 0.2 +
                           (bb_position - 0.85) / 0.15 * 0.2)

            if volume_ratio > 1.5:
                strength = min(1.0, strength * 1.3)
            elif volume_ratio < 0.8:
                strength *= 0.7

            if strength > 0.6:
                return TradeSignal(
                    timestamp=df.index[i],
                    direction=-1,
                    strength=strength,
                    strategy_type='mean_reversion',
                    indicators={'z_score': z_score, 'rsi': rsi, 'bb_position': bb_position}
                )

        return TradeSignal(
            timestamp=df.index[i],
            direction=0,
            strength=0,
            strategy_type='mean_reversion',
            indicators={}
        )

    def _generate_tf_signal(self, df: pd.DataFrame, features: pd.DataFrame, i: int) -> TradeSignal:
        sma_fast = features['sma_fast'].iloc[i]
        sma_slow = features['sma_slow'].iloc[i]
        ema_fast = features['ema_fast'].iloc[i]
        ema_slow = features['ema_slow'].iloc[i]
        macd = features['macd'].iloc[i]
        macd_signal = features['macd_signal'].iloc[i]
        macd_hist = features['macd_hist'].iloc[i]
        volume_ratio = features['volume_ratio'].iloc[i]
        atr = features['atr'].iloc[i]

        price = df['close'].iloc[i]
        trend_strength = abs(sma_fast - sma_slow) / sma_slow if sma_slow > 0 else 0

        price_momentum = features['returns'].iloc[max(0, i - 10):i].mean() if i >= 10 else 0

        sma_bull = sma_fast > sma_slow * 1.002
        ema_bull = ema_fast > ema_slow * 1.002
        macd_bull = macd > macd_signal and macd > 0

        if (sma_bull and ema_bull and macd_bull and
                trend_strength > 0.005 and
                price_momentum > 0.001 and
                macd_hist > 0):

            strength = min(1.0,
                           min(trend_strength * 50, 1.0) * 0.3 +
                           min(abs(macd - macd_signal) / (abs(macd_signal) + 1e-8), 1.0) * 0.2 +
                           min(price_momentum * 100, 1.0) * 0.3 +
                           min(volume_ratio / 1.5, 1.0) * 0.2)

            atr_ratio = atr / price
            if atr_ratio < 0.001:
                strength *= 0.5
            elif atr_ratio > 0.03:
                strength *= 0.8

            if strength > 0.65:
                return TradeSignal(
                    timestamp=df.index[i],
                    direction=1,
                    strength=strength,
                    strategy_type='trend_following',
                    indicators={'trend_strength': trend_strength, 'macd': macd, 'momentum': price_momentum}
                )

        elif (not sma_bull and not ema_bull and not macd_bull and
              trend_strength > 0.005 and
              price_momentum < -0.001 and
              macd_hist < 0):

            strength = min(1.0,
                           min(trend_strength * 50, 1.0) * 0.3 +
                           min(abs(macd_signal - macd) / (abs(macd_signal) + 1e-8), 1.0) * 0.2 +
                           min(abs(price_momentum) * 100, 1.0) * 0.3 +
                           min(volume_ratio / 1.5, 1.0) * 0.2)

            atr_ratio = atr / price
            if atr_ratio < 0.001:
                strength *= 0.5
            elif atr_ratio > 0.03:
                strength *= 0.8

            if strength > 0.65:
                return TradeSignal(
                    timestamp=df.index[i],
                    direction=-1,
                    strength=strength,
                    strategy_type='trend_following',
                    indicators={'trend_strength': -trend_strength, 'macd': macd, 'momentum': price_momentum}
                )

        return TradeSignal(
            timestamp=df.index[i],
            direction=0,
            strength=0,
            strategy_type='trend_following',
            indicators={}
        )

    def _should_generate_signal(self, i: int, signals: pd.DataFrame) -> bool:
        if i > 0 and signals['signal'].iloc[i - 1] != 0:
            return False

        last_signal_idx = signals[signals['signal'] != 0].index
        if len(last_signal_idx) > 0:
            last_signal_time = last_signal_idx[-1]
            current_time = signals.index[i]
            time_diff = (current_time - last_signal_time).total_seconds() / 60

            if time_diff < 60:
                return False

        return True

    def _check_exit_conditions(self, df: pd.DataFrame, signals: pd.DataFrame, i: int):
        if i == 0:
            return

        current_position = signals['signal'].iloc[i - 1]
        if current_position == 0:
            return

        current_price = df['close'].iloc[i]
        entry_idx = signals[signals['signal'] != 0].index.get_loc(signals.index[i - 1])
        entry_price = df['close'].loc[signals[signals['signal'] != 0].index[entry_idx]]

        stop_loss = signals['stop_loss'].iloc[i - 1]
        take_profit = signals['take_profit'].iloc[i - 1]

        if current_position == 1:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        exit_signal = False

        if current_position == 1 and current_price <= stop_loss:
            exit_signal = True
        elif current_position == -1 and current_price >= stop_loss:
            exit_signal = True

        elif current_position == 1 and current_price >= take_profit:
            exit_signal = True
        elif current_position == -1 and current_price <= take_profit:
            exit_signal = True

        elif pnl_pct > 0.02:
            if current_position == 1:
                trailing_stop = entry_price * (1 + pnl_pct * 0.5)
                if current_price <= trailing_stop:
                    exit_signal = True
            else:
                trailing_stop = entry_price * (1 - pnl_pct * 0.5)
                if current_price >= trailing_stop:
                    exit_signal = True

        holding_periods = i - entry_idx
        if holding_periods > 200:
            exit_signal = True

        if exit_signal:
            signals.loc[signals.index[i], 'signal'] = 0


class PerformanceAnalyzer:
    @staticmethod
    def analyze_strategy_performance(trades: List[Dict], equity_curve: np.ndarray) -> Dict:
        if not trades or len(equity_curve) < 2:
            return {}

        returns = [t['return'] for t in trades]
        win_trades = [t for t in trades if t['return'] > 0]
        loss_trades = [t for t in trades if t['return'] <= 0]

        metrics = {
            'total_return': (equity_curve[-1] / equity_curve[0] - 1) * 100,
            'cagr': PerformanceAnalyzer._calculate_cagr(equity_curve),
            'win_rate': len(win_trades) / len(trades),
            'avg_win': np.mean([t['return'] for t in win_trades]) * 100 if win_trades else 0,
            'avg_loss': np.mean([t['return'] for t in loss_trades]) * 100 if loss_trades else 0,
            'sharpe_ratio': PerformanceAnalyzer._calculate_sharpe(equity_curve),
            'sortino_ratio': PerformanceAnalyzer._calculate_sortino(equity_curve),
            'calmar_ratio': PerformanceAnalyzer._calculate_calmar(equity_curve),
            'max_drawdown': PerformanceAnalyzer._calculate_max_drawdown(equity_curve),
            'profit_factor': PerformanceAnalyzer._calculate_profit_factor(trades),
            'recovery_factor': PerformanceAnalyzer._calculate_recovery_factor(equity_curve),
            'ulcer_index': PerformanceAnalyzer._calculate_ulcer_index(equity_curve),
            'total_trades': len(trades),
            'avg_trade_duration': np.mean([t.get('duration', 0) for t in trades]),
            'max_consecutive_wins': PerformanceAnalyzer._max_consecutive(returns, True),
            'max_consecutive_losses': PerformanceAnalyzer._max_consecutive(returns, False),
            'risk_reward_ratio': abs(np.mean([t['return'] for t in win_trades]) /
                                     np.mean([t['return'] for t in loss_trades])) if loss_trades and win_trades else 0,
            'expectancy': np.mean(returns) * 100,
            'kelly_percentage': PerformanceAnalyzer._calculate_kelly(trades)
        }

        return metrics

    @staticmethod
    def _calculate_cagr(equity_curve: np.ndarray) -> float:
        years = len(equity_curve) / (252 * 96)
        return ((equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1) * 100

    @staticmethod
    def _calculate_sharpe(equity_curve: np.ndarray, risk_free_rate: float = 0.02) -> float:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        excess_returns = returns - risk_free_rate / (252 * 96)

        if returns.std() > 0:
            return np.mean(excess_returns) / returns.std() * np.sqrt(252 * 96)
        return 0

    @staticmethod
    def _calculate_sortino(equity_curve: np.ndarray, target_return: float = 0) -> float:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        downside_returns = returns[returns < target_return]

        if len(downside_returns) > 0 and downside_returns.std() > 0:
            return np.mean(returns - target_return) / downside_returns.std() * np.sqrt(252 * 96)
        return 0

    @staticmethod
    def _calculate_calmar(equity_curve: np.ndarray) -> float:
        cagr = PerformanceAnalyzer._calculate_cagr(equity_curve)
        max_dd = abs(PerformanceAnalyzer._calculate_max_drawdown(equity_curve))

        if max_dd > 0:
            return cagr / max_dd
        return 0

    @staticmethod
    def _calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return drawdown.min() * 100

    @staticmethod
    def _calculate_profit_factor(trades: List[Dict]) -> float:
        gross_profit = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))

        if gross_loss > 0:
            return gross_profit / gross_loss
        return float('inf') if gross_profit > 0 else 0

    @staticmethod
    def _calculate_recovery_factor(equity_curve: np.ndarray) -> float:
        total_return = equity_curve[-1] - equity_curve[0]
        max_dd = abs(PerformanceAnalyzer._calculate_max_drawdown(equity_curve) / 100 * equity_curve[0])

        if max_dd > 0:
            return total_return / max_dd
        return 0

    @staticmethod
    def _calculate_ulcer_index(equity_curve: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak * 100
        return np.sqrt(np.mean(drawdown ** 2))

    @staticmethod
    def _max_consecutive(returns: List[float], wins: bool) -> int:
        max_count = 0
        current_count = 0

        for r in returns:
            if (wins and r > 0) or (not wins and r <= 0):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    @staticmethod
    def _calculate_kelly(trades: List[Dict]) -> float:
        if not trades:
            return 0

        wins = [t['return'] for t in trades if t['return'] > 0]
        losses = [abs(t['return']) for t in trades if t['return'] < 0]

        if not wins or not losses:
            return 0

        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        if avg_loss > 0:
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            return max(0, min(kelly * 100, 25))

        return 0


def plot_strategy_results(df, results, signal_info):
    """Strategy results visualization"""

    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(df.index, df['close'], 'b-', alpha=0.7, linewidth=1)
    ax1.set_title('Price (15-min)', fontsize=14)
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 2, 2)
    for name, result in results.items():
        equity = result['equity_curve']
        ax2.plot(df.index[:len(equity)], equity, label=name, linewidth=2)
    ax2.set_title('Equity Curves Comparison', fontsize=14)
    ax2.set_ylabel('Equity ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 2, 3)
    for name, result in results.items():
        equity = result['equity_curve']
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        ax3.fill_between(df.index[:len(drawdown)], drawdown, 0,
                         alpha=0.3, label=name)
    ax3.set_title('Drawdown Analysis', fontsize=14)
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 2, 4)
    bar_width = 0.25
    x = np.arange(len(results))

    metrics = ['total_trades', 'win_rate', 'sharpe_ratio']
    colors = ['blue', 'green', 'orange']

    for i, metric in enumerate(metrics):
        if metric == 'win_rate':
            values = [r[metric] * 100 for r in results.values()]
            label = 'Win Rate (%)'
        elif metric == 'sharpe_ratio':
            values = [r[metric] * 10 for r in results.values()]
            label = 'Sharpe (x10)'
        else:
            values = [r[metric] for r in results.values()]
            label = 'Trades'

        ax4.bar(x + i * bar_width, values, bar_width,
                label=label, color=colors[i], alpha=0.7)

    ax4.set_xlabel('Strategy')
    ax4.set_xticks(x + bar_width)
    ax4.set_xticklabels(results.keys(), rotation=45)
    ax4.legend()
    ax4.set_title('Strategy Metrics Comparison', fontsize=14)

    ax5 = plt.subplot(3, 2, 5)
    for name, result in results.items():
        if result['trades']:
            returns = [t['return'] * 100 for t in result['trades']]
            ax5.hist(returns, bins=30, alpha=0.5, label=name)
    ax5.set_xlabel('Return per Trade (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Return Distribution', fontsize=14)
    ax5.legend()

    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('tight')
    ax6.axis('off')

    table_data = []
    table_data.append(['Metric'] + list(results.keys()))

    metrics_to_show = [
        ('Total Return', 'total_return', '{:.2f}%'),
        ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
        ('Max Drawdown', 'max_drawdown', '{:.2f}%'),
        ('Num Trades', 'total_trades', '{:d}'),
        ('Win Rate', 'win_rate', '{:.1%}'),
        ('Avg Win', 'avg_win', '{:.2f}%'),
        ('Avg Loss', 'avg_loss', '{:.2f}%')
    ]

    for metric_name, metric_key, fmt in metrics_to_show:
        row = [metric_name]
        for result in results.values():
            value = result[metric_key]
            if metric_key == 'win_rate':
                row.append(fmt.format(value))
            elif '%' in fmt:
                row.append(fmt.format(value))
            else:
                row.append(fmt.format(value))
        table_data.append(row)

    table = ax6.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax6.set_title('Performance Summary', fontsize=14)

    plt.tight_layout()
    plt.show()


def run_optimized_trading_system(filepath='TSLA_full_15min.csv',
                                 initial_capital=100000,
                                 commission=0.001):
    """Run optimized trading system"""

    print("=" * 80)
    print("CQF Practical Risk Management Trading System")
    print("=" * 80)

    df = pd.read_csv(filepath, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)

    spread = 0.0005
    df['bid'] = df['close'] * (1 - spread / 2)
    df['ask'] = df['close'] * (1 + spread / 2)

    df = df.tail(96 * 63)

    print(f"\nData range: {df.index[0]} to {df.index[-1]}")
    print(f"Data points: {len(df)}")

    strategies = {
        'Mean Reversion': OptimizedTradingStrategy(
            strategy_type='mean_reversion',
            mr_entry_z=1.8,
            mr_exit_z=0.5,
            risk_manager=PracticalRiskManager(
                base_risk_pct=0.015,
                max_drawdown=0.12,
                kelly_fraction=0.35,
                vol_target=0.20
            )
        ),
        'Trend Following': OptimizedTradingStrategy(
            strategy_type='trend_following',
            tf_fast=8,
            tf_slow=25,
            risk_manager=PracticalRiskManager(
                base_risk_pct=0.020,
                max_drawdown=0.18,
                kelly_fraction=0.40,
                vol_target=0.22
            )
        ),
        'Combined': OptimizedTradingStrategy(
            strategy_type='combined',
            mr_entry_z=2.0,
            tf_fast=10,
            tf_slow=30,
            risk_manager=PracticalRiskManager(
                base_risk_pct=0.018,
                max_drawdown=0.15,
                kelly_fraction=0.38,
                vol_target=0.21,
                regime_vol_threshold=0.025
            )
        ),
        'Part I Strategy': PartITradingStrategy(
            ema_fast=20,
            ema_slow=50,
            adx_threshold=25,
            z_lookback=20,
            z_entry_threshold=2.0,
            fx_ratio_bull=1.01,
            fx_ratio_bear=0.99,
            risk_manager=PracticalRiskManager(
                base_risk_pct=0.015,
                max_drawdown=0.15,
                kelly_fraction=0.35,
                vol_target=0.20
            )
        )
    }

    results = {}
    all_risk_metrics = {}

    print("\n" + "=" * 80)
    print("Running Backtests")
    print("=" * 80)

    for name, strategy in strategies.items():
        print(f"\nTesting: {name}")
        print("-" * 60)

        signals = strategy.generate_signals(df)
        trades, equity_curve = run_backtest_with_risk_management(
            df, signals, initial_capital, commission
        )

        performance = PerformanceAnalyzer.analyze_strategy_performance(trades, equity_curve)

        results[name] = {
            'trades': trades,
            'equity_curve': equity_curve,
            **performance
        }

        all_risk_metrics[name] = {
            'position_sizes': signals['position_size'],
            'risk_scores': signals['risk_score'],
            'market_regime': strategy.risk_manager.market_regime
        }

        print(f"\nPerformance:")
        print(f"  Total return: {performance.get('total_return', 0):.2f}%")
        print(f"  Sharpe ratio: {performance.get('sharpe_ratio', 0):.3f}")
        print(f"  Max drawdown: {performance.get('max_drawdown', 0):.2f}%")
        print(f"  Win rate: {performance.get('win_rate', 0):.1%}")
        print(f"  Profit factor: {performance.get('profit_factor', 0):.2f}")
        print(f"  Calmar ratio: {performance.get('calmar_ratio', 0):.3f}")

    print("\nGenerating charts...")
    plot_strategy_results(df, results, all_risk_metrics)

    return results, all_risk_metrics


def run_backtest_with_risk_management(df: pd.DataFrame,
                                      signals: pd.DataFrame,
                                      initial_capital: float,
                                      commission: float) -> Tuple[List[Dict], np.ndarray]:
    """Run backtest with risk management"""

    cash = initial_capital
    position = 0
    shares = 0
    trades = []
    equity_curve = [initial_capital]

    entry_price = 0
    entry_time = None
    entry_size = 0

    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        signal = signals['signal'].iloc[i]
        position_size = signals['position_size'].iloc[i]

        if position != 0:
            current_value = shares * current_price * np.sign(position)
            current_equity = cash + current_value
        else:
            current_equity = cash

        equity_curve.append(current_equity)

        if i > 0 and signal != signals['signal'].iloc[i - 1]:

            if position != 0:
                exit_value = shares * current_price * (1 - commission)
                pnl = exit_value - entry_size

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': position,
                    'shares': shares,
                    'pnl': pnl,
                    'return': (current_price / entry_price - 1) * np.sign(position),
                    'duration': (df.index[i] - entry_time).total_seconds() / 3600
                })

                cash += exit_value * np.sign(position)
                position = 0
                shares = 0

            if signal != 0 and position_size > 0:
                position = signal
                entry_price = current_price
                entry_time = df.index[i]

                position_value = current_equity * position_size
                shares = position_value / (current_price * (1 + commission))
                entry_size = shares * current_price

                cash -= position_value * np.sign(position)

    if position != 0:
        exit_value = shares * df['close'].iloc[-1] * (1 - commission)
        pnl = exit_value - entry_size

        trades.append({
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'entry_price': entry_price,
            'exit_price': df['close'].iloc[-1],
            'position': position,
            'shares': shares,
            'pnl': pnl,
            'return': (df['close'].iloc[-1] / entry_price - 1) * np.sign(position),
            'duration': (df.index[-1] - entry_time).total_seconds() / 3600
        })

    return trades, np.array(equity_curve)


if __name__ == "__main__":
    results, risk_metrics = run_optimized_trading_system()

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)