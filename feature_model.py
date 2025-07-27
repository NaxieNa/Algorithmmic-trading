"""
Systematic Feature Engineering Framework for SPY Trading
系统化特征工程框架 - 修正版本
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_classif
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

# ==================== 核心数据结构 ====================

@dataclass
class FeatureMetadata:
    """特征元数据"""
    name: str
    category: str  # 'technical', 'statistical', 'microstructure', 'interaction'
    market_dynamic: str  # 'trend', 'mean_reversion', 'volatility', 'volume'
    timeframe: str  # 'micro', 'meso', 'macro'
    math_type: str  # 'linear', 'nonlinear', 'frequency', 'information'
    factor_type: str  # 'risk', 'alpha', 'mixed'
    dependencies: List[str] = field(default_factory=list)
    compute_function: Optional[callable] = None
    version: str = "1.0"
    
@dataclass
class RegimeState:
    """市场状态"""
    name: str  # 'trending', 'ranging', 'volatile'
    probability: float
    features: List[str]
    timestamp: pd.Timestamp

# ==================== 特征基类 ====================

class BaseFeature(ABC):
    """所有特征的抽象基类"""
    
    def __init__(self, metadata: FeatureMetadata):
        self.metadata = metadata
        self._cache = {}
        
    @abstractmethod
    def compute(self, data: pd.DataFrame, **params) -> pd.Series:
        """计算特征值"""
        pass
        
    def validate_input(self, data: pd.DataFrame) -> bool:
        """验证输入数据包含必需列"""
        required_cols = self.metadata.dependencies
        return all(col in data.columns for col in required_cols)

# ==================== 特征注册中心 ====================

class FeatureRegistry:
    """特征中央注册表"""
    
    def __init__(self):
        self._features: Dict[str, BaseFeature] = {}
        self._metadata: Dict[str, FeatureMetadata] = {}
        self._hierarchy = self._initialize_hierarchy()
        
    def _initialize_hierarchy(self) -> Dict:
        """初始化4层特征层次结构"""
        return {
            'market_dynamics': {
                'trend': [],
                'mean_reversion': [],
                'volatility': [],
                'volume': []
            },
            'generation_method': {
                'technical': [],
                'statistical': [],
                'pattern': [],
                'microstructure': []
            },
            'timeframe': {
                'micro': [],  # 分钟到小时
                'meso': [],   # 天到周  
                'macro': []   # 月到季度
            },
            'mathematical': {
                'linear': [],
                'nonlinear': [],
                'frequency': [],
                'information': []
            }
        }
        
    def register(self, feature: BaseFeature):
        """注册特征到系统"""
        name = feature.metadata.name
        self._features[name] = feature
        self._metadata[name] = feature.metadata
        
        # 更新层次结构
        self._hierarchy['market_dynamics'][feature.metadata.market_dynamic].append(name)
        self._hierarchy['generation_method'][feature.metadata.category].append(name)
        self._hierarchy['timeframe'][feature.metadata.timeframe].append(name)
        self._hierarchy['mathematical'][feature.metadata.math_type].append(name)
        
    def get_feature(self, name: str) -> BaseFeature:
        """按名称获取特征"""
        return self._features.get(name)
        
    def get_features_by_category(self, category: str, subcategory: str) -> List[str]:
        """按层次类别获取特征"""
        return self._hierarchy.get(category, {}).get(subcategory, [])

# ==================== 技术指标特征 ====================

class TechnicalFeature(BaseFeature):
    """技术指标特征实现"""
    
    def compute(self, data: pd.DataFrame, **params) -> pd.Series:
        if not self.validate_input(data):
            raise ValueError(f"Missing dependencies for {self.metadata.name}")
            
        # 使用缓存结果（如果可用）
        cache_key = f"{len(data)}_{data.index[-1]}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # 根据特征名称计算
        result = self._compute_specific(data, **params)
        
        # 应用滞后以保证因果性
        result = result.shift(1)
        
        # 缓存结果
        self._cache[cache_key] = result
        return result
        
    def _compute_specific(self, data: pd.DataFrame, **params) -> pd.Series:
        """计算特定技术指标"""
        name = self.metadata.name
        
        if name == 'RSI_14':
            return talib.RSI(data['close'], timeperiod=14)
        elif name == 'MACD':
            macd, _, _ = talib.MACD(data['close'])
            return macd
        elif name == 'BB_width':
            upper, middle, lower = talib.BBANDS(data['close'], timeperiod=20)
            return upper - lower
        elif name == 'ADX_14':
            return talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        elif name == 'ATR_14':
            return talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        elif name == 'CCI_14':
            return talib.CCI(data['high'], data['low'], data['close'], timeperiod=14)
        elif name == 'STOCH_k':
            k, d = talib.STOCH(data['high'], data['low'], data['close'])
            return k
        elif name == 'OBV':
            return talib.OBV(data['close'], data['volume'])
        elif name == 'AD':
            return talib.AD(data['high'], data['low'], data['close'], data['volume'])
        elif name.startswith('SMA_'):
            period = int(name.split('_')[1])
            return talib.SMA(data['close'], timeperiod=period)
        elif name.startswith('EMA_'):
            period = int(name.split('_')[1])
            return talib.EMA(data['close'], timeperiod=period)
        elif name.startswith('close_ratio_SMA_'):
            period = int(name.split('_')[-1])
            sma = talib.SMA(data['close'], timeperiod=period)
            return data['close'] / sma
        else:
            raise NotImplementedError(f"Technical indicator {name} not implemented")

# ==================== 统计特征 ====================

class StatisticalFeature(BaseFeature):
    """统计特征实现"""
    
    def compute(self, data: pd.DataFrame, **params) -> pd.Series:
        if not self.validate_input(data):
            raise ValueError(f"Missing dependencies for {self.metadata.name}")
            
        name = self.metadata.name
        
        if name == 'z_score_20':
            close = data['close']
            rolling_mean = close.rolling(20).mean()
            rolling_std = close.rolling(20).std()
            z_score = (close - rolling_mean) / rolling_std
            return z_score.shift(1)
            
        elif name == 'realized_vol_20':
            returns = data['close'].pct_change()
            vol = returns.rolling(20).std() * np.sqrt(252)
            return vol.shift(1)
            
        elif name == 'realized_vol_5':
            returns = data['close'].pct_change()
            vol = returns.rolling(5).std() * np.sqrt(252)
            return vol.shift(1)
            
        elif name == 'max_drawdown_20':
            close = data['close']
            rolling_max = close.rolling(20).max()
            drawdown = (close / rolling_max - 1.0)
            return drawdown.rolling(20).min().shift(1)
            
        elif name == 'close_open_ratio':
            return (data['close'] / data['open']).shift(1)
            
        elif name == 'gap_size':
            return ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)).shift(1)
            
        elif name == 'volume_ratio':
            vol_ma = data['volume'].rolling(10).mean()
            return (data['volume'] / vol_ma).shift(1)
            
        else:
            raise NotImplementedError(f"Statistical feature {name} not implemented")

# ==================== 市场状态检测 ====================

class MarketRegimeDetector:
    """使用隐马尔科夫模型检测市场状态"""
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=1000,  # 增加迭代次数
            random_state=42,
            init_params='stm',  # 只初始化开始概率、转移矩阵和均值
            params='stmc'  # 更新所有参数
        )
        self.regime_names = {0: 'trending', 1: 'ranging', 2: 'volatile'}
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame):
        """拟合HMM模型"""
        features = self._extract_regime_features(data)
        if len(features) > 100:  # 确保有足够的数据
            try:
                self.model.fit(features)
                self.is_fitted = True
            except:
                print("HMM fitting failed, using default parameters")
                self.is_fitted = False
        
    def detect_regime(self, data: pd.DataFrame) -> RegimeState:
        """检测当前市场状态"""
        if not self.is_fitted:
            self.fit(data)
            
        features = self._extract_regime_features(data)
        
        if self.is_fitted and len(features) > 0:
            try:
                states = self.model.predict(features)
                current_state = states[-1]
                
                # 获取状态概率
                probas = self.model.predict_proba(features)
                current_proba = probas[-1, current_state]
            except:
                # 如果预测失败，使用默认值
                current_state = 1  # ranging
                current_proba = 1.0
        else:
            # 默认状态
            current_state = 1  # ranging
            current_proba = 1.0
        
        # 为每个状态定义最优特征
        regime_features = {
            'trending': ['MACD', 'ADX_14', 'close_ratio_SMA_20', 'volume_ratio'],
            'ranging': ['RSI_14', 'CCI_14', 'BB_width', 'z_score_20'],
            'volatile': ['ATR_14', 'realized_vol_5', 'gap_size', 'TRANGE']
        }
        
        return RegimeState(
            name=self.regime_names[current_state],
            probability=current_proba,
            features=regime_features[self.regime_names[current_state]],
            timestamp=data.index[-1] if len(data) > 0 else pd.Timestamp.now()
        )
        
    def _extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """提取用于状态检测的特征"""
        if len(data) < 50:
            return np.array([])
            
        returns = data['close'].pct_change()
        
        features = pd.DataFrame()
        features['return'] = returns
        features['volatility'] = returns.rolling(20).std()
        features['volume_change'] = data['volume'].pct_change()
        features['high_low_ratio'] = data['high'] / data['low'] - 1
        features['trend'] = data['close'].rolling(20).mean() / data['close'].rolling(50).mean() - 1
        
        # 填充NaN值
        features = features.fillna(method='ffill').fillna(0)
        
        return features.dropna().values

# ==================== 动态特征选择 ====================

class DynamicFeatureSelector:
    """基于市场状态动态选择特征"""
    
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self.performance_history = {}
        self.regime_features = {
            'trending': {
                'primary': ['MACD', 'ADX_14', 'close_ratio_SMA_20'],
                'secondary': ['volume_ratio', 'OBV', 'AD']
            },
            'ranging': {
                'primary': ['RSI_14', 'CCI_14', 'BB_width'],
                'secondary': ['z_score_20', 'STOCH_k', 'close_open_ratio']
            },
            'volatile': {
                'primary': ['ATR_14', 'realized_vol_5', 'gap_size'],
                'secondary': ['NATR_14', 'TRANGE', 'max_drawdown_20']
            }
        }
        
    def select_features(self, 
                       regime: RegimeState,
                       data: pd.DataFrame,
                       n_features: int = 10) -> List[str]:
        """为当前状态选择最优特征"""
        
        # 从状态特定特征开始
        primary = self.regime_features[regime.name]['primary'].copy()
        secondary = self.regime_features[regime.name]['secondary'].copy()
        
        # 如果状态置信度低，添加其他状态的特征
        if regime.probability < 0.7:
            for other_regime in self.regime_features:
                if other_regime != regime.name:
                    secondary.extend(self.regime_features[other_regime]['primary'][:1])
                    
        # 按信息系数对特征排序
        all_features = list(set(primary + secondary))
        scores = self._calculate_feature_scores(all_features, data)
        
        # 选择前N个特征
        ranked_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in ranked_features[:n_features]]
        
        # 确保用户选择的5个特征总是包含在内
        user_features = ['BB_width', 'gap_size', 'RSI_14', 'close_open_ratio', 'z_score_20']
        for feature in user_features:
            if feature not in selected:
                selected.append(feature)
                
        return selected[:n_features]
        
    def _calculate_feature_scores(self, 
                                 features: List[str],
                                 data: pd.DataFrame) -> Dict[str, float]:
        """计算特征的信息系数"""
        scores = {}
        
        # 简单的IC计算（与未来收益的相关性）
        if 'close' in data.columns:
            forward_returns = data['close'].pct_change().shift(-1)
        else:
            return {f: 0 for f in features}
        
        for feature_name in features:
            try:
                feature = self.registry.get_feature(feature_name)
                if feature:
                    values = feature.compute(data)
                    # 计算秩相关
                    ic = values.corr(forward_returns, method='spearman')
                    scores[feature_name] = abs(ic) if not np.isnan(ic) else 0
                else:
                    scores[feature_name] = 0
            except:
                scores[feature_name] = 0
                
        return scores

# ==================== 特征正交化 ====================

class OrthogonalizationProcessor:
    """通过正交化消除冗余"""
    
    def __init__(self, correlation_threshold: float = 0.85):
        self.correlation_threshold = correlation_threshold
        self.pca = PCA(n_components=0.95)
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_mapping = None
        
    def fit_transform(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换特征到正交空间"""
        
        if len(feature_matrix.columns) < 2:
            return feature_matrix
            
        # 移除高度相关的特征
        reduced_features = self._remove_correlated(feature_matrix)
        
        if len(reduced_features.columns) < 2:
            return reduced_features
            
        # 标准化
        scaled = self.scaler.fit_transform(reduced_features)
        
        # 应用PCA
        components = self.pca.fit_transform(scaled)
        
        # 创建带组件名称的DataFrame
        component_names = [f'PC{i+1}' for i in range(components.shape[1])]
        result = pd.DataFrame(
            components,
            index=feature_matrix.index,
            columns=component_names
        )
        
        # 存储映射以保持可解释性
        self.feature_mapping = pd.DataFrame(
            self.pca.components_,
            columns=reduced_features.columns,
            index=component_names
        )
        
        return result
        
    def _remove_correlated(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """移除高度相关的特征"""
        # 计算相关矩阵
        corr_matrix = feature_matrix.corr().abs()
        
        # 找到要移除的特征
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_tri.columns 
                  if any(upper_tri[column] > self.correlation_threshold)]
        
        self.selected_features = [col for col in feature_matrix.columns 
                                 if col not in to_drop]
        
        return feature_matrix[self.selected_features]

# ==================== 微观结构增强 ====================

class MicrostructureEnhancer:
    """使用市场微观结构信息增强特征"""
    
    def __init__(self):
        self.enhancements = {
            'spread_adjusted': self._spread_adjust,
            'volume_weighted': self._volume_weight,
            'order_flow': self._order_flow_enhance
        }
        
    def enhance_features(self, 
                        features: pd.DataFrame,
                        original_data: pd.DataFrame = None,
                        microstructure_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """使用微观结构信息增强传统特征"""
        
        enhanced = features.copy()
        
        if microstructure_data is None:
            # 从原始数据生成合成微观结构特征
            if original_data is not None:
                microstructure_data = self._generate_synthetic_microstructure(original_data)
            else:
                # 如果没有原始数据，返回原始特征
                return enhanced
                
        # 应用增强
        for feature_name in features.columns:
            if 'RSI' in feature_name:
                enhanced[f'{feature_name}_flow'] = self._order_flow_enhance(
                    features[feature_name], microstructure_data
                )
            elif 'BB' in feature_name:
                enhanced[f'{feature_name}_spread'] = self._spread_adjust(
                    features[feature_name], microstructure_data
                )
                
        return enhanced
        
    def _spread_adjust(self, feature: pd.Series, micro_data: pd.DataFrame) -> pd.Series:
        """按买卖价差调整特征"""
        if 'spread' in micro_data.columns and 'close' in micro_data.columns:
            spread_factor = 1 + micro_data['spread'] / micro_data['close']
            return feature * spread_factor
        return feature
        
    def _volume_weight(self, feature: pd.Series, micro_data: pd.DataFrame) -> pd.Series:
        """按成交量加权特征"""
        if 'volume' in micro_data.columns:
            vol_weight = micro_data['volume'] / micro_data['volume'].rolling(20).mean()
            return feature * vol_weight.clip(0.5, 2.0)
        return feature
        
    def _order_flow_enhance(self, feature: pd.Series, micro_data: pd.DataFrame) -> pd.Series:
        """使用订单流失衡增强"""
        # 从成交量和价格变化模拟订单流
        if 'volume' in micro_data.columns and 'close' in micro_data.columns:
            price_change = micro_data['close'].pct_change()
            signed_volume = micro_data['volume'] * np.sign(price_change)
            flow_imbalance = signed_volume.rolling(10).sum() / micro_data['volume'].rolling(10).sum()
            return feature * (1 + flow_imbalance.fillna(0).clip(-0.5, 0.5))
        return feature
        
    def _generate_synthetic_microstructure(self, data: pd.DataFrame) -> pd.DataFrame:
        """在真实数据不可用时生成合成微观结构数据"""
        micro = pd.DataFrame(index=data.index)
        
        # 确保有必要的列
        if 'close' in data.columns:
            micro['close'] = data['close']
        
        if 'volume' in data.columns:
            micro['volume'] = data['volume']
            
        # 基于波动率的合成价差
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            vol = returns.rolling(20).std()
            micro['spread'] = vol * 0.001  # 简单的价差模型
        else:
            micro['spread'] = 0.001
            
        return micro

# ==================== 特征质量监控 ====================

class FeatureQualityMonitor:
    """监控特征质量和性能"""
    
    def __init__(self):
        self.metrics_history = {}
        self.alert_thresholds = {
            'ic': 0.01,           # 最小信息系数
            'stability': 0.5,     # 最小稳定性得分
            'decay_rate': 0.3,    # 最大月度衰减
            'coverage': 0.95      # 最小数据覆盖率
        }
        
    def evaluate_feature(self, 
                        feature_name: str,
                        feature_values: pd.Series,
                        target: pd.Series) -> Dict[str, float]:
        """评估特征质量指标"""
        
        metrics = {}
        
        # 信息系数(IC)
        ic = feature_values.corr(target, method='spearman')
        metrics['ic'] = ic if not np.isnan(ic) else 0
        
        # 稳定性（滚动IC稳定性）
        if len(feature_values) > 252:
            rolling_ic = pd.Series([
                feature_values.iloc[i:i+252].corr(target.iloc[i:i+252], method='spearman')
                for i in range(0, len(feature_values)-252, 21)
                if not feature_values.iloc[i:i+252].isna().all()
            ])
            metrics['stability'] = 1 - rolling_ic.std() if len(rolling_ic) > 1 else 0
        else:
            metrics['stability'] = 0
        
        # 覆盖率（非缺失比率）
        metrics['coverage'] = 1 - feature_values.isna().sum() / len(feature_values)
        
        # 衰减率（IC趋势）
        if feature_name in self.metrics_history:
            history = self.metrics_history[feature_name]
            if len(history) > 5:
                recent_ics = [h['ic'] for h in history[-5:]]
                decay = (recent_ics[0] - recent_ics[-1]) / (abs(recent_ics[0]) + 1e-6)
                metrics['decay_rate'] = max(0, decay)
            else:
                metrics['decay_rate'] = 0
        else:
            metrics['decay_rate'] = 0
            
        # 存储到历史记录
        if feature_name not in self.metrics_history:
            self.metrics_history[feature_name] = []
        self.metrics_history[feature_name].append(metrics)
        
        # 检查警报
        self._check_alerts(feature_name, metrics)
        
        return metrics
        
    def _check_alerts(self, feature_name: str, metrics: Dict[str, float]):
        """检查指标是否触发警报"""
        alerts = []
        
        if metrics['ic'] < self.alert_thresholds['ic']:
            alerts.append(f"Low IC: {metrics['ic']:.4f}")
            
        if metrics['stability'] < self.alert_thresholds['stability']:
            alerts.append(f"Low stability: {metrics['stability']:.4f}")
            
        if metrics['decay_rate'] > self.alert_thresholds['decay_rate']:
            alerts.append(f"High decay rate: {metrics['decay_rate']:.4f}")
            
        if alerts:
            print(f"⚠️ Alerts for {feature_name}: {', '.join(alerts)}")

# ==================== 主特征系统 ====================

class FeatureSystem:
    """协调所有组件的主系统"""
    
    def __init__(self):
        self.registry = FeatureRegistry()
        self.regime_detector = MarketRegimeDetector()
        self.feature_selector = DynamicFeatureSelector(self.registry)
        self.orthogonalizer = OrthogonalizationProcessor()
        self.enhancer = MicrostructureEnhancer()
        self.monitor = FeatureQualityMonitor()
        
        # 初始化默认特征
        self._initialize_default_features()
        
    def _initialize_default_features(self):
        """注册与用户选择匹配的默认特征"""
        
        # 用户选择的特征
        selected_features = [
            ('BB_width', 'technical', 'volatility', 'meso', 'nonlinear', 'risk'),
            ('gap_size', 'statistical', 'volatility', 'micro', 'linear', 'alpha'),
            ('RSI_14', 'technical', 'mean_reversion', 'meso', 'nonlinear', 'alpha'),
            ('close_open_ratio', 'statistical', 'trend', 'micro', 'linear', 'alpha'),
            ('z_score_20', 'statistical', 'mean_reversion', 'meso', 'linear', 'alpha')
        ]
        
        # 额外的有用特征
        additional_features = [
            ('MACD', 'technical', 'trend', 'meso', 'nonlinear', 'alpha'),
            ('ADX_14', 'technical', 'trend', 'meso', 'nonlinear', 'risk'),
            ('ATR_14', 'technical', 'volatility', 'meso', 'linear', 'risk'),
            ('CCI_14', 'technical', 'mean_reversion', 'meso', 'nonlinear', 'alpha'),
            ('volume_ratio', 'statistical', 'volume', 'micro', 'linear', 'alpha'),
            ('realized_vol_5', 'statistical', 'volatility', 'micro', 'linear', 'risk'),
            ('realized_vol_20', 'statistical', 'volatility', 'meso', 'linear', 'risk'),
            ('max_drawdown_20', 'statistical', 'volatility', 'meso', 'nonlinear', 'risk'),
            ('close_ratio_SMA_20', 'technical', 'trend', 'meso', 'linear', 'alpha')
        ]
        
        all_features = selected_features + additional_features
        
        for name, category, dynamic, timeframe, math_type, factor_type in all_features:
            metadata = FeatureMetadata(
                name=name,
                category=category,
                market_dynamic=dynamic,
                timeframe=timeframe,
                math_type=math_type,
                factor_type=factor_type,
                dependencies=['open', 'high', 'low', 'close', 'volume']
            )
            
            if category == 'technical':
                feature = TechnicalFeature(metadata)
            else:
                feature = StatisticalFeature(metadata)
                
            self.registry.register(feature)
            
    def compute_features(self, 
                        data: pd.DataFrame,
                        feature_names: Optional[List[str]] = None,
                        regime_aware: bool = True,
                        enhance: bool = True,
                        orthogonalize: bool = True) -> pd.DataFrame:
        """计算带有所有增强的特征"""
        
        # 检测当前状态（如需要）
        if regime_aware:
            regime = self.regime_detector.detect_regime(data)
            if feature_names is None:
                feature_names = self.feature_selector.select_features(regime, data)
            print(f"📊 Current regime: {regime.name} (confidence: {regime.probability:.2%})")
        else:
            if feature_names is None:
                # 使用所有已注册特征
                feature_names = list(self.registry._features.keys())
                
        # 计算基础特征
        features = pd.DataFrame(index=data.index)
        for feature_name in feature_names:
            feature_obj = self.registry.get_feature(feature_name)
            if feature_obj:
                try:
                    features[feature_name] = feature_obj.compute(data)
                except Exception as e:
                    print(f"⚠️ Error computing {feature_name}: {e}")
                    
        # 使用微观结构增强
        if enhance:
            features = self.enhancer.enhance_features(features, original_data=data)
            
        # 正交化
        if orthogonalize and len(features.columns) > 5:
            # 移除NaN值进行正交化
            clean_features = features.dropna()
            if len(clean_features) > 10:
                orthogonal = self.orthogonalizer.fit_transform(clean_features)
                # 结合原始和正交特征
                features = pd.concat([features, orthogonal], axis=1)
            
        return features
        
    def evaluate_features(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """评估所有特征并返回质量指标"""
        
        results = []
        
        for feature_name in features.columns:
            if feature_name.startswith('PC'):  # 跳过主成分
                continue
                
            metrics = self.monitor.evaluate_feature(
                feature_name,
                features[feature_name],
                target
            )
            
            metrics['feature'] = feature_name
            results.append(metrics)
            
        return pd.DataFrame(results).set_index('feature')
        
    def get_feature_importance(self, 
                              features: pd.DataFrame,
                              target: pd.Series,
                              method: str = 'mutual_info') -> pd.Series:
        """计算特征重要性得分"""
        
        # 移除NaN值
        mask = features.notna().all(axis=1) & target.notna()
        clean_features = features[mask]
        clean_target = target[mask]
        
        if len(clean_features) == 0:
            return pd.Series()
            
        if method == 'mutual_info':
            # 离散化目标以计算互信息
            target_binary = (clean_target > clean_target.median()).astype(int)
            scores = mutual_info_classif(clean_features, target_binary)
            importance = pd.Series(scores, index=features.columns)
        elif method == 'correlation':
            importance = features.corrwith(target).abs()
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return importance.sort_values(ascending=False)
        
    def generate_report(self, data: pd.DataFrame) -> Dict:
        """生成综合特征分析报告"""
        
        report = {
            'timestamp': pd.Timestamp.now(),
            'data_shape': data.shape,
            'regime': self.regime_detector.detect_regime(data),
            'feature_count': len(self.registry._features),
            'hierarchy': self.registry._hierarchy
        }
        
        # 计算所有特征
        features = self.compute_features(data, regime_aware=True)
        report['computed_features'] = list(features.columns)
        
        # 计算特征重要性
        if 'future_return' in data.columns:
            target = data['future_return']
            importance = self.get_feature_importance(features, target)
            report['feature_importance'] = importance.to_dict()
            
            # 评估质量
            quality = self.evaluate_features(features, target)
            report['feature_quality'] = quality.to_dict()
            
        return report

def quick_start():
    """快速开始使用框架"""
    
    # 1. 基本使用 - 使用您的5个选定特征
    system = FeatureSystem()
    
    # 加载数据
    data = pd.read_csv('SPY_10Y_daily.csv')
    data.columns = [col.lower() for col in data.columns]
    
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    
    # 仅计算您选择的5个特征
    selected_5 = ['BB_width', 'gap_size', 'RSI_14', 'close_open_ratio', 'z_score_20']
    features = system.compute_features(data, feature_names=selected_5, regime_aware=False)
    
    print("✅ 基本特征计算完成")
    print(f"Features shape: {features.shape}")
    print(f"Features: {features.columns.tolist()}")
    
    # 2. 与现有代码集成（如果存在）
    try:
        # 尝试导入您原有的特征生成器
        from Feature_engineering import FeatureGenerator
        
        # 使用原有的特征生成器
        legacy_features = FeatureGenerator.generate(data)
        
        # 将原有特征和新系统结合
        combined_features = pd.concat([features, legacy_features], axis=1)
        
        # 移除重复列
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
        
        print(f"\n✅ 结合后的特征数量: {len(combined_features.columns)}")
        
        return combined_features
        
    except ImportError:
        print("\n💡 提示: 原有的 Feature_engineering 模块未找到")
        print("   如需集成原有代码，请将您的 FeatureGenerator 类保存为 Feature_engineering.py")
        print("   当前系统已独立运行，无需依赖原有代码")
        
        # 返回新系统计算的特征
        return features

# ==================== 展示更多功能 ====================

def demonstrate_advanced_features():
    """展示系统的高级功能"""
    
    print("\n" + "="*60)
    print("🔬 系统高级功能演示")
    print("="*60)
    
    # 初始化系统
    system = FeatureSystem()
    
    # 加载数据
    data = pd.read_csv('SPY_10Y_daily.csv')
    data.columns = [col.lower() for col in data.columns]
    
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    
    # 如果没有future_return，创建它
    if 'future_return' not in data.columns:
        data['future_return'] = data['close'].pct_change().shift(-1)
    
    # 1. 展示市场状态检测
    print("\n1️⃣ 市场状态检测")
    regime = system.regime_detector.detect_regime(data)
    print(f"   当前市场状态: {regime.name}")
    print(f"   置信度: {regime.probability:.2%}")
    print(f"   推荐特征: {regime.features}")
    
    # 2. 展示动态特征选择
    print("\n2️⃣ 动态特征选择")
    selected_features = system.feature_selector.select_features(regime, data, n_features=10)
    print(f"   根据当前市场状态选择的特征: {selected_features}")
    
    # 3. 计算完整特征集
    print("\n3️⃣ 计算完整特征集（包含增强和正交化）")
    features_full = system.compute_features(
        data,
        regime_aware=True,
        enhance=True,
        orthogonalize=True
    )
    print(f"   总特征数: {len(features_full.columns)}")
    print(f"   包含增强特征: {[col for col in features_full.columns if '_' in col and any(x in col for x in ['flow', 'spread'])]}")
    print(f"   包含主成分: {[col for col in features_full.columns if col.startswith('PC')]}")
    
    # 4. 特征质量评估
    print("\n4️⃣ 特征质量评估")
    target = data['future_return']
    quality_metrics = system.evaluate_features(features_full[selected_5], target)
    
    print("\n   特征质量指标:")
    print(quality_metrics[['ic', 'stability', 'coverage']].round(4))
    
    # 5. 特征重要性排序
    print("\n5️⃣ 特征重要性排序（互信息方法）")
    importance = system.get_feature_importance(features_full, target, method='mutual_info')
    print("\n   前10个最重要的特征:")
    for i, (feat, score) in enumerate(importance.head(10).items(), 1):
        print(f"   {i}. {feat}: {score:.4f}")
    
    # 6. 展示特征层次结构
    print("\n6️⃣ 特征层次结构")
    hierarchy = system.registry._hierarchy
    print("\n   按市场动态分类:")
    for dynamic, features in hierarchy['market_dynamics'].items():
        if features:
            print(f"   - {dynamic}: {features[:3]}...")  # 只显示前3个
    
    return features_full

# ==================== 如何集成原有代码 ====================

def integration_guide():
    """展示如何集成原有的 FeatureGenerator"""
    
    print("\n" + "="*60)
    print("📚 集成指南：如何整合您原有的 FeatureGenerator")
    print("="*60)
    
    print("""
    方法1: 创建 Feature_engineering.py 文件
    ----------------------------------------
    将您原有的 FeatureGenerator 类保存为 Feature_engineering.py:
    
    # Feature_engineering.py
    class FeatureGenerator:
        @staticmethod
        def generate(df):
            # 您原有的特征生成代码
            factors = pd.DataFrame(index=df.index)
            # ... 
            return factors
    
    方法2: 直接在当前代码中使用
    ----------------------------------------
    """)
    
    # 展示如何直接集成
    print("示例代码:")
    print("""
    # 创建一个包装器来使用您原有的特征逻辑
    class YourFeatureGenerator:
        @staticmethod
        def generate(df):
            factors = pd.DataFrame(index=df.index)
            o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
            
            # 您原有的特征计算逻辑
            factors['RSI_14'] = talib.RSI(c, 14).shift(1)
            factors['BB_width'] = (talib.BBANDS(c)[0] - talib.BBANDS(c)[2]).shift(1)
            # ... 添加更多特征
            
            return factors
    
    # 使用新系统增强原有特征
    system = FeatureSystem()
    legacy_features = YourFeatureGenerator.generate(data)
    enhanced_features = system.enhancer.enhance_features(legacy_features, data)
    """)

def demonstrate_advanced_features():
    """展示系统的高级功能"""
    
    print("\n" + "="*60)
    print("🔬 系统高级功能演示")
    print("="*60)
    
    # 初始化系统
    system = FeatureSystem()
    
    # 加载数据
    data = pd.read_csv('SPY_10Y_daily.csv')
    data.columns = [col.lower() for col in data.columns]
    
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
    
    # 如果没有future_return，创建它
    if 'future_return' not in data.columns:
        data['future_return'] = data['close'].pct_change().shift(-1)
    
    # 定义用户选择的5个特征
    selected_5 = ['BB_width', 'gap_size', 'RSI_14', 'close_open_ratio', 'z_score_20']
    
    # 1. 展示市场状态检测
    print("\n1️⃣ 市场状态检测")
    regime = system.regime_detector.detect_regime(data)
    print(f"   当前市场状态: {regime.name}")
    print(f"   置信度: {regime.probability:.2%}")
    print(f"   推荐特征: {regime.features}")
    
    # 2. 展示动态特征选择
    print("\n2️⃣ 动态特征选择")
    selected_features = system.feature_selector.select_features(regime, data, n_features=10)
    print(f"   根据当前市场状态选择的特征: {selected_features}")
    
    # 3. 计算完整特征集
    print("\n3️⃣ 计算完整特征集（包含增强和正交化）")
    features_full = system.compute_features(
        data,
        regime_aware=True,
        enhance=True,
        orthogonalize=True
    )
    print(f"   总特征数: {len(features_full.columns)}")
    print(f"   包含增强特征: {[col for col in features_full.columns if '_' in col and any(x in col for x in ['flow', 'spread'])]}")
    print(f"   包含主成分: {[col for col in features_full.columns if col.startswith('PC')]}")
    
    # 4. 特征质量评估（仅评估实际计算的特征）
    print("\n4️⃣ 特征质量评估")
    target = data['future_return']
    
    # 只评估实际存在的特征
    available_features = [f for f in selected_5 if f in features_full.columns]
    if available_features:
        quality_metrics = system.evaluate_features(features_full[available_features], target)
        print("\n   特征质量指标:")
        print(quality_metrics[['ic', 'stability', 'coverage']].round(4))
    else:
        print("   注意：由于市场状态为trending，某些mean_reversion特征可能未被计算")
        
    # 5. 单独计算用户选择的5个特征
    print("\n5️⃣ 计算用户指定的5个特征")
    user_features = system.compute_features(
        data,
        feature_names=selected_5,
        regime_aware=False,  # 不使用状态感知，确保计算所有指定特征
        enhance=True,
        orthogonalize=False
    )
    print(f"   用户特征数: {len(user_features.columns)}")
    print(f"   特征列表: {user_features.columns.tolist()}")
    
    # 评估用户特征质量
    if len(user_features.columns) > 0:
        quality_metrics_user = system.evaluate_features(user_features[selected_5], target)
        print("\n   用户特征质量指标:")
        print(quality_metrics_user[['ic', 'stability', 'coverage']].round(4))
    
    # 6. 特征重要性排序
    print("\n6️⃣ 特征重要性排序（互信息方法）")
    importance = system.get_feature_importance(features_full, target, method='mutual_info')
    if len(importance) > 0:
        print("\n   前10个最重要的特征:")
        for i, (feat, score) in enumerate(importance.head(10).items(), 1):
            print(f"   {i}. {feat}: {score:.4f}")
    
    # 7. 展示特征层次结构
    print("\n7️⃣ 特征层次结构")
    hierarchy = system.registry._hierarchy
    print("\n   按市场动态分类:")
    for dynamic, features in hierarchy['market_dynamics'].items():
        if features:
            print(f"   - {dynamic}: {features[:3]}...")  # 只显示前3个
    
    # 8. 比较不同市场状态下的特征选择
    print("\n8️⃣ 不同市场状态下的最优特征")
    for state_name in ['trending', 'ranging', 'volatile']:
        regime_test = RegimeState(
            name=state_name,
            probability=1.0,
            features=[],
            timestamp=pd.Timestamp.now()
        )
        state_features = system.feature_selector.select_features(regime_test, data, n_features=5)
        print(f"   {state_name}: {state_features[:5]}")
    
    return features_full, user_features

# 修改主函数，更好地处理结果
if __name__ == "__main__":
    print("🚀 启动系统化特征工程框架...")
    
    try:
        # 快速开始
        features_basic = quick_start()
        print("\n✅ 快速开始完成!")
        # 定义选择的5个特征（全局使用）
        selected_5 = ['BB_width', 'gap_size', 'RSI_14', 'close_open_ratio', 'z_score_20']
        # 展示高级功能
        features_advanced, user_features = demonstrate_advanced_features()
        print("\n✅ 高级功能演示完成!")
        
        # 显示集成指南
        integration_guide()
        
        print("\n" + "="*60)
        print("🎉 系统已成功运行！")
        print("="*60)
        
        # 展示最终结果
        print(f"\n📊 特征计算结果汇总:")
        print(f"1. 基础特征（快速开始）: {features_basic.shape[1]} 个特征")
        print(f"2. 高级特征（状态感知）: {features_advanced.shape[1]} 个特征")
        print(f"3. 用户指定特征: {user_features.shape[1]} 个特征")
        
        # 显示特征对比
        print(f"\n📈 特征对比:")
        print(f"- 基础特征包含: {features_basic.columns.tolist()}")
        print(f"- 用户5个核心特征: {['BB_width', 'gap_size', 'RSI_14', 'close_open_ratio', 'z_score_20']}")
        print(f"- 增强特征: {[col for col in features_basic.columns if 'flow' in col or 'spread' in col]}")
        print(f"- 主成分: {[col for col in features_basic.columns if col.startswith('PC')]}")
        
        # 保存不同版本的特征
        features_basic.to_csv('features_basic.csv')
        features_advanced.to_csv('features_advanced.csv')
        user_features.to_csv('features_user_selected.csv')
        
        print(f"\n💾 特征已保存:")
        print(f"   - features_basic.csv (基础特征)")
        print(f"   - features_advanced.csv (高级特征)")
        print(f"   - features_user_selected.csv (用户选择的特征)")
        
        # 提供使用建议
        print(f"\n💡 使用建议:")
        print(f"1. 对于简单模型，使用 features_basic.csv")
        print(f"2. 对于复杂策略，使用 features_advanced.csv")
        print(f"3. 对于与原有模型对比，使用 features_user_selected.csv")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()