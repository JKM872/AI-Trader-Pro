"""
Feature Engineering - Creates ML-ready features from market data.

Features include:
- Technical indicators
- Price patterns
- Volume analysis
- Market structure
- Time-based features
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features."""
    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    PATTERN = "pattern"
    TIME = "time"
    MICROSTRUCTURE = "microstructure"


@dataclass
class FeatureSet:
    """A set of features for ML model."""
    features: pd.DataFrame
    target: Optional[pd.Series] = None
    feature_names: list[str] = field(default_factory=list)
    feature_types: dict[str, FeatureType] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def shape(self) -> tuple[int, int]:
        """Shape of feature matrix."""
        return self.features.shape
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.feature_names)
    
    def get_features_by_type(self, feature_type: FeatureType) -> pd.DataFrame:
        """Get features of a specific type."""
        cols = [k for k, v in self.feature_types.items() if v == feature_type]
        return self.features[cols]
    
    def to_numpy(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert to numpy arrays."""
        X = self.features.values
        y = self.target.values if self.target is not None else None
        return X, y


class FeatureEngineer:
    """
    Creates features from OHLCV data for ML models.
    
    Features:
    - Rolling statistics (SMA, EMA, std)
    - Technical indicators (RSI, MACD, Bollinger)
    - Price patterns (gaps, doji, engulfing)
    - Volume analysis (OBV, volume ratio)
    - Time features (day of week, month)
    """
    
    def __init__(
        self,
        lookback_periods: list[int] = None,
        include_lags: bool = True,
        n_lags: int = 5,
        target_horizon: int = 1,
        target_type: str = "returns"  # "returns", "direction", "volatility"
    ):
        """
        Initialize Feature Engineer.
        
        Args:
            lookback_periods: Periods for rolling calculations
            include_lags: Whether to include lagged features
            n_lags: Number of lag periods
            target_horizon: Prediction horizon in bars
            target_type: Type of target variable
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        self.include_lags = include_lags
        self.n_lags = n_lags
        self.target_horizon = target_horizon
        self.target_type = target_type
    
    def _add_price_features(self, df: pd.DataFrame, features: dict) -> None:
        """Add price-based features."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_price = df['Open']
        
        # Basic price features
        features['log_return'] = np.log(close / close.shift(1))
        features['high_low_range'] = (high - low) / close
        features['close_open_range'] = (close - open_price) / open_price
        features['upper_shadow'] = (high - np.maximum(close, open_price)) / close
        features['lower_shadow'] = (np.minimum(close, open_price) - low) / close
        features['body_size'] = abs(close - open_price) / close
        
        # Gap features
        features['gap_up'] = (open_price > high.shift(1)).astype(int)
        features['gap_down'] = (open_price < low.shift(1)).astype(int)
        features['gap_size'] = (open_price - close.shift(1)) / close.shift(1)
        
        for col in ['log_return', 'high_low_range', 'close_open_range', 
                    'upper_shadow', 'lower_shadow', 'body_size', 'gap_size']:
            self.feature_types[col] = FeatureType.PRICE
    
    def _add_rolling_features(self, df: pd.DataFrame, features: dict) -> None:
        """Add rolling statistics features."""
        close = df['Close']
        
        for period in self.lookback_periods:
            # Moving averages
            sma = close.rolling(period).mean()
            ema = close.ewm(span=period).mean()
            
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            features[f'price_vs_sma_{period}'] = (close - sma) / sma
            features[f'price_vs_ema_{period}'] = (close - ema) / ema
            
            # Rolling statistics
            features[f'std_{period}'] = close.rolling(period).std() / close
            features[f'skew_{period}'] = close.rolling(period).skew()
            features[f'kurt_{period}'] = close.rolling(period).kurt()
            
            # Min/max features
            features[f'high_pct_{period}'] = (
                (close - close.rolling(period).min()) /
                (close.rolling(period).max() - close.rolling(period).min() + 1e-10)
            )
            
            # Mark feature types
            for col in [f'sma_{period}', f'ema_{period}', f'price_vs_sma_{period}',
                       f'price_vs_ema_{period}', f'std_{period}', f'skew_{period}',
                       f'kurt_{period}', f'high_pct_{period}']:
                self.feature_types[col] = FeatureType.TREND
    
    def _add_momentum_features(self, df: pd.DataFrame, features: dict) -> None:
        """Add momentum indicators."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # RSI
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            self.feature_types[f'rsi_{period}'] = FeatureType.MOMENTUM
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Rate of Change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = close.pct_change(period)
            self.feature_types[f'roc_{period}'] = FeatureType.MOMENTUM
        
        # Stochastic
        for period in [14]:
            lowest = low.rolling(period).min()
            highest = high.rolling(period).max()
            features[f'stoch_k_{period}'] = 100 * (close - lowest) / (highest - lowest + 1e-10)
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()
            self.feature_types[f'stoch_k_{period}'] = FeatureType.MOMENTUM
            self.feature_types[f'stoch_d_{period}'] = FeatureType.MOMENTUM
        
        for col in ['macd', 'macd_signal', 'macd_histogram']:
            self.feature_types[col] = FeatureType.MOMENTUM
    
    def _add_volatility_features(self, df: pd.DataFrame, features: dict) -> None:
        """Add volatility indicators."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # ATR
        for period in [7, 14, 21]:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            features[f'atr_{period}'] = tr.rolling(period).mean() / close
            self.feature_types[f'atr_{period}'] = FeatureType.VOLATILITY
        
        # Bollinger Bands
        for period in [20]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'bb_upper_{period}'] = (sma + 2 * std - close) / close
            features[f'bb_lower_{period}'] = (close - (sma - 2 * std)) / close
            features[f'bb_width_{period}'] = 4 * std / sma
            features[f'bb_position_{period}'] = (close - (sma - 2 * std)) / (4 * std + 1e-10)
            
            for col in [f'bb_upper_{period}', f'bb_lower_{period}', 
                       f'bb_width_{period}', f'bb_position_{period}']:
                self.feature_types[col] = FeatureType.VOLATILITY
        
        # Historical volatility
        for period in [5, 10, 20]:
            log_returns = np.log(close / close.shift(1))
            features[f'hist_vol_{period}'] = log_returns.rolling(period).std() * np.sqrt(252)
            self.feature_types[f'hist_vol_{period}'] = FeatureType.VOLATILITY
        
        # Parkinson volatility (high-low based)
        for period in [5, 10, 20]:
            features[f'parkinson_vol_{period}'] = (
                np.sqrt(1 / (4 * np.log(2)) * 
                       (np.log(high / low) ** 2).rolling(period).mean())
            )
            self.feature_types[f'parkinson_vol_{period}'] = FeatureType.VOLATILITY
    
    def _add_volume_features(self, df: pd.DataFrame, features: dict) -> None:
        """Add volume-based features."""
        if 'Volume' not in df.columns:
            return
        
        volume = df['Volume']
        close = df['Close']
        
        # Volume ratios
        for period in [5, 10, 20]:
            vol_sma = volume.rolling(period).mean()
            features[f'vol_ratio_{period}'] = volume / (vol_sma + 1)
            self.feature_types[f'vol_ratio_{period}'] = FeatureType.VOLUME
        
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).cumsum()
        features['obv'] = obv
        features['obv_sma_20'] = obv.rolling(20).mean()
        features['obv_vs_sma'] = (obv - features['obv_sma_20']) / (abs(features['obv_sma_20']) + 1)
        
        # Volume-price relationship
        features['vol_price_corr_20'] = (
            volume.rolling(20).corr(close)
        )
        
        # VWAP approximation
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        features['vwap_20'] = (typical_price * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1)
        features['price_vs_vwap'] = (close - features['vwap_20']) / features['vwap_20']
        
        for col in ['obv', 'obv_sma_20', 'obv_vs_sma', 'vol_price_corr_20', 
                   'vwap_20', 'price_vs_vwap']:
            self.feature_types[col] = FeatureType.VOLUME
    
    def _add_pattern_features(self, df: pd.DataFrame, features: dict) -> None:
        """Add candlestick pattern features."""
        open_price = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        body = close - open_price
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        body_size = abs(body)
        
        # Doji
        avg_body = body_size.rolling(20).mean()
        features['doji'] = (body_size < avg_body * 0.1).astype(int)
        
        # Hammer (long lower shadow, small body at top)
        features['hammer'] = (
            (lower_shadow > body_size * 2) & 
            (upper_shadow < body_size * 0.5) &
            (body > 0)
        ).astype(int)
        
        # Shooting star (long upper shadow, small body at bottom)
        features['shooting_star'] = (
            (upper_shadow > body_size * 2) & 
            (lower_shadow < body_size * 0.5) &
            (body < 0)
        ).astype(int)
        
        # Engulfing patterns
        features['bullish_engulfing'] = (
            (body.shift(1) < 0) &  # Previous bearish
            (body > 0) &  # Current bullish
            (open_price < close.shift(1)) &  # Opens below prev close
            (close > open_price.shift(1))  # Closes above prev open
        ).astype(int)
        
        features['bearish_engulfing'] = (
            (body.shift(1) > 0) &  # Previous bullish
            (body < 0) &  # Current bearish
            (open_price > close.shift(1)) &  # Opens above prev close
            (close < open_price.shift(1))  # Closes below prev open
        ).astype(int)
        
        # Three consecutive patterns
        features['three_green'] = (
            (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0)
        ).astype(int)
        
        features['three_red'] = (
            (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0)
        ).astype(int)
        
        for col in ['doji', 'hammer', 'shooting_star', 'bullish_engulfing',
                   'bearish_engulfing', 'three_green', 'three_red']:
            self.feature_types[col] = FeatureType.PATTERN
    
    def _add_time_features(self, df: pd.DataFrame, features: dict) -> None:
        """Add time-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        
        # Day of week (one-hot)
        for day in range(5):  # Mon-Fri
            features[f'dow_{day}'] = (df.index.dayofweek == day).astype(int)
            self.feature_types[f'dow_{day}'] = FeatureType.TIME
        
        # Month features
        features['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Day of month features
        features['dom_sin'] = np.sin(2 * np.pi * df.index.day / 31)
        features['dom_cos'] = np.cos(2 * np.pi * df.index.day / 31)
        
        # Quarter features
        features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        features['is_month_end'] = df.index.is_month_end.astype(int)
        
        for col in ['month_sin', 'month_cos', 'dom_sin', 'dom_cos',
                   'is_quarter_end', 'is_month_end']:
            self.feature_types[col] = FeatureType.TIME
    
    def _add_lag_features(self, features: dict) -> None:
        """Add lagged features."""
        if not self.include_lags:
            return
        
        key_features = ['log_return', 'rsi_14', 'macd', 'atr_14']
        
        for feat in key_features:
            if feat not in features:
                continue
            for lag in range(1, self.n_lags + 1):
                lag_name = f'{feat}_lag_{lag}'
                features[lag_name] = features[feat].shift(lag)
                self.feature_types[lag_name] = self.feature_types.get(feat, FeatureType.PRICE)
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable."""
        close = df['Close']
        
        if self.target_type == "returns":
            # Future returns
            target = close.pct_change(self.target_horizon).shift(-self.target_horizon)
        elif self.target_type == "direction":
            # Binary direction (1 = up, 0 = down)
            returns = close.pct_change(self.target_horizon).shift(-self.target_horizon)
            target = (returns > 0).astype(int)
        elif self.target_type == "volatility":
            # Future volatility
            log_returns = np.log(close / close.shift(1))
            target = log_returns.rolling(self.target_horizon).std().shift(-self.target_horizon)
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")
        
        return target
    
    def create_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True
    ) -> FeatureSet:
        """
        Create complete feature set from OHLCV data.
        
        Args:
            df: OHLCV DataFrame
            include_target: Whether to create target variable
            
        Returns:
            FeatureSet with all features
        """
        if len(df) < max(self.lookback_periods) + self.n_lags + 10:
            logger.warning("Insufficient data for feature engineering")
            return FeatureSet(
                features=pd.DataFrame(),
                target=None,
                feature_names=[],
                feature_types={}
            )
        
        # Reset feature types
        self.feature_types = {}
        
        # Create features dictionary
        features = {}
        
        # Add all feature groups
        self._add_price_features(df, features)
        self._add_rolling_features(df, features)
        self._add_momentum_features(df, features)
        self._add_volatility_features(df, features)
        self._add_volume_features(df, features)
        self._add_pattern_features(df, features)
        self._add_time_features(df, features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features, index=df.index)
        
        # Add lag features
        features_dict = feature_df.to_dict('series')
        self._add_lag_features(features_dict)
        feature_df = pd.DataFrame(features_dict, index=df.index)
        
        # Create target
        target = None
        if include_target:
            target = self._create_target(df)
        
        # Drop NaN rows
        valid_idx = feature_df.dropna().index
        if target is not None:
            valid_idx = valid_idx.intersection(target.dropna().index)
        
        feature_df = feature_df.loc[valid_idx]
        if target is not None:
            target = target.loc[valid_idx]
        
        feature_names = list(feature_df.columns)
        
        logger.info(f"Created {len(feature_names)} features from {len(df)} bars")
        
        return FeatureSet(
            features=feature_df,
            target=target,
            feature_names=feature_names,
            feature_types=self.feature_types
        )
    
    def get_feature_importance_groups(self) -> dict[FeatureType, list[str]]:
        """Get features grouped by type."""
        groups = {}
        for feat, ftype in self.feature_types.items():
            if ftype not in groups:
                groups[ftype] = []
            groups[ftype].append(feat)
        return groups
