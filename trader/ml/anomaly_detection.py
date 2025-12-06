"""
Anomaly Detection Module for Trading.

Detects unusual patterns in market data:
- Price anomalies (flash crashes, spikes)
- Volume anomalies
- Volatility regime changes
- Correlation breakdown
- Order flow anomalies
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of market anomalies."""
    PRICE_SPIKE = "price_spike"
    PRICE_CRASH = "price_crash"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLATILITY_CRUSH = "volatility_crush"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    TREND_REVERSAL = "trend_reversal"
    REGIME_CHANGE = "regime_change"
    OUTLIER = "outlier"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: datetime
    symbol: str
    value: float
    expected_value: float
    z_score: float
    description: str
    confidence: float
    additional_data: dict = field(default_factory=dict)
    
    @property
    def deviation_pct(self) -> float:
        """Calculate percentage deviation from expected."""
        if self.expected_value == 0:
            return 0.0
        return (self.value - self.expected_value) / abs(self.expected_value) * 100


@dataclass
class AnomalyAlert:
    """Alert for detected anomaly."""
    anomaly: Anomaly
    action_required: bool
    suggested_actions: list[str]
    risk_impact: str
    related_anomalies: list[Anomaly] = field(default_factory=list)


class StatisticalDetector:
    """
    Statistical anomaly detection using z-scores and quantiles.
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        quantile_threshold: float = 0.01,
        lookback_window: int = 60,
    ):
        """
        Initialize Statistical Detector.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection
            quantile_threshold: Quantile threshold (e.g., 0.01 = 1%)
            lookback_window: Days for calculating statistics
        """
        self.z_threshold = z_threshold
        self.quantile_threshold = quantile_threshold
        self.lookback_window = lookback_window
    
    def detect_zscore_anomalies(
        self,
        series: pd.Series,
        threshold: Optional[float] = None,
    ) -> list[tuple[datetime, float, float]]:
        """
        Detect anomalies using z-score.
        
        Args:
            series: Time series data
            threshold: Z-score threshold (uses default if None)
            
        Returns:
            List of (timestamp, value, z_score) tuples
        """
        threshold = threshold or self.z_threshold
        
        # Calculate rolling statistics
        rolling_mean = series.rolling(window=self.lookback_window).mean()
        rolling_std = series.rolling(window=self.lookback_window).std()
        
        # Calculate z-scores
        z_scores = (series - rolling_mean) / (rolling_std + 1e-10)
        
        # Find anomalies
        anomalies = []
        for idx in z_scores.index:
            z = z_scores.loc[idx]
            if abs(z) > threshold:
                anomalies.append((
                    idx if isinstance(idx, datetime) else datetime.now(),
                    series.loc[idx],
                    z,
                ))
        
        return anomalies
    
    def detect_quantile_anomalies(
        self,
        series: pd.Series,
        lower_quantile: Optional[float] = None,
        upper_quantile: Optional[float] = None,
    ) -> list[tuple[datetime, float, str]]:
        """
        Detect anomalies using quantile thresholds.
        
        Args:
            series: Time series data
            lower_quantile: Lower quantile threshold
            upper_quantile: Upper quantile threshold
            
        Returns:
            List of (timestamp, value, direction) tuples
        """
        lower_q = lower_quantile or self.quantile_threshold
        upper_q = upper_quantile or (1 - self.quantile_threshold)
        
        lower_threshold = series.quantile(lower_q)
        upper_threshold = series.quantile(upper_q)
        
        anomalies = []
        for idx, value in series.items():
            if value < lower_threshold:
                anomalies.append((
                    idx if isinstance(idx, datetime) else datetime.now(),
                    value,
                    "below",
                ))
            elif value > upper_threshold:
                anomalies.append((
                    idx if isinstance(idx, datetime) else datetime.now(),
                    value,
                    "above",
                ))
        
        return anomalies


class IsolationForestDetector:
    """
    Anomaly detection using Isolation Forest algorithm.
    
    Uses scikit-learn's IsolationForest when available.
    """
    
    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 100,
        max_features: float = 1.0,
    ):
        """
        Initialize Isolation Forest Detector.
        
        Args:
            contamination: Expected proportion of outliers
            n_estimators: Number of trees in the forest
            max_features: Features to draw from X to train each tree
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.model = None
        self._sklearn_available = False
        
        try:
            from sklearn.ensemble import IsolationForest
            self._sklearn_available = True
        except ImportError:
            logger.warning("scikit-learn not available, using simple method")
    
    def fit(self, data: np.ndarray) -> None:
        """
        Fit the Isolation Forest model.
        
        Args:
            data: Training data (n_samples, n_features)
        """
        if not self._sklearn_available:
            return
        
        from sklearn.ensemble import IsolationForest
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=42,
        )
        self.model.fit(data)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            data: Data to predict on (n_samples, n_features)
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        if not self._sklearn_available or self.model is None:
            # Fallback to simple z-score based detection
            z_scores = np.abs(stats.zscore(data, axis=0))
            return np.where(np.any(z_scores > 3, axis=1), -1, 1)
        
        return self.model.predict(data)
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores.
        
        Args:
            data: Data to score
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self._sklearn_available or self.model is None:
            z_scores = np.abs(stats.zscore(data, axis=0))
            return -np.max(z_scores, axis=1)
        
        return self.model.score_samples(data)


class MarketAnomalyDetector:
    """
    Comprehensive market anomaly detection.
    
    Combines multiple detection methods:
    - Statistical (z-score, quantile)
    - Machine learning (Isolation Forest)
    - Rule-based (gaps, limits)
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        volume_multiplier: float = 3.0,
        gap_threshold: float = 0.02,
        volatility_lookback: int = 20,
        correlation_window: int = 60,
    ):
        """
        Initialize Market Anomaly Detector.
        
        Args:
            z_threshold: Z-score threshold for anomalies
            volume_multiplier: Volume spike multiplier
            gap_threshold: Gap threshold (as decimal, e.g., 0.02 = 2%)
            volatility_lookback: Days for volatility calculation
            correlation_window: Days for correlation calculation
        """
        self.z_threshold = z_threshold
        self.volume_multiplier = volume_multiplier
        self.gap_threshold = gap_threshold
        self.volatility_lookback = volatility_lookback
        self.correlation_window = correlation_window
        
        self.statistical_detector = StatisticalDetector(z_threshold=z_threshold)
        self.isolation_detector = IsolationForestDetector()
    
    def detect_all_anomalies(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> list[Anomaly]:
        """
        Run all anomaly detection methods on OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Normalize column names
        df = self._normalize_columns(df)
        
        # Price anomalies
        price_anomalies = self.detect_price_anomalies(df, symbol)
        anomalies.extend(price_anomalies)
        
        # Volume anomalies
        volume_anomalies = self.detect_volume_anomalies(df, symbol)
        anomalies.extend(volume_anomalies)
        
        # Gap anomalies
        gap_anomalies = self.detect_gaps(df, symbol)
        anomalies.extend(gap_anomalies)
        
        # Volatility anomalies
        volatility_anomalies = self.detect_volatility_anomalies(df, symbol)
        anomalies.extend(volatility_anomalies)
        
        # Sort by timestamp
        anomalies.sort(key=lambda x: x.timestamp, reverse=True)
        
        return anomalies
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df
    
    def detect_price_anomalies(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> list[Anomaly]:
        """
        Detect price anomalies.
        
        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            
        Returns:
            List of price anomalies
        """
        anomalies = []
        
        if 'close' not in df.columns:
            return anomalies
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        # Detect z-score anomalies
        zscore_anomalies = self.statistical_detector.detect_zscore_anomalies(
            returns, threshold=self.z_threshold
        )
        
        for timestamp, value, z_score in zscore_anomalies:
            if z_score > 0:
                anomaly_type = AnomalyType.PRICE_SPIKE
                description = f"Price spiked {value*100:.2f}% (z-score: {z_score:.2f})"
            else:
                anomaly_type = AnomalyType.PRICE_CRASH
                description = f"Price crashed {value*100:.2f}% (z-score: {z_score:.2f})"
            
            severity = self._calculate_severity(abs(z_score))
            
            anomalies.append(Anomaly(
                anomaly_type=anomaly_type,
                severity=severity,
                timestamp=timestamp,
                symbol=symbol,
                value=value,
                expected_value=0.0,  # Expected return
                z_score=z_score,
                description=description,
                confidence=min(1.0, abs(z_score) / 5),
            ))
        
        return anomalies
    
    def detect_volume_anomalies(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> list[Anomaly]:
        """
        Detect volume anomalies.
        
        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            
        Returns:
            List of volume anomalies
        """
        anomalies = []
        
        if 'volume' not in df.columns:
            return anomalies
        
        volume = df['volume']
        
        # Calculate rolling average volume
        avg_volume = volume.rolling(window=20).mean()
        
        # Find volume spikes
        volume_ratio = volume / (avg_volume + 1)
        
        for idx, ratio in volume_ratio.items():
            if ratio > self.volume_multiplier:
                z_score = (ratio - 1) / 0.5  # Approximate z-score
                
                timestamp = idx if isinstance(idx, datetime) else datetime.now()
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    severity=self._calculate_severity(z_score),
                    timestamp=timestamp,
                    symbol=symbol,
                    value=volume.loc[idx],
                    expected_value=avg_volume.loc[idx] if pd.notna(avg_volume.loc[idx]) else 0,
                    z_score=z_score,
                    description=f"Volume {ratio:.1f}x above average",
                    confidence=min(1.0, ratio / (self.volume_multiplier * 2)),
                ))
        
        return anomalies
    
    def detect_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> list[Anomaly]:
        """
        Detect price gaps.
        
        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            
        Returns:
            List of gap anomalies
        """
        anomalies = []
        
        if 'open' not in df.columns or 'close' not in df.columns:
            return anomalies
        
        # Calculate gaps (open vs previous close)
        prev_close = df['close'].shift(1)
        gap_pct = (df['open'] - prev_close) / prev_close
        
        for idx, gap in gap_pct.items():
            if pd.isna(gap):
                continue
            
            if abs(gap) > self.gap_threshold:
                timestamp = idx if isinstance(idx, datetime) else datetime.now()
                
                if gap > 0:
                    anomaly_type = AnomalyType.GAP_UP
                    description = f"Gap up {gap*100:.2f}%"
                else:
                    anomaly_type = AnomalyType.GAP_DOWN
                    description = f"Gap down {gap*100:.2f}%"
                
                z_score = gap / 0.01  # Approximate z-score
                
                anomalies.append(Anomaly(
                    anomaly_type=anomaly_type,
                    severity=self._calculate_severity(abs(z_score)),
                    timestamp=timestamp,
                    symbol=symbol,
                    value=df['open'].loc[idx],
                    expected_value=prev_close.loc[idx] if pd.notna(prev_close.loc[idx]) else 0,
                    z_score=z_score,
                    description=description,
                    confidence=min(1.0, abs(gap) / (self.gap_threshold * 3)),
                ))
        
        return anomalies
    
    def detect_volatility_anomalies(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> list[Anomaly]:
        """
        Detect volatility regime changes.
        
        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            
        Returns:
            List of volatility anomalies
        """
        anomalies = []
        
        if 'close' not in df.columns:
            return anomalies
        
        # Calculate returns and rolling volatility
        returns = df['close'].pct_change()
        rolling_vol = returns.rolling(window=self.volatility_lookback).std() * np.sqrt(252)
        long_term_vol = returns.rolling(window=60).std() * np.sqrt(252)
        
        # Detect volatility regime changes
        vol_ratio = rolling_vol / (long_term_vol + 1e-10)
        
        for idx, ratio in vol_ratio.items():
            if pd.isna(ratio):
                continue
            
            timestamp = idx if isinstance(idx, datetime) else datetime.now()
            
            if ratio > 2.0:
                z_score = (ratio - 1) / 0.3
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VOLATILITY_SPIKE,
                    severity=self._calculate_severity(z_score),
                    timestamp=timestamp,
                    symbol=symbol,
                    value=rolling_vol.loc[idx] if pd.notna(rolling_vol.loc[idx]) else 0,
                    expected_value=long_term_vol.loc[idx] if pd.notna(long_term_vol.loc[idx]) else 0,
                    z_score=z_score,
                    description=f"Volatility {ratio:.1f}x above normal",
                    confidence=min(1.0, ratio / 4),
                ))
            elif ratio < 0.5:
                z_score = (ratio - 1) / 0.3
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VOLATILITY_CRUSH,
                    severity=AnomalySeverity.MEDIUM,
                    timestamp=timestamp,
                    symbol=symbol,
                    value=rolling_vol.loc[idx] if pd.notna(rolling_vol.loc[idx]) else 0,
                    expected_value=long_term_vol.loc[idx] if pd.notna(long_term_vol.loc[idx]) else 0,
                    z_score=z_score,
                    description=f"Volatility {ratio:.1f}x below normal",
                    confidence=min(1.0, (1 - ratio) / 0.5),
                ))
        
        return anomalies
    
    def detect_correlation_breakdown(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        symbol1: str,
        symbol2: str,
        expected_correlation: float = 0.8,
    ) -> list[Anomaly]:
        """
        Detect correlation breakdown between two assets.
        
        Args:
            df1: OHLCV for first asset
            df2: OHLCV for second asset
            symbol1: First symbol
            symbol2: Second symbol
            expected_correlation: Expected correlation level
            
        Returns:
            List of correlation anomalies
        """
        anomalies = []
        
        # Normalize and align data
        df1 = self._normalize_columns(df1)
        df2 = self._normalize_columns(df2)
        
        if 'close' not in df1.columns or 'close' not in df2.columns:
            return anomalies
        
        returns1 = df1['close'].pct_change()
        returns2 = df2['close'].pct_change()
        
        # Align data
        aligned = pd.concat([returns1, returns2], axis=1, join='inner')
        aligned.columns = [symbol1, symbol2]
        
        if len(aligned) < self.correlation_window:
            return anomalies
        
        # Calculate rolling correlation
        rolling_corr = aligned[symbol1].rolling(
            window=self.correlation_window
        ).corr(aligned[symbol2])
        
        # Detect significant deviations
        for idx, corr in rolling_corr.items():
            if pd.isna(corr):
                continue
            
            deviation = abs(corr - expected_correlation)
            
            if deviation > 0.3:
                timestamp = idx if isinstance(idx, datetime) else datetime.now()
                z_score = deviation / 0.1
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.CORRELATION_BREAKDOWN,
                    severity=self._calculate_severity(z_score),
                    timestamp=timestamp,
                    symbol=f"{symbol1}/{symbol2}",
                    value=corr,
                    expected_value=expected_correlation,
                    z_score=z_score,
                    description=f"Correlation broke down: {corr:.2f} vs expected {expected_correlation:.2f}",
                    confidence=min(1.0, deviation),
                    additional_data={
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                    },
                ))
        
        return anomalies
    
    def detect_regime_change(
        self,
        df: pd.DataFrame,
        symbol: str,
        window: int = 20,
    ) -> list[Anomaly]:
        """
        Detect market regime changes.
        
        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol
            window: Window for regime detection
            
        Returns:
            List of regime change anomalies
        """
        anomalies = []
        
        df = self._normalize_columns(df)
        
        if 'close' not in df.columns:
            return anomalies
        
        returns = df['close'].pct_change()
        
        # Calculate regime indicators
        rolling_mean = returns.rolling(window=window).mean()
        rolling_vol = returns.rolling(window=window).std()
        
        # Detect trend changes
        trend = rolling_mean.rolling(window=5).apply(lambda x: 1 if x.mean() > 0 else -1)
        trend_change = trend.diff().abs()
        
        # Detect volatility regime changes
        vol_ma = rolling_vol.rolling(window=20).mean()
        vol_regime = rolling_vol / (vol_ma + 1e-10)
        
        for idx in df.index:
            timestamp = idx if isinstance(idx, datetime) else datetime.now()
            
            # Check for trend reversal
            if pd.notna(trend_change.get(idx)) and trend_change.get(idx) == 2:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.TREND_REVERSAL,
                    severity=AnomalySeverity.MEDIUM,
                    timestamp=timestamp,
                    symbol=symbol,
                    value=rolling_mean.get(idx, 0) if pd.notna(rolling_mean.get(idx)) else 0,
                    expected_value=0.0,
                    z_score=2.0,
                    description="Trend reversal detected",
                    confidence=0.7,
                ))
            
            # Check for volatility regime change
            vr = vol_regime.get(idx)
            if pd.notna(vr) and (vr > 2.0 or vr < 0.5):
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.REGIME_CHANGE,
                    severity=AnomalySeverity.HIGH if vr > 2.0 else AnomalySeverity.MEDIUM,
                    timestamp=timestamp,
                    symbol=symbol,
                    value=rolling_vol.get(idx, 0) if pd.notna(rolling_vol.get(idx)) else 0,
                    expected_value=vol_ma.get(idx, 0) if pd.notna(vol_ma.get(idx)) else 0,
                    z_score=abs(vr - 1) / 0.3,
                    description=f"Volatility regime change ({vr:.1f}x)",
                    confidence=min(1.0, abs(vr - 1) / 2),
                ))
        
        return anomalies
    
    def _calculate_severity(self, z_score: float) -> AnomalySeverity:
        """Calculate anomaly severity from z-score."""
        abs_z = abs(z_score)
        
        if abs_z >= 5:
            return AnomalySeverity.CRITICAL
        elif abs_z >= 4:
            return AnomalySeverity.HIGH
        elif abs_z >= 3:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def generate_alert(
        self,
        anomaly: Anomaly,
        portfolio_context: Optional[dict] = None,
    ) -> AnomalyAlert:
        """
        Generate an alert for an anomaly.
        
        Args:
            anomaly: Detected anomaly
            portfolio_context: Optional portfolio context
            
        Returns:
            AnomalyAlert with suggested actions
        """
        suggested_actions = []
        risk_impact = "low"
        action_required = False
        
        if anomaly.severity in [AnomalySeverity.CRITICAL, AnomalySeverity.HIGH]:
            action_required = True
        
        # Generate suggestions based on anomaly type
        if anomaly.anomaly_type == AnomalyType.PRICE_CRASH:
            risk_impact = "high"
            suggested_actions.extend([
                "Review stop-loss levels",
                "Check for breaking news",
                "Consider hedging or position reduction",
            ])
        
        elif anomaly.anomaly_type == AnomalyType.PRICE_SPIKE:
            risk_impact = "medium"
            suggested_actions.extend([
                "Consider taking partial profits",
                "Adjust trailing stops",
                "Monitor for reversal signals",
            ])
        
        elif anomaly.anomaly_type == AnomalyType.VOLUME_SPIKE:
            risk_impact = "medium"
            suggested_actions.extend([
                "Analyze order flow for direction",
                "Check for institutional activity",
                "Monitor for breakout confirmation",
            ])
        
        elif anomaly.anomaly_type == AnomalyType.VOLATILITY_SPIKE:
            risk_impact = "high"
            suggested_actions.extend([
                "Reduce position sizes",
                "Widen stop-losses",
                "Consider volatility hedges",
            ])
        
        elif anomaly.anomaly_type == AnomalyType.GAP_UP:
            suggested_actions.extend([
                "Consider gap-fill strategy",
                "Adjust entry prices for orders",
                "Monitor for continuation",
            ])
        
        elif anomaly.anomaly_type == AnomalyType.GAP_DOWN:
            risk_impact = "high" if anomaly.severity == AnomalySeverity.HIGH else "medium"
            suggested_actions.extend([
                "Review overnight news",
                "Check for gap-fill potential",
                "Reassess position sizing",
            ])
        
        elif anomaly.anomaly_type == AnomalyType.CORRELATION_BREAKDOWN:
            suggested_actions.extend([
                "Review pair trade positions",
                "Assess hedge effectiveness",
                "Investigate fundamental changes",
            ])
        
        elif anomaly.anomaly_type == AnomalyType.REGIME_CHANGE:
            suggested_actions.extend([
                "Adjust strategy parameters",
                "Review volatility targeting",
                "Consider regime-appropriate strategies",
            ])
        
        return AnomalyAlert(
            anomaly=anomaly,
            action_required=action_required,
            suggested_actions=suggested_actions,
            risk_impact=risk_impact,
        )


class RealTimeAnomalyMonitor:
    """
    Real-time anomaly monitoring for live trading.
    """
    
    def __init__(
        self,
        detector: Optional[MarketAnomalyDetector] = None,
        alert_callback: Optional[Callable[[AnomalyAlert], None]] = None,
    ):
        """
        Initialize Real-Time Monitor.
        
        Args:
            detector: Anomaly detector instance
            alert_callback: Callback function for alerts
        """
        self.detector = detector or MarketAnomalyDetector()
        self.alert_callback = alert_callback
        self.recent_anomalies: dict[str, list[Anomaly]] = {}
        self.alert_history: list[AnomalyAlert] = []
    
    def process_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[list[AnomalyAlert]]:
        """
        Process a single tick for anomaly detection.
        
        Args:
            symbol: Stock symbol
            price: Current price
            volume: Current volume
            timestamp: Tick timestamp
            
        Returns:
            List of alerts if any anomalies detected
        """
        timestamp = timestamp or datetime.now()
        
        # Initialize symbol tracking
        if symbol not in self.recent_anomalies:
            self.recent_anomalies[symbol] = []
        
        # Simple tick-based detection (for demonstration)
        # In production, would maintain rolling windows
        
        alerts = []
        
        # This would be enhanced with actual tick processing
        # For now, return empty (full implementation would use rolling data)
        
        return alerts if alerts else None
    
    def process_bar(
        self,
        symbol: str,
        ohlcv: dict,
        history: Optional[pd.DataFrame] = None,
    ) -> list[AnomalyAlert]:
        """
        Process a new bar for anomaly detection.
        
        Args:
            symbol: Stock symbol
            ohlcv: OHLCV data for the bar
            history: Historical data for context
            
        Returns:
            List of alerts
        """
        alerts = []
        
        if history is None:
            return alerts
        
        # Append new bar to history
        new_row = pd.DataFrame([ohlcv])
        combined = pd.concat([history, new_row], ignore_index=True)
        
        # Run anomaly detection
        anomalies = self.detector.detect_all_anomalies(combined, symbol)
        
        # Filter to only new anomalies (last bar)
        recent = [a for a in anomalies if a.timestamp == combined.index[-1]]
        
        for anomaly in recent:
            alert = self.detector.generate_alert(anomaly)
            alerts.append(alert)
            self.alert_history.append(alert)
            
            if self.alert_callback:
                self.alert_callback(alert)
        
        return alerts
    
    def get_summary(self, symbol: Optional[str] = None) -> dict:
        """
        Get summary of recent anomalies.
        
        Args:
            symbol: Filter by symbol (all if None)
            
        Returns:
            Summary dictionary
        """
        if symbol:
            anomalies = self.recent_anomalies.get(symbol, [])
        else:
            anomalies = [a for anomalies in self.recent_anomalies.values() for a in anomalies]
        
        return {
            "total_anomalies": len(anomalies),
            "by_type": self._count_by_type(anomalies),
            "by_severity": self._count_by_severity(anomalies),
            "recent_alerts": len(self.alert_history),
            "critical_alerts": sum(
                1 for a in self.alert_history 
                if a.anomaly.severity == AnomalySeverity.CRITICAL
            ),
        }
    
    def _count_by_type(self, anomalies: list[Anomaly]) -> dict[str, int]:
        """Count anomalies by type."""
        counts: dict[str, int] = {}
        for a in anomalies:
            type_name = a.anomaly_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def _count_by_severity(self, anomalies: list[Anomaly]) -> dict[str, int]:
        """Count anomalies by severity."""
        counts: dict[str, int] = {}
        for a in anomalies:
            severity = a.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts


def scan_for_anomalies(
    symbols: list[str],
    period: str = "6mo",
    fetcher=None,
) -> dict[str, list[Anomaly]]:
    """
    Convenience function to scan multiple symbols for anomalies.
    
    Args:
        symbols: List of symbols to scan
        period: Data period
        fetcher: Optional DataFetcher
        
    Returns:
        Dictionary of symbol -> list of anomalies
    """
    if fetcher is None:
        try:
            from trader.data.fetcher import DataFetcher
            fetcher = DataFetcher()
        except ImportError:
            raise ImportError("DataFetcher required")
    
    detector = MarketAnomalyDetector()
    results = {}
    
    for symbol in symbols:
        try:
            df = fetcher.get_stock_data(symbol, period=period)
            if df is not None and not df.empty:
                anomalies = detector.detect_all_anomalies(df, symbol)
                results[symbol] = anomalies
        except Exception as e:
            logger.error(f"Failed to scan {symbol}: {e}")
            results[symbol] = []
    
    return results
