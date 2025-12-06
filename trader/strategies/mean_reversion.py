"""
Mean Reversion Strategy - Trading based on price returning to mean.

Principle: Prices tend to revert to their historical average.
Buy when price is significantly below the mean, sell when above.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .base import TradingStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class MeanReversionStrategy(TradingStrategy):
    """
    Mean Reversion trading strategy.
    
    Uses statistical measures to identify when price deviates
    significantly from its mean and is likely to revert.
    
    Indicators used:
    - Z-Score: Standard deviations from mean
    - Bollinger Bands %B: Position within bands
    - RSI extremes: Overbought/oversold conditions
    - Price distance from moving averages
    
    Usage:
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal('AAPL', price_data)
    """
    
    def __init__(self,
                 lookback_period: int = 20,
                 z_score_threshold: float = 2.0,
                 rsi_period: int = 14,
                 rsi_oversold: int = 25,
                 rsi_overbought: int = 75,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 **kwargs):
        """
        Initialize Mean Reversion Strategy.
        
        Args:
            lookback_period: Period for mean calculation
            z_score_threshold: Z-score threshold for signal (2.0 = 2 std deviations)
            rsi_period: RSI calculation period
            rsi_oversold: RSI level indicating oversold (buy signal)
            rsi_overbought: RSI level indicating overbought (sell signal)
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviations
        """
        super().__init__(name="MeanReversionStrategy", **kwargs)
        
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                        **kwargs) -> Signal:
        """
        Generate trading signal based on mean reversion indicators.
        """
        min_rows = max(self.lookback_period, self.bb_period, self.rsi_period) + 10
        if not self.validate_data(data, min_rows=min_rows):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=float(data['Close'].iloc[-1]) if not data.empty else 0.0,
                reasons=[f"Insufficient data (need {min_rows} rows)"]
            )
        
        df = self._calculate_indicators(data)
        
        # Extract scalar values from the last row using .iloc[-1] with column indexing
        price: float = df['Close'].iloc[-1]
        z_score_raw = df['z_score'].iloc[-1]
        percent_b_raw = df['percent_b'].iloc[-1]
        rsi_raw = df['rsi'].iloc[-1]
        sma_raw = df['sma'].iloc[-1]
        std_raw = df['std'].iloc[-1]
        
        # Convert to proper floats with NaN handling
        z_score_val = float(z_score_raw) if pd.notna(z_score_raw) else 0.0
        percent_b_val = float(percent_b_raw) if pd.notna(percent_b_raw) else 0.0
        rsi_val = float(rsi_raw) if pd.notna(rsi_raw) else 50.0
        sma_val = float(sma_raw) if pd.notna(sma_raw) else price
        std_val = float(std_raw) if pd.notna(std_raw) else 0.0
        
        signals = []
        reasons = []
        
        # Z-Score analysis
        z_signal, z_reason = self._analyze_zscore(z_score_val)
        signals.append(z_signal)
        reasons.append(z_reason)
        
        # Bollinger %B analysis
        bb_signal, bb_reason = self._analyze_percent_b(percent_b_val)
        signals.append(bb_signal)
        reasons.append(bb_reason)
        
        # RSI extremes
        rsi_signal, rsi_reason = self._analyze_rsi_extremes(rsi_val)
        signals.append(rsi_signal)
        reasons.append(rsi_reason)
        
        # Distance from MA
        ma_signal, ma_reason = self._analyze_ma_distance(price, sma_val, std_val)
        signals.append(ma_signal)
        reasons.append(ma_reason)
        
        # Mean reversion probability
        prob_signal, prob_reason = self._calculate_reversion_probability(df)
        signals.append(prob_signal)
        reasons.append(prob_reason)
        
        # Aggregate
        final_signal, confidence = self._aggregate_signals(signals)
        active_reasons = [r for i, r in enumerate(reasons) if signals[i] != 0]
        
        return Signal(
            symbol=symbol,
            signal_type=final_signal,
            confidence=confidence,
            price=price,
            stop_loss=self.calculate_stop_loss(price, final_signal),
            take_profit=self.calculate_take_profit(price, final_signal),
            reasons=active_reasons if active_reasons else reasons[:3],
            metadata={
                'z_score': z_score_val,
                'percent_b': percent_b_val,
                'rsi': rsi_val,
                'distance_from_mean': (price - sma_val) / sma_val if sma_val != 0 else 0.0
            }
        )
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators."""
        df = data.copy()
        
        # Simple Moving Average and Standard Deviation
        df['sma'] = df['Close'].rolling(window=self.lookback_period).mean()
        df['std'] = df['Close'].rolling(window=self.lookback_period).std()
        
        # Z-Score
        df['z_score'] = (df['Close'] - df['sma']) / df['std']
        
        # Bollinger Bands
        df['bb_upper'] = df['sma'] + (self.bb_std * df['std'])
        df['bb_lower'] = df['sma'] - (self.bb_std * df['std'])
        
        # Percent B (%B) - position within Bollinger Bands
        df['percent_b'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _analyze_zscore(self, z_score: float) -> Tuple[int, str]:
        """Analyze Z-Score for mean reversion signal."""
        if pd.isna(z_score):
            return 0, "Z-Score: Insufficient data"
        
        if z_score < -self.z_score_threshold:
            return 1, f"Z-Score ({z_score:.2f}) below -{self.z_score_threshold} - price likely to revert UP"
        elif z_score > self.z_score_threshold:
            return -1, f"Z-Score ({z_score:.2f}) above +{self.z_score_threshold} - price likely to revert DOWN"
        elif abs(z_score) < 0.5:
            return 0, f"Z-Score ({z_score:.2f}) near mean - no reversion expected"
        else:
            return 0, f"Z-Score ({z_score:.2f}) within normal range"
    
    def _analyze_percent_b(self, percent_b: float) -> Tuple[int, str]:
        """Analyze Bollinger %B for mean reversion."""
        if pd.isna(percent_b):
            return 0, "%B: Insufficient data"
        
        if percent_b < 0:
            return 1, f"%B ({percent_b:.2f}) below lower band - oversold, expect bounce"
        elif percent_b > 1:
            return -1, f"%B ({percent_b:.2f}) above upper band - overbought, expect pullback"
        elif percent_b < 0.2:
            return 1, f"%B ({percent_b:.2f}) near lower band - potential mean reversion up"
        elif percent_b > 0.8:
            return -1, f"%B ({percent_b:.2f}) near upper band - potential mean reversion down"
        else:
            return 0, f"%B ({percent_b:.2f}) near middle band"
    
    def _analyze_rsi_extremes(self, rsi: float) -> Tuple[int, str]:
        """Analyze RSI for extreme conditions."""
        if pd.isna(rsi):
            return 0, "RSI: Insufficient data"
        
        if rsi < self.rsi_oversold:
            return 1, f"RSI ({rsi:.1f}) extremely oversold - reversal likely"
        elif rsi > self.rsi_overbought:
            return -1, f"RSI ({rsi:.1f}) extremely overbought - reversal likely"
        else:
            return 0, f"RSI ({rsi:.1f}) in normal range"
    
    def _analyze_ma_distance(self, price: float, sma: float, 
                             std: float) -> Tuple[int, str]:
        """Analyze distance from moving average."""
        if pd.isna(sma) or pd.isna(std) or std == 0:
            return 0, "MA Distance: Insufficient data"
        
        distance_pct = (price - sma) / sma * 100
        
        if distance_pct < -5:
            return 1, f"Price {distance_pct:.1f}% below MA - mean reversion opportunity"
        elif distance_pct > 5:
            return -1, f"Price {distance_pct:.1f}% above MA - mean reversion opportunity"
        else:
            return 0, f"Price {distance_pct:.1f}% from MA - near equilibrium"
    
    def _calculate_reversion_probability(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Calculate historical reversion probability."""
        if len(df) < 50:
            return 0, "Reversion probability: Insufficient history"
        
        # Count how often price reverted after extreme moves
        z_scores = df['z_score'].dropna()
        
        extreme_low = z_scores < -1.5
        extreme_high = z_scores > 1.5
        
        # Check if price moved toward mean in next 5 days after extremes
        reversions = 0
        total_extremes = 0
        
        for i in range(len(z_scores) - 5):
            if extreme_low.iloc[i]:
                total_extremes += 1
                if z_scores.iloc[i+5] > z_scores.iloc[i]:
                    reversions += 1
            elif extreme_high.iloc[i]:
                total_extremes += 1
                if z_scores.iloc[i+5] < z_scores.iloc[i]:
                    reversions += 1
        
        if total_extremes > 0:
            prob = reversions / total_extremes
            if prob > 0.6:
                current_z = z_scores.iloc[-1]
                if current_z < -1.5:
                    return 1, f"Historical reversion rate: {prob:.0%} - favors upward reversion"
                elif current_z > 1.5:
                    return -1, f"Historical reversion rate: {prob:.0%} - favors downward reversion"
        
        return 0, "No strong historical reversion pattern"
    
    def _aggregate_signals(self, signals: List[int]) -> Tuple[SignalType, float]:
        """Aggregate signals for mean reversion."""
        valid_signals = [s for s in signals if s is not None]
        
        if not valid_signals:
            return SignalType.HOLD, 0.0
        
        avg_signal = sum(valid_signals) / len(valid_signals)
        agreement = sum(1 for s in valid_signals if s == np.sign(avg_signal)) / len(valid_signals)
        
        # Mean reversion requires stronger confirmation
        if avg_signal > 0.4:
            return SignalType.BUY, min(agreement, 0.9)
        elif avg_signal < -0.4:
            return SignalType.SELL, min(agreement, 0.9)
        else:
            return SignalType.HOLD, 0.5


if __name__ == "__main__":
    import yfinance as yf
    logging.basicConfig(level=logging.INFO)
    
    data = yf.Ticker('AAPL').history(period='6mo')
    strategy = MeanReversionStrategy()
    signal = strategy.generate_signal('AAPL', data)
    
    print(f"\nMean Reversion Signal for AAPL:")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Z-Score: {signal.metadata.get('z_score', 'N/A'):.2f}")
    print(f"\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
