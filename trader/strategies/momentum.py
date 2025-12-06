"""
Momentum Strategy - Trading strategy based on price momentum.

Implements momentum-based trading following the principle:
"Trend is your friend" - stocks that are rising tend to continue rising.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .base import TradingStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class MomentumStrategy(TradingStrategy):
    """
    Momentum-based trading strategy.
    
    Uses multiple timeframe momentum analysis:
    - Short-term momentum (5-10 days)
    - Medium-term momentum (20-30 days)
    - Long-term momentum (50-100 days)
    
    Also considers:
    - Volume confirmation
    - Rate of Change (ROC)
    - Price acceleration/deceleration
    
    Usage:
        strategy = MomentumStrategy()
        signal = strategy.generate_signal('AAPL', price_data)
    """
    
    def __init__(self,
                 short_period: int = 10,
                 medium_period: int = 30,
                 long_period: int = 90,
                 roc_period: int = 14,
                 volume_ma_period: int = 20,
                 momentum_threshold: float = 0.02,
                 **kwargs):
        """
        Initialize MomentumStrategy.
        
        Args:
            short_period: Short-term momentum period (days)
            medium_period: Medium-term momentum period (days)
            long_period: Long-term momentum period (days)
            roc_period: Rate of Change calculation period
            volume_ma_period: Volume moving average period
            momentum_threshold: Minimum momentum for signal (0.02 = 2%)
        """
        super().__init__(name="MomentumStrategy", **kwargs)
        
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.roc_period = roc_period
        self.volume_ma_period = volume_ma_period
        self.momentum_threshold = momentum_threshold
    
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                        **kwargs) -> Signal:
        """
        Generate trading signal based on momentum indicators.
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
        
        Returns:
            Signal with BUY, SELL, or HOLD recommendation
        """
        min_rows = self.long_period + 10
        if not self.validate_data(data, min_rows=min_rows):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0,
                reasons=[f"Insufficient data (need {min_rows} rows)"]
            )
        
        # Calculate momentum indicators
        df = self._calculate_indicators(data)
        current = df.iloc[-1]
        price = current['Close']
        
        signals = []
        reasons = []
        
        # Multi-timeframe momentum
        short_mom, short_reason = self._analyze_momentum(
            current['mom_short'], "Short-term"
        )
        signals.append(short_mom)
        reasons.append(short_reason)
        
        medium_mom, medium_reason = self._analyze_momentum(
            current['mom_medium'], "Medium-term"
        )
        signals.append(medium_mom)
        reasons.append(medium_reason)
        
        long_mom, long_reason = self._analyze_momentum(
            current['mom_long'], "Long-term"
        )
        signals.append(long_mom)
        reasons.append(long_reason)
        
        # Rate of Change analysis
        roc_signal, roc_reason = self._analyze_roc(current['roc'])
        signals.append(roc_signal)
        reasons.append(roc_reason)
        
        # Volume confirmation
        vol_signal, vol_reason = self._analyze_volume(
            current['Volume'], current['volume_ma'],
            current['mom_short']
        )
        signals.append(vol_signal)
        reasons.append(vol_reason)
        
        # Trend consistency check
        trend_signal, trend_reason = self._check_trend_consistency(
            current['mom_short'], current['mom_medium'], current['mom_long']
        )
        signals.append(trend_signal)
        reasons.append(trend_reason)
        
        # Aggregate signals
        final_signal, confidence = self._aggregate_signals(signals)
        
        # Filter active reasons
        active_reasons = [r for i, r in enumerate(reasons) 
                         if signals[i] != 0 or 'Insufficient' in r]
        if not active_reasons:
            active_reasons = reasons[:3]  # At least show momentum reasons
        
        return Signal(
            symbol=symbol,
            signal_type=final_signal,
            confidence=confidence,
            price=price,
            stop_loss=self.calculate_stop_loss(price, final_signal),
            take_profit=self.calculate_take_profit(price, final_signal),
            reasons=active_reasons,
            metadata={
                'momentum_short': current['mom_short'],
                'momentum_medium': current['mom_medium'],
                'momentum_long': current['mom_long'],
                'roc': current['roc'],
                'volume_ratio': current['Volume'] / current['volume_ma'] if current['volume_ma'] > 0 else 1.0
            }
        )
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        df = data.copy()
        
        # Price momentum (percentage change over period)
        df['mom_short'] = df['Close'].pct_change(periods=self.short_period)
        df['mom_medium'] = df['Close'].pct_change(periods=self.medium_period)
        df['mom_long'] = df['Close'].pct_change(periods=self.long_period)
        
        # Rate of Change
        df['roc'] = ((df['Close'] - df['Close'].shift(self.roc_period)) / 
                     df['Close'].shift(self.roc_period)) * 100
        
        # Volume analysis
        df['volume_ma'] = df['Volume'].rolling(window=self.volume_ma_period).mean()
        
        # Momentum acceleration
        df['mom_acceleration'] = df['mom_short'].diff()
        
        return df
    
    def _analyze_momentum(self, momentum: float, 
                          timeframe: str) -> Tuple[int, str]:
        """Analyze momentum for a given timeframe."""
        if pd.isna(momentum):
            return 0, f"{timeframe} momentum: Insufficient data"
        
        if momentum > self.momentum_threshold:
            return 1, f"{timeframe} momentum: +{momentum:.1%} (bullish)"
        elif momentum < -self.momentum_threshold:
            return -1, f"{timeframe} momentum: {momentum:.1%} (bearish)"
        else:
            return 0, f"{timeframe} momentum: {momentum:.1%} (neutral)"
    
    def _analyze_roc(self, roc: float) -> Tuple[int, str]:
        """Analyze Rate of Change."""
        if pd.isna(roc):
            return 0, "ROC: Insufficient data"
        
        if roc > 5:
            return 1, f"ROC: {roc:.1f}% - strong upward momentum"
        elif roc > 2:
            return 1, f"ROC: {roc:.1f}% - moderate upward momentum"
        elif roc < -5:
            return -1, f"ROC: {roc:.1f}% - strong downward momentum"
        elif roc < -2:
            return -1, f"ROC: {roc:.1f}% - moderate downward momentum"
        else:
            return 0, f"ROC: {roc:.1f}% - low momentum"
    
    def _analyze_volume(self, volume: float, volume_ma: float,
                        momentum: float) -> Tuple[int, str]:
        """Analyze volume for trend confirmation."""
        if pd.isna(volume_ma) or volume_ma == 0:
            return 0, "Volume: Insufficient data"
        
        volume_ratio = volume / volume_ma
        
        # Volume should confirm momentum
        if volume_ratio > 1.5:
            if momentum > 0:
                return 1, f"Volume {volume_ratio:.1f}x average confirms uptrend"
            elif momentum < 0:
                return -1, f"Volume {volume_ratio:.1f}x average confirms downtrend"
        elif volume_ratio < 0.5:
            return 0, f"Low volume ({volume_ratio:.1f}x) - weak signal"
        
        return 0, f"Volume at {volume_ratio:.1f}x average"
    
    def _check_trend_consistency(self, short: float, medium: float, 
                                  long: float) -> Tuple[int, str]:
        """Check if momentum is consistent across timeframes."""
        if pd.isna(short) or pd.isna(medium) or pd.isna(long):
            return 0, "Trend consistency: Insufficient data"
        
        # All positive = strong bullish
        if short > 0 and medium > 0 and long > 0:
            return 1, "Bullish momentum across all timeframes"
        
        # All negative = strong bearish
        if short < 0 and medium < 0 and long < 0:
            return -1, "Bearish momentum across all timeframes"
        
        # Mixed signals
        if short > 0 and medium < 0:
            return 0, "Short-term reversal in downtrend - caution"
        elif short < 0 and medium > 0:
            return 0, "Short-term pullback in uptrend - watch"
        
        return 0, "Mixed momentum signals across timeframes"
    
    def _aggregate_signals(self, signals: List[int]) -> Tuple[SignalType, float]:
        """Aggregate signals with weighted importance."""
        valid_signals = [s for s in signals if s is not None]
        
        if not valid_signals:
            return SignalType.HOLD, 0.0
        
        # Weight: volume and trend consistency are most important
        weights = [0.15, 0.2, 0.15, 0.2, 0.15, 0.15]  # Adjust based on signal order
        
        if len(valid_signals) != len(weights):
            weights = [1.0 / len(valid_signals)] * len(valid_signals)
        
        weighted_sum = sum(s * w for s, w in zip(valid_signals, weights))
        
        # Calculate confidence from agreement
        bullish = sum(1 for s in valid_signals if s > 0)
        bearish = sum(1 for s in valid_signals if s < 0)
        total = len(valid_signals)
        
        if weighted_sum > 0.2:
            confidence = bullish / total
            return SignalType.BUY, confidence
        elif weighted_sum < -0.2:
            confidence = bearish / total
            return SignalType.SELL, confidence
        else:
            return SignalType.HOLD, 0.5


if __name__ == "__main__":
    # Test
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    # Fetch test data
    stock = yf.Ticker('TSLA')
    data = stock.history(period='1y')
    
    # Generate signal
    strategy = MomentumStrategy()
    signal = strategy.generate_signal('TSLA', data)
    
    print(f"\nMomentum Signal for TSLA:")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Price: ${signal.price:.2f}")
    print(f"  Stop Loss: ${signal.stop_loss:.2f}")
    print(f"  Take Profit: ${signal.take_profit:.2f}")
    print(f"\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
    print(f"\nMetadata:")
    for key, value in signal.metadata.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 'momentum' in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
