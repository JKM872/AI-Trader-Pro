"""
Breakout Strategy - Trading based on price breaking key levels.

Principle: When price breaks through significant support/resistance levels
with strong volume, it tends to continue in that direction.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .base import TradingStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class BreakoutStrategy(TradingStrategy):
    """
    Breakout trading strategy.
    
    Identifies and trades breakouts from:
    - Consolidation ranges
    - Support/Resistance levels
    - Channel boundaries
    - Volatility squeezes (Bollinger Band squeeze)
    
    Key confirmations:
    - Volume surge on breakout
    - Momentum confirmation
    - Multiple timeframe alignment
    
    Usage:
        strategy = BreakoutStrategy()
        signal = strategy.generate_signal('AAPL', price_data)
    """
    
    def __init__(self,
                 lookback_period: int = 20,
                 breakout_threshold: float = 0.02,
                 volume_surge_multiplier: float = 1.5,
                 atr_period: int = 14,
                 squeeze_threshold: float = 0.1,
                 consolidation_periods: int = 10,
                 **kwargs):
        """
        Initialize Breakout Strategy.
        
        Args:
            lookback_period: Period for support/resistance calculation
            breakout_threshold: Minimum price move for breakout (2% default)
            volume_surge_multiplier: Volume must be X times average
            atr_period: ATR period for volatility
            squeeze_threshold: Bollinger Band width threshold for squeeze
            consolidation_periods: Min periods of consolidation before breakout
        """
        super().__init__(name="BreakoutStrategy", **kwargs)
        
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volume_surge_multiplier = volume_surge_multiplier
        self.atr_period = atr_period
        self.squeeze_threshold = squeeze_threshold
        self.consolidation_periods = consolidation_periods
    
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                        **kwargs) -> Signal:
        """
        Generate trading signal based on breakout patterns.
        """
        min_rows = max(self.lookback_period, self.atr_period) + 20
        if not self.validate_data(data, min_rows=min_rows):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0,
                reasons=[f"Insufficient data (need {min_rows} rows)"]
            )
        
        df = self._calculate_indicators(data)
        current = df.iloc[-1]
        prev = df.iloc[-2]
        price = current['Close']
        
        signals = []
        reasons = []
        
        # Resistance/Support breakout
        sr_signal, sr_reason = self._analyze_sr_breakout(
            df, price, current['resistance'], current['support']
        )
        signals.append(sr_signal)
        reasons.append(sr_reason)
        
        # Channel breakout
        ch_signal, ch_reason = self._analyze_channel_breakout(
            price, prev['Close'], current['upper_channel'], 
            current['lower_channel'], prev['upper_channel'], prev['lower_channel']
        )
        signals.append(ch_signal)
        reasons.append(ch_reason)
        
        # Volatility squeeze breakout
        sq_signal, sq_reason = self._analyze_squeeze_breakout(df)
        signals.append(sq_signal)
        reasons.append(sq_reason)
        
        # Volume confirmation
        vol_signal, vol_reason = self._analyze_volume_confirmation(
            current['Volume'], current['volume_ma'],
            sum(signals)
        )
        signals.append(vol_signal)
        reasons.append(vol_reason)
        
        # Momentum confirmation
        mom_signal, mom_reason = self._analyze_momentum(
            current['roc'], sum(signals)
        )
        signals.append(mom_signal)
        reasons.append(mom_reason)
        
        # Aggregate with breakout weighting
        final_signal, confidence = self._aggregate_breakout_signals(signals)
        active_reasons = [r for i, r in enumerate(reasons) if signals[i] != 0]
        
        # Adjust stop loss for breakouts (tighter)
        if final_signal == SignalType.BUY:
            stop_loss = max(current['support'], price * 0.97)
            take_profit = price + (price - stop_loss) * 2  # 2:1 R/R
        elif final_signal == SignalType.SELL:
            stop_loss = min(current['resistance'], price * 1.03)
            take_profit = price - (stop_loss - price) * 2
        else:
            stop_loss = self.calculate_stop_loss(price, final_signal)
            take_profit = self.calculate_take_profit(price, final_signal)
        
        return Signal(
            symbol=symbol,
            signal_type=final_signal,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=active_reasons if active_reasons else reasons[:3],
            metadata={
                'resistance': current['resistance'],
                'support': current['support'],
                'atr': current['atr'],
                'bb_width': current['bb_width'],
                'volume_ratio': current['Volume'] / current['volume_ma'] if current['volume_ma'] else 1
            }
        )
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout indicators."""
        df = data.copy()
        
        # Support and Resistance (rolling high/low)
        df['resistance'] = df['High'].rolling(window=self.lookback_period).max()
        df['support'] = df['Low'].rolling(window=self.lookback_period).min()
        
        # Donchian Channels
        df['upper_channel'] = df['High'].rolling(window=self.lookback_period).max()
        df['lower_channel'] = df['Low'].rolling(window=self.lookback_period).min()
        df['mid_channel'] = (df['upper_channel'] + df['lower_channel']) / 2
        
        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # Bollinger Bands for squeeze detection
        df['sma'] = df['Close'].rolling(window=20).mean()
        df['std'] = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['sma'] + (2 * df['std'])
        df['bb_lower'] = df['sma'] - (2 * df['std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma']
        
        # Volume
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        
        # Rate of Change for momentum
        df['roc'] = df['Close'].pct_change(periods=5) * 100
        
        # Consolidation detection (price range shrinking)
        df['range'] = df['High'] - df['Low']
        df['avg_range'] = df['range'].rolling(window=self.consolidation_periods).mean()
        df['range_shrinking'] = df['avg_range'] < df['avg_range'].shift(self.consolidation_periods)
        
        return df
    
    def _analyze_sr_breakout(self, df: pd.DataFrame, price: float,
                              resistance: float, support: float) -> Tuple[int, str]:
        """Analyze support/resistance breakout."""
        if pd.isna(resistance) or pd.isna(support):
            return 0, "S/R Breakout: Insufficient data"
        
        prev_close = df['Close'].iloc[-2]
        
        # Resistance breakout
        if price > resistance and prev_close <= resistance:
            return 1, f"BREAKOUT: Price broke above resistance ${resistance:.2f}"
        
        # Support breakdown
        if price < support and prev_close >= support:
            return -1, f"BREAKDOWN: Price broke below support ${support:.2f}"
        
        # Near levels but not broken
        if price > resistance * 0.98:
            return 0, f"Approaching resistance ${resistance:.2f} - watch for breakout"
        if price < support * 1.02:
            return 0, f"Approaching support ${support:.2f} - watch for breakdown"
        
        return 0, "Price within S/R range"
    
    def _analyze_channel_breakout(self, price: float, prev_price: float,
                                   upper: float, lower: float,
                                   prev_upper: float, prev_lower: float) -> Tuple[int, str]:
        """Analyze Donchian channel breakout."""
        if pd.isna(upper) or pd.isna(lower):
            return 0, "Channel: Insufficient data"
        
        # New high (upper channel breakout)
        if price >= upper and prev_price < prev_upper:
            return 1, f"New {self.lookback_period}-day HIGH - bullish breakout"
        
        # New low (lower channel breakdown)
        if price <= lower and prev_price > prev_lower:
            return -1, f"New {self.lookback_period}-day LOW - bearish breakdown"
        
        return 0, "Price within channel"
    
    def _analyze_squeeze_breakout(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Analyze Bollinger Band squeeze and breakout."""
        recent_width = df['bb_width'].iloc[-5:]
        current_width = df['bb_width'].iloc[-1]
        prev_width = df['bb_width'].iloc[-2]
        
        if pd.isna(current_width):
            return 0, "BB Squeeze: Insufficient data"
        
        # Detect squeeze (low volatility period)
        was_squeezed = prev_width < self.squeeze_threshold
        is_expanding = current_width > prev_width * 1.2
        
        if was_squeezed and is_expanding:
            price = df['Close'].iloc[-1]
            sma = df['sma'].iloc[-1]
            
            if price > sma:
                return 1, "Volatility SQUEEZE breakout to UPSIDE"
            else:
                return -1, "Volatility SQUEEZE breakout to DOWNSIDE"
        
        if current_width < self.squeeze_threshold:
            return 0, f"BB Squeeze forming (width: {current_width:.3f}) - breakout imminent"
        
        return 0, "No volatility squeeze"
    
    def _analyze_volume_confirmation(self, volume: float, volume_ma: float,
                                      direction: int) -> Tuple[int, str]:
        """Check volume confirms the breakout."""
        if pd.isna(volume_ma) or volume_ma == 0:
            return 0, "Volume: Insufficient data"
        
        volume_ratio = volume / volume_ma
        
        if volume_ratio >= self.volume_surge_multiplier:
            if direction > 0:
                return 1, f"Volume SURGE ({volume_ratio:.1f}x) confirms bullish breakout"
            elif direction < 0:
                return -1, f"Volume SURGE ({volume_ratio:.1f}x) confirms bearish breakdown"
            else:
                return 0, f"High volume ({volume_ratio:.1f}x) but no clear direction"
        
        if direction != 0 and volume_ratio < 1.0:
            return 0, f"Low volume ({volume_ratio:.1f}x) - breakout may fail"
        
        return 0, f"Volume at {volume_ratio:.1f}x average"
    
    def _analyze_momentum(self, roc: float, direction: int) -> Tuple[int, str]:
        """Check momentum confirms the breakout."""
        if pd.isna(roc):
            return 0, "Momentum: Insufficient data"
        
        if direction > 0 and roc > 3:
            return 1, f"Strong momentum ({roc:.1f}%) confirms breakout"
        elif direction < 0 and roc < -3:
            return -1, f"Strong momentum ({roc:.1f}%) confirms breakdown"
        elif direction != 0 and abs(roc) < 1:
            return 0, f"Weak momentum ({roc:.1f}%) - breakout may fail"
        
        return 0, f"Momentum: {roc:.1f}%"
    
    def _aggregate_breakout_signals(self, signals: List[int]) -> Tuple[SignalType, float]:
        """Aggregate signals with breakout-specific logic."""
        valid_signals = [s for s in signals if s is not None]
        
        if not valid_signals:
            return SignalType.HOLD, 0.0
        
        # For breakouts, we need strong confirmation
        bullish = sum(1 for s in valid_signals if s > 0)
        bearish = sum(1 for s in valid_signals if s < 0)
        total = len(valid_signals)
        
        # Need at least 3 confirming signals for breakout
        if bullish >= 3:
            confidence = bullish / total
            return SignalType.BUY, min(confidence, 0.95)
        elif bearish >= 3:
            confidence = bearish / total
            return SignalType.SELL, min(confidence, 0.95)
        else:
            return SignalType.HOLD, 0.3


if __name__ == "__main__":
    import yfinance as yf
    logging.basicConfig(level=logging.INFO)
    
    data = yf.Ticker('NVDA').history(period='6mo')
    strategy = BreakoutStrategy()
    signal = strategy.generate_signal('NVDA', data)
    
    print(f"\nBreakout Signal for NVDA:")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Support: ${signal.metadata.get('support', 0):.2f}")
    print(f"  Resistance: ${signal.metadata.get('resistance', 0):.2f}")
    print(f"\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
