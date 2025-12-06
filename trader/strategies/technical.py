"""
Technical Strategy - Trading strategy based on technical indicators.

Implements:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .base import TradingStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class TechnicalStrategy(TradingStrategy):
    """
    Technical analysis based trading strategy.
    
    Combines multiple indicators to generate trading signals:
    - RSI for overbought/oversold conditions
    - MACD for trend confirmation
    - Bollinger Bands for volatility and mean reversion
    - Moving Averages for trend direction
    
    Usage:
        strategy = TechnicalStrategy()
        signal = strategy.generate_signal('AAPL', price_data)
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 sma_short: int = 20,
                 sma_long: int = 50,
                 **kwargs):
        """
        Initialize TechnicalStrategy with indicator parameters.
        
        Args:
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold (buy signal)
            rsi_overbought: RSI overbought threshold (sell signal)
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
            sma_short: Short-term SMA period
            sma_long: Long-term SMA period
        """
        super().__init__(name="TechnicalStrategy", **kwargs)
        
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.sma_short = sma_short
        self.sma_long = sma_long
    
    def generate_signal(self, symbol: str, data: pd.DataFrame, 
                        **kwargs) -> Signal:
        """
        Generate trading signal based on technical indicators.
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
        
        Returns:
            Signal with BUY, SELL, or HOLD recommendation
        """
        if not self.validate_data(data, min_rows=self.sma_long + 10):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0,
                reasons=["Insufficient data for analysis"]
            )
        
        # Calculate all indicators
        df = self._calculate_indicators(data)
        
        # Get the latest values
        current = df.iloc[-1]
        price = current['Close']
        
        # Analyze each indicator
        signals = []
        reasons = []
        
        # RSI Analysis
        rsi_signal, rsi_reason = self._analyze_rsi(current['rsi'])
        signals.append(rsi_signal)
        reasons.append(rsi_reason)
        
        # MACD Analysis
        macd_signal, macd_reason = self._analyze_macd(
            current['macd'], current['macd_signal'], current['macd_hist'],
            df['macd_hist'].iloc[-2] if len(df) > 1 else 0
        )
        signals.append(macd_signal)
        reasons.append(macd_reason)
        
        # Bollinger Bands Analysis
        bb_signal, bb_reason = self._analyze_bollinger(
            price, current['bb_upper'], current['bb_lower'], current['bb_middle']
        )
        signals.append(bb_signal)
        reasons.append(bb_reason)
        
        # Moving Average Analysis
        ma_signal, ma_reason = self._analyze_moving_averages(
            price, current['sma_short'], current['sma_long']
        )
        signals.append(ma_signal)
        reasons.append(ma_reason)
        
        # Aggregate signals
        final_signal, confidence = self._aggregate_signals(signals)
        
        # Filter out neutral reasons
        active_reasons = [r for i, r in enumerate(reasons) if signals[i] != 0]
        if not active_reasons:
            active_reasons = reasons
        
        return Signal(
            symbol=symbol,
            signal_type=final_signal,
            confidence=confidence,
            price=price,
            stop_loss=self.calculate_stop_loss(price, final_signal),
            take_profit=self.calculate_take_profit(price, final_signal),
            reasons=active_reasons,
            metadata={
                'rsi': current['rsi'],
                'macd': current['macd'],
                'macd_signal': current['macd_signal'],
                'bb_position': (price - current['bb_lower']) / (current['bb_upper'] - current['bb_lower']) if current['bb_upper'] != current['bb_lower'] else 0.5,
                'sma_short': current['sma_short'],
                'sma_long': current['sma_long']
            }
        )
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = data.copy()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['Close'], self.rsi_period)
        
        # MACD
        df['ema_fast'] = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=self.bb_period).mean()
        bb_std = df['Close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std)
        
        # Moving Averages
        df['sma_short'] = df['Close'].rolling(window=self.sma_short).mean()
        df['sma_long'] = df['Close'].rolling(window=self.sma_long).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _analyze_rsi(self, rsi: float) -> Tuple[int, str]:
        """
        Analyze RSI for trading signal.
        
        Returns:
            Tuple of (signal [-1, 0, 1], reason string)
        """
        if pd.isna(rsi):
            return 0, "RSI: Insufficient data"
        
        if rsi < self.rsi_oversold:
            return 1, f"RSI ({rsi:.1f}) indicates oversold - potential buy"
        elif rsi > self.rsi_overbought:
            return -1, f"RSI ({rsi:.1f}) indicates overbought - potential sell"
        else:
            return 0, f"RSI ({rsi:.1f}) is neutral"
    
    def _analyze_macd(self, macd: float, signal: float, hist: float, 
                      prev_hist: float) -> Tuple[int, str]:
        """
        Analyze MACD for trading signal.
        
        Looks for:
        - MACD crossing above signal line (bullish)
        - MACD crossing below signal line (bearish)
        - Histogram momentum
        """
        if pd.isna(macd) or pd.isna(signal):
            return 0, "MACD: Insufficient data"
        
        # Check for crossover
        if prev_hist < 0 and hist > 0:
            return 1, "MACD crossed above signal line - bullish"
        elif prev_hist > 0 and hist < 0:
            return -1, "MACD crossed below signal line - bearish"
        elif hist > 0 and macd > 0:
            return 1, "MACD positive with bullish histogram"
        elif hist < 0 and macd < 0:
            return -1, "MACD negative with bearish histogram"
        else:
            return 0, "MACD shows no clear signal"
    
    def _analyze_bollinger(self, price: float, upper: float, 
                           lower: float, middle: float) -> Tuple[int, str]:
        """
        Analyze Bollinger Bands for trading signal.
        
        Looks for:
        - Price near lower band (potential buy)
        - Price near upper band (potential sell)
        - Price crossing middle band
        """
        if pd.isna(upper) or pd.isna(lower):
            return 0, "Bollinger Bands: Insufficient data"
        
        band_width = upper - lower
        position = (price - lower) / band_width if band_width > 0 else 0.5
        
        if price <= lower:
            return 1, f"Price at lower Bollinger Band - potential reversal up"
        elif price >= upper:
            return -1, f"Price at upper Bollinger Band - potential reversal down"
        elif position < 0.2:
            return 1, f"Price near lower band ({position:.0%}) - consider buying"
        elif position > 0.8:
            return -1, f"Price near upper band ({position:.0%}) - consider selling"
        else:
            return 0, f"Price within Bollinger Bands ({position:.0%})"
    
    def _analyze_moving_averages(self, price: float, sma_short: float, 
                                  sma_long: float) -> Tuple[int, str]:
        """
        Analyze Moving Averages for trading signal.
        
        Looks for:
        - Golden cross (short MA crosses above long MA)
        - Death cross (short MA crosses below long MA)
        - Price relative to MAs
        """
        if pd.isna(sma_short) or pd.isna(sma_long):
            return 0, "Moving Averages: Insufficient data"
        
        if sma_short > sma_long and price > sma_short:
            return 1, "Golden cross pattern - bullish trend"
        elif sma_short < sma_long and price < sma_short:
            return -1, "Death cross pattern - bearish trend"
        elif price > sma_long:
            return 1, "Price above long-term MA - uptrend"
        elif price < sma_long:
            return -1, "Price below long-term MA - downtrend"
        else:
            return 0, "Moving averages show no clear trend"
    
    def _aggregate_signals(self, signals: List[int]) -> Tuple[SignalType, float]:
        """
        Aggregate multiple indicator signals into final signal.
        
        Args:
            signals: List of signals (-1, 0, or 1 for each indicator)
        
        Returns:
            Tuple of (SignalType, confidence)
        """
        # Filter out None/NaN
        valid_signals = [s for s in signals if s is not None]
        
        if not valid_signals:
            return SignalType.HOLD, 0.0
        
        avg_signal = sum(valid_signals) / len(valid_signals)
        
        # Calculate confidence based on signal agreement
        agreement = sum(1 for s in valid_signals if s == np.sign(avg_signal)) / len(valid_signals)
        
        if avg_signal > 0.3:
            return SignalType.BUY, min(agreement, abs(avg_signal))
        elif avg_signal < -0.3:
            return SignalType.SELL, min(agreement, abs(avg_signal))
        else:
            return SignalType.HOLD, 1.0 - abs(avg_signal)


if __name__ == "__main__":
    # Test
    import yfinance as yf
    
    logging.basicConfig(level=logging.INFO)
    
    # Fetch test data
    stock = yf.Ticker('AAPL')
    data = stock.history(period='6mo')
    
    # Generate signal
    strategy = TechnicalStrategy()
    signal = strategy.generate_signal('AAPL', data)
    
    print(f"\nSignal for AAPL:")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Price: ${signal.price:.2f}")
    print(f"  Stop Loss: ${signal.stop_loss:.2f}")
    print(f"  Take Profit: ${signal.take_profit:.2f}")
    print(f"  Risk/Reward: {signal.risk_reward_ratio:.2f}" if signal.risk_reward_ratio else "")
    print(f"\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
    print(f"\nIndicators:")
    for key, value in signal.metadata.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
