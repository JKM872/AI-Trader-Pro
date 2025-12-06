"""
Advanced Technical Indicators - TradingView-style indicators.

Implements professional-grade indicators used by top traders:
- Supertrend (trend following)
- ADX/DMI (trend strength)
- VWAP (volume weighted)
- Ichimoku Cloud
- Squeeze Momentum
- Hull Moving Average
- Keltner Channels
- Pivot Points
- Order Flow indicators
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TrendDirection(Enum):
    """Trend direction enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class IndicatorResult:
    """Result container for indicator calculations."""
    value: float
    signal: TrendDirection
    strength: float  # 0.0 to 1.0
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TradingViewIndicators:
    """
    Collection of TradingView-style indicators for accurate signal generation.
    
    Based on:
    - TradingView Pine Script implementations
    - Professional trading algorithms
    - Quantitative research papers
    """
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        ATR measures volatility and is used for:
        - Stop loss placement
        - Position sizing
        - Breakout confirmation
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, 
                   multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Supertrend indicator - the most reliable trend-following indicator.
        
        Used by professional traders for:
        - Trend identification
        - Dynamic stop loss
        - Entry/Exit signals
        
        Returns:
            Tuple of (supertrend_line, trend_direction)
            trend_direction: 1 = bullish, -1 = bearish
        """
        hl2 = (df['High'] + df['Low']) / 2
        atr = TradingViewIndicators.calculate_atr(df, period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(1, len(df)):
            if pd.isna(atr.iloc[i]):
                supertrend.iloc[i] = np.nan
                direction.iloc[i] = 1
                continue
                
            # Calculate bands
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            prev_upper = upper_band.iloc[i-1] if not pd.isna(upper_band.iloc[i-1]) else curr_upper
            prev_lower = lower_band.iloc[i-1] if not pd.isna(lower_band.iloc[i-1]) else curr_lower
            prev_close = df['Close'].iloc[i-1]
            curr_close = df['Close'].iloc[i]
            
            # Final bands (prevent expansion in wrong direction)
            final_upper = curr_upper if curr_upper < prev_upper or prev_close > prev_upper else prev_upper
            final_lower = curr_lower if curr_lower > prev_lower or prev_close < prev_lower else prev_lower
            
            # Determine trend
            if i == 1:
                prev_st = final_upper
                prev_dir = 1
            else:
                prev_st = supertrend.iloc[i-1] if not pd.isna(supertrend.iloc[i-1]) else final_upper
                prev_dir = direction.iloc[i-1] if not pd.isna(direction.iloc[i-1]) else 1
            
            if prev_st == prev_upper if not pd.isna(prev_upper) else True:
                if curr_close > final_upper:
                    supertrend.iloc[i] = final_lower
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = final_upper
                    direction.iloc[i] = -1
            else:
                if curr_close < final_lower:
                    supertrend.iloc[i] = final_upper
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = final_lower
                    direction.iloc[i] = 1
        
        return supertrend, direction
    
    @staticmethod
    def adx_dmi(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX (Average Directional Index) and DMI (Directional Movement Index).
        
        Measures trend strength:
        - ADX > 25: Strong trend
        - ADX > 50: Very strong trend
        - ADX < 20: No trend (ranging)
        
        +DI > -DI: Bullish
        +DI < -DI: Bearish
        
        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm_raw = high.diff()
        minus_dm_raw = (-low.diff())
        
        # Apply conditions with clip to avoid type issues
        plus_dm = plus_dm_raw.clip(lower=0)
        minus_dm = minus_dm_raw.clip(lower=0)
        
        # Zero out where condition not met
        plus_dm = plus_dm.where(plus_dm_raw > minus_dm_raw, 0.0)
        minus_dm = minus_dm.where(minus_dm_raw > plus_dm_raw, 0.0)
        
        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        
        # ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def vwap(df: pd.DataFrame, anchor: str = 'D') -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Institutional traders use VWAP as:
        - Fair value reference
        - Support/Resistance level
        - Trade execution benchmark
        
        Args:
            df: OHLCV DataFrame
            anchor: Reset period ('D' daily, 'W' weekly, 'M' monthly)
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        tp_volume = typical_price * df['Volume']
        
        # Group by anchor period
        dt_index = pd.DatetimeIndex(df.index)
        if anchor == 'D':
            groups = dt_index.date
        elif anchor == 'W':
            groups = dt_index.isocalendar().week
        else:
            groups = dt_index.month
        
        cumulative_tp_vol = tp_volume.groupby(groups).cumsum()
        cumulative_vol = df['Volume'].groupby(groups).cumsum()
        
        vwap = cumulative_tp_vol / cumulative_vol
        return vwap
    
    @staticmethod
    def vwap_bands(df: pd.DataFrame, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate VWAP with standard deviation bands.
        
        Returns:
            Tuple of (VWAP, Upper Band, Lower Band)
        """
        vwap = TradingViewIndicators.vwap(df)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Calculate standard deviation from VWAP
        squared_diff = (typical_price - vwap) ** 2
        variance = squared_diff.expanding().mean()
        std = np.sqrt(variance)
        
        upper = vwap + (std_dev * std)
        lower = vwap - (std_dev * std)
        
        return vwap, upper, lower
    
    @staticmethod
    def ichimoku_cloud(df: pd.DataFrame, 
                       tenkan_period: int = 9,
                       kijun_period: int = 26,
                       senkou_b_period: int = 52,
                       displacement: int = 26) -> Dict[str, pd.Series]:
        """
        Calculate Ichimoku Cloud (Ichimoku Kinko Hyo).
        
        Complete trading system with:
        - Tenkan-sen (Conversion Line): Short-term trend
        - Kijun-sen (Base Line): Medium-term trend
        - Senkou Span A (Leading Span A): Future cloud boundary
        - Senkou Span B (Leading Span B): Future cloud boundary
        - Chikou Span (Lagging Span): Momentum confirmation
        
        Returns:
            Dict with all Ichimoku components
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A) - displaced forward
        senkou_a = ((tenkan + kijun) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B) - displaced forward
        senkou_b_high = high.rolling(window=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span) - displaced backward
        chikou = close.shift(-displacement)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou,
            'cloud_top': pd.concat([senkou_a, senkou_b], axis=1).max(axis=1),
            'cloud_bottom': pd.concat([senkou_a, senkou_b], axis=1).min(axis=1),
            'cloud_color': (senkou_a > senkou_b).astype(int)  # 1 = green (bullish), 0 = red (bearish)
        }
    
    @staticmethod
    def squeeze_momentum(df: pd.DataFrame, 
                         bb_length: int = 20, 
                         bb_mult: float = 2.0,
                         kc_length: int = 20, 
                         kc_mult: float = 1.5) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Squeeze Momentum Indicator (TTM Squeeze).
        
        Identifies low volatility setups before explosive moves:
        - Squeeze ON: Bollinger Bands inside Keltner Channels
        - Squeeze OFF: Breakout imminent
        - Momentum: Direction of the move
        
        Returns:
            Tuple of (squeeze_on, momentum)
            squeeze_on: True when in squeeze
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Bollinger Bands
        bb_basis = close.rolling(window=bb_length).mean()
        bb_dev = close.rolling(window=bb_length).std() * bb_mult
        bb_upper = bb_basis + bb_dev
        bb_lower = bb_basis - bb_dev
        
        # Keltner Channels
        atr = TradingViewIndicators.calculate_atr(df, kc_length)
        kc_basis = close.rolling(window=kc_length).mean()
        kc_upper = kc_basis + (kc_mult * atr)
        kc_lower = kc_basis - (kc_mult * atr)
        
        # Squeeze detection
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        # Momentum (Linear Regression of price from mean)
        highest = high.rolling(window=kc_length).max()
        lowest = low.rolling(window=kc_length).min()
        avg_hl = (highest + lowest) / 2
        avg_close = close.rolling(window=kc_length).mean()
        
        momentum = close - ((avg_hl + avg_close) / 2)
        
        return squeeze_on, momentum
    
    @staticmethod
    def hull_moving_average(series: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Hull Moving Average (HMA).
        
        Faster and smoother than traditional MAs:
        - Reduces lag significantly
        - Identifies trend changes earlier
        - Great for entries/exits
        """
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = series.rolling(window=half_period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
            raw=True
        )
        wma_full = series.rolling(window=period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
            raw=True
        )
        
        raw_hma = (2 * wma_half) - wma_full
        hma = raw_hma.rolling(window=sqrt_period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
            raw=True
        )
        
        return hma
    
    @staticmethod
    def keltner_channels(df: pd.DataFrame, 
                         ema_period: int = 20, 
                         atr_period: int = 10,
                         multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Keltner Channels.
        
        Similar to Bollinger Bands but uses ATR:
        - More stable bands
        - Better for trending markets
        
        Returns:
            Tuple of (middle, upper, lower)
        """
        middle = df['Close'].ewm(span=ema_period, adjust=False).mean()
        atr = TradingViewIndicators.calculate_atr(df, atr_period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return middle, upper, lower
    
    @staticmethod
    def pivot_points(df: pd.DataFrame, 
                     pivot_type: str = 'standard') -> Dict[str, float]:
        """
        Calculate Pivot Points for support/resistance levels.
        
        Types:
        - standard: Classic pivot points
        - fibonacci: Fibonacci pivot points
        - camarilla: Camarilla pivot points
        - woodie: Woodie's pivot points
        
        Returns:
            Dict with pivot, supports (S1-S3), resistances (R1-R3)
        """
        high = df['High'].iloc[-1]
        low = df['Low'].iloc[-1]
        close = df['Close'].iloc[-1]
        open_price = df['Open'].iloc[-1]
        
        if pivot_type == 'standard':
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
        elif pivot_type == 'fibonacci':
            pivot = (high + low + close) / 3
            range_hl = high - low
            r1 = pivot + 0.382 * range_hl
            s1 = pivot - 0.382 * range_hl
            r2 = pivot + 0.618 * range_hl
            s2 = pivot - 0.618 * range_hl
            r3 = pivot + 1.000 * range_hl
            s3 = pivot - 1.000 * range_hl
            
        elif pivot_type == 'camarilla':
            pivot = (high + low + close) / 3
            range_hl = high - low
            r1 = close + range_hl * 1.1 / 12
            s1 = close - range_hl * 1.1 / 12
            r2 = close + range_hl * 1.1 / 6
            s2 = close - range_hl * 1.1 / 6
            r3 = close + range_hl * 1.1 / 4
            s3 = close - range_hl * 1.1 / 4
            
        else:  # woodie
            pivot = (high + low + 2 * close) / 4
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = r1 + (high - low)
            s3 = s1 - (high - low)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def stochastic_rsi(df: pd.DataFrame, 
                       rsi_period: int = 14,
                       stoch_period: int = 14,
                       k_smooth: int = 3,
                       d_smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic RSI.
        
        Combines RSI with Stochastic for better signals:
        - More sensitive than RSI alone
        - Good for identifying overbought/oversold in trends
        
        Returns:
            Tuple of (%K, %D)
        """
        close = df['Close']
        
        # Calculate RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Apply Stochastic to RSI
        rsi_low = rsi.rolling(window=stoch_period).min()
        rsi_high = rsi.rolling(window=stoch_period).max()
        
        stoch_rsi = 100 * (rsi - rsi_low) / (rsi_high - rsi_low)
        
        # Smooth
        k = stoch_rsi.rolling(window=k_smooth).mean()
        d = k.rolling(window=d_smooth).mean()
        
        return k, d
    
    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Momentum indicator showing overbought/oversold levels:
        - Above -20: Overbought
        - Below -80: Oversold
        """
        high = df['High'].rolling(window=period).max()
        low = df['Low'].rolling(window=period).min()
        close = df['Close']
        
        wr = -100 * (high - close) / (high - low)
        return wr
    
    @staticmethod
    def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF).
        
        Volume-based indicator for buying/selling pressure:
        - Positive: Buying pressure
        - Negative: Selling pressure
        - Near zero: Indecision
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # CMF
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    
    @staticmethod
    def on_balance_volume(df: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Cumulative volume indicator:
        - Rising OBV + Rising Price = Bullish
        - Falling OBV + Falling Price = Bearish
        - Divergence = Potential reversal
        """
        close = df['Close']
        volume = df['Volume']
        
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = (volume * direction).cumsum()
        
        return obv
    
    @staticmethod
    def volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
        """
        Calculate Volume Profile.
        
        Shows volume distribution at different price levels:
        - Point of Control (POC): Highest volume price
        - Value Area: 70% of volume range
        
        Returns:
            DataFrame with price levels and volume
        """
        price_range = df['Close'].max() - df['Close'].min()
        bin_size = price_range / bins
        
        df_copy = df.copy()
        df_copy['price_bin'] = ((df_copy['Close'] - df_copy['Close'].min()) / bin_size).astype(int)
        
        volume_profile = df_copy.groupby('price_bin').agg({
            'Volume': 'sum',
            'Close': 'mean'
        }).reset_index()
        
        volume_profile.columns = ['bin', 'volume', 'price']
        volume_profile['volume_pct'] = volume_profile['volume'] / volume_profile['volume'].sum() * 100
        
        # Point of Control
        poc_idx = volume_profile['volume'].idxmax()
        volume_profile['is_poc'] = volume_profile.index == poc_idx
        
        return volume_profile
    
    @staticmethod
    def market_structure(df: pd.DataFrame, 
                         swing_length: int = 5) -> Dict[str, Any]:
        """
        Analyze market structure for trend identification.
        
        Identifies:
        - Higher Highs (HH) / Lower Highs (LH)
        - Higher Lows (HL) / Lower Lows (LL)
        - Break of Structure (BOS)
        - Change of Character (CHoCH)
        
        Returns:
            Dict with structure analysis
        """
        high = df['High']
        low = df['Low']
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(swing_length, len(df) - swing_length):
            # Swing High
            if high.iloc[i] == high.iloc[i-swing_length:i+swing_length+1].max():
                swing_highs.append((df.index[i], high.iloc[i]))
            
            # Swing Low
            if low.iloc[i] == low.iloc[i-swing_length:i+swing_length+1].min():
                swing_lows.append((df.index[i], low.iloc[i]))
        
        # Analyze structure
        structure = {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'trend': TrendDirection.NEUTRAL,
            'last_hh': None,
            'last_ll': None,
            'bos_detected': False
        }
        
        if len(swing_highs) >= 2:
            if swing_highs[-1][1] > swing_highs[-2][1]:
                structure['last_hh'] = swing_highs[-1]
            
        if len(swing_lows) >= 2:
            if swing_lows[-1][1] < swing_lows[-2][1]:
                structure['last_ll'] = swing_lows[-1]
        
        # Determine trend
        if structure['last_hh'] and len(swing_lows) >= 2:
            if swing_lows[-1][1] > swing_lows[-2][1]:
                structure['trend'] = TrendDirection.BULLISH
        elif structure['last_ll'] and len(swing_highs) >= 2:
            if swing_highs[-1][1] < swing_highs[-2][1]:
                structure['trend'] = TrendDirection.BEARISH
        
        return structure
    
    @staticmethod
    def order_block_detection(df: pd.DataFrame, 
                              sensitivity: int = 3) -> Dict[str, list]:
        """
        Detect Order Blocks (Smart Money Concepts).
        
        Order blocks are areas where institutions placed large orders:
        - Bullish OB: Last bearish candle before impulsive move up
        - Bearish OB: Last bullish candle before impulsive move down
        
        Returns:
            Dict with bullish and bearish order blocks
        """
        order_blocks = {
            'bullish': [],
            'bearish': []
        }
        
        for i in range(sensitivity + 1, len(df)):
            # Check for impulsive move
            current_close = df['Close'].iloc[i]
            prev_close = df['Close'].iloc[i-1]
            
            # Bullish Order Block
            if current_close > prev_close:
                move_size = (current_close - prev_close) / prev_close * 100
                
                if move_size > 1.0:  # Significant move (>1%)
                    # Find last bearish candle
                    for j in range(i-1, max(0, i-sensitivity-1), -1):
                        if df['Close'].iloc[j] < df['Open'].iloc[j]:
                            order_blocks['bullish'].append({
                                'index': df.index[j],
                                'high': df['High'].iloc[j],
                                'low': df['Low'].iloc[j],
                                'strength': move_size
                            })
                            break
            
            # Bearish Order Block
            elif current_close < prev_close:
                move_size = (prev_close - current_close) / prev_close * 100
                
                if move_size > 1.0:
                    for j in range(i-1, max(0, i-sensitivity-1), -1):
                        if df['Close'].iloc[j] > df['Open'].iloc[j]:
                            order_blocks['bearish'].append({
                                'index': df.index[j],
                                'high': df['High'].iloc[j],
                                'low': df['Low'].iloc[j],
                                'strength': move_size
                            })
                            break
        
        return order_blocks
    
    @staticmethod
    def fair_value_gap(df: pd.DataFrame) -> Dict[str, list]:
        """
        Detect Fair Value Gaps (FVG) / Imbalances.
        
        FVGs are price inefficiencies that often get filled:
        - Bullish FVG: Gap between candle 1 high and candle 3 low
        - Bearish FVG: Gap between candle 1 low and candle 3 high
        
        Returns:
            Dict with bullish and bearish FVGs
        """
        fvg = {
            'bullish': [],
            'bearish': []
        }
        
        for i in range(2, len(df)):
            # Bullish FVG
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                fvg['bullish'].append({
                    'index': df.index[i-1],
                    'top': df['Low'].iloc[i],
                    'bottom': df['High'].iloc[i-2],
                    'size': df['Low'].iloc[i] - df['High'].iloc[i-2]
                })
            
            # Bearish FVG
            if df['High'].iloc[i] < df['Low'].iloc[i-2]:
                fvg['bearish'].append({
                    'index': df.index[i-1],
                    'top': df['Low'].iloc[i-2],
                    'bottom': df['High'].iloc[i],
                    'size': df['Low'].iloc[i-2] - df['High'].iloc[i]
                })
        
        return fvg


class SignalStrength:
    """Calculate signal strength from multiple indicators."""
    
    @staticmethod
    def calculate_confluence(indicators: Dict[str, IndicatorResult]) -> Tuple[TrendDirection, float]:
        """
        Calculate confluence of multiple indicators.
        
        Args:
            indicators: Dict of indicator name to IndicatorResult
            
        Returns:
            Tuple of (overall direction, strength 0-1)
        """
        bullish_count = 0
        bearish_count = 0
        total_strength = 0
        
        for name, result in indicators.items():
            if result.signal == TrendDirection.BULLISH:
                bullish_count += 1
                total_strength += result.strength
            elif result.signal == TrendDirection.BEARISH:
                bearish_count += 1
                total_strength += result.strength
        
        total = bullish_count + bearish_count
        if total == 0:
            return TrendDirection.NEUTRAL, 0.0
        
        avg_strength = total_strength / total
        
        if bullish_count > bearish_count:
            confluence = bullish_count / len(indicators)
            return TrendDirection.BULLISH, confluence * avg_strength
        elif bearish_count > bullish_count:
            confluence = bearish_count / len(indicators)
            return TrendDirection.BEARISH, confluence * avg_strength
        else:
            return TrendDirection.NEUTRAL, 0.0


if __name__ == "__main__":
    import yfinance as yf
    
    # Test indicators
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="6mo")
    
    print("Testing TradingView Indicators on AAPL")
    print("=" * 50)
    
    # Supertrend
    st, direction = TradingViewIndicators.supertrend(df)
    print(f"\nSupertrend: {st.iloc[-1]:.2f}")
    print(f"Direction: {'Bullish' if direction.iloc[-1] == 1 else 'Bearish'}")
    
    # ADX
    adx, plus_di, minus_di = TradingViewIndicators.adx_dmi(df)
    print(f"\nADX: {adx.iloc[-1]:.2f}")
    print(f"+DI: {plus_di.iloc[-1]:.2f}")
    print(f"-DI: {minus_di.iloc[-1]:.2f}")
    trend_strength = "Strong" if adx.iloc[-1] > 25 else "Weak"
    print(f"Trend Strength: {trend_strength}")
    
    # VWAP
    vwap = TradingViewIndicators.vwap(df)
    print(f"\nVWAP: {vwap.iloc[-1]:.2f}")
    
    # Ichimoku
    ichimoku = TradingViewIndicators.ichimoku_cloud(df)
    print(f"\nIchimoku Cloud:")
    print(f"  Tenkan: {ichimoku['tenkan'].iloc[-1]:.2f}")
    print(f"  Kijun: {ichimoku['kijun'].iloc[-1]:.2f}")
    
    # Pivot Points
    pivots = TradingViewIndicators.pivot_points(df)
    print(f"\nPivot Points (Standard):")
    print(f"  Pivot: {pivots['pivot']:.2f}")
    print(f"  R1: {pivots['r1']:.2f}, R2: {pivots['r2']:.2f}")
    print(f"  S1: {pivots['s1']:.2f}, S2: {pivots['s2']:.2f}")
