"""
Market Regime Detector - Identifies current market conditions.

Detects:
- Trend direction and strength
- Volatility levels
- Market phases (accumulation, distribution, markup, markdown)
- Risk-on vs risk-off environments
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    # Trending regimes
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    DOWNTREND = "downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    
    # Sideways regimes
    RANGING = "ranging"
    CONSOLIDATING = "consolidating"
    CHOPPY = "choppy"
    
    # Volatility regimes
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    
    # Market phases (Wyckoff)
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    
    UNKNOWN = "unknown"


class VolatilityLevel(Enum):
    """Volatility classification."""
    EXTREME = "extreme"    # VIX > 30 or ATR ratio > 3
    HIGH = "high"          # VIX 20-30 or ATR ratio 2-3
    NORMAL = "normal"      # VIX 12-20 or ATR ratio 1-2
    LOW = "low"            # VIX < 12 or ATR ratio < 1
    

class TrendStrength(Enum):
    """Trend strength classification."""
    VERY_STRONG = "very_strong"  # ADX > 50
    STRONG = "strong"            # ADX 25-50
    MODERATE = "moderate"        # ADX 20-25
    WEAK = "weak"                # ADX 15-20
    ABSENT = "absent"            # ADX < 15


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    symbol: str
    regime: MarketRegime
    volatility: VolatilityLevel
    trend_strength: TrendStrength
    trend_direction: float  # -1 (bearish) to +1 (bullish)
    
    # Technical indicators
    adx: float
    atr: float
    atr_percent: float
    rsi: float
    
    # Moving average analysis
    price_vs_sma20: float  # % above/below
    price_vs_sma50: float
    price_vs_sma200: float
    sma20_vs_sma50: float  # Golden/death cross proximity
    
    # Volume analysis
    volume_ratio: float  # Current vs average
    volume_trend: str  # "increasing", "decreasing", "stable"
    
    # Market structure
    higher_highs: int  # Count in lookback
    lower_lows: int
    
    # Recommendations
    recommended_strategies: list[str]
    risk_level: str  # "low", "medium", "high"
    position_sizing_modifier: float  # 0.5 to 1.5
    
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'regime': self.regime.value,
            'volatility': self.volatility.value,
            'trend_strength': self.trend_strength.value,
            'trend_direction': self.trend_direction,
            'adx': self.adx,
            'atr': self.atr,
            'atr_percent': self.atr_percent,
            'rsi': self.rsi,
            'recommended_strategies': self.recommended_strategies,
            'risk_level': self.risk_level,
            'position_sizing_modifier': self.position_sizing_modifier,
            'confidence': self.confidence
        }


class MarketRegimeDetector:
    """
    Detects market regime using multiple technical indicators.
    
    Analyzes:
    - Trend (ADX, moving averages, higher highs/lower lows)
    - Volatility (ATR, Bollinger Band width, historical volatility)
    - Market structure (support/resistance, swing points)
    - Volume profile
    """
    
    # Strategy recommendations by regime
    REGIME_STRATEGIES = {
        MarketRegime.STRONG_UPTREND: ["momentum", "breakout", "trend_following"],
        MarketRegime.UPTREND: ["momentum", "trend_following", "pullback"],
        MarketRegime.WEAK_UPTREND: ["pullback", "range_breakout"],
        MarketRegime.STRONG_DOWNTREND: ["momentum_short", "breakdown"],
        MarketRegime.DOWNTREND: ["mean_reversion", "pullback_short"],
        MarketRegime.WEAK_DOWNTREND: ["mean_reversion", "range"],
        MarketRegime.RANGING: ["mean_reversion", "range_trading"],
        MarketRegime.CONSOLIDATING: ["breakout_watch", "range"],
        MarketRegime.CHOPPY: ["reduce_size", "stay_out"],
        MarketRegime.HIGH_VOLATILITY: ["reduce_size", "wider_stops"],
        MarketRegime.LOW_VOLATILITY: ["breakout_watch", "increase_size"],
        MarketRegime.ACCUMULATION: ["accumulate_long", "watch"],
        MarketRegime.MARKUP: ["momentum", "trend_following"],
        MarketRegime.DISTRIBUTION: ["take_profits", "watch"],
        MarketRegime.MARKDOWN: ["short", "avoid_long"],
    }
    
    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        rsi_period: int = 14,
        sma_periods: tuple[int, int, int] = (20, 50, 200),
        lookback_swing: int = 20
    ):
        """
        Initialize Regime Detector.
        
        Args:
            adx_period: Period for ADX calculation
            atr_period: Period for ATR calculation
            rsi_period: Period for RSI calculation
            sma_periods: Tuple of (short, medium, long) SMA periods
            lookback_swing: Bars to look back for swing points
        """
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.sma_short, self.sma_medium, self.sma_long = sma_periods
        self.lookback_swing = lookback_swing
    
    def _calculate_adx(self, df: pd.DataFrame) -> tuple[float, float, float]:
        """Calculate ADX and directional indicators."""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        n = len(df)
        if n < self.adx_period * 2:
            return 0.0, 0.0, 0.0
        
        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Smoothed averages using Wilder's smoothing
        period = self.adx_period
        atr = np.zeros(n)
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        
        # First value is simple average
        atr[period] = np.mean(tr[1:period+1])
        smooth_plus_dm = np.mean(plus_dm[1:period+1])
        smooth_minus_dm = np.mean(minus_dm[1:period+1])
        
        if atr[period] > 0:
            plus_di[period] = 100 * smooth_plus_dm / atr[period]
            minus_di[period] = 100 * smooth_minus_dm / atr[period]
        
        # Wilder's smoothing for rest
        for i in range(period + 1, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            smooth_plus_dm = (smooth_plus_dm * (period - 1) + plus_dm[i]) / period
            smooth_minus_dm = (smooth_minus_dm * (period - 1) + minus_dm[i]) / period
            
            if atr[i] > 0:
                plus_di[i] = 100 * smooth_plus_dm / atr[i]
                minus_di[i] = 100 * smooth_minus_dm / atr[i]
        
        # Calculate DX and ADX
        dx = np.zeros(n)
        for i in range(period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        
        # First ADX is simple average of DX
        adx = np.zeros(n)
        if 2 * period < n:
            adx[2*period] = np.mean(dx[period:2*period+1])
            
            # Wilder's smoothing for ADX
            for i in range(2*period + 1, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return (
            float(adx[-1]) if adx[-1] > 0 else float(np.mean(dx[-period:])),
            float(plus_di[-1]),
            float(minus_di[-1])
        )
    
    def _calculate_atr(self, df: pd.DataFrame) -> tuple[float, float]:
        """Calculate ATR and ATR as percentage of price."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        atr_value = atr.iloc[-1]
        atr_percent = (atr_value / close.iloc[-1]) * 100
        
        return atr_value, atr_percent
    
    def _calculate_rsi(self, df: pd.DataFrame) -> float:
        """Calculate RSI."""
        close = df['Close']
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> dict:
        """Calculate moving averages and price relationships."""
        close = df['Close']
        current_price = close.iloc[-1]
        
        sma20 = close.rolling(self.sma_short).mean().iloc[-1]
        sma50 = close.rolling(self.sma_medium).mean().iloc[-1]
        sma200 = close.rolling(self.sma_long).mean().iloc[-1] if len(df) >= self.sma_long else sma50
        
        return {
            'sma20': sma20,
            'sma50': sma50,
            'sma200': sma200,
            'price_vs_sma20': ((current_price - sma20) / sma20) * 100,
            'price_vs_sma50': ((current_price - sma50) / sma50) * 100,
            'price_vs_sma200': ((current_price - sma200) / sma200) * 100,
            'sma20_vs_sma50': ((sma20 - sma50) / sma50) * 100
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> tuple[float, str]:
        """Analyze volume patterns."""
        if 'Volume' not in df.columns:
            return 1.0, "unknown"
        
        volume = df['Volume']
        avg_volume = volume.rolling(20).mean()
        current_volume = volume.iloc[-1]
        
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
        
        # Determine volume trend
        recent_avg = volume.iloc[-5:].mean()
        older_avg = volume.iloc[-20:-5].mean()
        
        if recent_avg > older_avg * 1.2:
            volume_trend = "increasing"
        elif recent_avg < older_avg * 0.8:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"
        
        return volume_ratio, volume_trend
    
    def _count_swing_points(self, df: pd.DataFrame) -> tuple[int, int]:
        """Count higher highs and lower lows in lookback period."""
        high = df['High'].iloc[-self.lookback_swing:]
        low = df['Low'].iloc[-self.lookback_swing:]
        
        higher_highs = 0
        lower_lows = 0
        
        for i in range(2, len(high)):
            # Check for higher high
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i-1] > high.iloc[i-2]:
                higher_highs += 1
            # Check for lower low
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i-1] < low.iloc[i-2]:
                lower_lows += 1
        
        return higher_highs, lower_lows
    
    def _classify_trend_strength(self, adx: float) -> TrendStrength:
        """Classify trend strength based on ADX."""
        if adx >= 50:
            return TrendStrength.VERY_STRONG
        elif adx >= 25:
            return TrendStrength.STRONG
        elif adx >= 20:
            return TrendStrength.MODERATE
        elif adx >= 15:
            return TrendStrength.WEAK
        else:
            return TrendStrength.ABSENT
    
    def _classify_volatility(self, atr_percent: float) -> VolatilityLevel:
        """Classify volatility based on ATR percentage."""
        if atr_percent >= 4.0:
            return VolatilityLevel.EXTREME
        elif atr_percent >= 2.5:
            return VolatilityLevel.HIGH
        elif atr_percent >= 1.0:
            return VolatilityLevel.NORMAL
        else:
            return VolatilityLevel.LOW
    
    def _determine_regime(
        self,
        adx: float,
        plus_di: float,
        minus_di: float,
        ma_data: dict,
        higher_highs: int,
        lower_lows: int,
        volatility: VolatilityLevel
    ) -> tuple[MarketRegime, float]:
        """Determine market regime from indicators."""
        trend_direction = (plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        trend_strength = self._classify_trend_strength(adx)
        
        # High volatility override
        if volatility == VolatilityLevel.EXTREME:
            return MarketRegime.HIGH_VOLATILITY, trend_direction
        
        # Low volatility/consolidation
        if volatility == VolatilityLevel.LOW and trend_strength == TrendStrength.ABSENT:
            return MarketRegime.CONSOLIDATING, trend_direction
        
        # Strong trends
        if trend_strength in (TrendStrength.VERY_STRONG, TrendStrength.STRONG):
            if trend_direction > 0.3:
                if higher_highs > lower_lows:
                    return MarketRegime.STRONG_UPTREND, trend_direction
                return MarketRegime.UPTREND, trend_direction
            elif trend_direction < -0.3:
                if lower_lows > higher_highs:
                    return MarketRegime.STRONG_DOWNTREND, trend_direction
                return MarketRegime.DOWNTREND, trend_direction
        
        # Moderate/weak trends
        if trend_strength in (TrendStrength.MODERATE, TrendStrength.WEAK):
            if trend_direction > 0.2:
                return MarketRegime.WEAK_UPTREND, trend_direction
            elif trend_direction < -0.2:
                return MarketRegime.WEAK_DOWNTREND, trend_direction
        
        # No clear trend
        if trend_strength == TrendStrength.ABSENT:
            # Check for Wyckoff phases using price position relative to MAs
            price_position = ma_data['price_vs_sma50']
            
            if -2 < price_position < 2:
                if higher_highs > lower_lows:
                    return MarketRegime.ACCUMULATION, trend_direction
                elif lower_lows > higher_highs:
                    return MarketRegime.DISTRIBUTION, trend_direction
                return MarketRegime.RANGING, trend_direction
        
        # Choppy market (conflicting signals)
        if higher_highs > 0 and lower_lows > 0:
            return MarketRegime.CHOPPY, trend_direction
        
        return MarketRegime.RANGING, trend_direction
    
    def _get_risk_level(self, regime: MarketRegime, volatility: VolatilityLevel) -> str:
        """Determine risk level."""
        high_risk_regimes = {
            MarketRegime.STRONG_DOWNTREND, MarketRegime.DOWNTREND,
            MarketRegime.HIGH_VOLATILITY, MarketRegime.CHOPPY,
            MarketRegime.MARKDOWN
        }
        
        low_risk_regimes = {
            MarketRegime.STRONG_UPTREND, MarketRegime.UPTREND,
            MarketRegime.MARKUP, MarketRegime.ACCUMULATION
        }
        
        if regime in high_risk_regimes or volatility == VolatilityLevel.EXTREME:
            return "high"
        elif regime in low_risk_regimes and volatility != VolatilityLevel.HIGH:
            return "low"
        else:
            return "medium"
    
    def _get_position_modifier(self, risk_level: str, volatility: VolatilityLevel) -> float:
        """Get position sizing modifier."""
        base_modifier = {
            "low": 1.2,
            "medium": 1.0,
            "high": 0.6
        }[risk_level]
        
        volatility_adjustment = {
            VolatilityLevel.EXTREME: 0.5,
            VolatilityLevel.HIGH: 0.75,
            VolatilityLevel.NORMAL: 1.0,
            VolatilityLevel.LOW: 1.1
        }[volatility]
        
        return min(1.5, max(0.3, base_modifier * volatility_adjustment))
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Analyze market regime for a symbol.
        
        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            
        Returns:
            RegimeAnalysis with full regime detection
        """
        if len(df) < max(self.sma_long, self.adx_period * 2):
            logger.warning(f"Insufficient data for {symbol}, need {self.sma_long} bars")
            return RegimeAnalysis(
                symbol=symbol,
                regime=MarketRegime.UNKNOWN,
                volatility=VolatilityLevel.NORMAL,
                trend_strength=TrendStrength.ABSENT,
                trend_direction=0.0,
                adx=0.0,
                atr=0.0,
                atr_percent=0.0,
                rsi=50.0,
                price_vs_sma20=0.0,
                price_vs_sma50=0.0,
                price_vs_sma200=0.0,
                sma20_vs_sma50=0.0,
                volume_ratio=1.0,
                volume_trend="unknown",
                higher_highs=0,
                lower_lows=0,
                recommended_strategies=["insufficient_data"],
                risk_level="high",
                position_sizing_modifier=0.5,
                confidence=0.0
            )
        
        # Calculate indicators
        adx, plus_di, minus_di = self._calculate_adx(df)
        atr, atr_percent = self._calculate_atr(df)
        rsi = self._calculate_rsi(df)
        ma_data = self._calculate_moving_averages(df)
        volume_ratio, volume_trend = self._analyze_volume(df)
        higher_highs, lower_lows = self._count_swing_points(df)
        
        # Classify volatility
        volatility = self._classify_volatility(atr_percent)
        
        # Determine regime
        regime, trend_direction = self._determine_regime(
            adx, plus_di, minus_di, ma_data,
            higher_highs, lower_lows, volatility
        )
        
        # Get trend strength
        trend_strength = self._classify_trend_strength(adx)
        
        # Get recommendations
        recommended_strategies = self.REGIME_STRATEGIES.get(regime, ["watch"])
        risk_level = self._get_risk_level(regime, volatility)
        position_modifier = self._get_position_modifier(risk_level, volatility)
        
        # Calculate confidence based on indicator agreement
        confidence = min(1.0, adx / 50 + abs(trend_direction) * 0.3)
        
        return RegimeAnalysis(
            symbol=symbol,
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            adx=adx,
            atr=atr,
            atr_percent=atr_percent,
            rsi=rsi,
            price_vs_sma20=ma_data['price_vs_sma20'],
            price_vs_sma50=ma_data['price_vs_sma50'],
            price_vs_sma200=ma_data['price_vs_sma200'],
            sma20_vs_sma50=ma_data['sma20_vs_sma50'],
            volume_ratio=volume_ratio,
            volume_trend=volume_trend,
            higher_highs=higher_highs,
            lower_lows=lower_lows,
            recommended_strategies=recommended_strategies,
            risk_level=risk_level,
            position_sizing_modifier=position_modifier,
            confidence=confidence
        )
    
    def get_regime_description(self, regime: MarketRegime) -> str:
        """Get human-readable description of regime."""
        descriptions = {
            MarketRegime.STRONG_UPTREND: "Strong bullish momentum with clear higher highs",
            MarketRegime.UPTREND: "Bullish trend with good momentum",
            MarketRegime.WEAK_UPTREND: "Mild bullish bias, momentum fading",
            MarketRegime.STRONG_DOWNTREND: "Strong bearish momentum with lower lows",
            MarketRegime.DOWNTREND: "Bearish trend with selling pressure",
            MarketRegime.WEAK_DOWNTREND: "Mild bearish bias, may be reversing",
            MarketRegime.RANGING: "Sideways movement between support/resistance",
            MarketRegime.CONSOLIDATING: "Low volatility consolidation, breakout pending",
            MarketRegime.CHOPPY: "Conflicting signals, high noise",
            MarketRegime.HIGH_VOLATILITY: "Extreme volatility, use caution",
            MarketRegime.LOW_VOLATILITY: "Compressed volatility, expect expansion",
            MarketRegime.ACCUMULATION: "Smart money accumulating, potential bottom",
            MarketRegime.MARKUP: "Breakout from accumulation, strong buying",
            MarketRegime.DISTRIBUTION: "Smart money distributing, potential top",
            MarketRegime.MARKDOWN: "Breakdown from distribution, strong selling",
            MarketRegime.UNKNOWN: "Insufficient data for analysis"
        }
        return descriptions.get(regime, "Unknown regime")
