"""
Liquidity Mapper - Identifies key liquidity zones and order flow areas.

Features:
- Support/Resistance zones
- Order block detection
- Fair Value Gap (FVG) identification
- Liquidity pools (swing highs/lows)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Types of liquidity zones."""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    ORDER_BLOCK_BULLISH = "order_block_bullish"
    ORDER_BLOCK_BEARISH = "order_block_bearish"
    FVG_BULLISH = "fvg_bullish"
    FVG_BEARISH = "fvg_bearish"
    LIQUIDITY_POOL_HIGH = "liquidity_pool_high"
    LIQUIDITY_POOL_LOW = "liquidity_pool_low"
    PIVOT_POINT = "pivot_point"


@dataclass
class LiquidityZone:
    """A liquidity zone on the chart."""
    zone_type: ZoneType
    price_low: float
    price_high: float
    strength: float  # 0-1, how significant
    touches: int  # Times price has touched
    created_at: datetime
    last_tested: Optional[datetime] = None
    broken: bool = False
    
    @property
    def mid_price(self) -> float:
        """Middle of the zone."""
        return (self.price_low + self.price_high) / 2
    
    @property
    def width(self) -> float:
        """Width of zone in price."""
        return self.price_high - self.price_low
    
    @property
    def width_percent(self) -> float:
        """Width as percentage of mid price."""
        return (self.width / self.mid_price) * 100
    
    def contains_price(self, price: float) -> bool:
        """Check if price is within zone."""
        return self.price_low <= price <= self.price_high
    
    def distance_to_price(self, price: float) -> float:
        """Distance from price to nearest zone edge."""
        if self.contains_price(price):
            return 0.0
        return min(abs(price - self.price_low), abs(price - self.price_high))
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'type': self.zone_type.value,
            'low': self.price_low,
            'high': self.price_high,
            'mid': self.mid_price,
            'strength': self.strength,
            'touches': self.touches,
            'broken': self.broken
        }


@dataclass
class LiquidityAnalysis:
    """Complete liquidity analysis result."""
    symbol: str
    current_price: float
    zones: list[LiquidityZone]
    nearest_support: Optional[LiquidityZone]
    nearest_resistance: Optional[LiquidityZone]
    fvgs: list[LiquidityZone]
    order_blocks: list[LiquidityZone]
    liquidity_pools: list[LiquidityZone]
    
    # Analysis metrics
    support_strength: float  # Aggregate support below
    resistance_strength: float  # Aggregate resistance above
    liquidity_bias: str  # "bullish", "bearish", "neutral"
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def support_distance_percent(self) -> float:
        """Distance to nearest support as percentage."""
        if not self.nearest_support:
            return float('inf')
        return ((self.current_price - self.nearest_support.price_high) / 
                self.current_price) * 100
    
    @property
    def resistance_distance_percent(self) -> float:
        """Distance to nearest resistance as percentage."""
        if not self.nearest_resistance:
            return float('inf')
        return ((self.nearest_resistance.price_low - self.current_price) / 
                self.current_price) * 100
    
    def get_zones_in_range(
        self, 
        price_low: float, 
        price_high: float
    ) -> list[LiquidityZone]:
        """Get zones within price range."""
        return [
            z for z in self.zones
            if z.price_high >= price_low and z.price_low <= price_high
        ]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'nearest_support': self.nearest_support.to_dict() if self.nearest_support else None,
            'nearest_resistance': self.nearest_resistance.to_dict() if self.nearest_resistance else None,
            'support_strength': self.support_strength,
            'resistance_strength': self.resistance_strength,
            'liquidity_bias': self.liquidity_bias,
            'zone_count': len(self.zones),
            'fvg_count': len(self.fvgs),
            'order_block_count': len(self.order_blocks)
        }


class LiquidityMapper:
    """
    Maps liquidity zones and key price levels.
    
    Identifies:
    - Support/Resistance from swing points
    - Order blocks from reversal candles
    - Fair Value Gaps from price imbalance
    - Liquidity pools at swing extremes
    """
    
    def __init__(
        self,
        swing_lookback: int = 10,
        zone_merge_percent: float = 0.5,
        min_zone_touches: int = 2,
        fvg_min_size_percent: float = 0.3
    ):
        """
        Initialize Liquidity Mapper.
        
        Args:
            swing_lookback: Bars to look for swing points
            zone_merge_percent: Merge zones within this % of each other
            min_zone_touches: Minimum touches to confirm zone
            fvg_min_size_percent: Minimum FVG size as % of price
        """
        self.swing_lookback = swing_lookback
        self.zone_merge_percent = zone_merge_percent
        self.min_zone_touches = min_zone_touches
        self.fvg_min_size_percent = fvg_min_size_percent
    
    def _find_swing_highs(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        """Find swing high points."""
        highs = df['High'].values
        swing_highs = []
        
        for i in range(self.swing_lookback, len(highs) - self.swing_lookback):
            is_swing = True
            for j in range(1, self.swing_lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swing_highs.append((i, highs[i]))
        
        return swing_highs
    
    def _find_swing_lows(self, df: pd.DataFrame) -> list[tuple[int, float]]:
        """Find swing low points."""
        lows = df['Low'].values
        swing_lows = []
        
        for i in range(self.swing_lookback, len(lows) - self.swing_lookback):
            is_swing = True
            for j in range(1, self.swing_lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swing_lows.append((i, lows[i]))
        
        return swing_lows
    
    def _create_support_resistance_zones(
        self,
        df: pd.DataFrame,
        swing_highs: list[tuple[int, float]],
        swing_lows: list[tuple[int, float]]
    ) -> list[LiquidityZone]:
        """Create S/R zones from swing points."""
        zones = []
        current_price = df['Close'].iloc[-1]
        avg_range = (df['High'] - df['Low']).mean()
        zone_buffer = avg_range * 0.5
        
        # Create resistance zones from swing highs
        for idx, price in swing_highs:
            zones.append(LiquidityZone(
                zone_type=ZoneType.RESISTANCE if price > current_price else ZoneType.SUPPORT,
                price_low=price - zone_buffer,
                price_high=price + zone_buffer,
                strength=0.5,
                touches=1,
                created_at=datetime.now(timezone.utc)
            ))
        
        # Create support zones from swing lows
        for idx, price in swing_lows:
            zones.append(LiquidityZone(
                zone_type=ZoneType.SUPPORT if price < current_price else ZoneType.RESISTANCE,
                price_low=price - zone_buffer,
                price_high=price + zone_buffer,
                strength=0.5,
                touches=1,
                created_at=datetime.now(timezone.utc)
            ))
        
        return zones
    
    def _merge_overlapping_zones(
        self,
        zones: list[LiquidityZone]
    ) -> list[LiquidityZone]:
        """Merge zones that are close together."""
        if not zones:
            return []
        
        # Sort by price
        zones = sorted(zones, key=lambda z: z.mid_price)
        merged = [zones[0]]
        
        for zone in zones[1:]:
            last = merged[-1]
            
            # Check if zones should merge
            distance_percent = abs(zone.mid_price - last.mid_price) / last.mid_price * 100
            
            if distance_percent <= self.zone_merge_percent and zone.zone_type == last.zone_type:
                # Merge zones
                last.price_low = min(last.price_low, zone.price_low)
                last.price_high = max(last.price_high, zone.price_high)
                last.touches += zone.touches
                last.strength = min(1.0, last.strength + 0.2)
            else:
                merged.append(zone)
        
        return merged
    
    def _find_order_blocks(self, df: pd.DataFrame) -> list[LiquidityZone]:
        """Find order blocks (last opposing candle before strong move)."""
        order_blocks = []
        
        opens = df['Open'].values
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        
        avg_body = abs(closes - opens).mean()
        
        for i in range(2, len(df) - 1):
            # Check for bullish order block (bearish candle before bullish move)
            prev_bearish = closes[i-1] < opens[i-1]
            curr_bullish = closes[i] > opens[i]
            strong_move = (closes[i] - opens[i]) > avg_body * 2
            
            if prev_bearish and curr_bullish and strong_move:
                order_blocks.append(LiquidityZone(
                    zone_type=ZoneType.ORDER_BLOCK_BULLISH,
                    price_low=lows[i-1],
                    price_high=opens[i-1],
                    strength=0.7,
                    touches=0,
                    created_at=datetime.now(timezone.utc)
                ))
            
            # Check for bearish order block (bullish candle before bearish move)
            prev_bullish = closes[i-1] > opens[i-1]
            curr_bearish = closes[i] < opens[i]
            strong_down = (opens[i] - closes[i]) > avg_body * 2
            
            if prev_bullish and curr_bearish and strong_down:
                order_blocks.append(LiquidityZone(
                    zone_type=ZoneType.ORDER_BLOCK_BEARISH,
                    price_low=closes[i-1],
                    price_high=highs[i-1],
                    strength=0.7,
                    touches=0,
                    created_at=datetime.now(timezone.utc)
                ))
        
        return order_blocks
    
    def _find_fair_value_gaps(self, df: pd.DataFrame) -> list[LiquidityZone]:
        """Find Fair Value Gaps (imbalances)."""
        fvgs = []
        
        highs = df['High'].values
        lows = df['Low'].values
        current_price = df['Close'].iloc[-1]
        
        for i in range(2, len(df)):
            # Bullish FVG: gap between candle 1 high and candle 3 low
            if lows[i] > highs[i-2]:
                gap_size = lows[i] - highs[i-2]
                gap_percent = (gap_size / current_price) * 100
                
                if gap_percent >= self.fvg_min_size_percent:
                    fvgs.append(LiquidityZone(
                        zone_type=ZoneType.FVG_BULLISH,
                        price_low=highs[i-2],
                        price_high=lows[i],
                        strength=min(1.0, gap_percent / 2),
                        touches=0,
                        created_at=datetime.now(timezone.utc)
                    ))
            
            # Bearish FVG: gap between candle 3 high and candle 1 low
            if highs[i] < lows[i-2]:
                gap_size = lows[i-2] - highs[i]
                gap_percent = (gap_size / current_price) * 100
                
                if gap_percent >= self.fvg_min_size_percent:
                    fvgs.append(LiquidityZone(
                        zone_type=ZoneType.FVG_BEARISH,
                        price_low=highs[i],
                        price_high=lows[i-2],
                        strength=min(1.0, gap_percent / 2),
                        touches=0,
                        created_at=datetime.now(timezone.utc)
                    ))
        
        return fvgs
    
    def _find_liquidity_pools(
        self,
        swing_highs: list[tuple[int, float]],
        swing_lows: list[tuple[int, float]]
    ) -> list[LiquidityZone]:
        """Find liquidity pools at swing extremes."""
        pools = []
        
        # Group similar swing highs (stop hunt targets)
        if swing_highs:
            sorted_highs = sorted(swing_highs, key=lambda x: x[1], reverse=True)
            for _, price in sorted_highs[:5]:  # Top 5 highs
                pools.append(LiquidityZone(
                    zone_type=ZoneType.LIQUIDITY_POOL_HIGH,
                    price_low=price,
                    price_high=price * 1.002,  # Small buffer above
                    strength=0.6,
                    touches=1,
                    created_at=datetime.now(timezone.utc)
                ))
        
        # Group similar swing lows (stop hunt targets)
        if swing_lows:
            sorted_lows = sorted(swing_lows, key=lambda x: x[1])
            for _, price in sorted_lows[:5]:  # Bottom 5 lows
                pools.append(LiquidityZone(
                    zone_type=ZoneType.LIQUIDITY_POOL_LOW,
                    price_low=price * 0.998,  # Small buffer below
                    price_high=price,
                    strength=0.6,
                    touches=1,
                    created_at=datetime.now(timezone.utc)
                ))
        
        return pools
    
    def _count_zone_touches(
        self,
        zone: LiquidityZone,
        df: pd.DataFrame
    ) -> int:
        """Count how many times price touched a zone."""
        touches = 0
        highs = df['High'].values
        lows = df['Low'].values
        
        for i in range(len(df)):
            # Check if candle touched zone
            if lows[i] <= zone.price_high and highs[i] >= zone.price_low:
                touches += 1
        
        return touches
    
    def _find_nearest_zones(
        self,
        zones: list[LiquidityZone],
        current_price: float
    ) -> tuple[Optional[LiquidityZone], Optional[LiquidityZone]]:
        """Find nearest support and resistance."""
        supports = [z for z in zones if z.price_high < current_price and 
                   z.zone_type in (ZoneType.SUPPORT, ZoneType.ORDER_BLOCK_BULLISH, 
                                   ZoneType.FVG_BULLISH)]
        resistances = [z for z in zones if z.price_low > current_price and
                      z.zone_type in (ZoneType.RESISTANCE, ZoneType.ORDER_BLOCK_BEARISH,
                                      ZoneType.FVG_BEARISH)]
        
        nearest_support = None
        nearest_resistance = None
        
        if supports:
            nearest_support = max(supports, key=lambda z: z.price_high)
        
        if resistances:
            nearest_resistance = min(resistances, key=lambda z: z.price_low)
        
        return nearest_support, nearest_resistance
    
    def _calculate_liquidity_bias(
        self,
        current_price: float,
        support_strength: float,
        resistance_strength: float,
        nearest_support: Optional[LiquidityZone],
        nearest_resistance: Optional[LiquidityZone]
    ) -> str:
        """Determine liquidity bias."""
        # Consider strength and distance
        support_factor = support_strength
        resistance_factor = resistance_strength
        
        if nearest_support:
            support_distance = (current_price - nearest_support.price_high) / current_price
            support_factor *= (1 - min(support_distance, 0.1) * 5)
        
        if nearest_resistance:
            resistance_distance = (nearest_resistance.price_low - current_price) / current_price
            resistance_factor *= (1 - min(resistance_distance, 0.1) * 5)
        
        if support_factor > resistance_factor * 1.2:
            return "bullish"
        elif resistance_factor > support_factor * 1.2:
            return "bearish"
        else:
            return "neutral"
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> LiquidityAnalysis:
        """
        Analyze liquidity zones for a symbol.
        
        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            
        Returns:
            LiquidityAnalysis with all zones
        """
        if len(df) < self.swing_lookback * 2 + 5:
            logger.warning(f"Insufficient data for {symbol}")
            return LiquidityAnalysis(
                symbol=symbol,
                current_price=df['Close'].iloc[-1] if len(df) > 0 else 0,
                zones=[],
                nearest_support=None,
                nearest_resistance=None,
                fvgs=[],
                order_blocks=[],
                liquidity_pools=[],
                support_strength=0.0,
                resistance_strength=0.0,
                liquidity_bias="neutral"
            )
        
        current_price = df['Close'].iloc[-1]
        
        # Find swing points
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)
        
        # Create S/R zones
        sr_zones = self._create_support_resistance_zones(df, swing_highs, swing_lows)
        sr_zones = self._merge_overlapping_zones(sr_zones)
        
        # Find order blocks
        order_blocks = self._find_order_blocks(df)
        
        # Find FVGs
        fvgs = self._find_fair_value_gaps(df)
        
        # Find liquidity pools
        liquidity_pools = self._find_liquidity_pools(swing_highs, swing_lows)
        
        # Update touch counts
        all_zones = sr_zones + order_blocks + fvgs + liquidity_pools
        for zone in all_zones:
            zone.touches = self._count_zone_touches(zone, df)
        
        # Filter by minimum touches for S/R zones
        sr_zones = [z for z in sr_zones if z.touches >= self.min_zone_touches]
        
        # Find nearest zones
        nearest_support, nearest_resistance = self._find_nearest_zones(all_zones, current_price)
        
        # Calculate aggregate strength
        support_zones = [z for z in all_zones if z.price_high < current_price]
        resistance_zones = [z for z in all_zones if z.price_low > current_price]
        
        support_strength = sum(z.strength for z in support_zones) / max(len(support_zones), 1)
        resistance_strength = sum(z.strength for z in resistance_zones) / max(len(resistance_zones), 1)
        
        # Determine bias
        liquidity_bias = self._calculate_liquidity_bias(
            current_price, support_strength, resistance_strength,
            nearest_support, nearest_resistance
        )
        
        return LiquidityAnalysis(
            symbol=symbol,
            current_price=current_price,
            zones=all_zones,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            fvgs=fvgs,
            order_blocks=order_blocks,
            liquidity_pools=liquidity_pools,
            support_strength=support_strength,
            resistance_strength=resistance_strength,
            liquidity_bias=liquidity_bias
        )
