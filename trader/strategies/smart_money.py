"""
Smart Money Strategy - Institutional trading strategy.

Based on Smart Money Concepts (SMC) used by professional traders:
- Order Blocks
- Fair Value Gaps
- Liquidity Sweeps
- Break of Structure
- Change of Character
- Multi-timeframe Analysis

This strategy aims to trade with institutional order flow.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .base import TradingStrategy, Signal, SignalType
from ..analysis.indicators import TradingViewIndicators, TrendDirection

logger = logging.getLogger(__name__)


class SMCPattern(Enum):
    """Smart Money Concept patterns."""
    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    BREAK_OF_STRUCTURE = "break_of_structure"
    CHANGE_OF_CHARACTER = "change_of_character"
    INDUCEMENT = "inducement"
    MITIGATION = "mitigation"


@dataclass
class SMCSignal:
    """Smart Money signal with additional context."""
    pattern: SMCPattern
    direction: TrendDirection
    price_zone: Tuple[float, float]  # (low, high)
    strength: float
    reason: str


class SmartMoneyStrategy(TradingStrategy):
    """
    Smart Money Concepts (SMC) based trading strategy.
    
    This strategy identifies institutional trading patterns:
    1. Market Structure Analysis (trend direction)
    2. Order Block Detection (entry zones)
    3. Fair Value Gap Analysis (imbalances)
    4. Liquidity Analysis (stop hunts)
    5. Multi-timeframe Confluence
    
    Entry Rules:
    - Trade in direction of higher timeframe trend
    - Enter at order blocks or FVGs
    - Confirm with lower timeframe structure
    
    Exit Rules:
    - Target opposing liquidity pools
    - Use ATR-based stops at structure
    """
    
    def __init__(self,
                 swing_length: int = 5,
                 ob_sensitivity: int = 3,
                 use_fvg: bool = True,
                 use_liquidity: bool = True,
                 min_rr_ratio: float = 2.0,
                 **kwargs):
        """
        Initialize Smart Money Strategy.
        
        Args:
            swing_length: Length for swing detection
            ob_sensitivity: Sensitivity for order block detection
            use_fvg: Whether to use Fair Value Gaps
            use_liquidity: Whether to analyze liquidity
            min_rr_ratio: Minimum risk/reward ratio
        """
        super().__init__(name="SmartMoneyStrategy", **kwargs)
        
        self.swing_length = swing_length
        self.ob_sensitivity = ob_sensitivity
        self.use_fvg = use_fvg
        self.use_liquidity = use_liquidity
        self.min_rr_ratio = min_rr_ratio
        self.indicators = TradingViewIndicators()
    
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                        **kwargs) -> Signal:
        """
        Generate trading signal based on Smart Money Concepts.
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            
        Returns:
            Signal with SMC-based recommendation
        """
        if not self.validate_data(data, min_rows=50):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0,
                reasons=["Insufficient data for SMC analysis"]
            )
        
        current_price = data['Close'].iloc[-1]
        
        # 1. Analyze Market Structure
        structure = self._analyze_structure(data)
        
        # 2. Identify Order Blocks
        order_blocks = self._identify_order_blocks(data)
        
        # 3. Find Fair Value Gaps
        fvg = self._find_fair_value_gaps(data) if self.use_fvg else {'bullish': [], 'bearish': []}
        
        # 4. Analyze Liquidity Zones
        liquidity = self._analyze_liquidity(data) if self.use_liquidity else {}
        
        # 5. Get Higher Timeframe Bias
        htf_bias = self._get_htf_bias(data)
        
        # 6. Check for entry signals
        smc_signals = self._find_smc_signals(
            data, structure, order_blocks, fvg, liquidity, htf_bias
        )
        
        # 7. Generate final signal
        return self._create_signal(
            symbol, current_price, smc_signals, structure, htf_bias, data
        )
    
    def _analyze_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure for trend direction."""
        structure = TradingViewIndicators.market_structure(data, self.swing_length)
        
        # Add recent price action
        recent_data = data.tail(20)
        higher_highs = 0
        lower_lows = 0
        
        highs = recent_data['High'].values
        lows = recent_data['Low'].values
        
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                higher_highs += 1
            if lows[i] < lows[i-1]:
                lower_lows += 1
        
        structure['higher_highs'] = higher_highs
        structure['lower_lows'] = lower_lows
        structure['bullish_structure'] = higher_highs > lower_lows
        
        return structure
    
    def _identify_order_blocks(self, data: pd.DataFrame) -> Dict:
        """Identify valid order blocks."""
        obs = TradingViewIndicators.order_block_detection(data, self.ob_sensitivity)
        
        current_price = data['Close'].iloc[-1]
        
        # Filter for relevant order blocks (within 5% of current price)
        relevant_obs = {
            'bullish': [],
            'bearish': []
        }
        
        for ob in obs['bullish'][-10:]:  # Last 10 bullish OBs
            distance_pct = abs(current_price - ob['high']) / current_price * 100
            if distance_pct < 5:
                ob['distance'] = distance_pct
                ob['is_tested'] = self._check_ob_tested(data, ob, 'bullish')
                relevant_obs['bullish'].append(ob)
        
        for ob in obs['bearish'][-10:]:  # Last 10 bearish OBs
            distance_pct = abs(current_price - ob['low']) / current_price * 100
            if distance_pct < 5:
                ob['distance'] = distance_pct
                ob['is_tested'] = self._check_ob_tested(data, ob, 'bearish')
                relevant_obs['bearish'].append(ob)
        
        return relevant_obs
    
    def _check_ob_tested(self, data: pd.DataFrame, ob: Dict, ob_type: str) -> bool:
        """Check if order block has been tested/mitigated."""
        ob_idx_result = data.index.get_loc(ob['index'])
        # Handle different return types from get_loc
        if isinstance(ob_idx_result, slice):
            ob_idx = ob_idx_result.start or 0
        elif isinstance(ob_idx_result, np.ndarray):
            ob_idx = int(ob_idx_result[0]) if len(ob_idx_result) > 0 else 0
        else:
            ob_idx = int(ob_idx_result)
        
        for i in range(ob_idx + 1, len(data)):
            if ob_type == 'bullish':
                if data['Low'].iloc[i] < ob['high']:
                    return True
            else:
                if data['High'].iloc[i] > ob['low']:
                    return True
        
        return False
    
    def _find_fair_value_gaps(self, data: pd.DataFrame) -> Dict:
        """Find unmitigated Fair Value Gaps."""
        fvg = TradingViewIndicators.fair_value_gap(data)
        
        current_price = data['Close'].iloc[-1]
        
        # Filter for unfilled FVGs within range
        unfilled_fvg = {
            'bullish': [],
            'bearish': []
        }
        
        for gap in fvg['bullish'][-5:]:
            if current_price > gap['bottom']:
                gap['filled'] = current_price < gap['bottom']
                if not gap['filled']:
                    unfilled_fvg['bullish'].append(gap)
        
        for gap in fvg['bearish'][-5:]:
            if current_price < gap['top']:
                gap['filled'] = current_price > gap['top']
                if not gap['filled']:
                    unfilled_fvg['bearish'].append(gap)
        
        return unfilled_fvg
    
    def _analyze_liquidity(self, data: pd.DataFrame) -> Dict:
        """Analyze liquidity zones (stop clusters)."""
        recent = data.tail(50)
        
        # Find equal highs/lows (liquidity pools)
        tolerance = 0.002  # 0.2% tolerance
        
        liquidity = {
            'buy_side': [],  # Above price (seller stops)
            'sell_side': []  # Below price (buyer stops)
        }
        
        highs = recent['High'].values
        lows = recent['Low'].values
        current_price = data['Close'].iloc[-1]
        
        # Find clusters of similar highs
        for i in range(len(highs) - 3):
            cluster = [highs[i]]
            for j in range(i + 1, min(i + 10, len(highs))):
                if abs(highs[j] - highs[i]) / highs[i] < tolerance:
                    cluster.append(highs[j])
            
            if len(cluster) >= 2 and np.mean(cluster) > current_price:
                liquidity['buy_side'].append({
                    'price': np.mean(cluster),
                    'touches': len(cluster),
                    'strength': len(cluster) / 5  # Normalized strength
                })
        
        # Find clusters of similar lows
        for i in range(len(lows) - 3):
            cluster = [lows[i]]
            for j in range(i + 1, min(i + 10, len(lows))):
                if abs(lows[j] - lows[i]) / lows[i] < tolerance:
                    cluster.append(lows[j])
            
            if len(cluster) >= 2 and np.mean(cluster) < current_price:
                liquidity['sell_side'].append({
                    'price': np.mean(cluster),
                    'touches': len(cluster),
                    'strength': len(cluster) / 5
                })
        
        return liquidity
    
    def _get_htf_bias(self, data: pd.DataFrame) -> TrendDirection:
        """Get higher timeframe bias using multiple indicators."""
        # Supertrend
        st, direction = TradingViewIndicators.supertrend(data)
        st_bias = TrendDirection.BULLISH if direction.iloc[-1] == 1 else TrendDirection.BEARISH
        
        # ADX/DMI
        adx, plus_di, minus_di = TradingViewIndicators.adx_dmi(data)
        dmi_bias = TrendDirection.BULLISH if plus_di.iloc[-1] > minus_di.iloc[-1] else TrendDirection.BEARISH
        
        # Moving Averages
        sma50 = data['Close'].rolling(50).mean().iloc[-1]
        sma200 = data['Close'].rolling(200).mean().iloc[-1] if len(data) >= 200 else sma50
        ma_bias = TrendDirection.BULLISH if sma50 > sma200 else TrendDirection.BEARISH
        
        # Vote
        bullish_votes = sum([
            st_bias == TrendDirection.BULLISH,
            dmi_bias == TrendDirection.BULLISH,
            ma_bias == TrendDirection.BULLISH
        ])
        
        if bullish_votes >= 2:
            return TrendDirection.BULLISH
        elif bullish_votes <= 1:
            return TrendDirection.BEARISH
        
        return TrendDirection.NEUTRAL
    
    def _find_smc_signals(self, data: pd.DataFrame, structure: Dict,
                          order_blocks: Dict, fvg: Dict, liquidity: Dict,
                          htf_bias: TrendDirection) -> List[SMCSignal]:
        """Find tradeable SMC signals."""
        signals = []
        current_price = data['Close'].iloc[-1]
        
        # Check for bullish setups (if HTF is bullish)
        if htf_bias == TrendDirection.BULLISH or htf_bias == TrendDirection.NEUTRAL:
            # Order block entry
            for ob in order_blocks['bullish']:
                if not ob.get('is_tested', True):  # Untested OB
                    if current_price <= ob['high'] * 1.01:  # Price at/near OB
                        signals.append(SMCSignal(
                            pattern=SMCPattern.ORDER_BLOCK,
                            direction=TrendDirection.BULLISH,
                            price_zone=(ob['low'], ob['high']),
                            strength=0.8 if ob['strength'] > 1.5 else 0.6,
                            reason=f"Bullish Order Block at ${ob['low']:.2f}-${ob['high']:.2f}"
                        ))
            
            # FVG entry
            for gap in fvg['bullish']:
                if current_price >= gap['bottom'] and current_price <= gap['top']:
                    signals.append(SMCSignal(
                        pattern=SMCPattern.FAIR_VALUE_GAP,
                        direction=TrendDirection.BULLISH,
                        price_zone=(gap['bottom'], gap['top']),
                        strength=0.7,
                        reason=f"Bullish FVG at ${gap['bottom']:.2f}-${gap['top']:.2f}"
                    ))
        
        # Check for bearish setups (if HTF is bearish)
        if htf_bias == TrendDirection.BEARISH or htf_bias == TrendDirection.NEUTRAL:
            for ob in order_blocks['bearish']:
                if not ob.get('is_tested', True):
                    if current_price >= ob['low'] * 0.99:
                        signals.append(SMCSignal(
                            pattern=SMCPattern.ORDER_BLOCK,
                            direction=TrendDirection.BEARISH,
                            price_zone=(ob['low'], ob['high']),
                            strength=0.8 if ob['strength'] > 1.5 else 0.6,
                            reason=f"Bearish Order Block at ${ob['low']:.2f}-${ob['high']:.2f}"
                        ))
            
            for gap in fvg['bearish']:
                if current_price <= gap['top'] and current_price >= gap['bottom']:
                    signals.append(SMCSignal(
                        pattern=SMCPattern.FAIR_VALUE_GAP,
                        direction=TrendDirection.BEARISH,
                        price_zone=(gap['bottom'], gap['top']),
                        strength=0.7,
                        reason=f"Bearish FVG at ${gap['bottom']:.2f}-${gap['top']:.2f}"
                    ))
        
        # Check for Break of Structure
        if structure['trend'] == TrendDirection.BULLISH and structure.get('last_hh'):
            signals.append(SMCSignal(
                pattern=SMCPattern.BREAK_OF_STRUCTURE,
                direction=TrendDirection.BULLISH,
                price_zone=(current_price * 0.98, current_price * 1.02),
                strength=0.75,
                reason="Bullish Break of Structure detected"
            ))
        elif structure['trend'] == TrendDirection.BEARISH and structure.get('last_ll'):
            signals.append(SMCSignal(
                pattern=SMCPattern.BREAK_OF_STRUCTURE,
                direction=TrendDirection.BEARISH,
                price_zone=(current_price * 0.98, current_price * 1.02),
                strength=0.75,
                reason="Bearish Break of Structure detected"
            ))
        
        return signals
    
    def _create_signal(self, symbol: str, price: float,
                       smc_signals: List[SMCSignal], structure: Dict,
                       htf_bias: TrendDirection, data: pd.DataFrame) -> Signal:
        """Create final trading signal from SMC analysis."""
        
        if not smc_signals:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.3,
                price=price,
                reasons=["No valid SMC setups found"],
                metadata={
                    'htf_bias': htf_bias.value,
                    'structure_trend': structure['trend'].value
                }
            )
        
        # Score signals
        bullish_score = 0
        bearish_score = 0
        reasons = []
        
        for sig in smc_signals:
            if sig.direction == TrendDirection.BULLISH:
                bullish_score += sig.strength
                reasons.append(sig.reason)
            else:
                bearish_score += sig.strength
                reasons.append(sig.reason)
        
        # Add HTF bias weight
        if htf_bias == TrendDirection.BULLISH:
            bullish_score += 0.5
            reasons.append("Higher timeframe bias is BULLISH")
        elif htf_bias == TrendDirection.BEARISH:
            bearish_score += 0.5
            reasons.append("Higher timeframe bias is BEARISH")
        
        # Calculate ATR for stops
        atr = TradingViewIndicators.calculate_atr(data).iloc[-1]
        
        # Determine final signal
        if bullish_score > bearish_score and bullish_score >= 1.0:
            signal_type = SignalType.BUY
            confidence = min(bullish_score / 3, 0.95)
            stop_loss = price - (2 * atr)
            take_profit = price + (3 * atr)  # 1.5 RR minimum
        elif bearish_score > bullish_score and bearish_score >= 1.0:
            signal_type = SignalType.SELL
            confidence = min(bearish_score / 3, 0.95)
            stop_loss = price + (2 * atr)
            take_profit = price - (3 * atr)
        else:
            signal_type = SignalType.HOLD
            confidence = 0.4
            stop_loss = self.calculate_stop_loss(price, SignalType.HOLD)
            take_profit = self.calculate_take_profit(price, SignalType.HOLD)
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
            metadata={
                'htf_bias': htf_bias.value,
                'structure_trend': structure['trend'].value,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'atr': atr,
                'smc_patterns': [sig.pattern.value for sig in smc_signals]
            }
        )


class MultiTimeframeStrategy(TradingStrategy):
    """
    Multi-Timeframe Confluence Strategy.
    
    Combines signals from multiple timeframes:
    - Weekly: Overall trend direction
    - Daily: Swing trading bias
    - 4H/1H: Entry timing
    
    Only trades when all timeframes align.
    """
    
    def __init__(self,
                 htf_weight: float = 0.4,
                 mtf_weight: float = 0.35,
                 ltf_weight: float = 0.25,
                 min_confluence: float = 0.7,
                 **kwargs):
        """
        Initialize Multi-Timeframe Strategy.
        
        Args:
            htf_weight: Weight for higher timeframe
            mtf_weight: Weight for medium timeframe
            ltf_weight: Weight for lower timeframe
            min_confluence: Minimum confluence score for trade
        """
        super().__init__(name="MultiTimeframeStrategy", **kwargs)
        
        self.htf_weight = htf_weight
        self.mtf_weight = mtf_weight
        self.ltf_weight = ltf_weight
        self.min_confluence = min_confluence
        self.indicators = TradingViewIndicators()
    
    def generate_signal(self, symbol: str, data: pd.DataFrame,
                        **kwargs) -> Signal:
        """
        Generate signal based on multi-timeframe analysis.
        
        For intraday data, simulates HTF by aggregating bars.
        """
        if not self.validate_data(data, min_rows=100):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0,
                reasons=["Insufficient data for MTF analysis"]
            )
        
        price = data['Close'].iloc[-1]
        
        # Analyze each "timeframe" by aggregating data
        htf_data = self._aggregate_to_htf(data, factor=5)  # 5x bars = HTF
        mtf_data = self._aggregate_to_htf(data, factor=2)  # 2x bars = MTF
        ltf_data = data  # Current data = LTF
        
        # Get bias from each timeframe
        htf_analysis = self._analyze_timeframe(htf_data, "HTF")
        mtf_analysis = self._analyze_timeframe(mtf_data, "MTF")
        ltf_analysis = self._analyze_timeframe(ltf_data, "LTF")
        
        # Calculate confluence
        confluence_score, direction = self._calculate_confluence(
            htf_analysis, mtf_analysis, ltf_analysis
        )
        
        # Generate signal
        reasons = [
            f"HTF ({htf_analysis['trend'].value}): {htf_analysis['reason']}",
            f"MTF ({mtf_analysis['trend'].value}): {mtf_analysis['reason']}",
            f"LTF ({ltf_analysis['trend'].value}): {ltf_analysis['reason']}",
            f"Confluence Score: {confluence_score:.0%}"
        ]
        
        if confluence_score >= self.min_confluence:
            atr = TradingViewIndicators.calculate_atr(data).iloc[-1]
            
            if direction == TrendDirection.BULLISH:
                signal_type = SignalType.BUY
                stop_loss = price - (2 * atr)
                take_profit = price + (3 * atr)
            else:
                signal_type = SignalType.SELL
                stop_loss = price + (2 * atr)
                take_profit = price - (3 * atr)
            
            confidence = confluence_score
        else:
            signal_type = SignalType.HOLD
            stop_loss = self.calculate_stop_loss(price, SignalType.HOLD)
            take_profit = self.calculate_take_profit(price, SignalType.HOLD)
            confidence = confluence_score
            reasons.append("Insufficient confluence - waiting for alignment")
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons,
            metadata={
                'htf_trend': htf_analysis['trend'].value,
                'mtf_trend': mtf_analysis['trend'].value,
                'ltf_trend': ltf_analysis['trend'].value,
                'confluence_score': confluence_score
            }
        )
    
    def _aggregate_to_htf(self, data: pd.DataFrame, factor: int) -> pd.DataFrame:
        """Aggregate data to simulate higher timeframe."""
        # Group every 'factor' rows
        n = len(data)
        if n < factor:
            return data
        
        # Calculate how many complete groups we have
        n_groups = n // factor
        
        # Use last n_groups * factor rows
        data = data.tail(n_groups * factor).copy()
        
        # Create group index
        data['group'] = np.repeat(range(n_groups), factor)
        
        htf = data.groupby('group').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        return htf
    
    def _analyze_timeframe(self, data: pd.DataFrame, tf_name: str) -> Dict:
        """Analyze a single timeframe."""
        if len(data) < 20:
            return {
                'trend': TrendDirection.NEUTRAL,
                'strength': 0,
                'reason': 'Insufficient data'
            }
        
        # Supertrend
        st, direction = TradingViewIndicators.supertrend(data)
        st_trend = TrendDirection.BULLISH if direction.iloc[-1] == 1 else TrendDirection.BEARISH
        
        # ADX for trend strength
        adx, plus_di, minus_di = TradingViewIndicators.adx_dmi(data)
        trend_strength = min(adx.iloc[-1] / 50, 1.0) if not pd.isna(adx.iloc[-1]) else 0.5
        
        # EMA alignment
        ema9 = data['Close'].ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = data['Close'].ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = data['Close'].ewm(span=50, adjust=False).mean().iloc[-1] if len(data) >= 50 else ema21
        
        ema_bullish = ema9 > ema21 > ema50
        ema_bearish = ema9 < ema21 < ema50
        
        # Determine overall trend
        if st_trend == TrendDirection.BULLISH and ema_bullish:
            trend = TrendDirection.BULLISH
            reason = f"Supertrend UP + EMA aligned bullish (ADX: {adx.iloc[-1]:.1f})"
        elif st_trend == TrendDirection.BEARISH and ema_bearish:
            trend = TrendDirection.BEARISH
            reason = f"Supertrend DOWN + EMA aligned bearish (ADX: {adx.iloc[-1]:.1f})"
        elif st_trend == TrendDirection.BULLISH:
            trend = TrendDirection.BULLISH
            reason = f"Supertrend UP (ADX: {adx.iloc[-1]:.1f})"
            trend_strength *= 0.7  # Reduce strength if EMAs not aligned
        elif st_trend == TrendDirection.BEARISH:
            trend = TrendDirection.BEARISH
            reason = f"Supertrend DOWN (ADX: {adx.iloc[-1]:.1f})"
            trend_strength *= 0.7
        else:
            trend = TrendDirection.NEUTRAL
            reason = "No clear trend"
            trend_strength = 0.3
        
        return {
            'trend': trend,
            'strength': trend_strength,
            'reason': reason,
            'supertrend': st.iloc[-1],
            'adx': adx.iloc[-1]
        }
    
    def _calculate_confluence(self, htf: Dict, mtf: Dict, ltf: Dict) -> Tuple[float, TrendDirection]:
        """Calculate confluence score and direction."""
        # Check alignment
        all_bullish = (htf['trend'] == TrendDirection.BULLISH and
                       mtf['trend'] == TrendDirection.BULLISH and
                       ltf['trend'] == TrendDirection.BULLISH)
        
        all_bearish = (htf['trend'] == TrendDirection.BEARISH and
                       mtf['trend'] == TrendDirection.BEARISH and
                       ltf['trend'] == TrendDirection.BEARISH)
        
        if all_bullish:
            weighted_strength = (
                htf['strength'] * self.htf_weight +
                mtf['strength'] * self.mtf_weight +
                ltf['strength'] * self.ltf_weight
            )
            return weighted_strength, TrendDirection.BULLISH
        
        elif all_bearish:
            weighted_strength = (
                htf['strength'] * self.htf_weight +
                mtf['strength'] * self.mtf_weight +
                ltf['strength'] * self.ltf_weight
            )
            return weighted_strength, TrendDirection.BEARISH
        
        else:
            # Partial alignment - calculate based on majority
            bullish_score = 0
            bearish_score = 0
            
            for tf, weight in [(htf, self.htf_weight), 
                               (mtf, self.mtf_weight), 
                               (ltf, self.ltf_weight)]:
                if tf['trend'] == TrendDirection.BULLISH:
                    bullish_score += tf['strength'] * weight
                elif tf['trend'] == TrendDirection.BEARISH:
                    bearish_score += tf['strength'] * weight
            
            if bullish_score > bearish_score:
                return bullish_score * 0.7, TrendDirection.BULLISH  # Reduced for partial alignment
            elif bearish_score > bullish_score:
                return bearish_score * 0.7, TrendDirection.BEARISH
            else:
                return 0.3, TrendDirection.NEUTRAL


if __name__ == "__main__":
    import yfinance as yf
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Test Smart Money Strategy
    print("Testing Smart Money Strategy")
    print("=" * 50)
    
    # Fetch test data
    stock = yf.Ticker('AAPL')
    data = stock.history(period='6mo')
    
    # Generate signal
    smc_strategy = SmartMoneyStrategy()
    signal = smc_strategy.generate_signal('AAPL', data)
    
    print(f"\nSmart Money Signal for AAPL:")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Price: ${signal.price:.2f}")
    if signal.stop_loss:
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
    if signal.take_profit:
        print(f"  Take Profit: ${signal.take_profit:.2f}")
    print(f"\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
    
    print("\n" + "=" * 50)
    print("Testing Multi-Timeframe Strategy")
    print("=" * 50)
    
    mtf_strategy = MultiTimeframeStrategy()
    signal = mtf_strategy.generate_signal('AAPL', data)
    
    print(f"\nMulti-Timeframe Signal for AAPL:")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Price: ${signal.price:.2f}")
    print(f"\nReasons:")
    for reason in signal.reasons:
        print(f"  - {reason}")
