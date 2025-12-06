"""
Options Flow Analysis - Detect unusual options activity and smart money flows.

Features:
- Unusual volume detection
- Large block trades
- Sweep detection
- Put/Call ratio analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Type of options flow."""
    CALL_BUY = "call_buy"
    CALL_SELL = "call_sell"
    PUT_BUY = "put_buy"
    PUT_SELL = "put_sell"
    SPREAD = "spread"
    UNKNOWN = "unknown"


class TradeType(Enum):
    """Type of options trade."""
    BLOCK = "block"  # Large single trade
    SWEEP = "sweep"  # Aggressive multi-exchange
    SPLIT = "split"  # Split across strikes
    NORMAL = "normal"


@dataclass
class OptionsFlow:
    """Individual options flow event."""
    
    symbol: str
    
    # Contract details
    strike: float
    expiry: datetime
    contract_type: str  # 'call' or 'put'
    
    # Trade details
    premium: float  # Total premium paid
    size: int  # Number of contracts
    price: float  # Price per contract
    
    # Classification
    flow_type: FlowType = FlowType.UNKNOWN
    trade_type: TradeType = TradeType.NORMAL
    
    # Context
    underlying_price: float = 0.0
    implied_volatility: float = 0.0
    open_interest: int = 0
    daily_volume: int = 0
    
    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_otm(self) -> bool:
        """Check if option is out of the money."""
        if self.underlying_price <= 0:
            return False
        
        if self.contract_type.lower() == 'call':
            return self.strike > self.underlying_price
        else:
            return self.strike < self.underlying_price
    
    @property
    def moneyness(self) -> float:
        """Calculate moneyness (strike / underlying)."""
        if self.underlying_price <= 0:
            return 1.0
        return self.strike / self.underlying_price
    
    @property
    def days_to_expiry(self) -> int:
        """Days until expiration."""
        delta = self.expiry - datetime.now(timezone.utc)
        return max(0, delta.days)
    
    @property
    def volume_to_oi_ratio(self) -> float:
        """Volume to open interest ratio."""
        if self.open_interest <= 0:
            return float('inf') if self.daily_volume > 0 else 0
        return self.daily_volume / self.open_interest


@dataclass
class UnusualActivity:
    """Unusual options activity alert."""
    
    symbol: str
    flow: OptionsFlow
    
    # Alert details
    alert_type: str  # 'unusual_volume', 'large_block', 'sweep', etc.
    significance: float  # 0-1 scale
    
    # Context
    reason: str = ""
    
    # Signals
    bullish_probability: float = 0.5
    
    @property
    def is_bullish(self) -> bool:
        """Check if activity suggests bullish sentiment."""
        return self.bullish_probability > 0.5


@dataclass
class OptionsFlowSummary:
    """Summary of options flow for a symbol."""
    
    symbol: str
    
    # Volume metrics
    total_call_volume: int = 0
    total_put_volume: int = 0
    total_call_premium: float = 0.0
    total_put_premium: float = 0.0
    
    # Ratios
    put_call_ratio: float = 1.0
    put_call_premium_ratio: float = 1.0
    
    # Unusual activity
    unusual_activity: List[UnusualActivity] = field(default_factory=list)
    
    # Summary metrics
    net_sentiment: float = 0.0  # -1 to 1
    large_trades_count: int = 0
    sweep_count: int = 0
    
    # Timeframe
    period: str = "daily"


class OptionsFlowAnalyzer:
    """
    Analyzes options flow for trading signals.
    
    Features:
    - Unusual volume detection
    - Block and sweep trade identification
    - Put/Call ratio analysis
    - Smart money flow tracking
    """
    
    # Thresholds
    LARGE_PREMIUM_THRESHOLD = 100000  # $100k+
    BLOCK_SIZE_THRESHOLD = 100  # 100+ contracts
    UNUSUAL_VOLUME_RATIO = 3.0  # 3x average
    HIGH_VOL_OI_RATIO = 2.0  # Volume > 2x OI
    
    def __init__(self):
        """Initialize options flow analyzer."""
        self.flows: Dict[str, List[OptionsFlow]] = defaultdict(list)
        self.unusual_activity: Dict[str, List[UnusualActivity]] = defaultdict(list)
        
        # Historical averages
        self.avg_volume: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def add_flow(self, flow: OptionsFlow):
        """
        Add an options flow event.
        
        Args:
            flow: OptionsFlow to add
        """
        self.flows[flow.symbol].append(flow)
        
        # Check for unusual activity
        unusual = self._check_unusual(flow)
        if unusual:
            self.unusual_activity[flow.symbol].append(unusual)
            logger.info(
                f"Unusual options activity: {flow.symbol} "
                f"{flow.contract_type} ${flow.strike} - {unusual.alert_type}"
            )
        
        # Cleanup old data (keep last 24 hours)
        self._cleanup_old_data(flow.symbol)
    
    def _check_unusual(self, flow: OptionsFlow) -> Optional[UnusualActivity]:
        """Check if flow is unusual."""
        # Large block trade
        if flow.premium >= self.LARGE_PREMIUM_THRESHOLD:
            # Significance: 0.6 at threshold, 1.0 at $250k+
            significance = min(1.0, 0.6 + (flow.premium - self.LARGE_PREMIUM_THRESHOLD) / 375000)
            return UnusualActivity(
                symbol=flow.symbol,
                flow=flow,
                alert_type='large_block',
                significance=significance,
                reason=f"Large premium: ${flow.premium:,.0f}",
                bullish_probability=0.7 if flow.flow_type in [FlowType.CALL_BUY, FlowType.PUT_SELL] else 0.3
            )
        
        # High volume to OI ratio
        if flow.volume_to_oi_ratio >= self.HIGH_VOL_OI_RATIO:
            return UnusualActivity(
                symbol=flow.symbol,
                flow=flow,
                alert_type='unusual_volume',
                significance=min(1.0, flow.volume_to_oi_ratio / 5),
                reason=f"Volume/OI ratio: {flow.volume_to_oi_ratio:.1f}x",
                bullish_probability=0.6 if flow.contract_type.lower() == 'call' else 0.4
            )
        
        # Sweep trade
        if flow.trade_type == TradeType.SWEEP:
            return UnusualActivity(
                symbol=flow.symbol,
                flow=flow,
                alert_type='sweep',
                significance=0.8,
                reason="Aggressive sweep order",
                bullish_probability=0.75 if flow.flow_type == FlowType.CALL_BUY else 0.25
            )
        
        return None
    
    def _cleanup_old_data(self, symbol: str, hours: int = 24):
        """Remove old flow data."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        self.flows[symbol] = [
            f for f in self.flows[symbol]
            if f.timestamp > cutoff
        ]
        
        self.unusual_activity[symbol] = [
            a for a in self.unusual_activity[symbol]
            if a.flow.timestamp > cutoff
        ]
    
    def classify_flow(
        self,
        contract_type: str,
        trade_side: str  # 'bid', 'ask', 'mid'
    ) -> FlowType:
        """
        Classify the flow type based on trade location.
        
        Args:
            contract_type: 'call' or 'put'
            trade_side: Where trade executed
            
        Returns:
            FlowType classification
        """
        is_call = contract_type.lower() == 'call'
        
        if trade_side == 'ask':
            # Bought at ask = aggressive buy
            return FlowType.CALL_BUY if is_call else FlowType.PUT_BUY
        elif trade_side == 'bid':
            # Sold at bid = aggressive sell
            return FlowType.CALL_SELL if is_call else FlowType.PUT_SELL
        else:
            return FlowType.UNKNOWN
    
    def get_flow_summary(
        self,
        symbol: str,
        hours: int = 24
    ) -> OptionsFlowSummary:
        """
        Get options flow summary for a symbol.
        
        Args:
            symbol: Stock symbol
            hours: Lookback period
            
        Returns:
            OptionsFlowSummary
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        flows = [f for f in self.flows.get(symbol, []) if f.timestamp > cutoff]
        
        if not flows:
            return OptionsFlowSummary(symbol=symbol)
        
        # Calculate metrics
        call_volume = sum(f.size for f in flows if f.contract_type.lower() == 'call')
        put_volume = sum(f.size for f in flows if f.contract_type.lower() == 'put')
        call_premium = sum(f.premium for f in flows if f.contract_type.lower() == 'call')
        put_premium = sum(f.premium for f in flows if f.contract_type.lower() == 'put')
        
        # Ratios
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
        put_call_premium_ratio = put_premium / call_premium if call_premium > 0 else 1.0
        
        # Large trades
        large_trades = [f for f in flows if f.premium >= self.LARGE_PREMIUM_THRESHOLD]
        sweeps = [f for f in flows if f.trade_type == TradeType.SWEEP]
        
        # Net sentiment (-1 to 1)
        bullish_premium = sum(
            f.premium for f in flows
            if f.flow_type in [FlowType.CALL_BUY, FlowType.PUT_SELL]
        )
        bearish_premium = sum(
            f.premium for f in flows
            if f.flow_type in [FlowType.PUT_BUY, FlowType.CALL_SELL]
        )
        
        total_premium = bullish_premium + bearish_premium
        if total_premium > 0:
            net_sentiment = (bullish_premium - bearish_premium) / total_premium
        else:
            net_sentiment = 0.0
        
        return OptionsFlowSummary(
            symbol=symbol,
            total_call_volume=call_volume,
            total_put_volume=put_volume,
            total_call_premium=call_premium,
            total_put_premium=put_premium,
            put_call_ratio=put_call_ratio,
            put_call_premium_ratio=put_call_premium_ratio,
            unusual_activity=self.unusual_activity.get(symbol, []),
            net_sentiment=net_sentiment,
            large_trades_count=len(large_trades),
            sweep_count=len(sweeps)
        )
    
    def get_unusual_activity(
        self,
        symbols: Optional[List[str]] = None,
        min_significance: float = 0.5
    ) -> List[UnusualActivity]:
        """
        Get unusual activity alerts.
        
        Args:
            symbols: Filter by symbols (None for all)
            min_significance: Minimum significance threshold
            
        Returns:
            List of UnusualActivity alerts
        """
        all_activity = []
        
        for symbol, activities in self.unusual_activity.items():
            if symbols is not None and symbol not in symbols:
                continue
            
            for activity in activities:
                if activity.significance >= min_significance:
                    all_activity.append(activity)
        
        # Sort by significance
        return sorted(all_activity, key=lambda a: a.significance, reverse=True)
    
    def get_put_call_ratio(
        self,
        symbol: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get put/call ratios for a symbol.
        
        Args:
            symbol: Stock symbol
            hours: Lookback period
            
        Returns:
            Dict with volume and premium P/C ratios
        """
        summary = self.get_flow_summary(symbol, hours)
        
        return {
            'volume_ratio': summary.put_call_ratio,
            'premium_ratio': summary.put_call_premium_ratio,
            'interpretation': self._interpret_pc_ratio(summary.put_call_ratio)
        }
    
    def _interpret_pc_ratio(self, ratio: float) -> str:
        """Interpret put/call ratio."""
        if ratio < 0.5:
            return "Very bullish (low put activity)"
        elif ratio < 0.8:
            return "Bullish (moderate put activity)"
        elif ratio < 1.2:
            return "Neutral"
        elif ratio < 1.5:
            return "Bearish (elevated put activity)"
        else:
            return "Very bearish (high put activity)"
    
    def get_smart_money_signals(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get smart money signals from options flow.
        
        Args:
            symbols: Symbols to analyze
            
        Returns:
            Dict of signals per symbol
        """
        signals = {}
        
        for symbol in symbols:
            summary = self.get_flow_summary(symbol)
            
            signal = 'neutral'
            confidence = 0.5
            reasons = []
            
            # Check unusual activity
            unusual = summary.unusual_activity
            bullish_unusual = [a for a in unusual if a.is_bullish]
            bearish_unusual = [a for a in unusual if not a.is_bullish]
            
            if len(bullish_unusual) > len(bearish_unusual) + 2:
                signal = 'bullish'
                confidence = 0.7
                reasons.append(f"{len(bullish_unusual)} bullish unusual activities")
            elif len(bearish_unusual) > len(bullish_unusual) + 2:
                signal = 'bearish'
                confidence = 0.7
                reasons.append(f"{len(bearish_unusual)} bearish unusual activities")
            
            # Check net sentiment
            if summary.net_sentiment > 0.5:
                if signal != 'bearish':
                    signal = 'bullish'
                    confidence = max(confidence, 0.6)
                reasons.append(f"Net bullish flow ({summary.net_sentiment:.1%})")
            elif summary.net_sentiment < -0.5:
                if signal != 'bullish':
                    signal = 'bearish'
                    confidence = max(confidence, 0.6)
                reasons.append(f"Net bearish flow ({summary.net_sentiment:.1%})")
            
            # Large trades
            if summary.large_trades_count > 0:
                reasons.append(f"{summary.large_trades_count} large trades")
            
            signals[symbol] = {
                'signal': signal,
                'confidence': confidence,
                'reasons': reasons,
                'put_call_ratio': summary.put_call_ratio,
                'net_sentiment': summary.net_sentiment,
                'unusual_count': len(unusual)
            }
        
        return signals
