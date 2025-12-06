"""
Alternative Data Manager - Unified interface for alternative data sources.

Combines:
- SEC Filings
- Economic Indicators
- Earnings Calendar
- Options Flow (future)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum

from .sec_filings import SECFilingsTracker, InsiderTrade, InstitutionalHolding, TransactionType
from .economic_data import (
    EconomicIndicators, EconomicRelease, EconomicEvent,
    IndicatorType, ImpactLevel
)

logger = logging.getLogger(__name__)


class AlternativeSignal(Enum):
    """Signal from alternative data."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class InsiderSentiment:
    """Insider trading sentiment analysis."""
    
    symbol: str
    
    # Transaction counts
    buy_count: int = 0
    sell_count: int = 0
    
    # Values
    buy_value: float = 0.0
    sell_value: float = 0.0
    
    # Insider counts
    unique_buyers: int = 0
    unique_sellers: int = 0
    
    # Timeframe
    period_days: int = 90
    
    @property
    def net_value(self) -> float:
        """Net insider buying."""
        return self.buy_value - self.sell_value
    
    @property
    def buy_sell_ratio(self) -> float:
        """Ratio of buys to sells."""
        if self.sell_count == 0:
            return float('inf') if self.buy_count > 0 else 1.0
        return self.buy_count / self.sell_count
    
    @property
    def signal(self) -> AlternativeSignal:
        """Get trading signal from insider activity."""
        ratio = self.buy_sell_ratio
        
        if ratio >= 3:
            return AlternativeSignal.STRONG_BULLISH
        elif ratio >= 1.5:
            return AlternativeSignal.BULLISH
        elif ratio <= 0.33:
            return AlternativeSignal.STRONG_BEARISH
        elif ratio <= 0.67:
            return AlternativeSignal.BEARISH
        else:
            return AlternativeSignal.NEUTRAL


@dataclass
class InstitutionalSentiment:
    """Institutional ownership sentiment analysis."""
    
    symbol: str
    
    # Ownership metrics
    institutional_pct: float = 0.0
    institutional_change: float = 0.0  # Change in %
    
    # New positions vs exits
    new_positions: int = 0
    increased_positions: int = 0
    decreased_positions: int = 0
    exited_positions: int = 0
    
    # Period
    period: str = "quarterly"
    
    @property
    def net_change_ratio(self) -> float:
        """Ratio of increases to decreases."""
        increases = self.new_positions + self.increased_positions
        decreases = self.decreased_positions + self.exited_positions
        
        if decreases == 0:
            return float('inf') if increases > 0 else 1.0
        return increases / decreases
    
    @property
    def signal(self) -> AlternativeSignal:
        """Get trading signal from institutional activity."""
        ratio = self.net_change_ratio
        change = self.institutional_change
        
        if ratio >= 2 and change > 2:
            return AlternativeSignal.STRONG_BULLISH
        elif ratio >= 1.5 or change > 1:
            return AlternativeSignal.BULLISH
        elif ratio <= 0.5 and change < -2:
            return AlternativeSignal.STRONG_BEARISH
        elif ratio <= 0.67 or change < -1:
            return AlternativeSignal.BEARISH
        else:
            return AlternativeSignal.NEUTRAL


@dataclass
class EarningsEvent:
    """Upcoming earnings event."""
    
    symbol: str
    company_name: str
    
    # Timing
    report_date: datetime
    report_time: str = "after_close"  # before_open, after_close, unknown
    
    # Estimates
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None
    
    # Previous
    eps_previous: Optional[float] = None
    revenue_previous: Optional[float] = None
    
    # Options implied move
    implied_move: Optional[float] = None
    
    @property
    def days_until(self) -> int:
        """Days until earnings."""
        delta = self.report_date - datetime.now(timezone.utc)
        # Use total_seconds for more accurate day calculation
        days = delta.total_seconds() / 86400  # seconds per day
        return max(0, round(days))


@dataclass
class AlternativeDataSummary:
    """Summary of all alternative data for a symbol."""
    
    symbol: str
    
    # Sentiments
    insider_sentiment: Optional[InsiderSentiment] = None
    institutional_sentiment: Optional[InstitutionalSentiment] = None
    
    # Upcoming events
    next_earnings: Optional[EarningsEvent] = None
    upcoming_filings: List[str] = field(default_factory=list)
    
    # Economic context
    economic_outlook: str = "neutral"
    
    # Overall signal
    overall_signal: AlternativeSignal = AlternativeSignal.NEUTRAL
    confidence: float = 0.5
    
    # Reasons
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)


class AlternativeDataManager:
    """
    Unified manager for alternative data sources.
    
    Combines:
    - SEC filings and insider trading
    - Economic indicators
    - Earnings calendar
    - Institutional ownership
    """
    
    def __init__(
        self,
        sec_tracker: Optional[SECFilingsTracker] = None,
        economic_indicators: Optional[EconomicIndicators] = None
    ):
        """Initialize data manager."""
        self.sec_tracker = sec_tracker or SECFilingsTracker()
        self.economic = economic_indicators or EconomicIndicators()
        
        # Earnings calendar cache
        self.earnings_calendar: Dict[str, EarningsEvent] = {}
        
        logger.info("AlternativeDataManager initialized")
    
    def get_insider_sentiment(
        self,
        symbol: str,
        days: int = 90
    ) -> InsiderSentiment:
        """
        Get insider trading sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Lookback period
            
        Returns:
            InsiderSentiment analysis
        """
        trades = self.sec_tracker.get_recent_trades(symbol, days=days)
        
        buy_count = 0
        sell_count = 0
        buy_value = 0.0
        sell_value = 0.0
        buyers = set()
        sellers = set()
        
        for trade in trades:
            if trade.transaction_type in [TransactionType.PURCHASE, TransactionType.AWARD]:  # Purchase or Award
                buy_count += 1
                buy_value += trade.value
                buyers.add(trade.insider_name)
            elif trade.transaction_type in [TransactionType.SALE]:  # Sale
                sell_count += 1
                sell_value += trade.value
                sellers.add(trade.insider_name)
        
        return InsiderSentiment(
            symbol=symbol,
            buy_count=buy_count,
            sell_count=sell_count,
            buy_value=buy_value,
            sell_value=sell_value,
            unique_buyers=len(buyers),
            unique_sellers=len(sellers),
            period_days=days
        )
    
    def get_institutional_sentiment(
        self,
        symbol: str
    ) -> InstitutionalSentiment:
        """
        Get institutional ownership sentiment.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            InstitutionalSentiment analysis
        """
        holdings = self.sec_tracker.get_institutional_holdings(symbol)
        
        if not holdings:
            return InstitutionalSentiment(symbol=symbol)
        
        # Calculate metrics
        total_shares = sum(h.shares for h in holdings)
        
        new_positions = 0
        increased = 0
        decreased = 0
        exited = 0
        
        for h in holdings:
            change_pct = h.shares_change_pct
            
            if change_pct >= 100:  # New position or doubled
                new_positions += 1
            elif change_pct > 5:
                increased += 1
            elif change_pct < -50:
                exited += 1
            elif change_pct < -5:
                decreased += 1
        
        # Estimate institutional ownership (simplified)
        institutional_pct = min(100, len(holdings) * 2)  # Rough estimate
        
        return InstitutionalSentiment(
            symbol=symbol,
            institutional_pct=institutional_pct,
            institutional_change=0.0,  # Would need historical data
            new_positions=new_positions,
            increased_positions=increased,
            decreased_positions=decreased,
            exited_positions=exited
        )
    
    def add_earnings_event(self, event: EarningsEvent):
        """Add an earnings event to calendar."""
        self.earnings_calendar[event.symbol] = event
        logger.info(f"Added earnings event: {event.symbol} on {event.report_date}")
    
    def get_upcoming_earnings(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 14
    ) -> List[EarningsEvent]:
        """
        Get upcoming earnings events.
        
        Args:
            symbols: Filter by symbols (None for all)
            days: Days to look ahead
            
        Returns:
            List of upcoming earnings events
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days)
        
        events = []
        for symbol, event in self.earnings_calendar.items():
            if symbols is not None and symbol not in symbols:
                continue
            
            if now <= event.report_date <= cutoff:
                events.append(event)
        
        return sorted(events, key=lambda e: e.report_date)
    
    def get_data_summary(self, symbol: str) -> AlternativeDataSummary:
        """
        Get comprehensive alternative data summary.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            AlternativeDataSummary with all data
        """
        bullish_factors = []
        bearish_factors = []
        
        # Get insider sentiment
        insider = self.get_insider_sentiment(symbol)
        if insider.signal == AlternativeSignal.STRONG_BULLISH:
            bullish_factors.append(f"Strong insider buying ({insider.buy_count} buys)")
        elif insider.signal == AlternativeSignal.BULLISH:
            bullish_factors.append("Net insider buying")
        elif insider.signal == AlternativeSignal.STRONG_BEARISH:
            bearish_factors.append(f"Heavy insider selling ({insider.sell_count} sells)")
        elif insider.signal == AlternativeSignal.BEARISH:
            bearish_factors.append("Net insider selling")
        
        # Get institutional sentiment
        institutional = self.get_institutional_sentiment(symbol)
        if institutional.signal == AlternativeSignal.STRONG_BULLISH:
            bullish_factors.append("Strong institutional accumulation")
        elif institutional.signal == AlternativeSignal.BULLISH:
            bullish_factors.append("Institutional buying")
        elif institutional.signal == AlternativeSignal.STRONG_BEARISH:
            bearish_factors.append("Heavy institutional selling")
        elif institutional.signal == AlternativeSignal.BEARISH:
            bearish_factors.append("Institutional distribution")
        
        # Get earnings
        next_earnings = self.earnings_calendar.get(symbol)
        if next_earnings and next_earnings.days_until <= 7:
            if next_earnings.implied_move:
                bearish_factors.append(
                    f"Earnings in {next_earnings.days_until} days "
                    f"(±{next_earnings.implied_move:.1f}% implied)"
                )
        
        # Get economic outlook
        econ_outlook = self.economic.get_economic_outlook()
        if econ_outlook['overall'] == 'bullish':
            bullish_factors.append("Bullish economic conditions")
        elif econ_outlook['overall'] == 'bearish':
            bearish_factors.append("Bearish economic conditions")
        
        # Calculate overall signal
        bullish_score = len(bullish_factors)
        bearish_score = len(bearish_factors)
        
        if bullish_score >= 3 and bearish_score <= 1:
            overall_signal = AlternativeSignal.STRONG_BULLISH
            confidence = 0.8
        elif bullish_score > bearish_score:
            overall_signal = AlternativeSignal.BULLISH
            confidence = 0.6
        elif bearish_score >= 3 and bullish_score <= 1:
            overall_signal = AlternativeSignal.STRONG_BEARISH
            confidence = 0.8
        elif bearish_score > bullish_score:
            overall_signal = AlternativeSignal.BEARISH
            confidence = 0.6
        else:
            overall_signal = AlternativeSignal.NEUTRAL
            confidence = 0.5
        
        return AlternativeDataSummary(
            symbol=symbol,
            insider_sentiment=insider,
            institutional_sentiment=institutional,
            next_earnings=next_earnings,
            economic_outlook=econ_outlook['overall'],
            overall_signal=overall_signal,
            confidence=confidence,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors
        )
    
    def get_high_conviction_signals(
        self,
        symbols: List[str],
        min_confidence: float = 0.7
    ) -> Dict[str, AlternativeDataSummary]:
        """
        Get high conviction signals from alternative data.
        
        Args:
            symbols: List of symbols to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dict of high conviction signals
        """
        signals = {}
        
        for symbol in symbols:
            try:
                summary = self.get_data_summary(symbol)
                
                if summary.confidence >= min_confidence:
                    signals[symbol] = summary
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def get_economic_calendar(
        self,
        days: int = 7,
        high_impact_only: bool = True
    ) -> List[EconomicEvent]:
        """
        Get economic calendar.
        
        Args:
            days: Days to look ahead
            high_impact_only: Only include high impact events
            
        Returns:
            List of economic events
        """
        impact_filter = ImpactLevel.HIGH if high_impact_only else None
        return self.economic.get_calendar(days=days, impact_filter=impact_filter)
    
    def analyze_economic_release(
        self,
        release: EconomicRelease
    ) -> Dict[str, Any]:
        """
        Analyze an economic release for trading implications.
        
        Args:
            release: EconomicRelease to analyze
            
        Returns:
            Analysis dict
        """
        impact = self.economic.analyze_release(release)
        
        return {
            'indicator': release.indicator.value,
            'actual': release.actual,
            'forecast': release.forecast,
            'surprise': release.surprise,
            'beat_expectations': release.beat_expectations,
            'equity_impact': impact.equity_impact,
            'bond_impact': impact.bond_impact,
            'dollar_impact': impact.dollar_impact,
            'sector_impacts': impact.sector_impacts,
            'bullish_for': impact.bullish_for,
            'bearish_for': impact.bearish_for,
            'summary': impact.summary
        }
