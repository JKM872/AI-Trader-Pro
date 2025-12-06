"""
SEC Filings Tracker - Monitor insider trading and institutional holdings.

Tracks:
- Form 4 (Insider trading)
- Form 13F (Institutional holdings)
- Form 8-K (Material events)
- Form 10-K/10-Q (Annual/Quarterly reports)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any
import re

logger = logging.getLogger(__name__)


class FilingType(Enum):
    """SEC filing types."""
    FORM_4 = "4"           # Insider trading
    FORM_13F = "13F-HR"    # Institutional holdings
    FORM_8K = "8-K"        # Material events
    FORM_10K = "10-K"      # Annual report
    FORM_10Q = "10-Q"      # Quarterly report
    FORM_S1 = "S-1"        # IPO registration
    FORM_DEF14A = "DEF 14A"  # Proxy statement


class TransactionType(Enum):
    """Insider transaction types."""
    PURCHASE = "P"
    SALE = "S"
    EXERCISE = "M"
    GIFT = "G"
    AWARD = "A"


@dataclass
class InsiderTrade:
    """Individual insider transaction."""
    
    symbol: str
    insider_name: str
    insider_title: str
    
    # Transaction details
    transaction_type: TransactionType
    transaction_date: datetime
    shares: int
    price: float
    
    # Position after transaction
    shares_owned_after: int
    
    # Filing info
    filing_date: datetime
    filing_url: Optional[str] = None
    
    # Computed
    @property
    def value(self) -> float:
        """Transaction value."""
        return self.shares * self.price
    
    @property
    def is_purchase(self) -> bool:
        """Check if this is a purchase."""
        return self.transaction_type == TransactionType.PURCHASE
    
    @property
    def is_sale(self) -> bool:
        """Check if this is a sale."""
        return self.transaction_type == TransactionType.SALE
    
    @property
    def ownership_change_pct(self) -> float:
        """Percentage change in ownership."""
        if self.shares_owned_after == 0:
            return -100.0  # Sold everything
        
        prev_shares = self.shares_owned_after - self.shares if self.is_purchase else self.shares_owned_after + self.shares
        if prev_shares <= 0:
            return 100.0
        
        return (self.shares / prev_shares) * 100


@dataclass
class InstitutionalHolding:
    """Institutional holding from 13F filing."""
    
    symbol: str
    institution_name: str
    
    # Position
    shares: int
    value: float
    
    # Change from previous quarter
    shares_change: int = 0
    shares_change_pct: float = 0.0
    
    # Filing info
    report_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filing_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Ownership percentage
    ownership_pct: float = 0.0
    
    @property
    def is_new_position(self) -> bool:
        """Check if this is a new position."""
        return self.shares_change == self.shares
    
    @property
    def is_increased(self) -> bool:
        """Check if position was increased."""
        return self.shares_change > 0
    
    @property
    def is_decreased(self) -> bool:
        """Check if position was decreased."""
        return self.shares_change < 0
    
    @property
    def is_closed(self) -> bool:
        """Check if position was closed."""
        return self.shares == 0 and self.shares_change < 0


@dataclass
class InsiderSentiment:
    """Aggregated insider sentiment for a symbol."""
    
    symbol: str
    period_days: int
    
    # Trade counts
    total_trades: int = 0
    purchases: int = 0
    sales: int = 0
    
    # Values
    purchase_value: float = 0.0
    sale_value: float = 0.0
    net_value: float = 0.0
    
    # Unique insiders
    unique_buyers: int = 0
    unique_sellers: int = 0
    
    # Sentiment
    sentiment_score: float = 0.0  # -1 to 1
    confidence: float = 0.0
    
    # Signal
    signal: str = "neutral"  # bullish, bearish, neutral
    
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SECFilingsTracker:
    """
    Tracks SEC filings for trading signals.
    
    Features:
    - Insider transaction monitoring
    - Institutional holdings tracking
    - Sentiment analysis from filings
    - Alert generation
    """
    
    def __init__(self):
        """Initialize SEC filings tracker."""
        self.insider_trades: Dict[str, List[InsiderTrade]] = {}
        self.institutional_holdings: Dict[str, List[InstitutionalHolding]] = {}
        self.sentiment_cache: Dict[str, InsiderSentiment] = {}
        
        # Notable insiders (CEOs, CFOs are more significant)
        self.key_titles = ['ceo', 'cfo', 'president', 'director', 'chairman']
    
    def add_insider_trade(self, trade: InsiderTrade):
        """Add an insider trade."""
        symbol = trade.symbol.upper()
        
        if symbol not in self.insider_trades:
            self.insider_trades[symbol] = []
        
        self.insider_trades[symbol].append(trade)
        
        # Keep sorted by date
        self.insider_trades[symbol].sort(
            key=lambda t: t.transaction_date,
            reverse=True
        )
        
        # Keep last 200 trades
        self.insider_trades[symbol] = self.insider_trades[symbol][:200]
        
        # Invalidate sentiment cache
        if symbol in self.sentiment_cache:
            del self.sentiment_cache[symbol]
        
        logger.info(
            f"Insider trade: {trade.insider_name} "
            f"{trade.transaction_type.value} {trade.shares} {symbol} "
            f"@ ${trade.price:.2f}"
        )
    
    def add_institutional_holding(self, holding: InstitutionalHolding):
        """Add institutional holding."""
        symbol = holding.symbol.upper()
        
        if symbol not in self.institutional_holdings:
            self.institutional_holdings[symbol] = []
        
        self.institutional_holdings[symbol].append(holding)
        
        # Keep sorted by filing date
        self.institutional_holdings[symbol].sort(
            key=lambda h: h.filing_date,
            reverse=True
        )
        
        # Keep last 100
        self.institutional_holdings[symbol] = self.institutional_holdings[symbol][:100]
    
    def get_institutional_holdings(self, symbol: str) -> List[InstitutionalHolding]:
        """
        Get institutional holdings for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of institutional holdings
        """
        symbol = symbol.upper()
        return self.institutional_holdings.get(symbol, [])
    
    def get_insider_sentiment(
        self,
        symbol: str,
        days: int = 90
    ) -> InsiderSentiment:
        """
        Calculate insider sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Days to analyze
            
        Returns:
            InsiderSentiment analysis
        """
        symbol = symbol.upper()
        cache_key = f"{symbol}_{days}"
        
        # Check cache
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            age = datetime.now(timezone.utc) - cached.analyzed_at
            if age < timedelta(hours=1):
                return cached
        
        if symbol not in self.insider_trades:
            return InsiderSentiment(symbol=symbol, period_days=days)
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        trades = [
            t for t in self.insider_trades[symbol]
            if t.transaction_date > cutoff
        ]
        
        if not trades:
            return InsiderSentiment(symbol=symbol, period_days=days)
        
        # Count and sum
        purchases = [t for t in trades if t.is_purchase]
        sales = [t for t in trades if t.is_sale]
        
        purchase_value = sum(t.value for t in purchases)
        sale_value = sum(t.value for t in sales)
        net_value = purchase_value - sale_value
        
        # Unique insiders
        unique_buyers = len(set(t.insider_name for t in purchases))
        unique_sellers = len(set(t.insider_name for t in sales))
        
        # Calculate sentiment score
        total_value = purchase_value + sale_value
        if total_value > 0:
            sentiment_score = net_value / total_value
        else:
            sentiment_score = 0.0
        
        # Boost for key insiders
        key_purchases = [
            t for t in purchases
            if any(title in t.insider_title.lower() for title in self.key_titles)
        ]
        if key_purchases:
            sentiment_score = min(sentiment_score + 0.2, 1.0)
        
        # Determine signal
        if sentiment_score > 0.3:
            signal = "bullish"
        elif sentiment_score < -0.3:
            signal = "bearish"
        else:
            signal = "neutral"
        
        # Confidence based on trade count
        confidence = min(len(trades) * 0.1, 0.9)
        
        result = InsiderSentiment(
            symbol=symbol,
            period_days=days,
            total_trades=len(trades),
            purchases=len(purchases),
            sales=len(sales),
            purchase_value=purchase_value,
            sale_value=sale_value,
            net_value=net_value,
            unique_buyers=unique_buyers,
            unique_sellers=unique_sellers,
            sentiment_score=sentiment_score,
            confidence=confidence,
            signal=signal
        )
        
        self.sentiment_cache[cache_key] = result
        return result
    
    def get_institutional_activity(
        self,
        symbol: str,
        quarters: int = 2
    ) -> Dict[str, Any]:
        """
        Get institutional activity summary.
        
        Args:
            symbol: Stock symbol
            quarters: Quarters to analyze
            
        Returns:
            Activity summary dict
        """
        symbol = symbol.upper()
        
        if symbol not in self.institutional_holdings:
            return {
                'symbol': symbol,
                'total_institutions': 0,
                'new_positions': 0,
                'increased': 0,
                'decreased': 0,
                'closed': 0,
                'net_change': 0,
                'sentiment': 'neutral'
            }
        
        holdings = self.institutional_holdings[symbol]
        
        new_positions = sum(1 for h in holdings if h.is_new_position)
        increased = sum(1 for h in holdings if h.is_increased and not h.is_new_position)
        decreased = sum(1 for h in holdings if h.is_decreased and not h.is_closed)
        closed = sum(1 for h in holdings if h.is_closed)
        
        net_change = sum(h.shares_change for h in holdings)
        
        # Determine sentiment
        positive = new_positions + increased
        negative = decreased + closed
        
        if positive > negative * 1.5:
            sentiment = "bullish"
        elif negative > positive * 1.5:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        return {
            'symbol': symbol,
            'total_institutions': len(holdings),
            'new_positions': new_positions,
            'increased': increased,
            'decreased': decreased,
            'closed': closed,
            'net_change': net_change,
            'sentiment': sentiment,
            'top_buyers': self._get_top_buyers(symbol),
            'top_sellers': self._get_top_sellers(symbol)
        }
    
    def _get_top_buyers(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Get top institutional buyers."""
        if symbol not in self.institutional_holdings:
            return []
        
        buyers = [
            h for h in self.institutional_holdings[symbol]
            if h.is_increased
        ]
        
        sorted_buyers = sorted(
            buyers,
            key=lambda h: h.shares_change,
            reverse=True
        )
        
        return [
            {
                'institution': h.institution_name,
                'shares_added': h.shares_change,
                'total_shares': h.shares,
                'value': h.value
            }
            for h in sorted_buyers[:limit]
        ]
    
    def _get_top_sellers(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Get top institutional sellers."""
        if symbol not in self.institutional_holdings:
            return []
        
        sellers = [
            h for h in self.institutional_holdings[symbol]
            if h.is_decreased
        ]
        
        sorted_sellers = sorted(
            sellers,
            key=lambda h: h.shares_change
        )
        
        return [
            {
                'institution': h.institution_name,
                'shares_sold': abs(h.shares_change),
                'remaining_shares': h.shares,
                'value': h.value
            }
            for h in sorted_sellers[:limit]
        ]
    
    def get_recent_trades(
        self,
        symbol: str,
        days: int = 30,
        limit: int = 20
    ) -> List[InsiderTrade]:
        """Get recent insider trades."""
        symbol = symbol.upper()
        
        if symbol not in self.insider_trades:
            return []
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        trades = [
            t for t in self.insider_trades[symbol]
            if t.transaction_date > cutoff
        ]
        
        return trades[:limit]
    
    def get_cluster_buying(
        self,
        min_insiders: int = 3,
        days: int = 30
    ) -> List[Dict]:
        """
        Find symbols with cluster buying (multiple insiders buying).
        
        Args:
            min_insiders: Minimum unique insider buyers
            days: Days to look back
            
        Returns:
            List of symbols with cluster buying
        """
        results = []
        
        for symbol in self.insider_trades:
            sentiment = self.get_insider_sentiment(symbol, days)
            
            if sentiment.unique_buyers >= min_insiders and sentiment.signal == "bullish":
                results.append({
                    'symbol': symbol,
                    'unique_buyers': sentiment.unique_buyers,
                    'purchases': sentiment.purchases,
                    'purchase_value': sentiment.purchase_value,
                    'sentiment_score': sentiment.sentiment_score
                })
        
        # Sort by number of unique buyers
        results.sort(key=lambda x: x['unique_buyers'], reverse=True)
        
        return results
