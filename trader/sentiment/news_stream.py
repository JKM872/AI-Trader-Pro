"""
News Stream Processor - Real-time news sentiment analysis.

Processes news from multiple sources and extracts sentiment
for trading signals.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Callable
import hashlib

logger = logging.getLogger(__name__)


class SentimentScore(Enum):
    """Sentiment classification."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    SLIGHTLY_BULLISH = "slightly_bullish"
    NEUTRAL = "neutral"
    SLIGHTLY_BEARISH = "slightly_bearish"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"
    
    @classmethod
    def from_score(cls, score: float) -> 'SentimentScore':
        """Convert numeric score (-1 to 1) to sentiment."""
        if score >= 0.6:
            return cls.VERY_BULLISH
        elif score >= 0.3:
            return cls.BULLISH
        elif score >= 0.1:
            return cls.SLIGHTLY_BULLISH
        elif score >= -0.1:
            return cls.NEUTRAL
        elif score >= -0.3:
            return cls.SLIGHTLY_BEARISH
        elif score >= -0.6:
            return cls.BEARISH
        else:
            return cls.VERY_BEARISH
    
    def to_numeric(self) -> float:
        """Convert sentiment to numeric score."""
        scores = {
            self.VERY_BULLISH: 0.8,
            self.BULLISH: 0.5,
            self.SLIGHTLY_BULLISH: 0.2,
            self.NEUTRAL: 0.0,
            self.SLIGHTLY_BEARISH: -0.2,
            self.BEARISH: -0.5,
            self.VERY_BEARISH: -0.8,
        }
        return scores.get(self, 0.0)


@dataclass
class NewsItem:
    """Individual news item."""
    
    title: str
    source: str
    url: str
    published_at: datetime
    
    # Content
    summary: Optional[str] = None
    full_text: Optional[str] = None
    
    # Metadata
    symbols: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    author: Optional[str] = None
    
    # Unique identifier
    id: str = field(default="")
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from title and source
            content = f"{self.title}:{self.source}:{self.published_at.isoformat()}"
            self.id = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class NewsSentiment:
    """Sentiment analysis result for a news item."""
    
    news_item: NewsItem
    sentiment: SentimentScore
    confidence: float  # 0.0 to 1.0
    
    # Detailed scores
    title_sentiment: float = 0.0
    content_sentiment: float = 0.0
    
    # Impact assessment
    relevance_score: float = 0.0  # How relevant to the symbol
    impact_score: float = 0.0     # Expected market impact
    
    # Keywords found
    bullish_keywords: List[str] = field(default_factory=list)
    bearish_keywords: List[str] = field(default_factory=list)
    
    # Analysis timestamp
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def weighted_score(self) -> float:
        """Get confidence-weighted sentiment score."""
        return self.sentiment.to_numeric() * self.confidence


class NewsStreamProcessor:
    """
    Real-time news stream processor with sentiment analysis.
    
    Features:
    - Keyword-based sentiment analysis
    - Symbol relevance scoring
    - News deduplication
    - Sentiment trend tracking
    """
    
    # Sentiment keywords
    BULLISH_KEYWORDS = [
        'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'climb', 'upgrade',
        'beat', 'exceed', 'outperform', 'breakthrough', 'bullish', 'buy',
        'strong', 'growth', 'profit', 'revenue', 'earnings', 'success',
        'record', 'high', 'boom', 'optimistic', 'positive', 'upside',
        'momentum', 'breakout', 'innovation', 'expansion', 'partnership',
        'acquisition', 'dividend', 'buyback', 'approval', 'launch'
    ]
    
    BEARISH_KEYWORDS = [
        'plunge', 'crash', 'drop', 'fall', 'decline', 'sink', 'tumble',
        'downgrade', 'miss', 'underperform', 'bearish', 'sell', 'weak',
        'loss', 'debt', 'lawsuit', 'investigation', 'scandal', 'fraud',
        'bankruptcy', 'layoff', 'cut', 'warning', 'risk', 'concern',
        'recession', 'inflation', 'default', 'delay', 'fail', 'reject',
        'pessimistic', 'negative', 'downside', 'crisis', 'trouble'
    ]
    
    def __init__(self, cache_hours: int = 24):
        """
        Initialize news stream processor.
        
        Args:
            cache_hours: Hours to cache processed news
        """
        self.cache_hours = cache_hours
        self.processed_cache: Dict[str, NewsSentiment] = {}
        self.sentiment_history: Dict[str, List[NewsSentiment]] = {}
        self.callbacks: List[Callable[[NewsSentiment], None]] = []
    
    def process_news(self, news: NewsItem) -> NewsSentiment:
        """
        Process a news item and extract sentiment.
        
        Args:
            news: NewsItem to process
            
        Returns:
            NewsSentiment analysis result
        """
        # Check cache
        if news.id in self.processed_cache:
            cached = self.processed_cache[news.id]
            # Check if still valid
            age = datetime.now(timezone.utc) - cached.analyzed_at
            if age < timedelta(hours=self.cache_hours):
                return cached
        
        # Analyze sentiment
        title_score, title_bullish, title_bearish = self._analyze_text(news.title)
        
        content_score = 0.0
        content_bullish, content_bearish = [], []
        if news.summary:
            content_score, content_bullish, content_bearish = self._analyze_text(news.summary)
        elif news.full_text:
            content_score, content_bullish, content_bearish = self._analyze_text(news.full_text[:1000])
        
        # Weight title more heavily (0.6 title, 0.4 content)
        combined_score = title_score * 0.6 + content_score * 0.4
        
        # Calculate confidence based on keyword density
        all_keywords = title_bullish + title_bearish + content_bullish + content_bearish
        keyword_count = len(all_keywords)
        confidence = min(0.3 + keyword_count * 0.1, 0.95)
        
        # Calculate relevance for symbols
        relevance = self._calculate_relevance(news)
        
        # Calculate impact score
        impact = self._calculate_impact(news, combined_score, keyword_count)
        
        result = NewsSentiment(
            news_item=news,
            sentiment=SentimentScore.from_score(combined_score),
            confidence=confidence,
            title_sentiment=title_score,
            content_sentiment=content_score,
            relevance_score=relevance,
            impact_score=impact,
            bullish_keywords=list(set(title_bullish + content_bullish)),
            bearish_keywords=list(set(title_bearish + content_bearish))
        )
        
        # Cache result
        self.processed_cache[news.id] = result
        
        # Add to history for each symbol
        for symbol in news.symbols:
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            self.sentiment_history[symbol].append(result)
            # Keep last 100 items
            self.sentiment_history[symbol] = self.sentiment_history[symbol][-100:]
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.debug(f"Callback failed: {e}")
        
        return result
    
    def _analyze_text(self, text: str) -> tuple:
        """Analyze text for sentiment keywords."""
        text_lower = text.lower()
        
        bullish_found = []
        bearish_found = []
        
        for word in self.BULLISH_KEYWORDS:
            if re.search(r'\b' + word + r'\b', text_lower):
                bullish_found.append(word)
        
        for word in self.BEARISH_KEYWORDS:
            if re.search(r'\b' + word + r'\b', text_lower):
                bearish_found.append(word)
        
        # Calculate score
        bullish_count = len(bullish_found)
        bearish_count = len(bearish_found)
        total = bullish_count + bearish_count
        
        if total == 0:
            score = 0.0
        else:
            score = (bullish_count - bearish_count) / total
        
        return score, bullish_found, bearish_found
    
    def _calculate_relevance(self, news: NewsItem) -> float:
        """Calculate relevance score for news item."""
        relevance = 0.5  # Base relevance
        
        # More relevant if mentions specific symbols
        if news.symbols:
            relevance += 0.2
        
        # More relevant if from major source
        major_sources = ['reuters', 'bloomberg', 'wsj', 'cnbc', 'ft']
        if any(source in news.source.lower() for source in major_sources):
            relevance += 0.2
        
        # Recent news is more relevant
        age = datetime.now(timezone.utc) - news.published_at
        if age < timedelta(hours=1):
            relevance += 0.1
        elif age < timedelta(hours=6):
            relevance += 0.05
        
        return min(relevance, 1.0)
    
    def _calculate_impact(
        self,
        news: NewsItem,
        sentiment_score: float,
        keyword_count: int
    ) -> float:
        """Calculate expected market impact."""
        # Base impact on sentiment strength
        impact = abs(sentiment_score) * 0.5
        
        # More keywords = higher impact
        impact += min(keyword_count * 0.05, 0.3)
        
        # Major sources have higher impact
        major_sources = ['reuters', 'bloomberg', 'wsj']
        if any(source in news.source.lower() for source in major_sources):
            impact *= 1.3
        
        return min(impact, 1.0)
    
    def get_symbol_sentiment(
        self,
        symbol: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            hours: Hours to look back
            
        Returns:
            Dict with aggregated sentiment data
        """
        if symbol not in self.sentiment_history:
            return {
                'symbol': symbol,
                'sentiment': SentimentScore.NEUTRAL,
                'score': 0.0,
                'confidence': 0.0,
                'news_count': 0,
                'trend': 'stable'
            }
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [
            s for s in self.sentiment_history[symbol]
            if s.analyzed_at > cutoff
        ]
        
        if not recent:
            return {
                'symbol': symbol,
                'sentiment': SentimentScore.NEUTRAL,
                'score': 0.0,
                'confidence': 0.0,
                'news_count': 0,
                'trend': 'stable'
            }
        
        # Calculate weighted average
        total_weight = sum(s.confidence for s in recent)
        if total_weight > 0:
            avg_score = sum(s.weighted_score for s in recent) / total_weight
        else:
            avg_score = 0.0
        
        avg_confidence = sum(s.confidence for s in recent) / len(recent)
        
        # Determine trend
        if len(recent) >= 3:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            
            first_avg = sum(s.sentiment.to_numeric() for s in first_half) / len(first_half)
            second_avg = sum(s.sentiment.to_numeric() for s in second_half) / len(second_half)
            
            diff = second_avg - first_avg
            if diff > 0.2:
                trend = 'improving'
            elif diff < -0.2:
                trend = 'deteriorating'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'symbol': symbol,
            'sentiment': SentimentScore.from_score(avg_score),
            'score': avg_score,
            'confidence': avg_confidence,
            'news_count': len(recent),
            'trend': trend,
            'bullish_news': len([s for s in recent if s.sentiment.to_numeric() > 0.1]),
            'bearish_news': len([s for s in recent if s.sentiment.to_numeric() < -0.1]),
            'neutral_news': len([s for s in recent if abs(s.sentiment.to_numeric()) <= 0.1])
        }
    
    def register_callback(self, callback: Callable[[NewsSentiment], None]):
        """Register callback for new sentiment results."""
        self.callbacks.append(callback)
    
    def clear_cache(self):
        """Clear processed news cache."""
        self.processed_cache.clear()
    
    def get_recent_news(
        self,
        symbol: Optional[str] = None,
        limit: int = 20
    ) -> List[NewsSentiment]:
        """Get recent processed news."""
        if symbol:
            items = self.sentiment_history.get(symbol, [])
        else:
            # Merge all
            all_items = []
            for items_list in self.sentiment_history.values():
                all_items.extend(items_list)
            items = sorted(all_items, key=lambda x: x.analyzed_at, reverse=True)
        
        return items[-limit:]
