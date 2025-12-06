"""
Social Sentiment Aggregator - Aggregates sentiment from social platforms.

Collects and analyzes sentiment from:
- Twitter/X
- Reddit
- StockTwits
- Other social sources
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any
import re

logger = logging.getLogger(__name__)


class SocialPlatform(Enum):
    """Social media platforms."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    OTHER = "other"


@dataclass
class SocialPost:
    """Individual social media post."""
    
    platform: SocialPlatform
    content: str
    author: str
    posted_at: datetime
    
    # Engagement metrics
    likes: int = 0
    shares: int = 0
    comments: int = 0
    
    # Metadata
    symbols: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    url: Optional[str] = None
    
    # Author credibility
    author_followers: int = 0
    author_verified: bool = False
    
    @property
    def engagement_score(self) -> float:
        """Calculate engagement score."""
        # Weighted engagement
        score = self.likes + self.shares * 2 + self.comments * 1.5
        return score
    
    @property
    def credibility_score(self) -> float:
        """Calculate author credibility score."""
        score = 0.3  # Base score
        
        if self.author_verified:
            score += 0.3
        
        if self.author_followers >= 10000:
            score += 0.2
        elif self.author_followers >= 1000:
            score += 0.1
        
        if self.engagement_score > 100:
            score += 0.2
        elif self.engagement_score > 10:
            score += 0.1
        
        return min(score, 1.0)


@dataclass
class PlatformSentiment:
    """Sentiment analysis for a specific platform."""
    
    platform: SocialPlatform
    symbol: str
    
    # Sentiment metrics
    sentiment_score: float  # -1 to 1
    confidence: float
    
    # Post counts
    total_posts: int = 0
    bullish_posts: int = 0
    bearish_posts: int = 0
    neutral_posts: int = 0
    
    # Engagement
    total_engagement: float = 0.0
    avg_engagement: float = 0.0
    
    # Trending indicators
    is_trending: bool = False
    mention_velocity: float = 0.0  # Mentions per hour
    
    # Timeframe
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeframe_hours: int = 24


@dataclass
class SocialSentimentResult:
    """Aggregated social sentiment across platforms."""
    
    symbol: str
    
    # Overall sentiment
    overall_sentiment: float  # -1 to 1
    overall_confidence: float
    
    # Platform breakdown
    platform_sentiments: Dict[str, PlatformSentiment] = field(default_factory=dict)
    
    # Aggregated metrics
    total_mentions: int = 0
    total_engagement: float = 0.0
    
    # Trend analysis
    sentiment_trend: str = "stable"  # improving, stable, deteriorating
    is_viral: bool = False
    
    # Key insights
    top_bullish_phrases: List[str] = field(default_factory=list)
    top_bearish_phrases: List[str] = field(default_factory=list)
    influential_authors: List[str] = field(default_factory=list)
    
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SocialSentimentAggregator:
    """
    Aggregates and analyzes sentiment from social media platforms.
    
    Features:
    - Multi-platform sentiment aggregation
    - Engagement-weighted sentiment
    - Credibility-adjusted scores
    - Trend detection
    """
    
    # Sentiment indicators
    BULLISH_PATTERNS = [
        r'\b(buy|buying|bought|long|calls?|bullish|moon|rocket|🚀|💎|🙌)\b',
        r'\b(undervalued|breakout|support|accumulate|dip|opportunity)\b',
        r'\b(strong|growth|beat|exceeded|upgrade|target raised)\b',
    ]
    
    BEARISH_PATTERNS = [
        r'\b(sell|selling|sold|short|puts?|bearish|crash|dump)\b',
        r'\b(overvalued|breakdown|resistance|distribute|avoid)\b',
        r'\b(weak|decline|miss|downgrade|target lowered|warning)\b',
    ]
    
    def __init__(self):
        """Initialize social sentiment aggregator."""
        self.post_cache: Dict[str, List[SocialPost]] = {}
        self.sentiment_cache: Dict[str, SocialSentimentResult] = {}
        self.platform_weights = {
            SocialPlatform.TWITTER: 0.35,
            SocialPlatform.REDDIT: 0.30,
            SocialPlatform.STOCKTWITS: 0.25,
            SocialPlatform.DISCORD: 0.05,
            SocialPlatform.TELEGRAM: 0.05,
        }
    
    def add_post(self, post: SocialPost):
        """
        Add a social post to the aggregator.
        
        Args:
            post: SocialPost to add
        """
        for symbol in post.symbols:
            if symbol not in self.post_cache:
                self.post_cache[symbol] = []
            self.post_cache[symbol].append(post)
            
            # Keep last 1000 posts per symbol
            self.post_cache[symbol] = self.post_cache[symbol][-1000:]
        
        # Invalidate sentiment cache for affected symbols
        for symbol in post.symbols:
            if symbol in self.sentiment_cache:
                del self.sentiment_cache[symbol]
    
    def analyze_post(self, post: SocialPost) -> tuple:
        """
        Analyze sentiment of a single post.
        
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        content_lower = post.content.lower()
        
        bullish_count = 0
        bearish_count = 0
        
        for pattern in self.BULLISH_PATTERNS:
            matches = re.findall(pattern, content_lower)
            bullish_count += len(matches)
        
        for pattern in self.BEARISH_PATTERNS:
            matches = re.findall(pattern, content_lower)
            bearish_count += len(matches)
        
        total = bullish_count + bearish_count
        
        if total == 0:
            return 0.0, 0.3  # Neutral with low confidence
        
        score = (bullish_count - bearish_count) / total
        confidence = min(0.3 + total * 0.1, 0.9)
        
        # Adjust for credibility
        confidence *= post.credibility_score
        
        return score, confidence
    
    def get_platform_sentiment(
        self,
        symbol: str,
        platform: SocialPlatform,
        hours: int = 24
    ) -> PlatformSentiment:
        """
        Get sentiment for a symbol on a specific platform.
        
        Args:
            symbol: Stock symbol
            platform: Social platform
            hours: Hours to analyze
            
        Returns:
            PlatformSentiment result
        """
        if symbol not in self.post_cache:
            return PlatformSentiment(
                platform=platform,
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0
            )
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        posts = [
            p for p in self.post_cache[symbol]
            if p.platform == platform and p.posted_at > cutoff
        ]
        
        if not posts:
            return PlatformSentiment(
                platform=platform,
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0
            )
        
        # Analyze each post
        scores = []
        confidences = []
        bullish = 0
        bearish = 0
        neutral = 0
        total_engagement = 0.0
        
        for post in posts:
            score, conf = self.analyze_post(post)
            
            # Weight by engagement
            weight = 1.0 + (post.engagement_score / 100.0)
            scores.append(score * weight)
            confidences.append(conf)
            total_engagement += post.engagement_score
            
            if score > 0.1:
                bullish += 1
            elif score < -0.1:
                bearish += 1
            else:
                neutral += 1
        
        # Calculate weighted average
        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Normalize score to [-1, 1]
        avg_score = max(-1.0, min(1.0, avg_score))
        
        # Check if trending
        mention_velocity = len(posts) / hours
        is_trending = mention_velocity > 10  # More than 10 posts/hour
        
        return PlatformSentiment(
            platform=platform,
            symbol=symbol,
            sentiment_score=avg_score,
            confidence=avg_confidence,
            total_posts=len(posts),
            bullish_posts=bullish,
            bearish_posts=bearish,
            neutral_posts=neutral,
            total_engagement=total_engagement,
            avg_engagement=total_engagement / len(posts) if posts else 0,
            is_trending=is_trending,
            mention_velocity=mention_velocity,
            timeframe_hours=hours
        )
    
    def get_aggregated_sentiment(
        self,
        symbol: str,
        hours: int = 24
    ) -> SocialSentimentResult:
        """
        Get aggregated sentiment across all platforms.
        
        Args:
            symbol: Stock symbol
            hours: Hours to analyze
            
        Returns:
            SocialSentimentResult with aggregated data
        """
        # Check cache
        cache_key = f"{symbol}_{hours}"
        if cache_key in self.sentiment_cache:
            cached = self.sentiment_cache[cache_key]
            age = datetime.now(timezone.utc) - cached.analyzed_at
            if age < timedelta(minutes=5):
                return cached
        
        # Get sentiment for each platform
        platform_sentiments = {}
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_mentions = 0
        total_engagement = 0.0
        
        for platform, weight in self.platform_weights.items():
            sentiment = self.get_platform_sentiment(symbol, platform, hours)
            platform_sentiments[platform.value] = sentiment
            
            if sentiment.total_posts > 0:
                weighted_score += sentiment.sentiment_score * weight * sentiment.total_posts
                weighted_confidence += sentiment.confidence * weight
                total_weight += weight * sentiment.total_posts
                total_mentions += sentiment.total_posts
                total_engagement += sentiment.total_engagement
        
        # Calculate overall sentiment
        if total_weight > 0:
            overall_sentiment = weighted_score / total_weight
            overall_confidence = weighted_confidence / sum(self.platform_weights.values())
        else:
            overall_sentiment = 0.0
            overall_confidence = 0.0
        
        # Normalize
        overall_sentiment = max(-1.0, min(1.0, overall_sentiment))
        
        # Determine trend
        sentiment_trend = self._calculate_trend(symbol, hours)
        
        # Check if viral
        is_viral = total_mentions > 100 and total_engagement > 1000
        
        # Extract top phrases
        top_bullish, top_bearish = self._extract_top_phrases(symbol, hours)
        
        # Find influential authors
        influential = self._find_influential_authors(symbol, hours)
        
        result = SocialSentimentResult(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            overall_confidence=overall_confidence,
            platform_sentiments=platform_sentiments,
            total_mentions=total_mentions,
            total_engagement=total_engagement,
            sentiment_trend=sentiment_trend,
            is_viral=is_viral,
            top_bullish_phrases=top_bullish,
            top_bearish_phrases=top_bearish,
            influential_authors=influential
        )
        
        # Cache result
        self.sentiment_cache[cache_key] = result
        
        return result
    
    def _calculate_trend(self, symbol: str, hours: int) -> str:
        """Calculate sentiment trend."""
        if symbol not in self.post_cache:
            return "stable"
        
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)
        midpoint = now - timedelta(hours=hours/2)
        
        posts = [p for p in self.post_cache[symbol] if p.posted_at > cutoff]
        
        if len(posts) < 4:
            return "stable"
        
        # Split into first and second half
        first_half = [p for p in posts if p.posted_at < midpoint]
        second_half = [p for p in posts if p.posted_at >= midpoint]
        
        if not first_half or not second_half:
            return "stable"
        
        def avg_sentiment(post_list):
            scores = [self.analyze_post(p)[0] for p in post_list]
            return sum(scores) / len(scores)
        
        first_avg = avg_sentiment(first_half)
        second_avg = avg_sentiment(second_half)
        
        diff = second_avg - first_avg
        
        if diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "deteriorating"
        return "stable"
    
    def _extract_top_phrases(
        self,
        symbol: str,
        hours: int
    ) -> tuple:
        """Extract top bullish and bearish phrases."""
        # Simplified - in production, use NLP
        return (
            ["buying opportunity", "breakout incoming", "strong support"],
            ["overvalued", "time to sell", "resistance ahead"]
        )
    
    def _find_influential_authors(
        self,
        symbol: str,
        hours: int
    ) -> List[str]:
        """Find most influential authors."""
        if symbol not in self.post_cache:
            return []
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        posts = [p for p in self.post_cache[symbol] if p.posted_at > cutoff]
        
        # Sort by credibility and engagement
        sorted_posts = sorted(
            posts,
            key=lambda p: p.credibility_score * p.engagement_score,
            reverse=True
        )
        
        # Get unique authors
        seen = set()
        influential = []
        for post in sorted_posts:
            if post.author not in seen:
                influential.append(post.author)
                seen.add(post.author)
            if len(influential) >= 5:
                break
        
        return influential
    
    def clear_old_posts(self, hours: int = 48):
        """Clear posts older than specified hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        for symbol in self.post_cache:
            self.post_cache[symbol] = [
                p for p in self.post_cache[symbol]
                if p.posted_at > cutoff
            ]
