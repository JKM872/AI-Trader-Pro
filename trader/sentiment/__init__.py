"""
Sentiment Analysis Module - Real-time sentiment streaming and analysis.

This module provides:
- Real-time news sentiment processing
- Social media sentiment aggregation
- Sentiment trend tracking
- Multi-source sentiment fusion
"""

from trader.sentiment.news_stream import (
    NewsStreamProcessor,
    NewsItem,
    NewsSentiment,
    SentimentScore
)
from trader.sentiment.social_sentiment import (
    SocialSentimentAggregator,
    SocialPost,
    SocialSentimentResult,
    PlatformSentiment,
    SocialPlatform
)
from trader.sentiment.sentiment_fusion import (
    SentimentFusion,
    FusedSentiment,
    SentimentTrend,
    SentimentAlert,
    AlertSeverity
)

__all__ = [
    # News streaming
    'NewsStreamProcessor',
    'NewsItem',
    'NewsSentiment',
    'SentimentScore',
    # Social sentiment
    'SocialSentimentAggregator',
    'SocialPost',
    'SocialSentimentResult',
    'PlatformSentiment',
    'SocialPlatform',
    # Sentiment fusion
    'SentimentFusion',
    'FusedSentiment',
    'SentimentTrend',
    'SentimentAlert',
    'AlertSeverity',
]
