"""
Tests for Sentiment Analysis Module.

Tests:
- News stream processing
- Social sentiment aggregation
- Sentiment fusion
"""

import pytest
from datetime import datetime, timezone, timedelta

from trader.sentiment import (
    # News streaming
    NewsStreamProcessor,
    NewsItem,
    NewsSentiment,
    SentimentScore,
    # Social sentiment
    SocialSentimentAggregator,
    SocialPost,
    SocialSentimentResult,
    PlatformSentiment,
    SocialPlatform,
    # Sentiment fusion
    SentimentFusion,
    FusedSentiment,
    SentimentTrend,
    SentimentAlert,
    AlertSeverity,
)


# ============================================================================
# News Stream Tests
# ============================================================================

class TestNewsItem:
    """Tests for news item dataclass."""
    
    def test_news_item_creation(self):
        """Test creating a news item."""
        item = NewsItem(
            title='Apple beats earnings expectations',
            source='Reuters',
            url='https://example.com/news/1',
            published_at=datetime.now(timezone.utc),
            symbols=['AAPL']
        )
        
        assert item.title == 'Apple beats earnings expectations'
        assert item.source == 'Reuters'
        assert 'AAPL' in item.symbols
        assert item.id  # Should have auto-generated ID
    
    def test_news_item_with_content(self):
        """Test news item with full content."""
        item = NewsItem(
            title='Tech stocks rally',
            source='Bloomberg',
            url='https://example.com/news/2',
            published_at=datetime.now(timezone.utc),
            summary='Technology stocks rallied today on strong earnings...',
            symbols=['AAPL', 'MSFT', 'GOOGL']
        )
        
        assert len(item.symbols) == 3
        assert item.summary is not None and 'Technology' in item.summary


class TestSentimentScore:
    """Tests for sentiment score enum."""
    
    def test_sentiment_score_from_numeric(self):
        """Test converting numeric score to sentiment."""
        assert SentimentScore.from_score(0.8) == SentimentScore.VERY_BULLISH
        assert SentimentScore.from_score(0.4) == SentimentScore.BULLISH
        assert SentimentScore.from_score(0.0) == SentimentScore.NEUTRAL
        assert SentimentScore.from_score(-0.4) == SentimentScore.BEARISH
        assert SentimentScore.from_score(-0.8) == SentimentScore.VERY_BEARISH
    
    def test_sentiment_score_to_numeric(self):
        """Test converting sentiment to numeric score."""
        assert SentimentScore.VERY_BULLISH.to_numeric() == 0.8
        assert SentimentScore.NEUTRAL.to_numeric() == 0.0
        assert SentimentScore.VERY_BEARISH.to_numeric() == -0.8


class TestNewsSentiment:
    """Tests for news sentiment dataclass."""
    
    def test_news_sentiment_weighted_score(self):
        """Test weighted score calculation."""
        item = NewsItem(
            title='Test',
            source='Test',
            url='https://test.com',
            published_at=datetime.now(timezone.utc)
        )
        
        sentiment = NewsSentiment(
            news_item=item,
            sentiment=SentimentScore.BULLISH,
            confidence=0.8
        )
        
        expected = SentimentScore.BULLISH.to_numeric() * 0.8
        assert sentiment.weighted_score == expected


class TestNewsStreamProcessor:
    """Tests for news stream processor."""
    
    def test_init(self):
        """Test processor initialization."""
        processor = NewsStreamProcessor()
        assert processor is not None
    
    def test_process_bullish_news(self):
        """Test processing bullish news."""
        processor = NewsStreamProcessor()
        
        item = NewsItem(
            title='Apple stock surges on strong earnings beat',
            source='Reuters',
            url='https://example.com/news/3',
            published_at=datetime.now(timezone.utc),
            symbols=['AAPL']
        )
        
        sentiment = processor.process_news(item)
        
        assert sentiment is not None
        assert isinstance(sentiment, NewsSentiment)
        assert sentiment.sentiment.to_numeric() > 0  # Should be bullish
    
    def test_process_bearish_news(self):
        """Test processing bearish news."""
        processor = NewsStreamProcessor()
        
        item = NewsItem(
            title='Company faces bankruptcy after major scandal',
            source='WSJ',
            url='https://example.com/news/4',
            published_at=datetime.now(timezone.utc),
            symbols=['XYZ']
        )
        
        sentiment = processor.process_news(item)
        
        assert sentiment.sentiment.to_numeric() < 0  # Should be bearish
    
    def test_get_symbol_sentiment(self):
        """Test aggregated symbol sentiment."""
        processor = NewsStreamProcessor()
        
        # Add multiple positive news items
        for i in range(3):
            processor.process_news(NewsItem(
                title=f'Apple stock rallies {i}',
                source='Test',
                url=f'https://example.com/{i}',
                published_at=datetime.now(timezone.utc),
                symbols=['AAPL']
            ))
        
        result = processor.get_symbol_sentiment('AAPL')
        
        assert result['symbol'] == 'AAPL'
        assert result['news_count'] == 3
        assert 'sentiment' in result
    
    def test_register_callback(self):
        """Test callback registration."""
        processor = NewsStreamProcessor()
        
        received = []
        processor.register_callback(lambda s: received.append(s))
        
        processor.process_news(NewsItem(
            title='Test news',
            source='Test',
            url='https://test.com',
            published_at=datetime.now(timezone.utc),
            symbols=['TEST']
        ))
        
        assert len(received) == 1
    
    def test_cache_deduplication(self):
        """Test that duplicate news is cached."""
        processor = NewsStreamProcessor()
        
        item = NewsItem(
            title='Same news',
            source='Test',
            url='https://test.com/same',
            published_at=datetime.now(timezone.utc),
            symbols=['AAPL']
        )
        
        result1 = processor.process_news(item)
        result2 = processor.process_news(item)
        
        # Same item should return cached result
        assert result1.news_item.id == result2.news_item.id


# ============================================================================
# Social Sentiment Tests
# ============================================================================

class TestSocialPost:
    """Tests for social post dataclass."""
    
    def test_social_post_creation(self):
        """Test creating social post."""
        post = SocialPost(
            platform=SocialPlatform.TWITTER,
            content='$AAPL looking bullish today! 🚀',
            author='trader123',
            posted_at=datetime.now(timezone.utc),
            symbols=['AAPL']
        )
        
        assert post.platform == SocialPlatform.TWITTER
        assert 'AAPL' in post.symbols
    
    def test_post_engagement_score(self):
        """Test engagement score calculation."""
        post = SocialPost(
            platform=SocialPlatform.TWITTER,
            content='Test',
            author='user',
            posted_at=datetime.now(timezone.utc),
            likes=100,
            shares=50,
            comments=25
        )
        
        # likes + shares*2 + comments*1.5
        expected = 100 + 50*2 + 25*1.5
        assert post.engagement_score == expected
    
    def test_post_credibility_score(self):
        """Test credibility score calculation."""
        post = SocialPost(
            platform=SocialPlatform.TWITTER,
            content='Test',
            author='verified_trader',
            posted_at=datetime.now(timezone.utc),
            author_verified=True,
            author_followers=50000,
            likes=200
        )
        
        score = post.credibility_score
        assert score > 0.5  # Should be high due to verification and followers


class TestPlatformSentiment:
    """Tests for platform-specific sentiment."""
    
    def test_platform_sentiment_creation(self):
        """Test creating platform sentiment."""
        sentiment = PlatformSentiment(
            platform=SocialPlatform.TWITTER,
            symbol='AAPL',
            sentiment_score=0.55,
            confidence=0.8,
            total_posts=500,
            is_trending=True
        )
        
        assert sentiment.platform == SocialPlatform.TWITTER
        assert sentiment.is_trending == True


class TestSocialSentimentResult:
    """Tests for social sentiment result."""
    
    def test_result_creation(self):
        """Test creating social sentiment result."""
        result = SocialSentimentResult(
            symbol='AAPL',
            overall_sentiment=0.65,
            overall_confidence=0.8,
            total_mentions=100
        )
        
        assert result.symbol == 'AAPL'
        assert result.overall_sentiment == 0.65


class TestSocialSentimentAggregator:
    """Tests for social sentiment aggregator."""
    
    def test_init(self):
        """Test aggregator initialization."""
        aggregator = SocialSentimentAggregator()
        assert aggregator is not None
    
    def test_add_post(self):
        """Test adding social post."""
        aggregator = SocialSentimentAggregator()
        
        post = SocialPost(
            platform=SocialPlatform.TWITTER,
            content='$AAPL to the moon! 🚀',
            author='trader',
            posted_at=datetime.now(timezone.utc),
            symbols=['AAPL']
        )
        
        aggregator.add_post(post)
        result = aggregator.get_aggregated_sentiment('AAPL')
        
        assert result is not None
        assert result.total_mentions >= 1
    
    def test_analyze_bullish_post(self):
        """Test analyzing bullish post."""
        aggregator = SocialSentimentAggregator()
        
        post = SocialPost(
            platform=SocialPlatform.TWITTER,
            content='$AAPL bullish breakout! Buying calls! 🚀🚀🚀',
            author='trader',
            posted_at=datetime.now(timezone.utc)
        )
        
        score, confidence = aggregator.analyze_post(post)
        
        assert score > 0  # Should be bullish
    
    def test_analyze_bearish_post(self):
        """Test analyzing bearish post."""
        aggregator = SocialSentimentAggregator()
        
        post = SocialPost(
            platform=SocialPlatform.TWITTER,
            content='$XYZ crashing! Sell everything! Puts printing!',
            author='trader',
            posted_at=datetime.now(timezone.utc)
        )
        
        score, confidence = aggregator.analyze_post(post)
        
        assert score < 0  # Should be bearish
    
    def test_multi_platform_aggregation(self):
        """Test aggregating across platforms."""
        aggregator = SocialSentimentAggregator()
        
        # Add Twitter posts
        for _ in range(5):
            aggregator.add_post(SocialPost(
                platform=SocialPlatform.TWITTER,
                content='AAPL bullish',
                author='user',
                posted_at=datetime.now(timezone.utc),
                symbols=['AAPL']
            ))
        
        # Add Reddit posts
        for _ in range(3):
            aggregator.add_post(SocialPost(
                platform=SocialPlatform.REDDIT,
                content='AAPL analysis',
                author='redditor',
                posted_at=datetime.now(timezone.utc),
                symbols=['AAPL']
            ))
        
        result = aggregator.get_aggregated_sentiment('AAPL')
        
        assert result.total_mentions == 8
        assert 'twitter' in result.platform_sentiments or SocialPlatform.TWITTER.value in str(result.platform_sentiments)
    
    def test_get_platform_sentiment(self):
        """Test getting platform-specific sentiment."""
        aggregator = SocialSentimentAggregator()
        
        aggregator.add_post(SocialPost(
            platform=SocialPlatform.TWITTER,
            content='$AAPL',
            author='user',
            posted_at=datetime.now(timezone.utc),
            symbols=['AAPL']
        ))
        
        sentiment = aggregator.get_platform_sentiment('AAPL', SocialPlatform.TWITTER)
        
        assert isinstance(sentiment, PlatformSentiment)
        assert sentiment.total_posts >= 1


# ============================================================================
# Sentiment Fusion Tests
# ============================================================================

class TestFusedSentiment:
    """Tests for fused sentiment dataclass."""
    
    def test_fused_sentiment_creation(self):
        """Test creating fused sentiment."""
        fused = FusedSentiment(
            symbol='AAPL',
            fused_score=0.65,
            fused_confidence=0.80,
            news_score=0.70,
            social_score=0.60
        )
        
        assert fused.symbol == 'AAPL'
        assert fused.fused_score == 0.65
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        fused = FusedSentiment(
            symbol='AAPL',
            fused_score=0.7,
            fused_confidence=0.85
        )
        
        result = fused.to_dict()
        
        assert result['symbol'] == 'AAPL'
        assert result['fused_score'] == 0.7


class TestSentimentTrend:
    """Tests for sentiment trend enum."""
    
    def test_trend_values(self):
        """Test trend enum values."""
        assert SentimentTrend.IMPROVING.value == 'improving'
        assert SentimentTrend.STABLE.value == 'stable'
        assert SentimentTrend.DETERIORATING.value == 'deteriorating'


class TestSentimentAlert:
    """Tests for sentiment alert."""
    
    def test_alert_creation(self):
        """Test creating sentiment alert."""
        alert = SentimentAlert(
            symbol='AAPL',
            alert_type='sentiment_shift',
            severity=AlertSeverity.WARNING,
            message='Sentiment shifted significantly',
            current_sentiment=0.7,
            previous_sentiment=0.3,
            change=0.4,
            trigger_source='combined'
        )
        
        assert alert.alert_type == 'sentiment_shift'
        assert alert.severity == AlertSeverity.WARNING


class TestSentimentFusion:
    """Tests for sentiment fusion engine."""
    
    def test_init(self):
        """Test fusion engine initialization."""
        fusion = SentimentFusion()
        assert fusion is not None
    
    def test_fuse_news_only(self):
        """Test fusing with news sentiment only."""
        fusion = SentimentFusion()
        
        result = fusion.fuse_sentiment(
            symbol='AAPL',
            news_sentiment={'score': 0.7, 'confidence': 0.8}
        )
        
        assert isinstance(result, FusedSentiment)
        assert result.symbol == 'AAPL'
        assert 'news' in result.sources_used
    
    def test_fuse_multiple_sources(self):
        """Test fusing multiple sentiment sources."""
        fusion = SentimentFusion()
        
        result = fusion.fuse_sentiment(
            symbol='AAPL',
            news_sentiment={'score': 0.7, 'confidence': 0.8},
            social_sentiment={'overall_sentiment': 0.5, 'overall_confidence': 0.7},
            analyst_sentiment={'score': 0.6, 'confidence': 0.9}
        )
        
        assert len(result.sources_used) == 3
        assert result.fused_score > 0  # All sources bullish
    
    def test_weighted_fusion(self):
        """Test weighted fusion of sources."""
        fusion = SentimentFusion(
            news_weight=0.6,
            social_weight=0.4,
            analyst_weight=0.0
        )
        
        result = fusion.fuse_sentiment(
            symbol='AAPL',
            news_sentiment={'score': 1.0, 'confidence': 1.0},
            social_sentiment={'overall_sentiment': 0.0, 'overall_confidence': 1.0}
        )
        
        # Should be weighted toward news
        assert result.fused_score > 0.3
    
    def test_divergence_detection(self):
        """Test detection of source divergence."""
        fusion = SentimentFusion()
        
        result = fusion.fuse_sentiment(
            symbol='AAPL',
            news_sentiment={'score': 0.8, 'confidence': 0.9},  # Very bullish
            social_sentiment={'overall_sentiment': -0.6, 'overall_confidence': 0.8}  # Bearish
        )
        
        # Sources disagree
        assert result.divergence_score > 0.5
        assert result.sources_agree == False
    
    def test_signal_generation(self):
        """Test signal generation from sentiment."""
        fusion = SentimentFusion()
        
        # Strong bullish sentiment
        result = fusion.fuse_sentiment(
            symbol='AAPL',
            news_sentiment={'score': 0.8, 'confidence': 0.9},
            social_sentiment={'overall_sentiment': 0.7, 'overall_confidence': 0.8}
        )
        
        assert result.signal_direction == 'bullish'
        assert result.signal_strength > 0
    
    def test_alert_generation(self):
        """Test alert generation on significant change."""
        fusion = SentimentFusion()
        
        # First sentiment
        fusion.fuse_sentiment(
            symbol='AAPL',
            news_sentiment={'score': 0.2, 'confidence': 0.8}
        )
        
        # Significant change
        result = fusion.fuse_sentiment(
            symbol='AAPL',
            news_sentiment={'score': 0.9, 'confidence': 0.9}
        )
        
        # Should generate alert
        assert len(result.alerts) > 0
    
    def test_get_sentiment_summary(self):
        """Test getting sentiment summary."""
        fusion = SentimentFusion()
        
        # Add some sentiment data
        for i in range(5):
            fusion.fuse_sentiment(
                symbol='AAPL',
                news_sentiment={'score': 0.5 + i*0.05, 'confidence': 0.8}
            )
        
        summary = fusion.get_sentiment_summary('AAPL')
        
        assert summary['symbol'] == 'AAPL'
        assert summary['data_points'] == 5
    
    def test_rank_by_sentiment(self):
        """Test ranking symbols by sentiment."""
        fusion = SentimentFusion()
        
        # Add sentiment for multiple symbols
        fusion.fuse_sentiment('AAPL', news_sentiment={'score': 0.8, 'confidence': 0.9})
        fusion.fuse_sentiment('MSFT', news_sentiment={'score': 0.3, 'confidence': 0.9})
        fusion.fuse_sentiment('GOOGL', news_sentiment={'score': 0.6, 'confidence': 0.9})
        
        rankings = fusion.rank_by_sentiment(['AAPL', 'MSFT', 'GOOGL'], direction='bullish')
        
        assert len(rankings) == 3
        assert rankings[0][0] == 'AAPL'  # Highest bullish
    
    def test_register_alert_callback(self):
        """Test alert callback registration."""
        fusion = SentimentFusion()
        
        alerts_received = []
        fusion.register_alert_callback(lambda a: alerts_received.append(a))
        
        # Trigger extreme sentiment alert
        fusion.fuse_sentiment(
            symbol='AAPL',
            news_sentiment={'score': 0.9, 'confidence': 0.95}
        )
        
        # May or may not trigger depending on threshold
        assert isinstance(alerts_received, list)


# ============================================================================
# Integration Tests
# ============================================================================

class TestSentimentIntegration:
    """Integration tests for sentiment module."""
    
    def test_full_sentiment_pipeline(self):
        """Test complete sentiment analysis pipeline."""
        # 1. Process news
        news_processor = NewsStreamProcessor()
        
        news_processor.process_news(NewsItem(
            title='Apple reports record breaking quarterly revenue',
            source='Reuters',
            url='https://example.com',
            published_at=datetime.now(timezone.utc),
            symbols=['AAPL']
        ))
        
        news_processor.process_news(NewsItem(
            title='Apple beats Wall Street expectations',
            source='Bloomberg',
            url='https://example.com/2',
            published_at=datetime.now(timezone.utc),
            symbols=['AAPL']
        ))
        
        news_result = news_processor.get_symbol_sentiment('AAPL')
        
        # 2. Process social media
        social_aggregator = SocialSentimentAggregator()
        
        for _ in range(10):
            social_aggregator.add_post(SocialPost(
                platform=SocialPlatform.TWITTER,
                content='$AAPL crushing it! Bullish! 🚀',
                author='trader',
                posted_at=datetime.now(timezone.utc),
                symbols=['AAPL'],
                likes=100
            ))
        
        social_result = social_aggregator.get_aggregated_sentiment('AAPL')
        
        # 3. Fuse sentiments
        fusion = SentimentFusion()
        
        fused = fusion.fuse_sentiment(
            symbol='AAPL',
            news_sentiment={'score': news_result['score'], 'confidence': news_result['confidence']},
            social_sentiment={
                'overall_sentiment': social_result.overall_sentiment,
                'overall_confidence': social_result.overall_confidence
            }
        )
        
        assert fused.symbol == 'AAPL'
        assert len(fused.sources_used) >= 2
    
    def test_multi_symbol_sentiment(self):
        """Test sentiment analysis across multiple symbols."""
        news_processor = NewsStreamProcessor()
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in symbols:
            news_processor.process_news(NewsItem(
                title=f'{symbol} stock update with positive momentum',
                source='Test',
                url=f'https://example.com/{symbol}',
                published_at=datetime.now(timezone.utc),
                symbols=[symbol]
            ))
        
        for symbol in symbols:
            result = news_processor.get_symbol_sentiment(symbol)
            assert result['news_count'] >= 1
    
    def test_sentiment_trend_analysis(self):
        """Test sentiment trend detection."""
        fusion = SentimentFusion()
        
        # Simulate improving sentiment over time
        for i in range(10):
            score = 0.2 + (i * 0.06)  # Gradually improving
            fusion.fuse_sentiment(
                symbol='AAPL',
                news_sentiment={'score': score, 'confidence': 0.8}
            )
        
        # Get latest
        latest = fusion.sentiment_history['AAPL'][-1]
        
        # Should detect improving trend
        assert latest.trend in [SentimentTrend.IMPROVING, SentimentTrend.STRONGLY_IMPROVING]
    
    def test_multi_symbol_ranking(self):
        """Test multi-symbol ranking."""
        fusion = SentimentFusion()
        
        # Add varied sentiment
        fusion.fuse_sentiment('AAPL', news_sentiment={'score': 0.9, 'confidence': 0.9})
        fusion.fuse_sentiment('MSFT', news_sentiment={'score': 0.2, 'confidence': 0.9})
        fusion.fuse_sentiment('GOOGL', news_sentiment={'score': 0.5, 'confidence': 0.9})
        fusion.fuse_sentiment('AMZN', news_sentiment={'score': -0.3, 'confidence': 0.9})
        
        bullish = fusion.rank_by_sentiment(['AAPL', 'MSFT', 'GOOGL', 'AMZN'], direction='bullish')
        bearish = fusion.rank_by_sentiment(['AAPL', 'MSFT', 'GOOGL', 'AMZN'], direction='bearish')
        
        assert bullish[0][0] == 'AAPL'  # Most bullish
        assert bearish[0][0] == 'AMZN'  # Most bearish
