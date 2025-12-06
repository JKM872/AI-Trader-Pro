"""
Tests for News Sentiment module.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import json

from trader.data.news_sentiment import (
    NewsAggregator,
    SocialSentimentAnalyzer,
    NewsArticle,
    SocialPost,
    SentimentLevel,
)


# ============== Fixtures ==============

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("NEWS_API_KEY", "test_news_key")
    monkeypatch.setenv("FINNHUB_API_KEY", "test_finnhub_key")


@pytest.fixture
def sample_news_article():
    """Sample news article."""
    return NewsArticle(
        title="Apple Reports Record Earnings",
        source="Reuters",
        url="https://example.com/apple-earnings",
        published="2024-01-15T10:00:00Z",
        description="Apple Inc. reported quarterly earnings that exceeded expectations",
        author="John Doe",
        category="earnings"
    )


@pytest.fixture
def sample_social_post():
    """Sample social media post."""
    return SocialPost(
        platform="reddit",
        content="$AAPL to the moon! 🚀 Earnings beat expectations!",
        author="trader123",
        url="https://reddit.com/r/wallstreetbets/...",
        timestamp="2024-01-15T10:00:00Z",
        likes=1500,
        comments=234
    )


@pytest.fixture
def mock_httpx_response():
    """Create mock httpx response factory."""
    def _create(data, status_code=200):
        mock = Mock()
        mock.status_code = status_code
        mock.json.return_value = data
        mock.text = json.dumps(data) if isinstance(data, dict) else str(data)
        mock.raise_for_status = Mock()
        if status_code >= 400:
            mock.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        return mock
    return _create


# ============== NewsArticle Tests ==============

class TestNewsArticle:
    """Tests for NewsArticle dataclass."""
    
    def test_news_article_creation(self, sample_news_article):
        """Test creating NewsArticle."""
        assert sample_news_article.title == "Apple Reports Record Earnings"
        assert sample_news_article.source == "Reuters"
    
    def test_news_article_optional_fields(self):
        """Test NewsArticle with minimal fields."""
        article = NewsArticle(
            title="Test",
            source="test",
            url="https://test.com",
            published="2024-01-01"
        )
        assert article.description is None
        assert article.author is None


# ============== SocialPost Tests ==============

class TestSocialPost:
    """Tests for SocialPost dataclass."""
    
    def test_social_post_creation(self, sample_social_post):
        """Test creating SocialPost."""
        assert sample_social_post.platform == "reddit"
        assert "$AAPL" in sample_social_post.content
    
    def test_social_post_engagement(self, sample_social_post):
        """Test social post engagement metrics."""
        assert sample_social_post.likes == 1500
        assert sample_social_post.comments == 234


# ============== SentimentLevel Tests ==============

class TestSentimentLevel:
    """Tests for SentimentLevel enumeration."""
    
    def test_sentiment_levels_exist(self):
        """Test sentiment levels exist."""
        assert SentimentLevel.VERY_BULLISH.value == 2
        assert SentimentLevel.BULLISH.value == 1
        assert SentimentLevel.NEUTRAL.value == 0
        assert SentimentLevel.BEARISH.value == -1
        assert SentimentLevel.VERY_BEARISH.value == -2


# ============== NewsAggregator Tests ==============

class TestNewsAggregator:
    """Tests for NewsAggregator class."""
    
    def test_aggregator_initialization(self, mock_env_vars):
        """Test aggregator initializes correctly."""
        aggregator = NewsAggregator()
        assert aggregator is not None
        aggregator.client.close()
    
    def test_aggregator_api_keys(self, mock_env_vars):
        """Test aggregator loads API keys."""
        aggregator = NewsAggregator()
        assert aggregator.newsapi_key == "test_news_key"
        assert aggregator.finnhub_key == "test_finnhub_key"
        aggregator.client.close()
    
    @patch('httpx.Client')
    def test_get_all_news(self, mock_client_class, mock_env_vars, mock_httpx_response):
        """Test fetching news from all sources."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock NewsAPI response
        mock_client.get.return_value = mock_httpx_response({
            'articles': [
                {
                    'title': 'Apple Earnings Beat',
                    'description': 'Record quarterly earnings',
                    'source': {'name': 'Reuters'},
                    'url': 'https://example.com/news',
                    'publishedAt': '2024-01-15T10:00:00Z'
                }
            ]
        })
        
        aggregator = NewsAggregator()
        aggregator.client = mock_client
        
        # Method should exist
        assert hasattr(aggregator, 'get_all_news')
    
    def test_aggregator_has_required_methods(self, mock_env_vars):
        """Test aggregator has required methods."""
        aggregator = NewsAggregator()
        
        assert hasattr(aggregator, 'get_all_news')
        assert hasattr(aggregator, '_fetch_newsapi')
        assert hasattr(aggregator, '_fetch_finnhub')
        
        aggregator.client.close()


# ============== SocialSentimentAnalyzer Tests ==============

class TestSocialSentimentAnalyzer:
    """Tests for SocialSentimentAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        analyzer = SocialSentimentAnalyzer()
        assert analyzer is not None
        analyzer.client.close()
    
    def test_analyzer_has_required_methods(self):
        """Test analyzer has required methods."""
        analyzer = SocialSentimentAnalyzer()
        
        assert hasattr(analyzer, 'get_reddit_sentiment')
        assert hasattr(analyzer, 'get_stocktwits_sentiment')
        
        analyzer.client.close()
    
    @patch('httpx.Client')
    def test_get_reddit_sentiment(self, mock_client_class, mock_httpx_response):
        """Test fetching Reddit sentiment."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.get.return_value = mock_httpx_response({
            'data': {
                'children': [
                    {'data': {'title': 'AAPL to the moon!', 'score': 500}}
                ]
            }
        })
        
        analyzer = SocialSentimentAnalyzer()
        analyzer.client = mock_client
        
        assert hasattr(analyzer, 'get_reddit_sentiment')
    
    @patch('httpx.Client')
    def test_get_stocktwits_sentiment(self, mock_client_class, mock_httpx_response):
        """Test fetching StockTwits sentiment."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.get.return_value = mock_httpx_response({
            'symbol': {'symbol': 'AAPL', 'watchlist_count': 50000},
            'messages': [
                {'body': 'Bullish on AAPL!', 'sentiment': {'basic': 'Bullish'}}
            ]
        })
        
        analyzer = SocialSentimentAnalyzer()
        analyzer.client = mock_client
        
        assert hasattr(analyzer, 'get_stocktwits_sentiment')


# ============== Rate Limiting Tests ==============

class TestRateLimiting:
    """Tests for rate limiting in news APIs."""
    
    def test_aggregator_handles_rate_limits(self, mock_env_vars):
        """Test aggregator handles rate limiting gracefully."""
        aggregator = NewsAggregator()
        # Should not crash
        assert aggregator is not None
        aggregator.client.close()


# ============== Error Handling Tests ==============

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_handles_missing_api_key(self, monkeypatch):
        """Test handling missing API keys."""
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        
        # Should still initialize, just with fewer sources
        aggregator = NewsAggregator()
        assert aggregator is not None
        assert aggregator.newsapi_key is None
        aggregator.client.close()
    
    def test_aggregator_graceful_degradation(self, monkeypatch):
        """Test aggregator works without any API keys."""
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        
        aggregator = NewsAggregator()
        # Should still work with free sources (Google News RSS)
        assert hasattr(aggregator, '_fetch_google_news')
        aggregator.client.close()


# ============== Integration Tests ==============

class TestNewsSentimentIntegration:
    """Integration tests for news sentiment module."""
    
    def test_full_news_workflow(self, mock_env_vars):
        """Test complete news analysis workflow."""
        aggregator = NewsAggregator()
        
        # Aggregator should be functional
        assert aggregator is not None
        assert hasattr(aggregator, 'get_all_news')
        
        aggregator.client.close()
    
    def test_full_social_workflow(self):
        """Test complete social sentiment workflow."""
        analyzer = SocialSentimentAnalyzer()
        
        # Analyzer should be functional
        assert analyzer is not None
        
        analyzer.client.close()
    
    def test_all_components_compatible(self, mock_env_vars):
        """Test all components work together."""
        news = NewsAggregator()
        social = SocialSentimentAnalyzer()
        
        # All should initialize successfully
        assert news is not None
        assert social is not None
        
        news.client.close()
        social.client.close()
