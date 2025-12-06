"""
Tests for Multi-Source Data Fetcher module.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime
import json

from trader.data.multi_source_fetcher import (
    MultiSourceFetcher,
    DataSource,
    DataSourceConfig,
    DATA_SOURCES,
)


# ============== Fixtures ==============

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_av_key")
    monkeypatch.setenv("FINNHUB_API_KEY", "test_finnhub_key")
    monkeypatch.setenv("POLYGON_API_KEY", "test_polygon_key")
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "test_twelve_key")
    monkeypatch.setenv("FMP_API_KEY", "test_fmp_key")
    monkeypatch.setenv("FRED_API_KEY", "test_fred_key")
    monkeypatch.setenv("NEWS_API_KEY", "test_news_key")


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV DataFrame."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'Open': np.random.uniform(140, 160, 100),
        'High': np.random.uniform(145, 165, 100),
        'Low': np.random.uniform(135, 155, 100),
        'Close': np.random.uniform(140, 160, 100),
        'Volume': np.random.randint(1000000, 10000000, 100),
    }, index=dates)


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


# ============== DataSource Enum Tests ==============

class TestDataSourceEnum:
    """Tests for DataSource enumeration."""
    
    def test_data_sources_exist(self):
        """Test all expected data sources exist."""
        expected = ['YFINANCE', 'ALPHA_VANTAGE', 'FINNHUB', 'POLYGON', 
                    'TWELVE_DATA', 'FMP', 'FRED', 'COINGECKO']
        
        for source in expected:
            assert hasattr(DataSource, source), f"Missing DataSource.{source}"
    
    def test_data_source_values(self):
        """Test data source enum values."""
        assert DataSource.YFINANCE.value == 'yfinance'
        assert DataSource.ALPHA_VANTAGE.value == 'alpha_vantage'


# ============== DataSourceConfig Tests ==============

class TestDataSourceConfig:
    """Tests for DataSourceConfig dataclass."""
    
    def test_config_creation(self):
        """Test creating DataSourceConfig."""
        config = DataSourceConfig(
            name="Test Source",
            env_key="TEST_API_KEY",
            base_url="https://api.test.com",
            rate_limit=60,
            daily_limit=1000,
            is_free=True,
            description="Test data source"
        )
        
        assert config.name == "Test Source"
        assert config.rate_limit == 60
        assert config.daily_limit == 1000


# ============== DATA_SOURCES Tests ==============

class TestDataSourcesConfig:
    """Tests for DATA_SOURCES configuration."""
    
    def test_all_sources_configured(self):
        """Test all expected sources are configured."""
        expected = ['alpha_vantage', 'finnhub', 'polygon', 'twelve_data', 'fmp', 'fred']
        
        for source in expected:
            assert source in DATA_SOURCES, f"Missing config for: {source}"
    
    def test_source_configs_have_required_fields(self):
        """Test each source config has required fields."""
        for name, config in DATA_SOURCES.items():
            assert config.name is not None
            assert config.env_key is not None
            assert config.base_url is not None
            assert config.rate_limit > 0


# ============== MultiSourceFetcher Tests ==============

class TestMultiSourceFetcher:
    """Tests for MultiSourceFetcher class."""
    
    def test_initialization(self, mock_env_vars):
        """Test fetcher initializes correctly."""
        fetcher = MultiSourceFetcher()
        assert fetcher is not None
        fetcher.close()
    
    def test_available_sources(self, mock_env_vars):
        """Test getting available data sources."""
        fetcher = MultiSourceFetcher()
        sources = fetcher.get_available_sources()
        assert isinstance(sources, dict)
        # coingecko is always available (no key needed)
        assert sources.get('coingecko') is True
        fetcher.close()
    
    @patch('yfinance.Ticker')
    def test_get_stock_data_multi(self, mock_ticker, mock_env_vars, sample_ohlcv_data):
        """Test fetching stock data from multiple sources."""
        mock_instance = Mock()
        mock_ticker.return_value = mock_instance
        mock_instance.history.return_value = sample_ohlcv_data
        
        fetcher = MultiSourceFetcher()
        data = fetcher.get_stock_data_multi('AAPL')
        
        assert data is not None
        assert isinstance(data, dict)
        fetcher.close()
    
    def test_get_fundamentals_multi_exists(self, mock_env_vars):
        """Test fundamentals_multi method exists."""
        fetcher = MultiSourceFetcher()
        assert hasattr(fetcher, 'get_fundamentals_multi')
        fetcher.close()
    
    def test_has_client(self, mock_env_vars):
        """Test fetcher has httpx client."""
        fetcher = MultiSourceFetcher()
        assert hasattr(fetcher, 'client')
        fetcher.close()


# ============== Rate Limiting Tests ==============

class TestRateLimiting:
    """Tests for rate limiting functionality."""
    
    def test_source_configs_have_rate_limits(self, mock_env_vars):
        """Test rate limits are configured for each source."""
        for name, config in DATA_SOURCES.items():
            assert config.rate_limit > 0, f"{name} has no rate limit"


# ============== Caching Tests ==============

class TestCaching:
    """Tests for data caching functionality."""
    
    def test_cache_exists(self, mock_env_vars):
        """Test cache is configured."""
        fetcher = MultiSourceFetcher()
        assert hasattr(fetcher, '_cache') or hasattr(fetcher, 'cache')
        fetcher.close()


# ============== Integration Tests ==============

class TestMultiSourceIntegration:
    """Integration tests for multi-source fetcher."""
    
    def test_fetcher_handles_missing_keys(self, monkeypatch):
        """Test fetcher works with missing API keys."""
        for key in ['ALPHA_VANTAGE_API_KEY', 'FINNHUB_API_KEY', 'POLYGON_API_KEY',
                    'TWELVE_DATA_API_KEY', 'FMP_API_KEY', 'FRED_API_KEY']:
            monkeypatch.delenv(key, raising=False)
        
        fetcher = MultiSourceFetcher()
        # Should still initialize
        assert fetcher is not None
        
        # coingecko should still be available (no key needed)
        sources = fetcher.get_available_sources()
        assert sources.get('coingecko') is True
        fetcher.close()
    
    def test_all_methods_exist(self, mock_env_vars):
        """Test all expected methods exist."""
        fetcher = MultiSourceFetcher()
        
        expected_methods = ['get_stock_data_multi', 'get_fundamentals_multi', 
                           'get_available_sources', 'close']
        
        for method in expected_methods:
            assert hasattr(fetcher, method), f"Missing method: {method}"
        
        fetcher.close()
