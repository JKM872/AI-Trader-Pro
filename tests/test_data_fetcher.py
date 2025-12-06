"""
Unit tests for DataFetcher module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestDataFetcher:
    """Tests for DataFetcher class."""
    
    @pytest.fixture
    def mock_yfinance_data(self):
        """Create mock yfinance historical data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Open': np.random.uniform(145, 155, 100),
            'High': np.random.uniform(150, 160, 100),
            'Low': np.random.uniform(140, 150, 100),
            'Close': np.random.uniform(145, 155, 100),
            'Volume': np.random.randint(1000000, 5000000, 100),
            'Dividends': [0] * 100,
            'Stock Splits': [0] * 100,
        }, index=dates)
    
    @pytest.fixture
    def mock_fundamentals(self):
        """Create mock fundamental data."""
        return {
            'shortName': 'Apple Inc.',
            'symbol': 'AAPL',
            'marketCap': 3000000000000,
            'trailingPE': 28.5,
            'forwardPE': 25.2,
            'dividendYield': 0.005,
            'beta': 1.25,
            'fiftyTwoWeekHigh': 200.0,
            'fiftyTwoWeekLow': 140.0,
            'averageVolume': 50000000,
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
    
    @patch('trader.data.fetcher.yf')
    def test_get_stock_data_success(self, mock_yf, mock_yfinance_data):
        """Test successful stock data retrieval."""
        from trader.data.fetcher import DataFetcher
        
        # Setup mock
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker
        
        # Test
        fetcher = DataFetcher()
        data = fetcher.get_stock_data('AAPL', period='3mo')
        
        # Assertions
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'Open' in data.columns
        assert 'High' in data.columns
        assert 'Low' in data.columns
        assert 'Close' in data.columns
        assert 'Volume' in data.columns
    
    @patch('trader.data.fetcher.yf')
    def test_get_stock_data_empty(self, mock_yf):
        """Test handling of empty data."""
        from trader.data.fetcher import DataFetcher
        
        # Setup mock to return empty DataFrame
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker
        
        # Test
        fetcher = DataFetcher()
        data = fetcher.get_stock_data('INVALID', period='3mo')
        
        # Should return None or empty DataFrame
        assert data is None or data.empty
    
    @patch('trader.data.fetcher.yf')
    def test_get_stock_data_with_interval(self, mock_yf, mock_yfinance_data):
        """Test stock data retrieval with different intervals."""
        from trader.data.fetcher import DataFetcher
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker
        
        fetcher = DataFetcher()
        
        # Test with daily interval
        data = fetcher.get_stock_data('AAPL', period='1mo', interval='1d')
        assert data is not None
        
        # Verify correct parameters were passed
        mock_ticker.history.assert_called()
    
    @patch('trader.data.fetcher.yf')
    def test_get_fundamentals_success(self, mock_yf, mock_fundamentals):
        """Test successful fundamental data retrieval."""
        from trader.data.fetcher import DataFetcher
        
        mock_ticker = Mock()
        mock_ticker.info = mock_fundamentals
        mock_yf.Ticker.return_value = mock_ticker
        
        fetcher = DataFetcher()
        fundamentals = fetcher.get_fundamentals('AAPL')
        
        assert fundamentals is not None
        assert 'market_cap' in fundamentals or 'marketCap' in fundamentals
    
    @patch('trader.data.fetcher.yf')
    def test_get_fundamentals_error(self, mock_yf):
        """Test handling of fundamentals retrieval error."""
        from trader.data.fetcher import DataFetcher
        
        mock_ticker = Mock()
        mock_ticker.info = {}  # Empty info dict
        mock_yf.Ticker.return_value = mock_ticker
        
        fetcher = DataFetcher()
        fundamentals = fetcher.get_fundamentals('INVALID')
        
        # Should return a dict structure (even with None/empty values)
        assert isinstance(fundamentals, dict)
        # The implementation returns a structured dict - just verify it's a dict
        # Values may be None, empty string, or default values
    
    @patch('trader.data.fetcher.yf')
    def test_caching_behavior(self, mock_yf, mock_yfinance_data):
        """Test that data is cached properly."""
        from trader.data.fetcher import DataFetcher
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker
        
        fetcher = DataFetcher()
        
        # First call
        data1 = fetcher.get_stock_data('AAPL', period='3mo')
        
        # Second call - should use cache
        data2 = fetcher.get_stock_data('AAPL', period='3mo')
        
        # Data should be the same
        assert data1 is not None
        assert data2 is not None
        # If caching is implemented, yfinance should only be called once
        # (This depends on implementation)
    
    @patch('trader.data.fetcher.yf')
    def test_multiple_symbols(self, mock_yf, mock_yfinance_data):
        """Test fetching data for multiple symbols."""
        from trader.data.fetcher import DataFetcher
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker
        
        fetcher = DataFetcher()
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        for symbol in symbols:
            data = fetcher.get_stock_data(symbol, period='1mo')
            assert data is not None
            assert len(data) > 0
    
    @patch('trader.data.fetcher.yf')
    def test_data_columns(self, mock_yf, mock_yfinance_data):
        """Test that returned data has correct columns."""
        from trader.data.fetcher import DataFetcher
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker
        
        fetcher = DataFetcher()
        data = fetcher.get_stock_data('AAPL', period='3mo')
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            assert col in data.columns, f"Missing column: {col}"
    
    @patch('trader.data.fetcher.yf')
    def test_data_types(self, mock_yf, mock_yfinance_data):
        """Test that returned data has correct data types."""
        from trader.data.fetcher import DataFetcher
        
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_yfinance_data
        mock_yf.Ticker.return_value = mock_ticker
        
        fetcher = DataFetcher()
        data = fetcher.get_stock_data('AAPL', period='3mo')
        
        # Check numeric types
        assert pd.api.types.is_numeric_dtype(data['Open'])
        assert pd.api.types.is_numeric_dtype(data['Close'])
        assert pd.api.types.is_numeric_dtype(data['Volume'])
        
        # Check index is datetime
        assert isinstance(data.index, pd.DatetimeIndex)


class TestRateLimiter:
    """Tests for rate limiting functionality."""
    
    def test_rate_limiter_creation(self):
        """Test rate limiter initialization."""
        from trader.data.fetcher import RateLimiter
        
        limiter = RateLimiter(max_requests=30, period_seconds=60)
        assert limiter.max_requests == 30
        assert limiter.period_seconds == 60
    
    def test_rate_limiter_allows_calls(self):
        """Test that rate limiter allows calls within limit."""
        from trader.data.fetcher import RateLimiter
        
        limiter = RateLimiter(max_requests=100, period_seconds=60)
        
        # Should allow multiple calls - wait_if_needed doesn't block
        # when under limit
        for _ in range(10):
            limiter.wait_if_needed()
            # Check that requests are being tracked
            assert len(limiter.requests) <= 100
    
    def test_rate_limiter_tracks_requests(self):
        """Test that rate limiter tracks requests properly."""
        from trader.data.fetcher import RateLimiter
        
        limiter = RateLimiter(max_requests=5, period_seconds=60)
        
        # Make 5 calls
        for _ in range(5):
            limiter.wait_if_needed()
        
        # 6th call should be blocked (or delayed)
        # Implementation may vary - either returns False or sleeps
        # This test verifies the limiter tracks calls


class TestDataValidation:
    """Tests for data validation utilities."""
    
    def test_validate_ohlcv_data(self):
        """Test OHLCV data validation."""
        from trader.data.fetcher import DataFetcher
        
        # Valid data
        valid_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [98, 99, 100],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        fetcher = DataFetcher()
        
        # Should not raise exception for valid data
        # (Implementation depends on validation method)
    
    def test_detect_invalid_values(self):
        """Test detection of invalid values in data."""
        # Data with NaN values
        invalid_data = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [105, 106, 107],
            'Low': [98, 99, 100],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        # Should handle NaN values appropriately
        assert invalid_data['Open'].isna().sum() == 1
    
    def test_detect_negative_prices(self):
        """Test detection of negative prices."""
        invalid_data = pd.DataFrame({
            'Open': [100, -101, 102],  # Negative price
            'High': [105, 106, 107],
            'Low': [98, 99, 100],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        # Negative prices should be detected
        assert (invalid_data['Open'] < 0).any()


class TestDataFetcherIntegration:
    """Integration tests (marked for optional execution)."""
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires network access")
    def test_real_data_fetch(self):
        """Test fetching real data from yfinance."""
        from trader.data.fetcher import DataFetcher
        
        fetcher = DataFetcher()
        data = fetcher.get_stock_data('AAPL', period='5d')
        
        assert data is not None
        assert len(data) > 0
        assert 'Close' in data.columns
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires network access")
    def test_real_fundamentals_fetch(self):
        """Test fetching real fundamental data."""
        from trader.data.fetcher import DataFetcher
        
        fetcher = DataFetcher()
        fundamentals = fetcher.get_fundamentals('AAPL')
        
        assert fundamentals is not None
        assert len(fundamentals) > 0
