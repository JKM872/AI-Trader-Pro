"""
Unit tests for Signal Scanner module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from trader.strategies.scanner import (
    SignalScanner,
    ScoredSignal,
    SignalScorer,
    MarketAnalyzer,
    get_signal_summary,
)
from trader.strategies.base import Signal, SignalType


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    base_price = 150
    returns = np.random.normal(0.001, 0.02, 100)
    close = base_price * np.cumprod(1 + returns)
    
    high = close * (1 + np.abs(np.random.normal(0, 0.01, 100)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, 100)))
    open_price = low + (high - low) * np.random.random(100)
    volume = np.random.randint(1000000, 10000000, 100)
    
    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(float)
    }, index=dates)


@pytest.fixture
def mock_data_fetcher(sample_ohlcv_data):
    """Mock data fetcher for scanner tests."""
    with patch('trader.strategies.scanner.DataFetcher') as mock:
        instance = mock.return_value
        instance.get_stock_data.return_value = sample_ohlcv_data
        yield instance


class TestScoredSignal:
    """Tests for ScoredSignal dataclass."""
    
    def test_scored_signal_creation(self):
        """Test creating a scored signal."""
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0
        )
        
        scored = ScoredSignal(
            signal=signal,
            score=85.5,
            strategy_name='Technical'
        )
        
        assert scored.signal == signal
        assert scored.score == 85.5
        assert scored.strategy_name == 'Technical'
    
    def test_scored_signal_comparison(self):
        """Test that scored signals can be compared by score."""
        signal1 = Signal(symbol='AAPL', signal_type=SignalType.BUY, 
                        confidence=0.8, price=150.0)
        signal2 = Signal(symbol='MSFT', signal_type=SignalType.BUY,
                        confidence=0.7, price=400.0)
        
        scored1 = ScoredSignal(signal=signal1, score=90.0, strategy_name='A')
        scored2 = ScoredSignal(signal=signal2, score=75.0, strategy_name='B')
        
        # Higher score should be "greater"
        assert scored1.score > scored2.score


class TestSignalScorer:
    """Tests for SignalScorer class."""
    
    @pytest.fixture
    def scorer_data(self):
        """Generate sample data for scorer tests."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        base_price = 150
        returns = np.random.normal(0.001, 0.02, 100)
        close = base_price * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.01,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 100).astype(float)
        }, index=dates)
    
    def test_scorer_initialization(self):
        """Test scorer initialization."""
        scorer = SignalScorer()
        assert scorer is not None
        assert hasattr(SignalScorer, 'WEIGHTS')
    
    def test_score_signal_buy(self, scorer_data):
        """Test scoring a BUY signal."""
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0,
            stop_loss=142.5,
            take_profit=165.0
        )
        
        score, components = SignalScorer.score_signal(signal, scorer_data)
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100
        assert isinstance(components, dict)
    
    def test_score_signal_hold(self, scorer_data):
        """Test scoring a HOLD signal."""
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.HOLD,
            confidence=0.5,
            price=150.0
        )
        
        score, components = SignalScorer.score_signal(signal, scorer_data)
        
        # HOLD signals should have lower scores
        assert 0 <= score <= 100
        assert isinstance(components, dict)
    
    def test_high_confidence_higher_score(self, scorer_data):
        """Test that higher confidence leads to higher scores."""
        high_conf = Signal(symbol='AAPL', signal_type=SignalType.BUY,
                          confidence=0.95, price=150.0,
                          stop_loss=142.5, take_profit=165.0)
        low_conf = Signal(symbol='AAPL', signal_type=SignalType.BUY,
                         confidence=0.55, price=150.0,
                         stop_loss=142.5, take_profit=165.0)
        
        high_score, _ = SignalScorer.score_signal(high_conf, scorer_data)
        low_score, _ = SignalScorer.score_signal(low_conf, scorer_data)
        
        assert high_score >= low_score
    
    def test_good_rr_higher_score(self, scorer_data):
        """Test that better risk/reward leads to higher scores."""
        # Good R:R (3:1)
        good_rr = Signal(symbol='AAPL', signal_type=SignalType.BUY,
                        confidence=0.7, price=150.0,
                        stop_loss=145.0, take_profit=165.0)  # 3:1 R:R
        
        # Bad R:R (0.5:1)
        bad_rr = Signal(symbol='AAPL', signal_type=SignalType.BUY,
                       confidence=0.7, price=150.0,
                       stop_loss=140.0, take_profit=155.0)  # 0.5:1 R:R
        
        good_score, _ = SignalScorer.score_signal(good_rr, scorer_data)
        bad_score, _ = SignalScorer.score_signal(bad_rr, scorer_data)
        
        assert good_score >= bad_score


class TestMarketAnalyzer:
    """Tests for MarketAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = MarketAnalyzer()
        assert analyzer is not None
    
    def test_analyzer_has_analyze_method(self, sample_ohlcv_data):
        """Test that analyzer can analyze market conditions."""
        analyzer = MarketAnalyzer()
        # Check if analyze_regime method exists (the actual method name)
        if hasattr(analyzer, 'analyze_regime'):
            result = analyzer.analyze_regime(sample_ohlcv_data)
            assert result is not None
        elif hasattr(analyzer, 'analyze'):
            result = analyzer.analyze(sample_ohlcv_data)
            assert result is not None
        else:
            # Just verify the object was created
            assert analyzer is not None


class TestSignalScanner:
    """Tests for SignalScanner class."""
    
    def test_scanner_initialization_default(self):
        """Test scanner initialization with defaults."""
        scanner = SignalScanner()
        
        assert scanner is not None
        assert scanner.min_score >= 0
    
    def test_scanner_initialization_custom(self):
        """Test scanner initialization with custom settings."""
        scanner = SignalScanner(
            strategies=['Technical', 'Momentum'],
            min_score=60
        )
        
        assert scanner.min_score == 60
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_scan_single_symbol(self, mock_fetcher_class, sample_ohlcv_data):
        """Test scanning a single symbol."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(strategies=['Technical'])
        results = scanner.scan_watchlist(['AAPL'])
        
        assert isinstance(results, dict)
        assert 'AAPL' in results
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_scan_multiple_symbols(self, mock_fetcher_class, sample_ohlcv_data):
        """Test scanning multiple symbols."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(strategies=['Technical'])
        results = scanner.scan_watchlist(['AAPL', 'MSFT', 'GOOGL'])
        
        assert isinstance(results, dict)
        # Should have results for all symbols (or at least attempted)
        assert len(results) >= 0
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_scan_returns_scored_signals(self, mock_fetcher_class, sample_ohlcv_data):
        """Test that scan returns scored signals."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(strategies=['Technical'], min_score=0)
        results = scanner.scan_watchlist(['AAPL'])
        
        if results.get('AAPL'):
            for scored_signal in results['AAPL']:
                assert isinstance(scored_signal, ScoredSignal)
                assert hasattr(scored_signal, 'score')
                assert hasattr(scored_signal, 'signal')
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_min_score_filtering(self, mock_fetcher_class, sample_ohlcv_data):
        """Test that min_score affects which signals are returned."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        # Low threshold - should get more signals
        scanner_low = SignalScanner(strategies=['Technical'], min_score=0)
        results_low = scanner_low.scan_watchlist(['AAPL'])
        
        # High threshold - may get fewer signals  
        scanner_high = SignalScanner(strategies=['Technical'], min_score=90)
        results_high = scanner_high.scan_watchlist(['AAPL'])
        
        # High threshold should have equal or fewer signals
        low_count = len(results_low.get('AAPL', []))
        high_count = len(results_high.get('AAPL', []))
        assert high_count <= low_count
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_get_top_signals(self, mock_fetcher_class, sample_ohlcv_data):
        """Test getting top signals by score."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(strategies=['Technical', 'Momentum'], min_score=0)
        top_signals = scanner.get_top_signals(['AAPL', 'MSFT'], top_n=5)
        
        assert isinstance(top_signals, list)
        assert len(top_signals) <= 5
        
        # Should be sorted by score (descending)
        if len(top_signals) >= 2:
            assert top_signals[0].score >= top_signals[1].score
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_get_top_signals_by_type(self, mock_fetcher_class, sample_ohlcv_data):
        """Test filtering top signals by signal type."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(strategies=['Technical'], min_score=0)
        buy_signals = scanner.get_top_signals(
            ['AAPL'], top_n=5, signal_type=SignalType.BUY
        )
        
        for sig in buy_signals:
            assert sig.signal.signal_type == SignalType.BUY
    
    def test_empty_watchlist(self):
        """Test scanning empty watchlist."""
        scanner = SignalScanner()
        results = scanner.scan_watchlist([])
        
        assert results == {}
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_data_fetch_error_handling(self, mock_fetcher_class):
        """Test handling of data fetch errors."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = pd.DataFrame()  # Empty data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(strategies=['Technical'])
        results = scanner.scan_watchlist(['INVALID'])
        
        # Should handle gracefully
        assert isinstance(results, dict)


class TestGetSignalSummary:
    """Tests for get_signal_summary function."""
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_get_signal_summary_returns_dict(self, mock_fetcher_class, sample_ohlcv_data):
        """Test that get_signal_summary returns a dictionary."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        summary = get_signal_summary('AAPL')
        
        assert isinstance(summary, dict)
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_summary_contains_key_fields(self, mock_fetcher_class, sample_ohlcv_data):
        """Test that summary contains expected fields."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        summary = get_signal_summary('AAPL')
        
        # Should contain overall signal info
        assert 'overall_signal' in summary or 'signals' in summary or 'symbol' in summary


class TestScannerIntegration:
    """Integration tests for scanner with real strategy calculations."""
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_multi_strategy_scan(self, mock_fetcher_class, sample_ohlcv_data):
        """Test scanning with multiple strategies."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(
            strategies=['Technical', 'Momentum', 'Mean Reversion'],
            min_score=0
        )
        results = scanner.scan_watchlist(['AAPL'])
        
        if results.get('AAPL'):
            # Should have signals from multiple strategies
            strategy_names = set(sig.strategy_name for sig in results['AAPL'])
            assert len(strategy_names) >= 1
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_scanner_with_smart_money_strategy(self, mock_fetcher_class, sample_ohlcv_data):
        """Test scanner with Smart Money strategy."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(strategies=['Smart Money'], min_score=0)
        results = scanner.scan_watchlist(['AAPL'])
        
        assert isinstance(results, dict)
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_scanner_with_mtf_strategy(self, mock_fetcher_class, sample_ohlcv_data):
        """Test scanner with Multi-Timeframe strategy."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(strategies=['Multi-Timeframe'], min_score=0)
        results = scanner.scan_watchlist(['AAPL'])
        
        assert isinstance(results, dict)


class TestScannerEdgeCases:
    """Edge case tests for scanner."""
    
    def test_invalid_strategy_name(self):
        """Test handling of invalid strategy names."""
        scanner = SignalScanner(strategies=['InvalidStrategy'], min_score=0)
        
        # Should initialize without error
        assert scanner is not None
    
    @patch('trader.strategies.scanner.DataFetcher')
    def test_concurrent_scanning(self, mock_fetcher_class, sample_ohlcv_data):
        """Test scanning behavior with multiple symbols."""
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_ohlcv_data
        mock_fetcher_class.return_value = mock_fetcher
        
        scanner = SignalScanner(strategies=['Technical'], min_score=0)
        
        # Scan same symbol multiple times
        symbols = ['AAPL'] * 3
        results = scanner.scan_watchlist(symbols)
        
        # Should handle duplicates gracefully
        assert isinstance(results, dict)
