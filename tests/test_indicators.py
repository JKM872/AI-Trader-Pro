"""
Unit tests for TradingView-style indicators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trader.analysis.indicators import (
    TradingViewIndicators,
    TrendDirection,
    IndicatorResult,
)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    base_price = 100
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
def trending_data():
    """Generate trending (bullish) price data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Strong uptrend
    close = 100 + np.arange(100) * 0.5 + np.random.normal(0, 1, 100)
    high = close + np.abs(np.random.normal(1, 0.5, 100))
    low = close - np.abs(np.random.normal(1, 0.5, 100))
    open_price = low + (high - low) * 0.5
    volume = np.random.randint(1000000, 10000000, 100)
    
    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(float)
    }, index=dates)


@pytest.fixture
def indicators():
    """Create TradingViewIndicators instance."""
    return TradingViewIndicators()


class TestTradingViewIndicators:
    """Tests for TradingViewIndicators class."""
    
    def test_initialization(self, indicators):
        """Test indicator instance creation."""
        assert indicators is not None
        assert isinstance(indicators, TradingViewIndicators)
    
    def test_calculate_atr(self, indicators, sample_ohlcv_data):
        """Test ATR calculation."""
        atr = indicators.calculate_atr(sample_ohlcv_data, period=14)
        
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(sample_ohlcv_data)
        assert atr.dropna().min() >= 0  # ATR should be positive
    
    def test_supertrend(self, indicators, sample_ohlcv_data):
        """Test Supertrend indicator."""
        supertrend, direction = indicators.supertrend(
            sample_ohlcv_data, period=10, multiplier=3.0
        )
        
        assert isinstance(supertrend, pd.Series)
        assert isinstance(direction, pd.Series)
        assert len(supertrend) == len(sample_ohlcv_data)
        assert len(direction) == len(sample_ohlcv_data)
        
        # Direction should be 1 (bullish) or -1 (bearish)
        valid_directions = direction.dropna().unique()
        assert all(d in [1, -1] for d in valid_directions)
    
    def test_supertrend_trending_market(self, indicators, trending_data):
        """Test Supertrend in trending market."""
        supertrend, direction = indicators.supertrend(trending_data)
        
        # In uptrend, most recent direction should be bullish (1)
        recent_direction = direction.iloc[-10:].mean()
        assert recent_direction > 0  # Mostly bullish
    
    def test_adx_dmi(self, indicators, sample_ohlcv_data):
        """Test ADX/DMI calculation."""
        adx, plus_di, minus_di = indicators.adx_dmi(sample_ohlcv_data, period=14)
        
        assert isinstance(adx, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)
        
        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        assert valid_adx.min() >= 0
        assert valid_adx.max() <= 100
        
        # DI values should be non-negative
        assert plus_di.dropna().min() >= 0
        assert minus_di.dropna().min() >= 0
    
    def test_vwap(self, indicators, sample_ohlcv_data):
        """Test VWAP calculation."""
        vwap = indicators.vwap(sample_ohlcv_data)
        
        assert isinstance(vwap, pd.Series)
        assert len(vwap) == len(sample_ohlcv_data)
        
        # VWAP should be within price range
        valid_vwap = vwap.dropna()
        assert valid_vwap.min() > 0
    
    def test_ichimoku_cloud(self, indicators, sample_ohlcv_data):
        """Test Ichimoku Cloud calculation."""
        ichimoku = indicators.ichimoku_cloud(sample_ohlcv_data)
        
        assert isinstance(ichimoku, dict)
        expected_keys = ['tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou']
        assert all(key in ichimoku for key in expected_keys)
        
        for key in expected_keys:
            assert isinstance(ichimoku[key], pd.Series)
    
    def test_squeeze_momentum(self, indicators, sample_ohlcv_data):
        """Test Squeeze Momentum indicator."""
        squeeze_on, momentum = indicators.squeeze_momentum(sample_ohlcv_data)
        
        assert isinstance(squeeze_on, pd.Series)
        assert isinstance(momentum, pd.Series)
        assert len(squeeze_on) == len(sample_ohlcv_data)
        
        # squeeze_on should be boolean-like (True/False or 1/0)
        valid_squeeze = squeeze_on.dropna()
        assert all(s in [True, False, 1, 0] for s in valid_squeeze)
    
    def test_hull_moving_average(self, indicators, sample_ohlcv_data):
        """Test Hull Moving Average calculation."""
        hma = indicators.hull_moving_average(sample_ohlcv_data['Close'], period=20)
        
        assert isinstance(hma, pd.Series)
        assert len(hma) == len(sample_ohlcv_data)
        
        # HMA should be close to price (not NaN at the end)
        assert not pd.isna(hma.iloc[-1])
    
    def test_keltner_channels(self, indicators, sample_ohlcv_data):
        """Test Keltner Channels calculation."""
        middle, upper, lower = indicators.keltner_channels(
            sample_ohlcv_data, ema_period=20, atr_period=10, multiplier=2.0
        )
        
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        
        # Upper >= Middle >= Lower (for valid values)
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_idx.any():
            # Allow for floating point precision
            assert (upper[valid_idx] >= middle[valid_idx] - 0.001).all()
            assert (middle[valid_idx] >= lower[valid_idx] - 0.001).all()
    
    def test_pivot_points_standard(self, indicators, sample_ohlcv_data):
        """Test standard pivot points."""
        pivots = indicators.pivot_points(sample_ohlcv_data, pivot_type='standard')
        
        assert isinstance(pivots, dict)
        expected_keys = ['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']
        assert all(key in pivots for key in expected_keys)
        
        # Resistance > Pivot > Support
        assert pivots['r1'] >= pivots['pivot']
        assert pivots['pivot'] >= pivots['s1']
    
    def test_pivot_points_fibonacci(self, indicators, sample_ohlcv_data):
        """Test Fibonacci pivot points."""
        pivots = indicators.pivot_points(sample_ohlcv_data, pivot_type='fibonacci')
        
        assert isinstance(pivots, dict)
        assert 'pivot' in pivots
        assert pivots['r1'] >= pivots['pivot']
    
    def test_pivot_points_camarilla(self, indicators, sample_ohlcv_data):
        """Test Camarilla pivot points."""
        pivots = indicators.pivot_points(sample_ohlcv_data, pivot_type='camarilla')
        
        assert isinstance(pivots, dict)
        assert 'pivot' in pivots
    
    def test_market_structure(self, indicators, sample_ohlcv_data):
        """Test market structure analysis."""
        structure = indicators.market_structure(sample_ohlcv_data)
        
        assert isinstance(structure, dict)
        assert 'trend' in structure
        assert 'swing_highs' in structure
        assert 'swing_lows' in structure
        
        # Trend should be a TrendDirection enum
        assert isinstance(structure['trend'], TrendDirection)
    
    def test_market_structure_trending(self, indicators, trending_data):
        """Test market structure in trending market."""
        structure = indicators.market_structure(trending_data)
        
        # Should detect bullish or neutral trend in uptrending data
        # (algorithm may be conservative with short data)
        assert structure['trend'] in [TrendDirection.BULLISH, TrendDirection.NEUTRAL]
    
    def test_order_block_detection(self, indicators, sample_ohlcv_data):
        """Test order block detection."""
        order_blocks = indicators.order_block_detection(sample_ohlcv_data)
        
        assert isinstance(order_blocks, dict)
        assert 'bullish' in order_blocks
        assert 'bearish' in order_blocks
        
        # Order blocks should be lists
        assert isinstance(order_blocks['bullish'], list)
        assert isinstance(order_blocks['bearish'], list)
    
    def test_fair_value_gap(self, indicators, sample_ohlcv_data):
        """Test Fair Value Gap detection."""
        fvg = indicators.fair_value_gap(sample_ohlcv_data)
        
        assert isinstance(fvg, dict)
        assert 'bullish' in fvg
        assert 'bearish' in fvg
        
        # FVGs should be lists
        assert isinstance(fvg['bullish'], list)
        assert isinstance(fvg['bearish'], list)


class TestIndicatorResult:
    """Tests for IndicatorResult dataclass."""
    
    def test_indicator_result_creation(self):
        """Test creating an indicator result."""
        result = IndicatorResult(
            value=50.0,
            signal=TrendDirection.BULLISH,
            strength=0.8
        )
        
        assert result.value == 50.0
        assert result.signal == TrendDirection.BULLISH
        assert result.strength == 0.8


class TestTrendDirection:
    """Tests for TrendDirection enum."""
    
    def test_trend_values(self):
        """Test TrendDirection enum values."""
        assert TrendDirection.BULLISH.value == 'bullish'
        assert TrendDirection.BEARISH.value == 'bearish'
        assert TrendDirection.NEUTRAL.value == 'neutral'


class TestIndicatorEdgeCases:
    """Test edge cases for indicators."""
    
    def test_short_data(self, indicators):
        """Test with minimal data."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Open': [100] * 10,
            'High': [101] * 10,
            'Low': [99] * 10,
            'Close': [100.5] * 10,
            'Volume': [1000000.0] * 10
        }, index=dates)
        
        # Should not raise errors
        atr = indicators.calculate_atr(df, period=5)
        assert isinstance(atr, pd.Series)
    
    def test_constant_prices(self, indicators):
        """Test with constant (non-moving) prices."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Open': [100.0] * 50,
            'High': [100.5] * 50,
            'Low': [99.5] * 50,
            'Close': [100.0] * 50,
            'Volume': [1000000.0] * 50
        }, index=dates)
        
        supertrend, direction = indicators.supertrend(df)
        assert isinstance(supertrend, pd.Series)
        
        adx, plus_di, minus_di = indicators.adx_dmi(df)
        assert isinstance(adx, pd.Series)
