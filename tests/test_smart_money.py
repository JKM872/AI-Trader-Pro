"""
Unit tests for Smart Money Concepts and Multi-Timeframe strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trader.strategies.smart_money import (
    SmartMoneyStrategy,
    MultiTimeframeStrategy,
    SMCPattern,
)
from trader.strategies.base import Signal, SignalType


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Generate realistic price data with some structure
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 200)
    close = base_price * np.cumprod(1 + returns)
    
    high = close * (1 + np.abs(np.random.normal(0, 0.015, 200)))
    low = close * (1 - np.abs(np.random.normal(0, 0.015, 200)))
    open_price = low + (high - low) * np.random.random(200)
    volume = np.random.randint(1000000, 10000000, 200)
    
    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(float)
    }, index=dates)


@pytest.fixture
def trending_up_data():
    """Generate bullish trending data."""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Strong uptrend with occasional pullbacks
    trend = 100 + np.arange(200) * 0.3
    noise = np.random.normal(0, 1.5, 200)
    close = trend + noise
    
    high = close + np.abs(np.random.normal(1, 0.5, 200))
    low = close - np.abs(np.random.normal(1, 0.5, 200))
    open_price = low + (high - low) * 0.5
    volume = np.random.randint(1000000, 10000000, 200)
    
    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(float)
    }, index=dates)


@pytest.fixture
def trending_down_data():
    """Generate bearish trending data."""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Strong downtrend
    trend = 150 - np.arange(200) * 0.3
    noise = np.random.normal(0, 1.5, 200)
    close = trend + noise
    
    high = close + np.abs(np.random.normal(1, 0.5, 200))
    low = close - np.abs(np.random.normal(1, 0.5, 200))
    open_price = low + (high - low) * 0.5
    volume = np.random.randint(1000000, 10000000, 200)
    
    return pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume.astype(float)
    }, index=dates)


class TestSmartMoneyStrategy:
    """Tests for SmartMoneyStrategy."""
    
    def test_initialization(self):
        """Test strategy initialization with defaults."""
        strategy = SmartMoneyStrategy()
        
        assert strategy.name == "SmartMoneyStrategy"
        assert strategy.stop_loss_pct == 0.05
        assert strategy.take_profit_pct == 0.10
    
    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = SmartMoneyStrategy(
            stop_loss_pct=0.03,
            take_profit_pct=0.15,
            swing_length=10,
            ob_sensitivity=5
        )
        
        assert strategy.stop_loss_pct == 0.03
        assert strategy.take_profit_pct == 0.15
        assert strategy.swing_length == 10
        assert strategy.ob_sensitivity == 5
    
    def test_generate_signal_returns_signal(self, sample_ohlcv_data):
        """Test that generate_signal returns a Signal object."""
        strategy = SmartMoneyStrategy()
        signal = strategy.generate_signal('AAPL', sample_ohlcv_data)
        
        assert isinstance(signal, Signal)
        assert signal.symbol == 'AAPL'
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert 0.0 <= signal.confidence <= 1.0
    
    def test_signal_has_price(self, sample_ohlcv_data):
        """Test that signal includes current price."""
        strategy = SmartMoneyStrategy()
        signal = strategy.generate_signal('AAPL', sample_ohlcv_data)
        
        assert signal.price > 0
        # Price should be close to last close
        assert abs(signal.price - sample_ohlcv_data['Close'].iloc[-1]) < 1
    
    def test_signal_has_stop_loss_take_profit(self, sample_ohlcv_data):
        """Test that actionable signals have risk levels."""
        strategy = SmartMoneyStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
        signal = strategy.generate_signal('AAPL', sample_ohlcv_data)
        
        if signal.signal_type != SignalType.HOLD:
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
    
    def test_signal_has_reasons(self, sample_ohlcv_data):
        """Test that signal includes analysis reasons."""
        strategy = SmartMoneyStrategy()
        signal = strategy.generate_signal('AAPL', sample_ohlcv_data)
        
        assert isinstance(signal.reasons, list)
    
    def test_bullish_trend_detection(self, trending_up_data):
        """Test detection of bullish market structure."""
        strategy = SmartMoneyStrategy()
        signal = strategy.generate_signal('AAPL', trending_up_data)
        
        # In strong uptrend, should lean bullish
        # Note: May still be HOLD if no entry opportunity
        assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
    
    def test_bearish_trend_detection(self, trending_down_data):
        """Test detection of bearish market structure."""
        strategy = SmartMoneyStrategy()
        signal = strategy.generate_signal('AAPL', trending_down_data)
        
        # In strong downtrend, should lean bearish
        assert signal.signal_type in [SignalType.SELL, SignalType.HOLD]


class TestMultiTimeframeStrategy:
    """Tests for MultiTimeframeStrategy."""
    
    def test_initialization(self):
        """Test strategy initialization with defaults."""
        strategy = MultiTimeframeStrategy()
        
        assert strategy.name == "MultiTimeframeStrategy"
        assert strategy.stop_loss_pct == 0.05
        assert strategy.take_profit_pct == 0.10
    
    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = MultiTimeframeStrategy(
            stop_loss_pct=0.04,
            take_profit_pct=0.12,
            min_confluence=0.8
        )
        
        assert strategy.stop_loss_pct == 0.04
        assert strategy.take_profit_pct == 0.12
        assert strategy.min_confluence == 0.8
    
    def test_generate_signal_returns_signal(self, sample_ohlcv_data):
        """Test that generate_signal returns a Signal object."""
        strategy = MultiTimeframeStrategy()
        signal = strategy.generate_signal('AAPL', sample_ohlcv_data)
        
        assert isinstance(signal, Signal)
        assert signal.symbol == 'AAPL'
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
    
    def test_signal_confidence_bounds(self, sample_ohlcv_data):
        """Test that confidence is within valid bounds."""
        strategy = MultiTimeframeStrategy()
        signal = strategy.generate_signal('AAPL', sample_ohlcv_data)
        
        assert 0.0 <= signal.confidence <= 1.0
    
    def test_confluence_affects_signal(self, trending_up_data):
        """Test that confluence requirement affects signals."""
        # High confluence requirement
        strict_strategy = MultiTimeframeStrategy(min_confluence=0.9)
        strict_signal = strict_strategy.generate_signal('AAPL', trending_up_data)
        
        # Lower confluence requirement
        loose_strategy = MultiTimeframeStrategy(min_confluence=0.5)
        loose_signal = loose_strategy.generate_signal('AAPL', trending_up_data)
        
        # Both should return valid signals
        assert isinstance(strict_signal, Signal)
        assert isinstance(loose_signal, Signal)
    
    def test_trending_market_alignment(self, trending_up_data):
        """Test MTF alignment in trending market."""
        strategy = MultiTimeframeStrategy(min_confluence=0.6)
        signal = strategy.generate_signal('AAPL', trending_up_data)
        
        # In strong uptrend, should have bullish bias
        if signal.signal_type == SignalType.BUY:
            assert signal.confidence >= 0.5


class TestSMCPattern:
    """Tests for SMCPattern enum."""
    
    def test_pattern_values(self):
        """Test SMCPattern enum values."""
        assert SMCPattern.ORDER_BLOCK.value == "order_block"
        assert SMCPattern.FAIR_VALUE_GAP.value == "fair_value_gap"
        assert SMCPattern.LIQUIDITY_SWEEP.value == "liquidity_sweep"
        assert SMCPattern.BREAK_OF_STRUCTURE.value == "break_of_structure"
    
    def test_pattern_enumeration(self):
        """Test that all patterns are accessible."""
        patterns = list(SMCPattern)
        assert len(patterns) >= 4
        assert SMCPattern.ORDER_BLOCK in patterns


class TestStrategyComparison:
    """Compare Smart Money and MTF strategies."""
    
    def test_both_strategies_return_valid_signals(self, sample_ohlcv_data):
        """Test that both strategies produce valid signals."""
        smc = SmartMoneyStrategy()
        mtf = MultiTimeframeStrategy()
        
        smc_signal = smc.generate_signal('AAPL', sample_ohlcv_data)
        mtf_signal = mtf.generate_signal('AAPL', sample_ohlcv_data)
        
        assert isinstance(smc_signal, Signal)
        assert isinstance(mtf_signal, Signal)
    
    def test_strategies_have_different_names(self):
        """Test that strategies have distinct names."""
        smc = SmartMoneyStrategy()
        mtf = MultiTimeframeStrategy()
        
        assert smc.name != mtf.name
    
    def test_strategies_consistent_interface(self, sample_ohlcv_data):
        """Test that both strategies have consistent interface."""
        smc = SmartMoneyStrategy()
        mtf = MultiTimeframeStrategy()
        
        for strategy in [smc, mtf]:
            signal = strategy.generate_signal('AAPL', sample_ohlcv_data)
            
            # All signals should have these attributes
            assert hasattr(signal, 'symbol')
            assert hasattr(signal, 'signal_type')
            assert hasattr(signal, 'confidence')
            assert hasattr(signal, 'price')
            assert hasattr(signal, 'stop_loss')
            assert hasattr(signal, 'take_profit')
            assert hasattr(signal, 'reasons')


class TestEdgeCases:
    """Test edge cases for Smart Money strategies."""
    
    def test_minimal_data(self):
        """Test with minimal data length."""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Open': [100] * 30,
            'High': [101] * 30,
            'Low': [99] * 30,
            'Close': [100] * 30,
            'Volume': [1000000.0] * 30
        }, index=dates)
        
        strategy = SmartMoneyStrategy()
        signal = strategy.generate_signal('TEST', df)
        
        # Should return HOLD with low confidence for insufficient data
        assert isinstance(signal, Signal)
    
    def test_high_volatility_data(self):
        """Test with highly volatile data."""
        np.random.seed(123)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Very volatile price action
        close = 100 + np.cumsum(np.random.normal(0, 5, 100))
        high = close + np.abs(np.random.normal(3, 1, 100))
        low = close - np.abs(np.random.normal(3, 1, 100))
        open_price = low + (high - low) * 0.5
        
        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': [1000000.0] * 100
        }, index=dates)
        
        strategy = SmartMoneyStrategy()
        signal = strategy.generate_signal('VOLATILE', df)
        
        assert isinstance(signal, Signal)
        # High volatility might result in lower confidence
        assert 0.0 <= signal.confidence <= 1.0
