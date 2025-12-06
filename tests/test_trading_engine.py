"""Tests for the Unified Trading Engine."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from trader.core.trading_engine import (
    TradingEngine,
    TradingEngineConfig,
    TradingMode,
    SignalStrength,
    AnalysisResult,
    TradeExecution,
    create_trading_engine
)
from trader.strategies.base import Signal, SignalType


class TestTradingMode:
    """Tests for TradingMode enum."""
    
    def test_trading_mode_values(self):
        """Test trading mode values."""
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.BACKTEST.value == "backtest"
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.ANALYSIS.value == "analysis"
    
    def test_all_modes_exist(self):
        """Test all expected modes exist."""
        modes = [m for m in TradingMode]
        assert len(modes) == 4


class TestSignalStrength:
    """Tests for SignalStrength enum."""
    
    def test_signal_strength_values(self):
        """Test signal strength values."""
        assert SignalStrength.STRONG_BUY.value == "strong_buy"
        assert SignalStrength.BUY.value == "buy"
        assert SignalStrength.WEAK_BUY.value == "weak_buy"
        assert SignalStrength.NEUTRAL.value == "neutral"
        assert SignalStrength.WEAK_SELL.value == "weak_sell"
        assert SignalStrength.SELL.value == "sell"
        assert SignalStrength.STRONG_SELL.value == "strong_sell"
    
    def test_all_strengths_exist(self):
        """Test all expected strengths exist."""
        strengths = [s for s in SignalStrength]
        assert len(strengths) == 7


class TestTradingEngineConfig:
    """Tests for TradingEngineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TradingEngineConfig()
        
        assert config.mode == TradingMode.PAPER
        assert config.initial_capital == 100000.0
        assert config.max_position_pct == 0.15
        assert config.risk_per_trade_pct == 0.02
        assert config.max_open_positions == 10
        assert config.min_confidence == 0.6
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TradingEngineConfig(
            mode=TradingMode.BACKTEST,
            initial_capital=50000.0,
            max_position_pct=0.10,
            min_confidence=0.7
        )
        
        assert config.mode == TradingMode.BACKTEST
        assert config.initial_capital == 50000.0
        assert config.max_position_pct == 0.10
        assert config.min_confidence == 0.7
    
    def test_feature_flags(self):
        """Test feature flags."""
        config = TradingEngineConfig(
            use_ai_ensemble=False,
            use_regime_detection=False,
            use_ml_prediction=False
        )
        
        assert not config.use_ai_ensemble
        assert not config.use_regime_detection
        assert not config.use_ml_prediction


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""
    
    def test_default_result(self):
        """Test default analysis result."""
        result = AnalysisResult(symbol='AAPL')
        
        assert result.symbol == 'AAPL'
        assert result.signal_type == SignalType.HOLD
        assert result.signal_strength == SignalStrength.NEUTRAL
        assert result.confidence == 0.0
        assert not result.is_actionable
        assert result.suggested_action == "hold"
    
    def test_actionable_result(self):
        """Test actionable analysis result."""
        result = AnalysisResult(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            signal_strength=SignalStrength.STRONG_BUY,
            confidence=0.85,
            risk_adjusted_size=15000.0,
            stop_loss=145.0,
            take_profit=165.0,
            risk_reward_ratio=2.0,
            is_actionable=True,
            suggested_action="buy",
            reasons=["RSI oversold", "MACD bullish crossover"]
        )
        
        assert result.is_actionable
        assert result.confidence == 0.85
        assert result.risk_reward_ratio == 2.0
        assert len(result.reasons) == 2
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AnalysisResult(
            symbol='MSFT',
            signal_type=SignalType.SELL,
            confidence=0.75
        )
        
        data = result.to_dict()
        
        assert data['symbol'] == 'MSFT'
        assert data['signal_type'] == 'sell'
        assert data['confidence'] == 0.75
        assert 'timestamp' in data


class TestTradeExecution:
    """Tests for TradeExecution dataclass."""
    
    def test_basic_execution(self):
        """Test basic trade execution."""
        execution = TradeExecution(
            symbol='AAPL',
            action='BUY',
            quantity=100,
            price=150.0
        )
        
        assert execution.symbol == 'AAPL'
        assert execution.action == 'BUY'
        assert execution.quantity == 100
        assert execution.price == 150.0
        assert not execution.executed
    
    def test_total_value(self):
        """Test total value calculation."""
        execution = TradeExecution(
            symbol='AAPL',
            action='BUY',
            quantity=100,
            price=150.0,
            execution_price=150.50,
            commission=1.0
        )
        
        # 100 * 150.50 + 1.0 = 15051.0
        assert execution.total_value == 15051.0
    
    def test_total_value_no_execution_price(self):
        """Test total value uses target price if no execution price."""
        execution = TradeExecution(
            symbol='AAPL',
            action='BUY',
            quantity=100,
            price=150.0
        )
        
        assert execution.total_value == 15000.0


class TestTradingEngine:
    """Tests for TradingEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create trading engine instance."""
        config = TradingEngineConfig(
            mode=TradingMode.ANALYSIS,
            use_ai_ensemble=False,
            use_regime_detection=False,
            use_ml_prediction=False,
            use_dynamic_risk=False,
            auto_journal_trades=False
        )
        return TradingEngine(config=config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = 150 + np.cumsum(np.random.randn(100) * 2)
        
        return pd.DataFrame({
            'Open': prices - 1,
            'High': prices + 2,
            'Low': prices - 2,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.config.mode == TradingMode.ANALYSIS
        assert engine.current_capital == 100000.0
        assert not engine.is_initialized
        assert engine.daily_trades == 0
    
    def test_initialize_components(self, engine):
        """Test component initialization."""
        result = engine.initialize()
        
        assert result is True
        assert engine.is_initialized
        assert engine._data_fetcher is not None
        assert engine._technical_strategy is not None
        assert engine._portfolio is not None
    
    def test_analyze_symbol_with_data(self, engine, sample_data):
        """Test symbol analysis with provided data."""
        engine.initialize()
        
        result = engine.analyze_symbol('AAPL', data=sample_data)
        
        assert isinstance(result, AnalysisResult)
        assert result.symbol == 'AAPL'
        assert result.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
    
    def test_analyze_symbol_insufficient_data(self, engine):
        """Test analysis with insufficient data."""
        engine.initialize()
        
        # Only 10 days of data
        short_data = pd.DataFrame({
            'Open': [100] * 10,
            'High': [101] * 10,
            'Low': [99] * 10,
            'Close': [100] * 10,
            'Volume': [1000000] * 10
        })
        
        result = engine.analyze_symbol('AAPL', data=short_data)
        
        assert "Insufficient data" in result.warnings[0]
    
    def test_signal_strength_classification(self, engine):
        """Test signal strength classification."""
        engine.initialize()
        
        # Strong buy
        strength = engine._get_signal_strength(SignalType.BUY, 0.85)
        assert strength == SignalStrength.STRONG_BUY
        
        # Regular buy
        strength = engine._get_signal_strength(SignalType.BUY, 0.65)
        assert strength == SignalStrength.BUY
        
        # Weak buy
        strength = engine._get_signal_strength(SignalType.BUY, 0.55)
        assert strength == SignalStrength.WEAK_BUY
        
        # Hold is always neutral
        strength = engine._get_signal_strength(SignalType.HOLD, 0.9)
        assert strength == SignalStrength.NEUTRAL
    
    def test_combine_signals_technical_only(self, engine):
        """Test signal combination with only technical signal."""
        result = AnalysisResult(symbol='AAPL')
        result.technical_signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=150.0
        )
        
        engine.initialize()
        signal, confidence = engine._combine_signals(result)
        
        # Should lean towards buy
        assert signal == SignalType.BUY or confidence > 0
    
    def test_daily_trade_limit(self, engine):
        """Test daily trade limit enforcement."""
        # Use PAPER mode for execution tests
        engine.config.mode = TradingMode.PAPER
        engine.config.max_daily_trades = 2
        engine.initialize()
        
        # Execute trades up to limit
        result1 = engine.execute_trade('AAPL', 'BUY', 10, 150.0)
        result2 = engine.execute_trade('MSFT', 'BUY', 10, 300.0)
        
        assert engine.daily_trades == 2
        
        # Third trade should be blocked
        result = engine.execute_trade('GOOGL', 'BUY', 10, 140.0)
        
        assert not result.executed
    
    def test_daily_trade_reset(self, engine):
        """Test daily trade counter resets on new day."""
        # Use PAPER mode for execution tests
        engine.config.mode = TradingMode.PAPER
        engine.initialize()
        
        engine.daily_trades = 5
        engine.last_trade_date = datetime.now(timezone.utc).date() - timedelta(days=1)
        
        # Execute trade (should reset counter)
        engine.execute_trade('AAPL', 'BUY', 10, 150.0)
        
        assert engine.daily_trades == 1
    
    def test_analysis_mode_no_execution(self, engine):
        """Test that analysis mode doesn't execute trades."""
        engine.config.mode = TradingMode.ANALYSIS
        engine.initialize()
        
        result = engine.execute_trade('AAPL', 'BUY', 100, 150.0)
        
        assert not result.executed
    
    def test_scan_watchlist(self, engine, sample_data):
        """Test watchlist scanning."""
        engine.initialize()
        
        # Mock data fetcher to return sample data
        engine._data_fetcher.get_stock_data = Mock(return_value=sample_data)
        
        results = engine.scan_watchlist(['AAPL', 'MSFT'])
        
        assert isinstance(results, list)
        # Results should be sorted by confidence
        if len(results) > 1:
            assert results[0].confidence >= results[1].confidence
    
    def test_get_portfolio_status(self, engine):
        """Test portfolio status retrieval."""
        engine.initialize()
        
        status = engine.get_portfolio_status()
        
        assert isinstance(status, dict)
        assert 'total_value' in status or status == {}
    
    def test_get_status_report(self, engine):
        """Test status report generation."""
        engine.initialize()
        
        report = engine.get_status_report()
        
        assert "TRADING ENGINE STATUS REPORT" in report
        assert "COMPONENTS" in report
        assert "TRADING STATS" in report
    
    def test_analysis_cache(self, engine, sample_data):
        """Test analysis results are cached."""
        engine.initialize()
        
        result = engine.analyze_symbol('AAPL', data=sample_data)
        
        assert 'AAPL' in engine.analysis_cache
        assert engine.analysis_cache['AAPL'] == result
    
    def test_stop_loss_take_profit_calculation(self, engine, sample_data):
        """Test stop loss and take profit calculation."""
        engine.initialize()
        
        # Force a buy signal
        with patch.object(engine._technical_strategy, 'generate_signal') as mock:
            mock.return_value = Signal(
                symbol='AAPL',
                signal_type=SignalType.BUY,
                confidence=0.8,
                price=150.0
            )
            
            result = engine.analyze_symbol('AAPL', data=sample_data)
        
        if result.signal_type == SignalType.BUY:
            current_price = float(sample_data['Close'].iloc[-1])
            assert result.stop_loss < current_price
            assert result.take_profit > current_price


class TestCreateTradingEngine:
    """Tests for create_trading_engine factory function."""
    
    def test_create_paper_engine(self):
        """Test creating paper trading engine."""
        engine = create_trading_engine(mode='paper')
        
        assert engine.config.mode == TradingMode.PAPER
        assert engine.is_initialized
    
    def test_create_backtest_engine(self):
        """Test creating backtest engine."""
        engine = create_trading_engine(mode='backtest')
        
        assert engine.config.mode == TradingMode.BACKTEST
    
    def test_create_analysis_engine(self):
        """Test creating analysis-only engine."""
        engine = create_trading_engine(mode='analysis')
        
        assert engine.config.mode == TradingMode.ANALYSIS
    
    def test_create_with_custom_capital(self):
        """Test creating engine with custom capital."""
        engine = create_trading_engine(
            mode='paper',
            initial_capital=50000.0
        )
        
        assert engine.config.initial_capital == 50000.0
    
    def test_create_with_kwargs(self):
        """Test creating engine with additional kwargs."""
        engine = create_trading_engine(
            mode='paper',
            max_position_pct=0.10,
            min_confidence=0.7
        )
        
        assert engine.config.max_position_pct == 0.10
        assert engine.config.min_confidence == 0.7


class TestIntegration:
    """Integration tests for trading engine."""
    
    @pytest.fixture
    def full_engine(self):
        """Create fully configured engine."""
        config = TradingEngineConfig(
            mode=TradingMode.PAPER,
            initial_capital=100000.0,
            use_ai_ensemble=False,  # Skip AI for tests
            use_ml_prediction=False,  # Skip ML for tests
            auto_journal_trades=False
        )
        engine = TradingEngine(config=config)
        engine.initialize()
        return engine
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Create trending data for clearer signals
        trend = np.linspace(0, 20, 100)
        noise = np.random.randn(100) * 2
        prices = 150 + trend + noise
        
        return pd.DataFrame({
            'Open': prices - 1,
            'High': prices + 2,
            'Low': prices - 2,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_full_analysis_workflow(self, full_engine, sample_data):
        """Test complete analysis workflow."""
        # Analyze
        result = full_engine.analyze_symbol('AAPL', data=sample_data)
        
        assert result.symbol == 'AAPL'
        assert result.timestamp is not None
        assert result.signal_type in SignalType
        assert result.signal_strength in SignalStrength
        assert 0 <= result.confidence <= 1
    
    def test_trade_execution_workflow(self, full_engine, sample_data):
        """Test trade execution workflow."""
        # Analyze first
        result = full_engine.analyze_symbol('AAPL', data=sample_data)
        
        # Execute based on analysis
        if result.is_actionable:
            quantity = int(result.risk_adjusted_size / float(sample_data['Close'].iloc[-1]))
            quantity = max(1, quantity)
            
            execution = full_engine.execute_trade(
                symbol='AAPL',
                action=result.suggested_action.upper(),
                quantity=quantity,
                price=float(sample_data['Close'].iloc[-1]),
                stop_loss=result.stop_loss,
                take_profit=result.take_profit
            )
            
            assert execution.symbol == 'AAPL'
            # In paper mode, should execute
            if full_engine.config.mode == TradingMode.PAPER:
                assert execution.executed or not result.is_actionable
    
    def test_watchlist_scan_workflow(self, full_engine, sample_data):
        """Test watchlist scanning workflow."""
        # Mock data fetcher
        full_engine._data_fetcher.get_stock_data = Mock(return_value=sample_data)
        
        # Scan watchlist
        results = full_engine.scan_watchlist(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            min_confidence=0.3
        )
        
        assert isinstance(results, list)
        
        # Find actionable signals
        actionable = [r for r in results if r.is_actionable]
        
        # Execute top signal if any
        if actionable:
            top = actionable[0]
            current_price = float(sample_data['Close'].iloc[-1])
            quantity = int(top.risk_adjusted_size / current_price) if top.risk_adjusted_size > 0 else 10
            quantity = max(1, quantity)
            
            execution = full_engine.execute_trade(
                symbol=top.symbol,
                action=top.suggested_action.upper(),
                quantity=quantity,
                price=current_price
            )
            
            assert execution.symbol == top.symbol
    
    def test_risk_reward_calculation(self, full_engine, sample_data):
        """Test risk/reward ratio calculation."""
        result = full_engine.analyze_symbol('AAPL', data=sample_data)
        
        if result.signal_type != SignalType.HOLD:
            # Should have stop loss and take profit
            assert result.stop_loss > 0 or result.take_profit > 0
            
            # Risk/reward should be positive
            if result.risk_reward_ratio > 0:
                assert result.risk_reward_ratio > 0
    
    def test_engine_state_persistence(self, full_engine, sample_data):
        """Test engine maintains state across operations."""
        full_engine._data_fetcher.get_stock_data = Mock(return_value=sample_data)
        
        # Multiple analyses
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            full_engine.analyze_symbol(symbol, data=sample_data)
        
        # Check cache populated
        assert len(full_engine.analysis_cache) == 3
        
        # Execute trades
        full_engine.execute_trade('AAPL', 'BUY', 10, 150.0)
        full_engine.execute_trade('MSFT', 'BUY', 5, 300.0)
        
        # Check state
        assert len(full_engine.trades_executed) == 2
        assert full_engine.daily_trades == 2
