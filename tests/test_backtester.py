"""
Unit tests for backtesting module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from trader.backtest.backtester import (
    Backtester, BacktestResult, BacktestMetrics,
    Trade, TradeStatus
)
from trader.strategies.technical import TechnicalStrategy
from trader.strategies.momentum import MomentumStrategy
from trader.strategies.mean_reversion import MeanReversionStrategy
from trader.strategies.breakout import BreakoutStrategy


class TestTrade:
    """Tests for Trade dataclass."""
    
    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            symbol='AAPL',
            entry_date=datetime(2024, 1, 1),
            entry_price=150.0,
            quantity=100,
            side='long',
            stop_loss=142.50,
            take_profit=165.00,
        )
        
        assert trade.symbol == 'AAPL'
        assert trade.entry_price == 150.0
        assert trade.quantity == 100
        assert trade.side == 'long'
        assert trade.status == TradeStatus.OPEN
    
    def test_trade_close_profit(self):
        """Test closing a profitable trade."""
        trade = Trade(
            symbol='AAPL',
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            quantity=100,
            side='long',
        )
        
        trade.close(datetime(2024, 1, 10), 110.0)
        
        assert trade.status == TradeStatus.CLOSED
        assert trade.exit_price == 110.0
        assert trade.pnl == 1000.0  # (110 - 100) * 100
        assert trade.pnl_percent == 10.0
    
    def test_trade_close_loss(self):
        """Test closing a losing trade."""
        trade = Trade(
            symbol='AAPL',
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            quantity=100,
            side='long',
        )
        
        trade.close(datetime(2024, 1, 10), 95.0)
        
        assert trade.pnl == -500.0  # (95 - 100) * 100
        assert trade.pnl_percent == -5.0
    
    def test_short_trade_profit(self):
        """Test profitable short trade."""
        trade = Trade(
            symbol='AAPL',
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            quantity=100,
            side='short',
        )
        
        trade.close(datetime(2024, 1, 10), 90.0)
        
        assert trade.pnl == 1000.0  # (100 - 90) * 100
        assert trade.pnl_percent == 10.0
    
    def test_short_trade_loss(self):
        """Test losing short trade."""
        trade = Trade(
            symbol='AAPL',
            entry_date=datetime(2024, 1, 1),
            entry_price=100.0,
            quantity=100,
            side='short',
        )
        
        trade.close(datetime(2024, 1, 10), 105.0)
        
        assert trade.pnl == -500.0  # (100 - 105) * 100
        assert trade.pnl_percent == -5.0


class TestBacktestMetrics:
    """Tests for BacktestMetrics."""
    
    def test_metrics_default_values(self):
        """Test default metric values."""
        metrics = BacktestMetrics()
        
        assert metrics.total_return == 0.0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.sharpe_ratio == 0.0
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = BacktestMetrics(
            total_return=1000.0,
            total_return_pct=10.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=5.0,
            win_rate=60.0,
            total_trades=20,
        )
        
        d = metrics.to_dict()
        assert 'Total Return' in d
        assert 'Sharpe Ratio' in d
        assert 'Win Rate' in d


class TestBacktester:
    """Tests for Backtester class."""
    
    def test_initialization(self):
        """Test backtester initialization."""
        backtester = Backtester(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
        )
        
        assert backtester.initial_capital == 100000
        assert backtester.commission == 0.001
        assert backtester.slippage == 0.0005
    
    def test_initialization_custom_params(self):
        """Test backtester with custom parameters."""
        backtester = Backtester(
            initial_capital=50000,
            commission=0.002,
            risk_per_trade=0.01,
            max_positions=3,
            allow_shorting=True,
        )
        
        assert backtester.initial_capital == 50000
        assert backtester.risk_per_trade == 0.01
        assert backtester.max_positions == 3
        assert backtester.allow_shorting == True
    
    def test_run_returns_result(self, sample_ohlcv_data):
        """Test that run returns BacktestResult."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        
        assert isinstance(result, BacktestResult)
        assert isinstance(result.metrics, BacktestMetrics)
        assert isinstance(result.trades, list)
        assert isinstance(result.equity_curve, pd.Series)
    
    def test_run_with_technical_strategy(self, sample_ohlcv_data):
        """Test backtest with technical strategy."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        
        assert result.symbol == 'AAPL'
        assert result.initial_capital == 100000
        assert result.strategy_name == 'TechnicalStrategy'
    
    def test_run_with_momentum_strategy(self, sample_ohlcv_data):
        """Test backtest with momentum strategy."""
        backtester = Backtester(initial_capital=100000)
        strategy = MomentumStrategy()
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        
        assert result.strategy_name == 'MomentumStrategy'
    
    def test_run_with_mean_reversion_strategy(self, sample_ohlcv_data):
        """Test backtest with mean reversion strategy."""
        backtester = Backtester(initial_capital=100000)
        strategy = MeanReversionStrategy()
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        
        assert result.strategy_name == 'MeanReversionStrategy'
    
    def test_run_with_breakout_strategy(self, sample_ohlcv_data):
        """Test backtest with breakout strategy."""
        backtester = Backtester(initial_capital=100000)
        strategy = BreakoutStrategy()
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        
        assert result.strategy_name == 'BreakoutStrategy'
    
    def test_equity_curve_length(self, sample_ohlcv_data):
        """Test equity curve has correct length."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        
        # Equity curve should have entries for each bar processed
        assert len(result.equity_curve) > 0
    
    def test_drawdown_calculation(self, sample_ohlcv_data):
        """Test drawdown is calculated correctly."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        
        # Drawdown should be non-positive
        assert result.metrics.max_drawdown >= 0
        assert result.metrics.max_drawdown_pct >= 0
    
    def test_win_rate_bounds(self, sample_ohlcv_data):
        """Test win rate is between 0 and 100."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        
        assert 0.0 <= result.metrics.win_rate <= 100.0
    
    def test_trade_count_consistency(self, sample_ohlcv_data):
        """Test trade counts are consistent."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        
        assert result.metrics.total_trades == len(result.trades)
        assert result.metrics.total_trades == result.metrics.winning_trades + result.metrics.losing_trades
    
    def test_commission_impact(self, trending_up_data):
        """Test that commission affects results."""
        strategy = TechnicalStrategy()
        
        # Run with no commission
        bt_no_comm = Backtester(initial_capital=100000, commission=0.0)
        result_no_comm = bt_no_comm.run(strategy, trending_up_data, 'AAPL')
        
        # Run with commission
        bt_comm = Backtester(initial_capital=100000, commission=0.01)
        result_comm = bt_comm.run(strategy, trending_up_data, 'AAPL')
        
        # If there are trades, commission version should have lower returns
        if result_no_comm.metrics.total_trades > 0 and result_comm.metrics.total_trades > 0:
            assert result_comm.metrics.total_return <= result_no_comm.metrics.total_return
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises error."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            backtester.run(strategy, empty_df, 'AAPL')
    
    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises error."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        incomplete_df = pd.DataFrame({
            'Close': [100, 101, 102],
        }, index=pd.date_range('2024-01-01', periods=3))
        
        with pytest.raises(ValueError):
            backtester.run(strategy, incomplete_df, 'AAPL')
    
    def test_result_summary(self, sample_ohlcv_data):
        """Test result summary generation."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert 'AAPL' in summary
        assert 'TechnicalStrategy' in summary
    
    def test_compare_strategies(self, sample_ohlcv_data):
        """Test comparing multiple strategies."""
        backtester = Backtester(initial_capital=100000)
        
        strategies = [
            TechnicalStrategy(),
            MomentumStrategy(),
        ]
        
        comparison = backtester.compare_strategies(
            strategies, sample_ohlcv_data, 'AAPL'
        )
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'Strategy' in comparison.columns
        assert 'Total Return %' in comparison.columns


class TestBacktestScenarios:
    """Test specific backtest scenarios."""
    
    def test_uptrend_performance(self, trending_up_data):
        """Test performance in uptrending market."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        result = backtester.run(strategy, trending_up_data, 'AAPL')
        
        # In strong uptrend, a reasonable strategy should not lose too much
        # (allowing for some loss due to timing/signals)
        assert result.metrics.total_return_pct > -50  # Should not lose more than 50%
    
    def test_downtrend_performance(self, trending_down_data):
        """Test performance in downtrending market."""
        backtester = Backtester(initial_capital=100000)
        strategy = TechnicalStrategy()
        
        result = backtester.run(strategy, trending_down_data, 'AAPL')
        
        # Should complete without errors
        assert isinstance(result, BacktestResult)
    
    def test_volatile_market_performance(self, volatile_data):
        """Test performance in volatile market."""
        backtester = Backtester(initial_capital=100000)
        strategy = BreakoutStrategy()  # Breakout should detect volatility
        
        result = backtester.run(strategy, volatile_data, 'AAPL')
        
        assert isinstance(result, BacktestResult)
    
    def test_range_bound_mean_reversion(self, range_bound_data):
        """Test mean reversion in range-bound market."""
        backtester = Backtester(initial_capital=100000)
        strategy = MeanReversionStrategy()
        
        result = backtester.run(strategy, range_bound_data, 'AAPL')
        
        # Mean reversion should work in range-bound markets
        assert isinstance(result, BacktestResult)
    
    def test_all_strategies_complete(self, sample_ohlcv_data):
        """Test all strategies complete backtest."""
        backtester = Backtester(initial_capital=100000)
        
        strategies = [
            TechnicalStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
        ]
        
        for strategy in strategies:
            result = backtester.run(strategy, sample_ohlcv_data, 'AAPL')
            assert isinstance(result, BacktestResult), f"{strategy.__class__.__name__} failed"
            assert result.final_capital > 0, f"{strategy.__class__.__name__} went bankrupt"
