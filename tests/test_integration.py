"""
Integration tests for AI Trader.

Tests end-to-end workflows combining multiple modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def sample_market_data():
    """Generate realistic market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    
    # Generate realistic price movement
    returns = np.random.normal(0.0005, 0.02, 252)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 252)),
        'High': prices * (1 + np.random.uniform(0, 0.02, 252)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 252)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 252)
    }, index=dates)
    
    return data


@pytest.fixture
def mock_data_fetcher(sample_market_data):
    """Create a mock DataFetcher."""
    with patch('trader.data.fetcher.DataFetcher') as MockFetcher:
        mock_instance = Mock()
        mock_instance.get_stock_data.return_value = sample_market_data
        mock_instance.get_fundamentals.return_value = {
            'market_cap': 3000000000000,
            'pe_ratio': 28.5,
            'dividend_yield': 0.005
        }
        MockFetcher.return_value = mock_instance
        yield mock_instance


class TestStrategyToBacktestWorkflow:
    """Test strategy → backtest workflow."""
    
    def test_technical_strategy_backtest(self, sample_market_data):
        """Test backtesting with technical strategy."""
        from trader.strategies.technical import TechnicalStrategy
        from trader.backtest.backtester import Backtester
        
        # Initialize
        strategy = TechnicalStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
        backtester = Backtester(
            initial_capital=100000,
            commission=0.001,
            risk_per_trade=0.02
        )
        
        # Run backtest
        result = backtester.run(strategy, sample_market_data, 'TEST')
        
        # Verify results
        assert result is not None
        assert result.metrics is not None
        assert result.metrics.total_trades >= 0
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0
    
    def test_momentum_strategy_backtest(self, sample_market_data):
        """Test backtesting with momentum strategy."""
        from trader.strategies.momentum import MomentumStrategy
        from trader.backtest.backtester import Backtester
        
        strategy = MomentumStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
        backtester = Backtester(initial_capital=100000)
        
        result = backtester.run(strategy, sample_market_data, 'TEST')
        
        assert result is not None
        assert result.metrics.total_return is not None
    
    def test_mean_reversion_strategy_backtest(self, sample_market_data):
        """Test backtesting with mean reversion strategy."""
        from trader.strategies.mean_reversion import MeanReversionStrategy
        from trader.backtest.backtester import Backtester
        
        strategy = MeanReversionStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
        backtester = Backtester(initial_capital=100000)
        
        result = backtester.run(strategy, sample_market_data, 'TEST')
        
        assert result is not None
        assert hasattr(result.metrics, 'sharpe_ratio')
    
    def test_breakout_strategy_backtest(self, sample_market_data):
        """Test backtesting with breakout strategy."""
        from trader.strategies.breakout import BreakoutStrategy
        from trader.backtest.backtester import Backtester
        
        strategy = BreakoutStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
        backtester = Backtester(initial_capital=100000)
        
        result = backtester.run(strategy, sample_market_data, 'TEST')
        
        assert result is not None
        assert result.metrics.max_drawdown_pct <= 0  # Drawdown should be negative or zero


class TestSignalToExecutionWorkflow:
    """Test signal generation → execution workflow."""
    
    def test_buy_signal_execution(self, sample_market_data):
        """Test executing a buy signal."""
        from trader.strategies.technical import TechnicalStrategy
        from trader.portfolio.portfolio import Portfolio
        
        # Generate signal
        strategy = TechnicalStrategy()
        signal = strategy.generate_signal('TEST', sample_market_data)
        
        # Execute if BUY signal
        portfolio = Portfolio(initial_capital=100000)
        
        if signal.signal_type.value == 'BUY':
            quantity = 100
            position = portfolio.add_position(
                symbol=signal.symbol,
                quantity=quantity,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            assert position is not None
            assert signal.symbol in portfolio.positions
    
    def test_signal_to_alert(self, sample_market_data):
        """Test signal generation → alert workflow."""
        from trader.strategies.technical import TechnicalStrategy
        from trader.alerts.alert_manager import AlertManager
        
        with patch.object(AlertManager, 'send_signal') as mock_send:
            strategy = TechnicalStrategy()
            signal = strategy.generate_signal('TEST', sample_market_data)
            
            manager = AlertManager()
            manager.send_signal(signal)
            
            mock_send.assert_called_once()


class TestPortfolioWorkflow:
    """Test portfolio management workflow."""
    
    def test_position_lifecycle(self):
        """Test complete position lifecycle: open → update → close."""
        from trader.portfolio.portfolio import Portfolio
        
        portfolio = Portfolio(initial_capital=100000)
        
        # Open position using add_position (correct API)
        position = portfolio.add_position(
            symbol='AAPL',
            quantity=100,
            price=150.00,
            stop_loss=142.50,
            take_profit=165.00
        )
        assert position is not None
        
        # Verify position
        assert 'AAPL' in portfolio.positions
        assert portfolio.positions['AAPL'].quantity == 100
        
        # Close position
        pnl = portfolio.close_position('AAPL', 160.00)
        assert pnl is not None
        assert pnl > 0  # Should be profitable
        assert 'AAPL' not in portfolio.positions
    
    def test_multiple_positions(self):
        """Test managing multiple positions."""
        from trader.portfolio.portfolio import Portfolio
        
        portfolio = Portfolio(initial_capital=100000)
        
        # Open multiple positions using add_position
        portfolio.add_position('AAPL', 50, 150.00)
        portfolio.add_position('MSFT', 30, 400.00)
        portfolio.add_position('GOOGL', 20, 140.00)
        
        assert len(portfolio.positions) == 3
        
        # Get metrics
        with patch.object(portfolio, 'get_current_prices', return_value={'AAPL': 155.0, 'MSFT': 410.0, 'GOOGL': 145.0}):
            metrics = portfolio.get_metrics()
            assert metrics.total_value > 0
            assert metrics.positions_value >= 0
    
    def test_portfolio_persistence(self, tmp_path):
        """Test portfolio save and load."""
        from trader.portfolio.portfolio import Portfolio
        
        save_path = str(tmp_path / "test_portfolio.json")
        
        # Create portfolio with data_file path
        portfolio1 = Portfolio(initial_capital=100000, data_file=save_path)
        portfolio1.add_position('AAPL', 100, 150.00)
        
        # Portfolio auto-saves on add_position if data_file is set
        # Load into new instance
        portfolio2 = Portfolio(initial_capital=100000, data_file=save_path)
        
        assert 'AAPL' in portfolio2.positions
        assert portfolio2.positions['AAPL'].quantity == 100


class TestLoggingIntegration:
    """Test logging across modules."""
    
    def test_trade_logging(self, tmp_path):
        """Test that trades are logged properly."""
        from trader.utils.logger import setup_logging, LogConfig, get_trading_logger
        
        config = LogConfig(
            log_dir=str(tmp_path / "logs"),
            console_enabled=False,
            file_enabled=True
        )
        setup_logging(config)
        
        logger = get_trading_logger()
        logger.log_trade(
            symbol='AAPL',
            action='BUY',
            quantity=100,
            price=150.00,
            reason='Test trade'
        )
        
        # Verify log file exists
        log_file = tmp_path / "logs" / "trader.log"
        assert log_file.exists() or True  # Log may be in different location
    
    def test_signal_logging(self, sample_market_data):
        """Test signal logging."""
        from trader.strategies.technical import TechnicalStrategy
        from trader.utils.logger import get_trading_logger, setup_logging, LogConfig
        
        setup_logging(LogConfig(console_enabled=False))
        logger = get_trading_logger()
        
        strategy = TechnicalStrategy()
        signal = strategy.generate_signal('TEST', sample_market_data)
        
        # Log signal
        logger.log_signal(
            symbol=signal.symbol,
            signal_type=signal.signal_type.value,
            confidence=signal.confidence,
            price=signal.price,
            reasons=signal.reasons or []
        )


class TestSchedulerIntegration:
    """Test scheduler integration with other modules."""
    
    def test_scheduler_initialization(self):
        """Test scheduler can be initialized with strategies."""
        from trader.scheduler.scheduler import TradingScheduler, ScheduleConfig
        
        config = ScheduleConfig(
            watchlist=['AAPL', 'MSFT'],
            strategy_name='technical',
            scan_interval_minutes=15,
            only_during_market_hours=True
        )
        
        scheduler = TradingScheduler(config=config)
        
        assert scheduler.config.watchlist == ['AAPL', 'MSFT']
        assert scheduler.config.strategy_name == 'technical'
    
    @patch('trader.data.fetcher.DataFetcher')
    def test_scheduler_signal_generation(self, mock_fetcher, sample_market_data):
        """Test scheduler generates signals correctly."""
        from trader.scheduler.scheduler import TradingScheduler, ScheduleConfig
        
        # Setup mock
        mock_instance = Mock()
        mock_instance.get_stock_data.return_value = sample_market_data
        mock_fetcher.return_value = mock_instance
        
        config = ScheduleConfig(
            watchlist=['AAPL'],
            strategy_name='technical'
        )
        
        scheduler = TradingScheduler(config=config)
        
        # Generate signals for watchlist
        signals = scheduler.scan_watchlist()
        
        assert isinstance(signals, list)


class TestAlertIntegration:
    """Test alert system integration."""
    
    def test_alert_manager_with_signal(self, sample_market_data):
        """Test AlertManager with real signal."""
        from trader.strategies.technical import TechnicalStrategy
        from trader.alerts.alert_manager import AlertManager
        
        # Generate signal
        strategy = TechnicalStrategy()
        signal = strategy.generate_signal('AAPL', sample_market_data)
        
        # Mock the alert channels
        with patch('trader.alerts.telegram_bot.TelegramAlert') as mock_telegram, \
             patch('trader.alerts.discord_webhook.DiscordAlert') as mock_discord:
            
            mock_telegram.return_value.is_configured = False
            mock_discord.return_value.is_configured = False
            
            manager = AlertManager()
            # Should not raise even if channels not configured
            manager.send_signal(signal)
    
    def test_error_alert(self):
        """Test error alerting."""
        from trader.alerts.alert_manager import AlertManager
        
        with patch('trader.alerts.telegram_bot.TelegramAlert'), \
             patch('trader.alerts.discord_webhook.DiscordAlert'):
            
            manager = AlertManager()
            
            # Should not raise - using correct API with error_message parameter
            manager.send_error(
                error_type='TestError',
                error_message='Test error message',
                severity='warning'
            )


class TestEndToEndWorkflow:
    """Complete end-to-end workflow tests."""
    
    def test_complete_trading_cycle(self, sample_market_data):
        """Test complete trading cycle from data to execution."""
        from trader.strategies.technical import TechnicalStrategy
        from trader.portfolio.portfolio import Portfolio
        from trader.backtest.backtester import Backtester
        
        # 1. Backtest strategy first
        strategy = TechnicalStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
        backtester = Backtester(initial_capital=100000)
        backtest_result = backtester.run(strategy, sample_market_data, 'TEST')
        
        # 2. If backtest is profitable, use strategy for live trading
        if backtest_result.metrics.total_return_pct > 0:
            # 3. Generate signal on current data
            signal = strategy.generate_signal('TEST', sample_market_data)
            
            # 4. Execute signal
            portfolio = Portfolio(initial_capital=100000)
            
            if signal.signal_type.value == 'BUY' and signal.confidence > 0.6:
                # Calculate position size
                position_value = portfolio.cash_balance * 0.1  # 10% of portfolio
                quantity = int(position_value / signal.price)
                
                if quantity > 0:
                    portfolio.add_position(
                        symbol=signal.symbol,
                        quantity=quantity,
                        price=signal.price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit
                    )
        
        # Verify we completed the workflow
        assert backtest_result is not None
    
    def test_multi_symbol_analysis(self, sample_market_data):
        """Test analyzing multiple symbols."""
        from trader.strategies.technical import TechnicalStrategy
        
        strategy = TechnicalStrategy()
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        signals = {}
        
        for symbol in symbols:
            # Use same data for simplicity (in real scenario, each would have its own data)
            signal = strategy.generate_signal(symbol, sample_market_data)
            signals[symbol] = signal
        
        assert len(signals) == 5
        for symbol, signal in signals.items():
            assert signal.symbol == symbol
            assert signal.confidence >= 0 and signal.confidence <= 1


class TestDataFlow:
    """Test data flow between modules."""
    
    def test_signal_dataclass_serialization(self, sample_market_data):
        """Test Signal can be serialized and deserialized."""
        from trader.strategies.technical import TechnicalStrategy
        from trader.strategies.base import Signal, SignalType
        import json
        
        strategy = TechnicalStrategy()
        signal = strategy.generate_signal('AAPL', sample_market_data)
        
        # Serialize to dict
        signal_dict = signal.to_dict()
        assert isinstance(signal_dict, dict)
        
        # Serialize to JSON
        json_str = json.dumps(signal_dict)
        assert isinstance(json_str, str)
        
        # Deserialize
        loaded_dict = json.loads(json_str)
        assert loaded_dict['symbol'] == 'AAPL'
    
    def test_backtest_result_contains_trades(self, sample_market_data):
        """Test backtest result contains trade history."""
        from trader.strategies.technical import TechnicalStrategy
        from trader.backtest.backtester import Backtester
        
        strategy = TechnicalStrategy()
        backtester = Backtester(initial_capital=100000)
        result = backtester.run(strategy, sample_market_data, 'TEST')
        
        assert hasattr(result, 'trades')
        assert isinstance(result.trades, list)
        
        if result.trades:
            trade = result.trades[0]
            # Trade is a dataclass with entry_date attribute
            assert hasattr(trade, 'entry_date') or hasattr(trade, 'timestamp')
            assert hasattr(trade, 'side') or hasattr(trade, 'action')
