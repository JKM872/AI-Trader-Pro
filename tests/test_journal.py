"""
Tests for Trade Journal Module.

Tests for trade journaling, performance tracking, and pattern analysis.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from trader.journal import (
    TradeJournal,
    TradeEntry,
    TradeStatus,
    TradeType,
    JournalStats,
    PerformanceTracker,
    PerformanceMetrics,
    TimeframedMetrics,
    DrawdownAnalysis,
    PatternAnalyzer,
    TradingPattern,
    PatternType,
    PatternStrength,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_journal_path():
    """Create a temporary path for journal storage."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def trade_journal(temp_journal_path):
    """Create a trade journal instance."""
    return TradeJournal(storage_path=temp_journal_path)


@pytest.fixture
def journal_with_trades(temp_journal_path):
    """Create a journal with sample trades."""
    journal = TradeJournal(storage_path=temp_journal_path)
    
    # Create sample trades
    base_time = datetime.now() - timedelta(days=30)
    
    trades_data = [
        ('AAPL', TradeType.LONG, 150.0, 155.0, 100, 'Technical', 0.75, ['RSI oversold']),
        ('MSFT', TradeType.LONG, 300.0, 295.0, 50, 'Momentum', 0.65, ['Strong momentum']),
        ('GOOGL', TradeType.LONG, 2800.0, 2900.0, 10, 'Technical', 0.80, ['Breakout']),
        ('AAPL', TradeType.LONG, 155.0, 160.0, 100, 'Technical', 0.70, ['Trend continuation']),
        ('TSLA', TradeType.SHORT, 250.0, 240.0, 40, 'Mean Reversion', 0.60, ['Overbought']),
        ('AMZN', TradeType.LONG, 3200.0, 3100.0, 5, 'Momentum', 0.55, ['Weak signal']),
        ('NVDA', TradeType.LONG, 450.0, 480.0, 20, 'Technical', 0.85, ['Multiple confirmations']),
        ('META', TradeType.LONG, 320.0, 330.0, 30, 'Smart Money', 0.72, ['Order block']),
    ]
    
    for i, (symbol, trade_type, entry, exit_p, qty, strategy, conf, reasons) in enumerate(trades_data):
        trade = journal.create_trade(
            symbol=symbol,
            trade_type=trade_type,
            entry_price=entry,
            quantity=qty,
            strategy_name=strategy,
            signal_confidence=conf,
            signal_reasons=reasons,
            market_regime='Trending' if i % 2 == 0 else 'Ranging',
        )
        
        # Set entry time for time-based analysis
        trade.entry_time = base_time + timedelta(days=i * 3, hours=9 + (i % 8))
        
        # Close the trade
        journal.close_trade(
            trade.trade_id,
            exit_price=exit_p,
        )
        
        # Update exit time
        trade.exit_time = trade.entry_time + timedelta(hours=2 + i)
    
    return journal


@pytest.fixture
def performance_tracker(journal_with_trades):
    """Create a performance tracker with trades."""
    tracker = PerformanceTracker(
        journal=journal_with_trades,
        initial_capital=100000.0
    )
    return tracker


@pytest.fixture
def pattern_analyzer(journal_with_trades):
    """Create a pattern analyzer with trades."""
    analyzer = PatternAnalyzer(journal=journal_with_trades)
    return analyzer


# ============================================================================
# TradeEntry Tests
# ============================================================================

class TestTradeEntry:
    """Tests for TradeEntry dataclass."""
    
    def test_create_trade_entry(self):
        """Test creating a trade entry."""
        entry = TradeEntry(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
            entry_quantity=100,
        )
        
        assert entry.symbol == 'AAPL'
        assert entry.trade_type == TradeType.LONG
        assert entry.entry_price == 150.0
        assert entry.entry_quantity == 100
        assert entry.trade_id is not None
    
    def test_calculate_pnl_long(self):
        """Test P/L calculation for long trade."""
        entry = TradeEntry(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
            entry_quantity=100,
            exit_price=160.0,
        )
        
        pnl = entry.calculate_pnl()
        assert pnl == 1000.0  # (160 - 150) * 100
    
    def test_calculate_pnl_short(self):
        """Test P/L calculation for short trade."""
        entry = TradeEntry(
            symbol='AAPL',
            trade_type=TradeType.SHORT,
            entry_price=160.0,
            entry_quantity=100,
            exit_price=150.0,
        )
        
        pnl = entry.calculate_pnl()
        assert pnl == 1000.0  # (160 - 150) * 100
    
    def test_calculate_pnl_pct(self):
        """Test P/L percentage calculation."""
        entry = TradeEntry(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=100.0,
            entry_quantity=100,
            exit_price=110.0,
        )
        
        pnl_pct = entry.calculate_pnl_pct()
        assert pnl_pct == 10.0  # 10% gain
    
    def test_duration(self):
        """Test trade duration calculation."""
        entry = TradeEntry(
            symbol='AAPL',
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 14, 0),
        )
        
        duration = entry.duration()
        assert duration == timedelta(hours=4)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = TradeEntry(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
        )
        
        data = entry.to_dict()
        
        assert data['symbol'] == 'AAPL'
        assert data['trade_type'] == 'long'
        assert data['entry_price'] == 150.0
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'trade_id': 'test123',
            'symbol': 'AAPL',
            'trade_type': 'long',
            'status': 'open',
            'entry_price': 150.0,
            'entry_quantity': 100,
            'entry_time': '2024-01-01T10:00:00',
            'exit_price': None,
            'exit_time': None,
            'exit_quantity': None,
            'stop_loss': 145.0,
            'take_profit': 160.0,
            'trailing_stop': None,
            'strategy_name': 'Technical',
            'signal_confidence': 0.75,
            'signal_reasons': ['RSI oversold'],
            'market_regime': 'Trending',
            'volatility_level': '',
            'trend_direction': '',
            'ai_predictions': {},
            'ensemble_consensus': None,
            'ml_confidence': None,
            'realized_pnl': None,
            'realized_pnl_pct': None,
            'max_favorable_excursion': None,
            'max_adverse_excursion': None,
            'notes': '',
            'tags': [],
            'screenshots': [],
            'created_at': '2024-01-01T10:00:00',
            'updated_at': '2024-01-01T10:00:00',
        }
        
        entry = TradeEntry.from_dict(data)
        
        assert entry.symbol == 'AAPL'
        assert entry.trade_type == TradeType.LONG
        assert entry.entry_price == 150.0


# ============================================================================
# TradeJournal Tests
# ============================================================================

class TestTradeJournal:
    """Tests for TradeJournal class."""
    
    def test_create_journal(self, temp_journal_path):
        """Test creating a journal."""
        journal = TradeJournal(storage_path=temp_journal_path)
        assert journal is not None
        assert len(journal.trades) == 0
    
    def test_create_trade(self, trade_journal):
        """Test creating a trade."""
        trade = trade_journal.create_trade(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
            quantity=100,
            strategy_name='Technical',
        )
        
        assert trade is not None
        assert trade.symbol == 'AAPL'
        assert trade.status == TradeStatus.OPEN
        assert len(trade_journal.trades) == 1
    
    def test_close_trade(self, trade_journal):
        """Test closing a trade."""
        trade = trade_journal.create_trade(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
            quantity=100,
        )
        
        closed = trade_journal.close_trade(trade.trade_id, exit_price=160.0)
        
        assert closed is not None
        assert closed.status == TradeStatus.CLOSED
        assert closed.exit_price == 160.0
        assert closed.realized_pnl == 1000.0
    
    def test_update_trade(self, trade_journal):
        """Test updating a trade."""
        trade = trade_journal.create_trade(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
            quantity=100,
        )
        
        updated = trade_journal.update_trade(
            trade.trade_id,
            notes='Updated note',
            tags=['momentum', 'breakout'],
        )
        
        assert updated.notes == 'Updated note'
        assert 'momentum' in updated.tags
    
    def test_update_excursions(self, trade_journal):
        """Test updating MFE/MAE."""
        trade = trade_journal.create_trade(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
            quantity=100,
        )
        
        # Price goes up (favorable)
        trade_journal.update_excursions(trade.trade_id, current_price=155.0)
        updated = trade_journal.get_trade(trade.trade_id)
        assert updated.max_favorable_excursion == 500.0  # (155-150)*100
        
        # Price goes down (adverse)
        trade_journal.update_excursions(trade.trade_id, current_price=145.0)
        updated = trade_journal.get_trade(trade.trade_id)
        assert updated.max_adverse_excursion == 500.0  # (150-145)*100
    
    def test_get_open_trades(self, journal_with_trades):
        """Test getting open trades."""
        # Create one more open trade
        journal_with_trades.create_trade(
            symbol='TEST',
            trade_type=TradeType.LONG,
            entry_price=100.0,
            quantity=10,
        )
        
        open_trades = journal_with_trades.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0].symbol == 'TEST'
    
    def test_get_closed_trades(self, journal_with_trades):
        """Test getting closed trades."""
        closed_trades = journal_with_trades.get_closed_trades()
        assert len(closed_trades) == 8  # All sample trades are closed
    
    def test_get_trades_by_symbol(self, journal_with_trades):
        """Test filtering trades by symbol."""
        aapl_trades = journal_with_trades.get_trades_by_symbol('AAPL')
        assert len(aapl_trades) == 2
    
    def test_get_trades_by_strategy(self, journal_with_trades):
        """Test filtering trades by strategy."""
        tech_trades = journal_with_trades.get_trades_by_strategy('Technical')
        assert len(tech_trades) == 4
    
    def test_search_trades(self, journal_with_trades):
        """Test searching trades with filters."""
        results = journal_with_trades.search_trades(
            strategy='Technical',
            min_confidence=0.7,
        )
        
        assert len(results) > 0
        assert all(t.strategy_name == 'Technical' for t in results)
        assert all(t.signal_confidence >= 0.7 for t in results)
    
    def test_get_statistics(self, journal_with_trades):
        """Test getting journal statistics."""
        stats = journal_with_trades.get_statistics()
        
        assert isinstance(stats, JournalStats)
        assert stats.total_trades == 8
        assert stats.closed_trades == 8
        assert stats.win_rate > 0
    
    def test_to_dataframe(self, journal_with_trades):
        """Test converting to DataFrame."""
        df = journal_with_trades.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8
        assert 'symbol' in df.columns
        assert 'realized_pnl' in df.columns
    
    def test_persistence(self, temp_journal_path):
        """Test that trades persist across instances."""
        # Create journal and add trade
        journal1 = TradeJournal(storage_path=temp_journal_path)
        journal1.create_trade(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
            quantity=100,
        )
        
        # Create new instance with same path
        journal2 = TradeJournal(storage_path=temp_journal_path)
        
        assert len(journal2.trades) == 1
        assert list(journal2.trades.values())[0].symbol == 'AAPL'


# ============================================================================
# PerformanceTracker Tests
# ============================================================================

class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""
    
    def test_initialization(self, performance_tracker):
        """Test tracker initialization."""
        assert performance_tracker is not None
        assert performance_tracker.initial_capital == 100000.0
    
    def test_get_equity_curve(self, performance_tracker):
        """Test equity curve generation."""
        equity_curve = performance_tracker.get_equity_curve()
        
        assert isinstance(equity_curve, pd.DataFrame)
        assert 'equity' in equity_curve.columns
    
    def test_calculate_returns(self, performance_tracker):
        """Test returns calculation."""
        returns = performance_tracker.calculate_returns()
        
        assert isinstance(returns, pd.Series)
    
    def test_analyze_drawdowns(self, performance_tracker):
        """Test drawdown analysis."""
        analysis = performance_tracker.analyze_drawdowns()
        
        assert isinstance(analysis, DrawdownAnalysis)
    
    def test_calculate_sharpe_ratio(self, performance_tracker):
        """Test Sharpe ratio calculation."""
        sharpe = performance_tracker.calculate_sharpe_ratio()
        
        assert isinstance(sharpe, float)
    
    def test_calculate_sortino_ratio(self, performance_tracker):
        """Test Sortino ratio calculation."""
        sortino = performance_tracker.calculate_sortino_ratio()
        
        assert isinstance(sortino, float)
    
    def test_calculate_calmar_ratio(self, performance_tracker):
        """Test Calmar ratio calculation."""
        calmar = performance_tracker.calculate_calmar_ratio()
        
        assert isinstance(calmar, float)
    
    def test_analyze_streaks(self, performance_tracker):
        """Test streak analysis."""
        win_streak, loss_streak, max_win, max_loss = performance_tracker.analyze_streaks()
        
        assert win_streak >= 0
        assert loss_streak >= 0
        assert max_win >= 0
        assert max_loss >= 0
    
    def test_get_timeframed_metrics(self, performance_tracker):
        """Test timeframed metrics."""
        metrics = performance_tracker.get_timeframed_metrics('monthly')
        
        assert isinstance(metrics, TimeframedMetrics)
        assert metrics.timeframe == 'monthly'
    
    def test_get_comprehensive_metrics(self, performance_tracker):
        """Test comprehensive metrics."""
        metrics = performance_tracker.get_comprehensive_metrics()
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 8
    
    def test_get_performance_report(self, performance_tracker):
        """Test performance report."""
        report = performance_tracker.get_performance_report()
        
        assert isinstance(report, dict)
        assert 'overview' in report
        assert 'risk_adjusted' in report
        assert 'drawdown' in report


# ============================================================================
# PatternAnalyzer Tests
# ============================================================================

class TestPatternAnalyzer:
    """Tests for PatternAnalyzer class."""
    
    def test_initialization(self, pattern_analyzer):
        """Test analyzer initialization."""
        assert pattern_analyzer is not None
    
    def test_analyze_by_strategy(self, pattern_analyzer):
        """Test strategy analysis."""
        patterns = pattern_analyzer.analyze_by_strategy()
        
        assert isinstance(patterns, list)
        # May be empty if not enough trades per strategy
        for p in patterns:
            assert p.pattern_type == PatternType.STRATEGY
    
    def test_analyze_by_symbol(self, pattern_analyzer):
        """Test symbol analysis."""
        patterns = pattern_analyzer.analyze_by_symbol()
        
        assert isinstance(patterns, list)
        assert all(p.pattern_type == PatternType.SYMBOL for p in patterns)
    
    def test_analyze_by_market_regime(self, pattern_analyzer):
        """Test market regime analysis."""
        patterns = pattern_analyzer.analyze_by_market_regime()
        
        assert isinstance(patterns, list)
        assert all(p.pattern_type == PatternType.MARKET_REGIME for p in patterns)
    
    def test_analyze_confidence_correlation(self, pattern_analyzer):
        """Test confidence correlation analysis."""
        pattern = pattern_analyzer.analyze_confidence_correlation()
        
        assert isinstance(pattern, TradingPattern)
        assert pattern.pattern_type == PatternType.CONFIDENCE_LEVEL
    
    def test_analyze_trade_duration(self, pattern_analyzer):
        """Test duration analysis."""
        pattern = pattern_analyzer.analyze_trade_duration()
        
        assert isinstance(pattern, TradingPattern)
        assert pattern.pattern_type == PatternType.TRADE_DURATION
    
    def test_run_full_analysis(self, pattern_analyzer):
        """Test full analysis."""
        patterns = pattern_analyzer.run_full_analysis()
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
    
    def test_get_strongest_patterns(self, pattern_analyzer):
        """Test getting strongest patterns."""
        patterns = pattern_analyzer.get_strongest_patterns(top_n=3)
        
        assert len(patterns) <= 3
    
    def test_get_all_recommendations(self, pattern_analyzer):
        """Test getting recommendations."""
        recommendations = pattern_analyzer.get_all_recommendations()
        
        assert isinstance(recommendations, list)
    
    def test_get_pattern_summary(self, pattern_analyzer):
        """Test pattern summary."""
        summary = pattern_analyzer.get_pattern_summary()
        
        assert isinstance(summary, dict)
        assert 'total_patterns' in summary
        assert 'patterns_by_type' in summary


# ============================================================================
# Integration Tests
# ============================================================================

class TestJournalIntegration:
    """Integration tests for the journal module."""
    
    def test_full_trade_lifecycle(self, temp_journal_path):
        """Test complete trade lifecycle."""
        journal = TradeJournal(storage_path=temp_journal_path)
        
        # Create trade
        trade = journal.create_trade(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
            quantity=100,
            strategy_name='Technical',
            signal_confidence=0.75,
            stop_loss=145.0,
            take_profit=165.0,
        )
        
        assert trade.status == TradeStatus.OPEN
        
        # Update excursions as price moves
        journal.update_excursions(trade.trade_id, 155.0)  # Favorable
        journal.update_excursions(trade.trade_id, 148.0)  # Adverse
        journal.update_excursions(trade.trade_id, 160.0)  # New favorable high
        
        # Close trade
        closed = journal.close_trade(trade.trade_id, exit_price=160.0)
        
        assert closed is not None
        assert closed.status == TradeStatus.CLOSED
        assert closed.realized_pnl == 1000.0
        assert closed.max_favorable_excursion == 1000.0  # (160-150)*100
        assert closed.max_adverse_excursion == 200.0  # (150-148)*100
    
    def test_journal_to_performance_analysis(self, journal_with_trades):
        """Test using journal data for performance analysis."""
        tracker = PerformanceTracker(
            journal=journal_with_trades,
            initial_capital=100000.0
        )
        
        metrics = tracker.get_comprehensive_metrics()
        
        assert metrics.total_trades == 8
        assert metrics.total_pnl != 0
    
    def test_journal_to_pattern_analysis(self, journal_with_trades):
        """Test using journal data for pattern analysis."""
        analyzer = PatternAnalyzer(journal=journal_with_trades)
        
        patterns = analyzer.run_full_analysis()
        summary = analyzer.get_pattern_summary()
        
        assert len(patterns) > 0
        assert summary['total_patterns'] > 0


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestJournalEdgeCases:
    """Edge case tests for journal module."""
    
    def test_empty_journal_statistics(self, trade_journal):
        """Test statistics for empty journal."""
        stats = trade_journal.get_statistics()
        
        assert stats.total_trades == 0
        assert stats.win_rate == 0.0
    
    def test_close_nonexistent_trade(self, trade_journal):
        """Test closing a non-existent trade."""
        result = trade_journal.close_trade('nonexistent', exit_price=100.0)
        assert result is None
    
    def test_update_nonexistent_trade(self, trade_journal):
        """Test updating a non-existent trade."""
        result = trade_journal.update_trade('nonexistent', notes='test')
        assert result is None
    
    def test_empty_journal_patterns(self, temp_journal_path):
        """Test pattern analysis on empty journal."""
        journal = TradeJournal(storage_path=temp_journal_path)
        analyzer = PatternAnalyzer(journal=journal)
        
        patterns = analyzer.run_full_analysis()
        # Should return patterns but with no data
        assert isinstance(patterns, list)
    
    def test_single_trade_analysis(self, temp_journal_path):
        """Test analysis with single trade."""
        journal = TradeJournal(storage_path=temp_journal_path)
        
        trade = journal.create_trade(
            symbol='AAPL',
            trade_type=TradeType.LONG,
            entry_price=150.0,
            quantity=100,
        )
        journal.close_trade(trade.trade_id, exit_price=160.0)
        
        stats = journal.get_statistics()
        
        assert stats.total_trades == 1
        assert stats.win_rate == 100.0
