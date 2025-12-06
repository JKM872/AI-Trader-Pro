"""
Performance Tracker - Advanced performance analysis and metrics.

Provides comprehensive performance analytics across multiple timeframes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np

from .trade_journal import TradeJournal, TradeEntry, TradeStatus


@dataclass
class DrawdownAnalysis:
    """Drawdown statistics."""
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_start: Optional[datetime] = None
    max_drawdown_end: Optional[datetime] = None
    max_drawdown_duration: Optional[timedelta] = None
    recovery_time: Optional[timedelta] = None
    drawdown_periods: int = 0
    average_drawdown: float = 0.0
    average_drawdown_duration: Optional[timedelta] = None


@dataclass
class TimeframedMetrics:
    """Metrics for a specific timeframe."""
    timeframe: str = ""  # daily, weekly, monthly, yearly
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    total_pnl: float = 0.0
    average_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    best_day: float = 0.0
    worst_day: float = 0.0
    
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Basic stats
    total_trades: int = 0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    
    # Win/Loss
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0  # Avg Win / Avg Loss
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    average_drawdown: float = 0.0
    
    # Consistency
    win_streak: int = 0
    loss_streak: int = 0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    
    # Time analysis
    average_hold_time: Optional[timedelta] = None
    average_winning_hold_time: Optional[timedelta] = None
    average_losing_hold_time: Optional[timedelta] = None
    
    # Recovery
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    
    # Timeframed
    daily_metrics: Optional[TimeframedMetrics] = None
    weekly_metrics: Optional[TimeframedMetrics] = None
    monthly_metrics: Optional[TimeframedMetrics] = None


class PerformanceTracker:
    """
    Advanced performance tracking and analysis.
    
    Features:
    - Multi-timeframe analysis
    - Risk-adjusted returns
    - Drawdown analysis
    - Consistency metrics
    - Equity curve analysis
    """
    
    def __init__(
        self,
        journal: Optional[TradeJournal] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.02  # 2% annual
    ):
        """Initialize performance tracker."""
        self.journal = journal
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self._equity_curve: List[Tuple[datetime, float]] = []
    
    def set_journal(self, journal: TradeJournal) -> None:
        """Set or update the journal."""
        self.journal = journal
        self._build_equity_curve()
    
    def _build_equity_curve(self) -> None:
        """Build equity curve from closed trades."""
        if not self.journal:
            return
        
        closed_trades = self.journal.get_closed_trades()
        
        # Sort by exit time
        sorted_trades = sorted(
            [t for t in closed_trades if t.exit_time],
            key=lambda t: t.exit_time
        )
        
        equity = self.initial_capital
        self._equity_curve = [(datetime.now() - timedelta(days=365), equity)]
        
        for trade in sorted_trades:
            if trade.realized_pnl:
                equity += trade.realized_pnl
                self._equity_curve.append((trade.exit_time, equity))
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self._equity_curve:
            self._build_equity_curve()
        
        if not self._equity_curve:
            return pd.DataFrame(columns=['datetime', 'equity'])
        
        df = pd.DataFrame(self._equity_curve, columns=['datetime', 'equity'])
        df = df.set_index('datetime')
        return df
    
    def calculate_returns(self) -> pd.Series:
        """Calculate daily returns from equity curve."""
        equity_df = self.get_equity_curve()
        
        if equity_df.empty:
            return pd.Series(dtype=float)
        
        # Resample to daily and forward fill
        daily = equity_df.resample('D').last().ffill()
        returns = daily['equity'].pct_change().dropna()
        
        return returns
    
    def analyze_drawdowns(self) -> DrawdownAnalysis:
        """Comprehensive drawdown analysis."""
        analysis = DrawdownAnalysis()
        
        equity_df = self.get_equity_curve()
        
        if equity_df.empty or len(equity_df) < 2:
            return analysis
        
        equity = equity_df['equity']
        
        # Calculate running maximum
        running_max = equity.expanding().max()
        
        # Calculate drawdown
        drawdown = equity - running_max
        drawdown_pct = drawdown / running_max * 100
        
        # Current drawdown
        analysis.current_drawdown = drawdown.iloc[-1]
        analysis.current_drawdown_pct = drawdown_pct.iloc[-1]
        
        # Maximum drawdown
        analysis.max_drawdown = drawdown.min()
        analysis.max_drawdown_pct = drawdown_pct.min()
        
        # Find max drawdown period
        max_dd_end_idx = drawdown.idxmin()
        max_dd_end = drawdown.index.get_loc(max_dd_end_idx)
        
        # Find start of max drawdown (last peak before the trough)
        if max_dd_end > 0:
            peak_before = running_max.iloc[:max_dd_end + 1].idxmax()
            analysis.max_drawdown_start = peak_before
            analysis.max_drawdown_end = max_dd_end_idx
            
            if isinstance(peak_before, datetime) and isinstance(max_dd_end_idx, datetime):
                analysis.max_drawdown_duration = max_dd_end_idx - peak_before
        
        # Count drawdown periods
        in_drawdown = drawdown < 0
        analysis.drawdown_periods = (in_drawdown.diff() == True).sum()
        
        # Average drawdown
        negative_drawdowns = drawdown[drawdown < 0]
        if len(negative_drawdowns) > 0:
            analysis.average_drawdown = negative_drawdowns.mean()
        
        return analysis
    
    def calculate_sharpe_ratio(
        self,
        returns: Optional[pd.Series] = None,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sharpe ratio."""
        if returns is None:
            returns = self.calculate_returns()
        
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (self.risk_free_rate / periods_per_year)
        sharpe = excess_returns / returns.std() * np.sqrt(periods_per_year)
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: Optional[pd.Series] = None,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Sortino ratio (uses only downside deviation)."""
        if returns is None:
            returns = self.calculate_returns()
        
        if returns.empty:
            return 0.0
        
        # Calculate downside deviation
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0.0
        
        downside_std = negative_returns.std()
        excess_returns = returns.mean() - (self.risk_free_rate / periods_per_year)
        sortino = excess_returns / downside_std * np.sqrt(periods_per_year)
        
        return sortino
    
    def calculate_calmar_ratio(
        self,
        returns: Optional[pd.Series] = None
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if returns is None:
            returns = self.calculate_returns()
        
        if returns.empty:
            return 0.0
        
        # Annualized return
        annualized_return = returns.mean() * 252
        
        # Max drawdown
        drawdown_analysis = self.analyze_drawdowns()
        max_drawdown_pct = abs(drawdown_analysis.max_drawdown_pct) / 100
        
        if max_drawdown_pct == 0:
            return 0.0
        
        return annualized_return / max_drawdown_pct
    
    def calculate_omega_ratio(
        self,
        returns: Optional[pd.Series] = None,
        threshold: float = 0.0
    ) -> float:
        """Calculate Omega ratio."""
        if returns is None:
            returns = self.calculate_returns()
        
        if returns.empty:
            return 0.0
        
        # Separate gains and losses relative to threshold
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        sum_losses = losses.sum()
        
        if sum_losses == 0:
            return float('inf') if gains.sum() > 0 else 0.0
        
        return gains.sum() / sum_losses
    
    def calculate_ulcer_index(
        self,
        equity_series: Optional[pd.Series] = None
    ) -> float:
        """Calculate Ulcer Index (measures downside volatility)."""
        if equity_series is None:
            equity_df = self.get_equity_curve()
            if equity_df.empty:
                return 0.0
            equity_series = equity_df['equity']
        
        if len(equity_series) < 2:
            return 0.0
        
        # Running maximum
        running_max = equity_series.expanding().max()
        
        # Percentage drawdown
        pct_drawdown = (equity_series - running_max) / running_max * 100
        
        # Ulcer Index
        ulcer = np.sqrt((pct_drawdown ** 2).mean())
        
        return ulcer
    
    def calculate_recovery_factor(self) -> float:
        """Calculate recovery factor (net profit / max drawdown)."""
        if not self.journal:
            return 0.0
        
        closed_trades = self.journal.get_closed_trades()
        net_profit = sum(t.realized_pnl for t in closed_trades if t.realized_pnl)
        
        drawdown_analysis = self.analyze_drawdowns()
        max_drawdown = abs(drawdown_analysis.max_drawdown)
        
        if max_drawdown == 0:
            return 0.0
        
        return net_profit / max_drawdown
    
    def analyze_streaks(self) -> Tuple[int, int, int, int]:
        """Analyze win/loss streaks. Returns (current_win, current_loss, max_win, max_loss)."""
        if not self.journal:
            return 0, 0, 0, 0
        
        closed_trades = self.journal.get_closed_trades()
        sorted_trades = sorted(
            [t for t in closed_trades if t.exit_time],
            key=lambda t: t.exit_time
        )
        
        if not sorted_trades:
            return 0, 0, 0, 0
        
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        win_streak = 0
        loss_streak = 0
        
        for trade in sorted_trades:
            if trade.realized_pnl and trade.realized_pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            elif trade.realized_pnl and trade.realized_pnl < 0:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
            else:
                win_streak = 0
                loss_streak = 0
        
        current_win_streak = win_streak
        current_loss_streak = loss_streak
        
        return current_win_streak, current_loss_streak, max_win_streak, max_loss_streak
    
    def get_timeframed_metrics(
        self,
        timeframe: str = 'monthly'
    ) -> TimeframedMetrics:
        """Get metrics for a specific timeframe."""
        metrics = TimeframedMetrics(timeframe=timeframe)
        
        if not self.journal:
            return metrics
        
        closed_trades = self.journal.get_closed_trades()
        
        if not closed_trades:
            return metrics
        
        # Get date range
        dates = [t.exit_time for t in closed_trades if t.exit_time]
        if not dates:
            return metrics
        
        metrics.start_date = min(dates)
        metrics.end_date = max(dates)
        
        # Group by timeframe
        trades_df = self.journal.to_dataframe()
        trades_df = trades_df[trades_df['status'] == 'closed']
        
        if trades_df.empty or 'exit_time' not in trades_df.columns:
            return metrics
        
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df = trades_df.set_index('exit_time')
        
        # Resample
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'ME',
            'yearly': 'YE'
        }
        freq = freq_map.get(timeframe, 'ME')
        
        if 'realized_pnl' in trades_df.columns:
            grouped = trades_df['realized_pnl'].resample(freq).sum()
            
            metrics.total_pnl = grouped.sum()
            if len(grouped) > 0:
                metrics.average_pnl = grouped.mean()
                metrics.best_day = grouped.max()
                metrics.worst_day = grouped.min()
                metrics.volatility = grouped.std()
        
        metrics.trades_count = len(closed_trades)
        
        winners = [t for t in closed_trades if t.realized_pnl and t.realized_pnl > 0]
        losers = [t for t in closed_trades if t.realized_pnl and t.realized_pnl < 0]
        
        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)
        
        if closed_trades:
            metrics.win_rate = len(winners) / len(closed_trades) * 100
        
        gross_profit = sum(t.realized_pnl for t in winners if t.realized_pnl)
        gross_loss = abs(sum(t.realized_pnl for t in losers if t.realized_pnl))
        
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss
        
        return metrics
    
    def get_comprehensive_metrics(self) -> PerformanceMetrics:
        """Get all performance metrics."""
        metrics = PerformanceMetrics()
        
        if not self.journal:
            return metrics
        
        self._build_equity_curve()
        
        closed_trades = self.journal.get_closed_trades()
        
        if not closed_trades:
            return metrics
        
        # Basic stats
        metrics.total_trades = len(closed_trades)
        metrics.total_pnl = sum(t.realized_pnl for t in closed_trades if t.realized_pnl)
        metrics.total_return_pct = (metrics.total_pnl / self.initial_capital) * 100
        
        # Win/Loss analysis
        winners = [t for t in closed_trades if t.realized_pnl and t.realized_pnl > 0]
        losers = [t for t in closed_trades if t.realized_pnl and t.realized_pnl < 0]
        
        if closed_trades:
            metrics.win_rate = len(winners) / len(closed_trades) * 100
        
        avg_win = sum(t.realized_pnl for t in winners if t.realized_pnl) / len(winners) if winners else 0
        avg_loss = abs(sum(t.realized_pnl for t in losers if t.realized_pnl) / len(losers)) if losers else 0
        
        if avg_loss > 0:
            metrics.payoff_ratio = avg_win / avg_loss
        
        gross_profit = sum(t.realized_pnl for t in winners if t.realized_pnl)
        gross_loss = abs(sum(t.realized_pnl for t in losers if t.realized_pnl))
        
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss
        
        # Expectancy
        if metrics.win_rate > 0:
            win_prob = metrics.win_rate / 100
            loss_prob = 1 - win_prob
            metrics.expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)
        
        # Risk metrics
        returns = self.calculate_returns()
        metrics.sharpe_ratio = self.calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self.calculate_sortino_ratio(returns)
        metrics.calmar_ratio = self.calculate_calmar_ratio(returns)
        metrics.omega_ratio = self.calculate_omega_ratio(returns)
        
        # Drawdown
        dd_analysis = self.analyze_drawdowns()
        metrics.max_drawdown = dd_analysis.max_drawdown
        metrics.max_drawdown_pct = dd_analysis.max_drawdown_pct
        metrics.average_drawdown = dd_analysis.average_drawdown
        
        # Streaks
        win_streak, loss_streak, max_win, max_loss = self.analyze_streaks()
        metrics.win_streak = win_streak
        metrics.loss_streak = loss_streak
        metrics.max_win_streak = max_win
        metrics.max_loss_streak = max_loss
        
        # Time analysis
        durations = [t.duration() for t in closed_trades if t.duration()]
        if durations:
            total_seconds = sum(d.total_seconds() for d in durations)
            metrics.average_hold_time = timedelta(seconds=total_seconds / len(durations))
        
        winner_durations = [t.duration() for t in winners if t.duration()]
        if winner_durations:
            total_seconds = sum(d.total_seconds() for d in winner_durations)
            metrics.average_winning_hold_time = timedelta(seconds=total_seconds / len(winner_durations))
        
        loser_durations = [t.duration() for t in losers if t.duration()]
        if loser_durations:
            total_seconds = sum(d.total_seconds() for d in loser_durations)
            metrics.average_losing_hold_time = timedelta(seconds=total_seconds / len(loser_durations))
        
        # Recovery and Ulcer
        metrics.recovery_factor = self.calculate_recovery_factor()
        metrics.ulcer_index = self.calculate_ulcer_index()
        
        # Timeframed metrics
        metrics.daily_metrics = self.get_timeframed_metrics('daily')
        metrics.weekly_metrics = self.get_timeframed_metrics('weekly')
        metrics.monthly_metrics = self.get_timeframed_metrics('monthly')
        
        return metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        metrics = self.get_comprehensive_metrics()
        dd_analysis = self.analyze_drawdowns()
        
        report = {
            'overview': {
                'total_trades': metrics.total_trades,
                'total_pnl': metrics.total_pnl,
                'total_return_pct': metrics.total_return_pct,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'expectancy': metrics.expectancy,
            },
            'risk_adjusted': {
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'omega_ratio': metrics.omega_ratio,
                'ulcer_index': metrics.ulcer_index,
                'recovery_factor': metrics.recovery_factor,
            },
            'drawdown': {
                'current_drawdown': dd_analysis.current_drawdown,
                'current_drawdown_pct': dd_analysis.current_drawdown_pct,
                'max_drawdown': dd_analysis.max_drawdown,
                'max_drawdown_pct': dd_analysis.max_drawdown_pct,
                'average_drawdown': dd_analysis.average_drawdown,
                'drawdown_periods': dd_analysis.drawdown_periods,
            },
            'consistency': {
                'current_win_streak': metrics.win_streak,
                'current_loss_streak': metrics.loss_streak,
                'max_win_streak': metrics.max_win_streak,
                'max_loss_streak': metrics.max_loss_streak,
                'payoff_ratio': metrics.payoff_ratio,
            },
            'time_analysis': {
                'average_hold_time': str(metrics.average_hold_time) if metrics.average_hold_time else None,
                'average_winning_hold_time': str(metrics.average_winning_hold_time) if metrics.average_winning_hold_time else None,
                'average_losing_hold_time': str(metrics.average_losing_hold_time) if metrics.average_losing_hold_time else None,
            }
        }
        
        return report
