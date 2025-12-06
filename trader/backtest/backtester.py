"""
Backtesting engine for strategy evaluation.
Simulates trades on historical data and calculates performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Type
from enum import Enum
import pandas as pd
import numpy as np

from trader.strategies.base import TradingStrategy, Signal, SignalType


class TradeStatus(Enum):
    """Trade status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"
    TAKE_PROFIT = "take_profit"


@dataclass
class Trade:
    """Represents a single trade in backtesting."""
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: float
    side: str  # 'long' or 'short'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    pnl_percent: float = 0.0
    
    def close(self, exit_date: datetime, exit_price: float, status: TradeStatus = TradeStatus.CLOSED):
        """Close the trade and calculate PnL."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.status = status
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_percent = (exit_price - self.entry_price) / self.entry_price * 100
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_percent = (self.entry_price - exit_price) / self.entry_price * 100


@dataclass
class BacktestMetrics:
    """Performance metrics from backtesting."""
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0  # in days
    best_trade: float = 0.0
    worst_trade: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    risk_reward_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'Total Return': f"${self.total_return:,.2f}",
            'Total Return %': f"{self.total_return_pct:.2f}%",
            'Annualized Return': f"{self.annualized_return:.2f}%",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{self.sortino_ratio:.2f}",
            'Max Drawdown': f"${self.max_drawdown:,.2f}",
            'Max Drawdown %': f"{self.max_drawdown_pct:.2f}%",
            'Win Rate': f"{self.win_rate:.2f}%",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Total Trades': self.total_trades,
            'Winning Trades': self.winning_trades,
            'Losing Trades': self.losing_trades,
            'Average Win': f"${self.avg_win:,.2f}",
            'Average Loss': f"${self.avg_loss:,.2f}",
            'Avg Trade Duration': f"{self.avg_trade_duration:.1f} days",
            'Best Trade': f"${self.best_trade:,.2f}",
            'Worst Trade': f"${self.worst_trade:,.2f}",
            'Max Consecutive Wins': self.consecutive_wins,
            'Max Consecutive Losses': self.consecutive_losses,
            'Risk/Reward Ratio': f"{self.risk_reward_ratio:.2f}",
            'Calmar Ratio': f"{self.calmar_ratio:.2f}",
        }


@dataclass
class BacktestResult:
    """Complete backtest results including trades and metrics."""
    metrics: BacktestMetrics
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    daily_returns: pd.Series
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    def summary(self) -> str:
        """Generate a text summary of backtest results."""
        lines = [
            f"=" * 60,
            f"BACKTEST RESULTS: {self.strategy_name}",
            f"=" * 60,
            f"Symbol: {self.symbol}",
            f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Capital: ${self.final_capital:,.2f}",
            f"-" * 60,
            "PERFORMANCE METRICS:",
            f"-" * 60,
        ]
        
        for key, value in self.metrics.to_dict().items():
            lines.append(f"  {key}: {value}")
        
        lines.append(f"=" * 60)
        return "\n".join(lines)


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    
    Features:
    - Supports long and short positions
    - Stop-loss and take-profit orders
    - Commission and slippage simulation
    - Comprehensive performance metrics
    - Position sizing based on risk
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,   # 0.05% slippage
        risk_per_trade: float = 0.02,  # 2% risk per trade
        max_positions: int = 5,
        allow_shorting: bool = False,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.allow_shorting = allow_shorting
        
        self._reset()
    
    def _reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.equity_history: List[Dict] = []
        
    def run(
        self,
        strategy: TradingStrategy,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            strategy: Trading strategy to test
            data: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            symbol: Stock symbol being tested
            
        Returns:
            BacktestResult with trades, metrics, and equity curve
        """
        self._reset()
        
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Process each bar
        for i in range(50, len(data)):  # Start at 50 for indicator warm-up
            current_bar = data.iloc[:i+1]
            current_date = data.index[i]
            current_price = data.iloc[i]['Close']
            high_price = data.iloc[i]['High']
            low_price = data.iloc[i]['Low']
            
            # Check stop-loss and take-profit for open positions
            self._check_exits(symbol, current_date, high_price, low_price)
            
            # Record equity
            self._record_equity(current_date, current_price)
            
            # Generate signal
            try:
                signal = strategy.generate_signal(symbol, current_bar)
            except Exception:
                continue
            
            if signal is None:
                continue
            
            # Process signal
            self._process_signal(
                signal=signal,
                symbol=symbol,
                current_date=current_date,
                current_price=current_price,
            )
        
        # Close any remaining positions at the end
        final_date = data.index[-1]
        final_price = data.iloc[-1]['Close']
        self._close_all_positions(final_date, final_price)
        self._record_equity(final_date, final_price)
        
        # Calculate metrics
        metrics = self._calculate_metrics(data)
        
        # Build equity curve
        equity_df = pd.DataFrame(self.equity_history)
        equity_df.set_index('date', inplace=True)
        
        equity_curve = equity_df['equity']
        drawdown_curve = self._calculate_drawdown_curve(equity_curve)
        daily_returns = equity_curve.pct_change().dropna()
        
        return BacktestResult(
            metrics=metrics,
            trades=self.trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            daily_returns=daily_returns,
            strategy_name=strategy.__class__.__name__,
            symbol=symbol,
            start_date=data.index[0].to_pydatetime() if hasattr(data.index[0], 'to_pydatetime') else data.index[0],
            end_date=data.index[-1].to_pydatetime() if hasattr(data.index[-1], 'to_pydatetime') else data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=self.capital,
        )
    
    def _process_signal(
        self,
        signal: Signal,
        symbol: str,
        current_date: datetime,
        current_price: float,
    ):
        """Process a trading signal."""
        
        if signal.signal_type == SignalType.BUY:
            # Check if we have an open position
            if symbol in self.open_positions:
                return  # Already in position
            
            # Check max positions
            if len(self.open_positions) >= self.max_positions:
                return
            
            # Check confidence threshold
            if signal.confidence < 0.5:
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(
                current_price=current_price,
                stop_loss=signal.stop_loss,
            )
            
            if position_size <= 0:
                return
            
            # Apply slippage (buy higher)
            entry_price = current_price * (1 + self.slippage)
            
            # Calculate commission
            commission_cost = entry_price * position_size * self.commission
            
            # Check if we have enough capital
            total_cost = entry_price * position_size + commission_cost
            if total_cost > self.capital:
                return
            
            # Open position
            trade = Trade(
                symbol=symbol,
                entry_date=current_date,
                entry_price=entry_price,
                quantity=position_size,
                side='long',
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
            
            self.open_positions[symbol] = trade
            self.capital -= total_cost
            
        elif signal.signal_type == SignalType.SELL:
            # Close long position if exists
            if symbol in self.open_positions:
                trade = self.open_positions[symbol]
                if trade.side == 'long':
                    # Apply slippage (sell lower)
                    exit_price = current_price * (1 - self.slippage)
                    
                    # Calculate commission
                    commission_cost = exit_price * trade.quantity * self.commission
                    
                    # Close trade
                    trade.close(current_date, exit_price)
                    
                    # Update capital
                    self.capital += exit_price * trade.quantity - commission_cost
                    
                    # Move to closed trades
                    self.trades.append(trade)
                    del self.open_positions[symbol]
            
            # Open short if allowed
            elif self.allow_shorting and symbol not in self.open_positions:
                if len(self.open_positions) >= self.max_positions:
                    return
                
                if signal.confidence < 0.5:
                    return
                
                position_size = self._calculate_position_size(
                    current_price=current_price,
                    stop_loss=signal.stop_loss,
                    is_short=True,
                )
                
                if position_size <= 0:
                    return
                
                # Apply slippage (short lower)
                entry_price = current_price * (1 - self.slippage)
                commission_cost = entry_price * position_size * self.commission
                
                # Reserve margin
                margin_required = entry_price * position_size * 1.5  # 150% margin
                if margin_required > self.capital:
                    return
                
                trade = Trade(
                    symbol=symbol,
                    entry_date=current_date,
                    entry_price=entry_price,
                    quantity=position_size,
                    side='short',
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                )
                
                self.open_positions[symbol] = trade
                self.capital -= margin_required + commission_cost
    
    def _check_exits(
        self,
        symbol: str,
        current_date: datetime,
        high_price: float,
        low_price: float,
    ):
        """Check stop-loss and take-profit for open positions."""
        if symbol not in self.open_positions:
            return
        
        trade = self.open_positions[symbol]
        
        if trade.side == 'long':
            # Check stop-loss (price went below)
            if trade.stop_loss and low_price <= trade.stop_loss:
                exit_price = trade.stop_loss * (1 - self.slippage)
                commission_cost = exit_price * trade.quantity * self.commission
                trade.close(current_date, exit_price, TradeStatus.STOPPED_OUT)
                self.capital += exit_price * trade.quantity - commission_cost
                self.trades.append(trade)
                del self.open_positions[symbol]
                return
            
            # Check take-profit (price went above)
            if trade.take_profit and high_price >= trade.take_profit:
                exit_price = trade.take_profit * (1 - self.slippage)
                commission_cost = exit_price * trade.quantity * self.commission
                trade.close(current_date, exit_price, TradeStatus.TAKE_PROFIT)
                self.capital += exit_price * trade.quantity - commission_cost
                self.trades.append(trade)
                del self.open_positions[symbol]
                return
        
        else:  # short
            # Check stop-loss (price went above)
            if trade.stop_loss and high_price >= trade.stop_loss:
                exit_price = trade.stop_loss * (1 + self.slippage)
                commission_cost = exit_price * trade.quantity * self.commission
                margin_return = trade.entry_price * trade.quantity * 1.5
                trade.close(current_date, exit_price, TradeStatus.STOPPED_OUT)
                self.capital += margin_return - exit_price * trade.quantity - commission_cost + trade.entry_price * trade.quantity
                self.trades.append(trade)
                del self.open_positions[symbol]
                return
            
            # Check take-profit (price went below)
            if trade.take_profit and low_price <= trade.take_profit:
                exit_price = trade.take_profit * (1 + self.slippage)
                commission_cost = exit_price * trade.quantity * self.commission
                margin_return = trade.entry_price * trade.quantity * 1.5
                trade.close(current_date, exit_price, TradeStatus.TAKE_PROFIT)
                self.capital += margin_return - exit_price * trade.quantity - commission_cost + trade.entry_price * trade.quantity
                self.trades.append(trade)
                del self.open_positions[symbol]
                return
    
    def _close_all_positions(self, date: datetime, price: float):
        """Close all open positions at current price."""
        for symbol in list(self.open_positions.keys()):
            trade = self.open_positions[symbol]
            exit_price = price * (1 - self.slippage) if trade.side == 'long' else price * (1 + self.slippage)
            commission_cost = exit_price * trade.quantity * self.commission
            
            if trade.side == 'long':
                trade.close(date, exit_price)
                self.capital += exit_price * trade.quantity - commission_cost
            else:
                margin_return = trade.entry_price * trade.quantity * 1.5
                trade.close(date, exit_price)
                self.capital += margin_return - exit_price * trade.quantity - commission_cost + trade.entry_price * trade.quantity
            
            self.trades.append(trade)
            del self.open_positions[symbol]
    
    def _calculate_position_size(
        self,
        current_price: float,
        stop_loss: Optional[float],
        is_short: bool = False,
    ) -> float:
        """Calculate position size based on risk per trade."""
        if stop_loss is None:
            # Default 5% stop-loss
            stop_loss = current_price * (0.95 if not is_short else 1.05)
        
        risk_amount = self.capital * self.risk_per_trade
        
        if is_short:
            price_risk = stop_loss - current_price
        else:
            price_risk = current_price - stop_loss
        
        if price_risk <= 0:
            return 0
        
        shares = risk_amount / price_risk
        
        # Limit to 20% of capital per position
        max_shares = (self.capital * 0.2) / current_price
        shares = min(shares, max_shares)
        
        return max(0, int(shares))
    
    def _record_equity(self, date: datetime, current_price: float):
        """Record current equity value."""
        # Calculate value of open positions
        positions_value = 0
        for symbol, trade in self.open_positions.items():
            if trade.side == 'long':
                positions_value += current_price * trade.quantity
            else:  # short
                positions_value += trade.entry_price * trade.quantity * 2 - current_price * trade.quantity
        
        total_equity = self.capital + positions_value
        
        self.equity_history.append({
            'date': date,
            'equity': total_equity,
            'capital': self.capital,
            'positions_value': positions_value,
        })
    
    def _calculate_drawdown_curve(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown curve from equity curve."""
        rolling_max = equity.expanding().max()
        drawdown = equity - rolling_max
        return drawdown
    
    def _calculate_metrics(self, data: pd.DataFrame) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        metrics = BacktestMetrics()
        
        if not self.trades:
            return metrics
        
        # Basic metrics
        metrics.total_trades = len(self.trades)
        
        # Separate winning and losing trades
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        
        if metrics.total_trades > 0:
            metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
        
        # PnL metrics
        total_pnl = sum(t.pnl for t in self.trades)
        metrics.total_return = total_pnl
        metrics.total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # Average win/loss
        if winning_trades:
            metrics.avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
            metrics.best_trade = max(t.pnl for t in self.trades)
        
        if losing_trades:
            metrics.avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades)
            metrics.worst_trade = min(t.pnl for t in self.trades)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            metrics.profit_factor = float('inf')
        
        # Risk/Reward ratio
        if metrics.avg_loss != 0:
            metrics.risk_reward_ratio = abs(metrics.avg_win / metrics.avg_loss)
        
        # Average trade duration
        durations = []
        for trade in self.trades:
            if trade.exit_date and trade.entry_date:
                duration = (trade.exit_date - trade.entry_date).days
                durations.append(duration)
        
        if durations:
            metrics.avg_trade_duration = sum(durations) / len(durations)
        
        # Consecutive wins/losses
        metrics.consecutive_wins = self._max_consecutive(self.trades, winning=True)
        metrics.consecutive_losses = self._max_consecutive(self.trades, winning=False)
        
        # Calculate from equity curve
        if self.equity_history:
            equity_df = pd.DataFrame(self.equity_history)
            equity_series = equity_df['equity']
            
            # Maximum drawdown
            rolling_max = equity_series.expanding().max()
            drawdown = equity_series - rolling_max
            metrics.max_drawdown = abs(drawdown.min())
            if rolling_max.max() > 0:
                metrics.max_drawdown_pct = (metrics.max_drawdown / rolling_max.max()) * 100
            
            # Daily returns for Sharpe/Sortino
            daily_returns = equity_series.pct_change().dropna()
            
            if len(daily_returns) > 1:
                # Annualized return
                total_days = (data.index[-1] - data.index[0]).days
                if total_days > 0:
                    years = total_days / 365.25
                    final_value = self.capital
                    metrics.annualized_return = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100
                
                # Sharpe Ratio (assuming risk-free rate of 2%)
                risk_free_daily = 0.02 / 252
                excess_returns = daily_returns - risk_free_daily
                if daily_returns.std() > 0:
                    metrics.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
                
                # Sortino Ratio (downside deviation)
                negative_returns = daily_returns[daily_returns < 0]
                if len(negative_returns) > 0 and negative_returns.std() > 0:
                    metrics.sortino_ratio = np.sqrt(252) * excess_returns.mean() / negative_returns.std()
                
                # Calmar Ratio
                if metrics.max_drawdown_pct > 0:
                    metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown_pct
        
        return metrics
    
    def _max_consecutive(self, trades: List[Trade], winning: bool) -> int:
        """Calculate maximum consecutive winning or losing trades."""
        max_count = 0
        current_count = 0
        
        for trade in trades:
            if (trade.pnl > 0) == winning:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def compare_strategies(
        self,
        strategies: List[TradingStrategy],
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data.
        
        Returns DataFrame with performance comparison.
        """
        results = []
        
        for strategy in strategies:
            result = self.run(strategy, data, symbol)
            results.append({
                'Strategy': result.strategy_name,
                'Total Return %': result.metrics.total_return_pct,
                'Sharpe Ratio': result.metrics.sharpe_ratio,
                'Max Drawdown %': result.metrics.max_drawdown_pct,
                'Win Rate %': result.metrics.win_rate,
                'Total Trades': result.metrics.total_trades,
                'Profit Factor': result.metrics.profit_factor,
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from trader.strategies.technical import TechnicalStrategy
    from trader.strategies.momentum import MomentumStrategy
    
    # Download test data
    print("Downloading test data...")
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="2y")
    
    # Initialize backtester
    backtester = Backtester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        risk_per_trade=0.02,
    )
    
    # Test Technical Strategy
    print("\nTesting TechnicalStrategy...")
    tech_strategy = TechnicalStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
    tech_result = backtester.run(tech_strategy, data, "AAPL")
    print(tech_result.summary())
    
    # Test Momentum Strategy
    print("\nTesting MomentumStrategy...")
    momentum_strategy = MomentumStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
    momentum_result = backtester.run(momentum_strategy, data, "AAPL")
    print(momentum_result.summary())
    
    # Compare strategies
    print("\nStrategy Comparison:")
    comparison = backtester.compare_strategies(
        [tech_strategy, momentum_strategy],
        data,
        "AAPL"
    )
    print(comparison.to_string(index=False))
