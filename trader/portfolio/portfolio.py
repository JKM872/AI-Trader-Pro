"""
Portfolio Manager for tracking positions, P/L, and risk metrics.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from trader.data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    quantity: float
    avg_cost: float
    side: str = 'long'  # 'long' or 'short'
    entry_date: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: str = ""
    
    @property
    def market_value(self) -> float:
        """Calculate market value (requires current price)."""
        return self.quantity * self.avg_cost
    
    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis."""
        return self.quantity * self.avg_cost
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P/L."""
        if self.side == 'long':
            return (current_price - self.avg_cost) * self.quantity
        else:
            return (self.avg_cost - current_price) * self.quantity
    
    def calculate_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P/L percentage."""
        if self.avg_cost == 0:
            return 0.0
        if self.side == 'long':
            return ((current_price - self.avg_cost) / self.avg_cost) * 100
        else:
            return ((self.avg_cost - current_price) / self.avg_cost) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'side': self.side,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'notes': self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create from dictionary."""
        entry_date = None
        if data.get('entry_date'):
            entry_date = datetime.fromisoformat(data['entry_date'])
        
        return cls(
            symbol=data['symbol'],
            quantity=data['quantity'],
            avg_cost=data['avg_cost'],
            side=data.get('side', 'long'),
            entry_date=entry_date,
            stop_loss=data.get('stop_loss'),
            take_profit=data.get('take_profit'),
            notes=data.get('notes', ''),
        )


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float = 0.0
    cash_balance: float = 0.0
    positions_value: float = 0.0
    total_cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    num_positions: int = 0
    num_winners: int = 0
    num_losers: int = 0
    largest_position_pct: float = 0.0
    cash_allocation_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    order_id: Optional[str] = None
    notes: str = ""
    
    @property
    def total_value(self) -> float:
        """Total trade value including commission."""
        return (self.quantity * self.price) + self.commission
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission,
            'order_id': self.order_id,
            'notes': self.notes,
        }


class Portfolio:
    """
    Portfolio manager for tracking positions and performance.
    
    Features:
    - Position tracking with P/L calculation
    - Trade history recording
    - Risk metrics calculation
    - Portfolio analytics
    - Persistence to JSON file
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        name: str = "Main Portfolio",
        data_file: Optional[str] = None,
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            name: Portfolio name
            data_file: Path to JSON file for persistence
        """
        self.name = name
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.realized_pnl = 0.0
        self.data_file = data_file
        self._data_fetcher = DataFetcher()
        
        # Load from file if exists
        if data_file and Path(data_file).exists():
            self.load()
    
    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str = 'long',
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Position:
        """
        Add or update a position.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Entry price
            side: 'long' or 'short'
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Updated position
        """
        cost = quantity * price
        
        if symbol in self.positions:
            # Update existing position (average cost)
            pos = self.positions[symbol]
            total_qty = pos.quantity + quantity
            total_cost = (pos.quantity * pos.avg_cost) + (quantity * price)
            pos.avg_cost = total_cost / total_qty
            pos.quantity = total_qty
            if stop_loss:
                pos.stop_loss = stop_loss
            if take_profit:
                pos.take_profit = take_profit
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                side=side,
                entry_date=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
        
        # Record trade
        self.trade_history.append(Trade(
            symbol=symbol,
            side='buy' if side == 'long' else 'sell',
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
        ))
        
        # Update cash
        self.cash_balance -= cost
        
        logger.info(f"Added position: {symbol} x {quantity} @ ${price:.2f}")
        self._save()
        
        return self.positions[symbol]
    
    def close_position(
        self,
        symbol: str,
        price: float,
        quantity: Optional[float] = None,
    ) -> float:
        """
        Close a position (fully or partially).
        
        Args:
            symbol: Stock symbol
            price: Exit price
            quantity: Quantity to close (None for full close)
            
        Returns:
            Realized P/L from the trade
        """
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")
        
        pos = self.positions[symbol]
        close_qty = quantity or pos.quantity
        
        if close_qty > pos.quantity:
            raise ValueError(f"Cannot close more than position size ({pos.quantity})")
        
        # Calculate P/L
        pnl = pos.calculate_pnl(price) * (close_qty / pos.quantity)
        self.realized_pnl += pnl
        
        # Record trade
        self.trade_history.append(Trade(
            symbol=symbol,
            side='sell' if pos.side == 'long' else 'buy',
            quantity=close_qty,
            price=price,
            timestamp=datetime.now(),
        ))
        
        # Update cash
        self.cash_balance += close_qty * price
        
        # Update or remove position
        if close_qty >= pos.quantity:
            del self.positions[symbol]
            logger.info(f"Closed position: {symbol} @ ${price:.2f} (P/L: ${pnl:.2f})")
        else:
            pos.quantity -= close_qty
            logger.info(f"Partial close: {symbol} x {close_qty} @ ${price:.2f} (P/L: ${pnl:.2f})")
        
        self._save()
        return pnl
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all positions."""
        prices = {}
        for symbol in self.positions:
            try:
                df = self._data_fetcher.get_stock_data(symbol, period='1d')
                if not df.empty:
                    prices[symbol] = df['Close'].iloc[-1]
            except Exception as e:
                logger.warning(f"Could not fetch price for {symbol}: {e}")
        return prices
    
    def get_metrics(self) -> PortfolioMetrics:
        """Calculate portfolio metrics."""
        prices = self.get_current_prices()
        
        metrics = PortfolioMetrics()
        metrics.cash_balance = self.cash_balance
        metrics.num_positions = len(self.positions)
        metrics.realized_pnl = self.realized_pnl
        
        total_cost_basis = 0.0
        unrealized_pnl = 0.0
        positions_value = 0.0
        
        for symbol, pos in self.positions.items():
            current_price = prices.get(symbol, pos.avg_cost)
            pos_value = pos.quantity * current_price
            pos_pnl = pos.calculate_pnl(current_price)
            
            positions_value += pos_value
            total_cost_basis += pos.cost_basis
            unrealized_pnl += pos_pnl
            
            if pos_pnl > 0:
                metrics.num_winners += 1
            else:
                metrics.num_losers += 1
        
        metrics.positions_value = positions_value
        metrics.total_cost_basis = total_cost_basis
        metrics.unrealized_pnl = unrealized_pnl
        
        if total_cost_basis > 0:
            metrics.unrealized_pnl_pct = (unrealized_pnl / total_cost_basis) * 100
        
        metrics.total_value = self.cash_balance + positions_value
        metrics.total_return = metrics.total_value - self.initial_capital
        
        if self.initial_capital > 0:
            metrics.total_return_pct = (metrics.total_return / self.initial_capital) * 100
        
        if metrics.total_value > 0:
            metrics.cash_allocation_pct = (self.cash_balance / metrics.total_value) * 100
            
            # Find largest position
            max_pos_value = 0
            for symbol, pos in self.positions.items():
                current_price = prices.get(symbol, pos.avg_cost)
                pos_value = pos.quantity * current_price
                max_pos_value = max(max_pos_value, pos_value)
            
            metrics.largest_position_pct = (max_pos_value / metrics.total_value) * 100
        
        return metrics
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions as DataFrame."""
        prices = self.get_current_prices()
        
        data = []
        for symbol, pos in self.positions.items():
            current_price = prices.get(symbol, pos.avg_cost)
            pnl = pos.calculate_pnl(current_price)
            pnl_pct = pos.calculate_pnl_pct(current_price)
            
            data.append({
                'Symbol': symbol,
                'Quantity': pos.quantity,
                'Avg Cost': f"${pos.avg_cost:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Market Value': f"${pos.quantity * current_price:,.2f}",
                'P/L': f"${pnl:+,.2f}",
                'P/L %': f"{pnl_pct:+.2f}%",
                'Side': pos.side.upper(),
            })
        
        return pd.DataFrame(data)
    
    def get_trade_history(self, limit: int = 50) -> pd.DataFrame:
        """Get recent trade history as DataFrame."""
        trades = self.trade_history[-limit:]
        
        data = []
        for trade in trades:
            data.append({
                'Date': trade.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Symbol': trade.symbol,
                'Side': trade.side.upper(),
                'Quantity': trade.quantity,
                'Price': f"${trade.price:.2f}",
                'Total': f"${trade.total_value:,.2f}",
            })
        
        return pd.DataFrame(data)
    
    def check_stop_losses(self) -> List[str]:
        """Check if any positions hit stop loss."""
        prices = self.get_current_prices()
        triggered = []
        
        for symbol, pos in self.positions.items():
            if pos.stop_loss is None:
                continue
            
            current_price = prices.get(symbol)
            if current_price is None:
                continue
            
            if pos.side == 'long' and current_price <= pos.stop_loss:
                triggered.append(symbol)
                logger.warning(f"Stop loss triggered: {symbol} @ ${current_price:.2f}")
            elif pos.side == 'short' and current_price >= pos.stop_loss:
                triggered.append(symbol)
                logger.warning(f"Stop loss triggered: {symbol} @ ${current_price:.2f}")
        
        return triggered
    
    def check_take_profits(self) -> List[str]:
        """Check if any positions hit take profit."""
        prices = self.get_current_prices()
        triggered = []
        
        for symbol, pos in self.positions.items():
            if pos.take_profit is None:
                continue
            
            current_price = prices.get(symbol)
            if current_price is None:
                continue
            
            if pos.side == 'long' and current_price >= pos.take_profit:
                triggered.append(symbol)
                logger.info(f"Take profit triggered: {symbol} @ ${current_price:.2f}")
            elif pos.side == 'short' and current_price <= pos.take_profit:
                triggered.append(symbol)
                logger.info(f"Take profit triggered: {symbol} @ ${current_price:.2f}")
        
        return triggered
    
    def get_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation by symbol."""
        prices = self.get_current_prices()
        total_value = self.cash_balance
        
        allocations = {'Cash': self.cash_balance}
        
        for symbol, pos in self.positions.items():
            current_price = prices.get(symbol, pos.avg_cost)
            pos_value = pos.quantity * current_price
            total_value += pos_value
            allocations[symbol] = pos_value
        
        # Convert to percentages
        if total_value > 0:
            allocations = {k: (v / total_value) * 100 for k, v in allocations.items()}
        
        return allocations
    
    def _save(self):
        """Save portfolio to file."""
        if not self.data_file:
            return
        
        data = {
            'name': self.name,
            'initial_capital': self.initial_capital,
            'cash_balance': self.cash_balance,
            'realized_pnl': self.realized_pnl,
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'trade_history': [t.to_dict() for t in self.trade_history[-500:]],  # Keep last 500
            'updated_at': datetime.now().isoformat(),
        }
        
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Portfolio saved to {self.data_file}")
        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")
    
    def load(self):
        """Load portfolio from file."""
        if not self.data_file or not Path(self.data_file).exists():
            return
        
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            self.name = data.get('name', self.name)
            self.initial_capital = data.get('initial_capital', self.initial_capital)
            self.cash_balance = data.get('cash_balance', self.cash_balance)
            self.realized_pnl = data.get('realized_pnl', 0.0)
            
            self.positions = {}
            for symbol, pos_data in data.get('positions', {}).items():
                self.positions[symbol] = Position.from_dict(pos_data)
            
            self.trade_history = []
            for trade_data in data.get('trade_history', []):
                self.trade_history.append(Trade(
                    symbol=trade_data['symbol'],
                    side=trade_data['side'],
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    timestamp=datetime.fromisoformat(trade_data['timestamp']),
                    commission=trade_data.get('commission', 0.0),
                    order_id=trade_data.get('order_id'),
                ))
            
            logger.info(f"Portfolio loaded from {self.data_file}")
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
    
    def summary(self) -> str:
        """Generate text summary of portfolio."""
        metrics = self.get_metrics()
        
        lines = [
            f"{'='*50}",
            f"PORTFOLIO: {self.name}",
            f"{'='*50}",
            f"Total Value: ${metrics.total_value:,.2f}",
            f"Cash: ${metrics.cash_balance:,.2f} ({metrics.cash_allocation_pct:.1f}%)",
            f"Positions Value: ${metrics.positions_value:,.2f}",
            f"{'-'*50}",
            f"Total Return: ${metrics.total_return:+,.2f} ({metrics.total_return_pct:+.2f}%)",
            f"Unrealized P/L: ${metrics.unrealized_pnl:+,.2f} ({metrics.unrealized_pnl_pct:+.2f}%)",
            f"Realized P/L: ${metrics.realized_pnl:+,.2f}",
            f"{'-'*50}",
            f"Positions: {metrics.num_positions} ({metrics.num_winners} ↑ / {metrics.num_losers} ↓)",
            f"Largest Position: {metrics.largest_position_pct:.1f}% of portfolio",
            f"{'='*50}",
        ]
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create portfolio
    portfolio = Portfolio(
        initial_capital=100000,
        name="Test Portfolio",
        data_file="portfolio.json"
    )
    
    # Add some positions
    portfolio.add_position('AAPL', 50, 175.00, stop_loss=165.00, take_profit=195.00)
    portfolio.add_position('MSFT', 30, 380.00, stop_loss=360.00, take_profit=420.00)
    portfolio.add_position('GOOGL', 20, 140.00, stop_loss=130.00, take_profit=160.00)
    
    # Print summary
    print(portfolio.summary())
    
    # Print positions
    print("\nPositions:")
    print(portfolio.get_position_summary().to_string(index=False))
    
    # Print allocation
    print("\nAllocation:")
    for symbol, pct in portfolio.get_allocation().items():
        print(f"  {symbol}: {pct:.1f}%")
