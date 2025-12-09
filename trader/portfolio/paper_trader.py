"""
Paper Trading System for AI Trader Pro.

Simulates real trading without actual money.
Tracks positions, P&L, and performance metrics.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    """Represents a single trade."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: str
    commission: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Total cost including commission."""
        return (self.quantity * self.price) + self.commission


@dataclass
class Position:
    """Represents a stock position."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return self.quantity * self.avg_entry_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


class PaperAccount:
    """
    Paper trading account.
    
    Features:
    - Virtual cash balance
    - Position tracking
    - Trade execution
    - P&L calculation
    - Performance metrics
    - Persistence to JSON
    """
    
    def __init__(
        self,
        initial_balance: float = 100000.0,
        commission_per_trade: float = 0.0,
        account_file: Optional[str] = None
    ):
        """
        Initialize paper account.
        
        Args:
            initial_balance: Starting cash balance
            commission_per_trade: Commission per trade (default free)
            account_file: JSON file to persist account state
        """
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.commission_per_trade = commission_per_trade
        self.account_file = account_file or "paper_account.json"
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.closed_trades: List[Dict] = []  # Completed round-trip trades
        
        # Try to load existing account
        if os.path.exists(self.account_file):
            self.load()
    
    def execute_trade(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET
    ) -> Tuple[bool, str]:
        """
        Execute a paper trade.
        
        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            price: Execution price
            order_type: Market or limit order
            
        Returns:
            (success, message)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if price <= 0:
            return False, "Price must be positive"
        
        total_cost = quantity * price + self.commission_per_trade
        
        if side == OrderSide.BUY:
            # Check if we have enough cash
            if total_cost > self.cash:
                return False, f"Insufficient funds. Need ${total_cost:.2f}, have ${self.cash:.2f}"
            
            # Execute buy
            self.cash -= total_cost
            
            # Update or create position
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_quantity = pos.quantity + quantity
                new_avg_price = ((pos.quantity * pos.avg_entry_price) + (quantity * price)) / new_quantity
                pos.quantity = new_quantity
                pos.avg_entry_price = new_avg_price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=price,
                    current_price=price
                )
            
            # Record trade
            trade = Trade(
                id=f"T{len(self.trades) + 1:06d}",
                symbol=symbol,
                side="buy",
                quantity=quantity,
                price=price,
                timestamp=datetime.now().isoformat(),
                commission=self.commission_per_trade
            )
            self.trades.append(trade)
            
            self.save()
            return True, f"Bought {quantity} shares of {symbol} at ${price:.2f}"
        
        elif side == OrderSide.SELL:
            # Check if we have the position
            if symbol not in self.positions:
                return False, f"No position in {symbol}"
            
            pos = self.positions[symbol]
            if quantity > pos.quantity:
                return False, f"Insufficient shares. Have {pos.quantity}, trying to sell {quantity}"
            
            # Execute sell
            proceeds = (quantity * price) - self.commission_per_trade
            self.cash += proceeds
            
            # Calculate realized P&L for this sale
            cost_basis = quantity * pos.avg_entry_price
            realized_pnl = proceeds - cost_basis
            
            # Record closed trade
            self.closed_trades.append({
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": pos.avg_entry_price,
                "exit_price": price,
                "pnl": realized_pnl,
                "pnl_pct": (realized_pnl / cost_basis) * 100 if cost_basis > 0 else 0,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update position
            pos.quantity -= quantity
            if pos.quantity == 0:
                del self.positions[symbol]
            
            # Record trade
            trade = Trade(
                id=f"T{len(self.trades) + 1:06d}",
                symbol=symbol,
                side="sell",
                quantity=quantity,
                price=price,
                timestamp=datetime.now().isoformat(),
                commission=self.commission_per_trade
            )
            self.trades.append(trade)
            
            self.save()
            return True, f"Sold {quantity} shares of {symbol} at ${price:.2f}. P&L: ${realized_pnl:+.2f}"
        
        return False, "Invalid order side"
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for positions.
        
        Args:
            prices: Dict of symbol -> current price
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
    
    def get_total_value(self) -> float:
        """Get total account value (cash + positions)."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_total_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def get_total_return(self) -> float:
        """Get total return percentage."""
        total_value = self.get_total_value()
        return ((total_value - self.initial_balance) / self.initial_balance) * 100
    
    def get_metrics(self) -> Dict:
        """
        Calculate performance metrics.
        
        Returns:
            Dict with performance metrics
        """
        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t["pnl"] > 0]
        losing_trades = [t for t in self.closed_trades if t["pnl"] < 0]
        
        total_realized_pnl = sum(t["pnl"] for t in self.closed_trades)
        
        return {
            "total_value": self.get_total_value(),
            "cash": self.cash,
            "positions_value": sum(pos.market_value for pos in self.positions.values()),
            "total_return_pct": self.get_total_return(),
            "total_pnl": self.get_total_pnl(),
            "realized_pnl": total_realized_pnl,
            "unrealized_pnl": self.get_total_pnl(),
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0,
            "avg_win": sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            "avg_loss": sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
        }
    
    def save(self):
        """Save account state to JSON."""
        data = {
            "initial_balance": self.initial_balance,
            "cash": self.cash,
            "commission_per_trade": self.commission_per_trade,
            "positions": {
                symbol: {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price
                }
                for symbol, pos in self.positions.items()
            },
            "trades": [asdict(t) for t in self.trades],
            "closed_trades": self.closed_trades,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.account_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load account state from JSON."""
        try:
            with open(self.account_file, 'r') as f:
                data = json.load(f)
            
            self.initial_balance = data.get("initial_balance", 100000.0)
            self.cash = data.get("cash", self.initial_balance)
            self.commission_per_trade = data.get("commission_per_trade", 0.0)
            
            # Load positions
            self.positions = {}
            for symbol, pos_data in data.get("positions", {}).items():
                self.positions[symbol] = Position(**pos_data)
            
            # Load trades
            self.trades = [Trade(**t) for t in data.get("trades", [])]
            self.closed_trades = data.get("closed_trades", [])
            
        except Exception as e:
            print(f"Error loading account: {e}")
    
    def reset(self):
        """Reset account to initial state."""
        self.cash = self.initial_balance
        self.positions = {}
        self.trades = []
        self.closed_trades = []
        self.save()
