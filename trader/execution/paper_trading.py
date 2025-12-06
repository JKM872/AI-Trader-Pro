"""
Paper Trading Executor - Simulated trading using Alpaca Paper Trading API.

IMPORTANT: This module is for PAPER TRADING only.
Never use for real trading without explicit confirmation and proper risk assessment.

Features:
- Submit paper trades via Alpaca
- Track positions and P&L
- Implement stop-loss and take-profit orders
- Log all trading activity
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    qty: int
    side: Literal['long', 'short']
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.qty * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost of position."""
        return self.qty * self.entry_price
    
    @property
    def unrealized_pl(self) -> float:
        """Unrealized profit/loss."""
        if self.side == 'long':
            return (self.current_price - self.entry_price) * self.qty
        else:  # short
            return (self.entry_price - self.current_price) * self.qty
    
    @property
    def unrealized_pl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pl / self.cost_basis


@dataclass 
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    qty: int
    side: Literal['buy', 'sell']
    type: Literal['market', 'limit', 'stop', 'stop_limit']
    time_in_force: str
    status: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_price: Optional[float] = None
    filled_qty: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None


class PaperTradingExecutor:
    """
    Paper trading executor using Alpaca API.
    
    Usage:
        executor = PaperTradingExecutor(api_key, secret_key)
        order = executor.submit_order('AAPL', 10, 'buy', 'market')
        positions = executor.get_positions()
        executor.close_position('AAPL')
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 base_url: str = 'https://paper-api.alpaca.markets'):
        """
        Initialize Paper Trading Executor.
        
        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
            base_url: Alpaca API base URL (paper trading by default)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = base_url
        
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys not set. Running in simulation mode.")
            self._simulation_mode = True
            self._simulated_positions: Dict[str, Position] = {}
            self._simulated_orders: List[Order] = []
            self._simulated_cash = 100000.0  # $100k paper money
        else:
            self._simulation_mode = False
            self._init_alpaca()
    
    def _init_alpaca(self):
        """Initialize Alpaca API client."""
        try:
            import alpaca_trade_api as tradeapi
            from alpaca_trade_api.rest import URL
            self.api = tradeapi.REST(
                str(self.api_key) if self.api_key else '',
                str(self.secret_key) if self.secret_key else '',
                base_url=URL(self.base_url),
                api_version='v2'
            )
            # Test connection
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca. Account status: {account.status}")
            logger.info(f"Buying power: ${float(account.buying_power):,.2f}")
        except ImportError:
            logger.error("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            self._simulation_mode = True
            self._simulated_positions = {}
            self._simulated_orders = []
            self._simulated_cash = 100000.0
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    def get_account(self) -> Dict:
        """Get account information."""
        if self._simulation_mode:
            total_value = self._simulated_cash + sum(
                p.market_value for p in self._simulated_positions.values()
            )
            return {
                'cash': self._simulated_cash,
                'buying_power': self._simulated_cash,
                'portfolio_value': total_value,
                'equity': total_value,
                'status': 'ACTIVE (SIMULATION)'
            }
        
        account = self.api.get_account()
        return {
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'status': account.status
        }
    
    def submit_order(self, 
                     symbol: str,
                     qty: int,
                     side: Literal['buy', 'sell'],
                     order_type: Literal['market', 'limit', 'stop', 'stop_limit'] = 'market',
                     time_in_force: str = 'day',
                     limit_price: Optional[float] = None,
                     stop_price: Optional[float] = None,
                     take_profit: Optional[float] = None,
                     stop_loss: Optional[float] = None) -> Order:
        """
        Submit a trading order.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', or 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            take_profit: Take profit price (creates bracket order)
            stop_loss: Stop loss price (creates bracket order)
        
        Returns:
            Order object with status
        """
        logger.info(f"Submitting order: {side.upper()} {qty} {symbol} @ {order_type}")
        
        if self._simulation_mode:
            return self._simulate_order(
                symbol, qty, side, order_type, time_in_force,
                limit_price, stop_price
            )
        
        try:
            # Prepare order parameters
            order_params = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force
            }
            
            if limit_price:
                order_params['limit_price'] = str(limit_price)
            if stop_price:
                order_params['stop_price'] = str(stop_price)
            
            # Bracket order (with take profit and stop loss)
            if take_profit and stop_loss and side == 'buy':
                order_params['order_class'] = 'bracket'
                order_params['take_profit'] = {'limit_price': str(take_profit)}
                order_params['stop_loss'] = {'stop_price': str(stop_loss)}
            
            # Submit order
            alpaca_order = self.api.submit_order(**order_params)
            
            if alpaca_order is None:
                raise Exception("Alpaca API returned None for order")
            
            order = Order(
                id=str(alpaca_order.id),
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                status=str(alpaca_order.status),
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            logger.info(f"Order submitted: {order.id} - Status: {order.status}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise
    
    def _simulate_order(self, symbol: str, qty: int, side: str,
                        order_type: str, time_in_force: str,
                        limit_price: Optional[float],
                        stop_price: Optional[float]) -> Order:
        """Simulate order execution for testing."""
        import uuid
        
        # Get current price (simulated as the limit price or a default)
        price = limit_price or 100.0  # Would need real price in production
        
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            qty=qty,
            side=side,  # type: ignore
            type=order_type,  # type: ignore
            time_in_force=time_in_force,
            status='filled',
            limit_price=limit_price,
            stop_price=stop_price,
            filled_price=price,
            filled_qty=qty,
            filled_at=datetime.now()
        )
        
        self._simulated_orders.append(order)
        
        # Update simulated positions
        if side == 'buy':
            cost = qty * price
            if cost > self._simulated_cash:
                order.status = 'rejected'
                logger.warning(f"Insufficient funds for order: need ${cost:.2f}, have ${self._simulated_cash:.2f}")
                return order
            
            self._simulated_cash -= cost
            
            if symbol in self._simulated_positions:
                pos = self._simulated_positions[symbol]
                # Average up
                total_qty = pos.qty + qty
                avg_price = (pos.entry_price * pos.qty + price * qty) / total_qty
                pos.qty = total_qty
                pos.entry_price = avg_price
            else:
                self._simulated_positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    side='long',
                    entry_price=price,
                    current_price=price
                )
        else:  # sell
            if symbol in self._simulated_positions:
                pos = self._simulated_positions[symbol]
                if qty > pos.qty:
                    order.status = 'rejected'
                    logger.warning(f"Cannot sell {qty} shares, only have {pos.qty}")
                    return order
                
                self._simulated_cash += qty * price
                pos.qty -= qty
                
                if pos.qty == 0:
                    del self._simulated_positions[symbol]
            else:
                # Short selling (simplified)
                self._simulated_positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    side='short',
                    entry_price=price,
                    current_price=price
                )
        
        logger.info(f"Simulated order filled: {order.id}")
        return order
    
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if self._simulation_mode:
            return list(self._simulated_positions.values())
        
        positions = []
        for p in self.api.list_positions():
            positions.append(Position(
                symbol=str(p.symbol),
                qty=int(p.qty),
                side='long' if str(p.side) == 'long' else 'short',
                entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price)
            ))
        return positions
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        if self._simulation_mode:
            return self._simulated_positions.get(symbol)
        
        try:
            p = self.api.get_position(symbol)
            return Position(
                symbol=str(p.symbol),
                qty=int(p.qty),
                side='long' if str(p.side) == 'long' else 'short',
                entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price)
            )
        except Exception:
            return None
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close entire position for a symbol."""
        logger.info(f"Closing position: {symbol}")
        
        if self._simulation_mode:
            if symbol in self._simulated_positions:
                pos = self._simulated_positions[symbol]
                side = 'sell' if pos.side == 'long' else 'buy'
                return self.submit_order(symbol, pos.qty, side, 'market')
            return None
        
        try:
            self.api.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return Order(
                id='close_' + symbol,
                symbol=symbol,
                qty=0,
                side='sell',
                type='market',
                time_in_force='day',
                status='closed'
            )
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return None
    
    def close_all_positions(self):
        """Close all open positions."""
        logger.warning("Closing ALL positions!")
        
        if self._simulation_mode:
            symbols = list(self._simulated_positions.keys())
            for symbol in symbols:
                self.close_position(symbol)
            return
        
        try:
            self.api.close_all_positions()
            logger.info("All positions closed")
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
    
    def get_orders(self, status: str = 'open') -> List[Order]:
        """Get orders by status."""
        if self._simulation_mode:
            return [o for o in self._simulated_orders 
                    if status == 'all' or o.status == status]
        
        orders = []
        for o in self.api.list_orders(status=status):
            orders.append(Order(
                id=str(o.id),
                symbol=str(o.symbol),
                qty=int(o.qty),
                side=str(o.side),  # type: ignore
                type=str(o.type),  # type: ignore
                time_in_force=str(o.time_in_force),
                status=str(o.status),
                limit_price=float(o.limit_price) if o.limit_price else None,
                stop_price=float(o.stop_price) if o.stop_price else None,
                filled_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                filled_qty=int(o.filled_qty) if o.filled_qty else 0
            ))
        return orders
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        logger.info(f"Cancelling order: {order_id}")
        
        if self._simulation_mode:
            for order in self._simulated_orders:
                if order.id == order_id and order.status == 'open':
                    order.status = 'cancelled'
                    return True
            return False
        
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self):
        """Cancel all open orders."""
        logger.warning("Cancelling ALL open orders!")
        
        if self._simulation_mode:
            for order in self._simulated_orders:
                if order.status == 'open':
                    order.status = 'cancelled'
            return
        
        try:
            self.api.cancel_all_orders()
            logger.info("All orders cancelled")
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")


if __name__ == "__main__":
    # Test in simulation mode
    logging.basicConfig(level=logging.INFO)
    
    executor = PaperTradingExecutor()  # Will run in simulation mode without API keys
    
    print("\n=== Account Info ===")
    account = executor.get_account()
    for key, value in account.items():
        print(f"  {key}: {value}")
    
    print("\n=== Submitting Orders ===")
    order1 = executor.submit_order('AAPL', 10, 'buy', 'market', limit_price=150.0)
    print(f"Order 1: {order1.id} - {order1.status}")
    
    order2 = executor.submit_order('MSFT', 5, 'buy', 'market', limit_price=380.0)
    print(f"Order 2: {order2.id} - {order2.status}")
    
    print("\n=== Positions ===")
    for pos in executor.get_positions():
        print(f"  {pos.symbol}: {pos.qty} shares @ ${pos.entry_price:.2f}")
        print(f"    P&L: ${pos.unrealized_pl:.2f} ({pos.unrealized_pl_pct:.2%})")
    
    print("\n=== Account After Trades ===")
    account = executor.get_account()
    print(f"  Cash: ${account['cash']:,.2f}")
    print(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
