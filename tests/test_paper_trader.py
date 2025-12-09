"""
Unit tests for Paper Trading System.
"""

import pytest
import os
import tempfile
from datetime import datetime

from trader.portfolio.paper_trader import (
    PaperAccount,
    Position,
    Trade,
    OrderSide,
    OrderType
)


class TestPosition:
    """Test Position dataclass."""
    
    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(
            symbol="AAPL",
            quantity=10,
            avg_entry_price=150.0,
            current_price=155.0
        )
        
        assert pos.symbol == "AAPL"
        assert pos.quantity == 10
        assert pos.avg_entry_price == 150.0
        assert pos.current_price == 155.0
    
    def test_market_value(self):
        """Test market value calculation."""
        pos = Position("AAPL", 10, 150.0, 155.0)
        assert pos.market_value == 1550.0
    
    def test_cost_basis(self):
        """Test cost basis calculation."""
        pos = Position("AAPL", 10, 150.0, 155.0)
        assert pos.cost_basis == 1500.0
    
    def test_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        pos = Position("AAPL", 10, 150.0, 155.0)
        assert pos.unrealized_pnl == 50.0
    
    def test_unrealized_pnl_pct(self):
        """Test unrealized P&L percentage."""
        pos = Position("AAPL", 10, 150.0, 155.0)
        assert abs(pos.unrealized_pnl_pct - 3.33) < 0.01


class TestPaperAccount:
    """Test PaperAccount class."""
    
    def setup_method(self):
        """Setup for each test."""
        # Use temp file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.account = PaperAccount(
            initial_balance=100000.0,
            commission_per_trade=1.0,
            account_file=self.temp_file.name
        )
    
    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """Test account initialization."""
        assert self.account.initial_balance == 100000.0
        assert self.account.cash == 100000.0
        assert self.account.commission_per_trade == 1.0
        assert len(self.account.positions) == 0
        assert len(self.account.trades) == 0
    
    def test_buy_order_success(self):
        """Test successful buy order."""
        success, msg = self.account.execute_trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=150.0
        )
        
        assert success is True
        assert "Bought" in msg
        assert self.account.cash == 100000.0 - (10 * 150.0) - 1.0  # Cost + commission
        assert "AAPL" in self.account.positions
        assert self.account.positions["AAPL"].quantity == 10
        assert len(self.account.trades) == 1
    
    def test_buy_order_insufficient_funds(self):
        """Test buy order with insufficient funds."""
        success, msg = self.account.execute_trade(
            symbol="TSLA",
            side=OrderSide.BUY,
            quantity=1000,
            price=1000.0  # Would cost $1,000,000
        )
        
        assert success is False
        assert "Insufficient funds" in msg
        assert self.account.cash == 100000.0
        assert "TSLA" not in self.account.positions
    
    def test_multiple_buys_same_symbol(self):
        """Test multiple buys of same symbol (averaging)."""
        # First buy: 10 shares at $150
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        
        # Second buy: 10 shares at $160
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 160.0)
        
        # Should have 20 shares at avg price of $155
        pos = self.account.positions["AAPL"]
        assert pos.quantity == 20
        assert pos.avg_entry_price == 155.0
    
    def test_sell_order_success(self):
        """Test successful sell order."""
        # First buy
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        
        # Then sell
        success, msg = self.account.execute_trade("AAPL", OrderSide.SELL, 5, 160.0)
        
        assert success is True
        assert "Sold" in msg
        assert self.account.positions["AAPL"].quantity == 5
        assert len(self.account.closed_trades) == 1
        
        # Check realized P&L
        closed = self.account.closed_trades[0]
        assert closed["quantity"] == 5
        assert closed["entry_price"] == 150.0
        assert closed["exit_price"] == 160.0
        # P&L = (160 * 5) - 1 (commission) - (150 * 5) = 800 - 1 - 750 = 49
        assert abs(closed["pnl"] - 49.0) < 0.01
    
    def test_sell_entire_position(self):
        """Test selling entire position."""
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        self.account.execute_trade("AAPL", OrderSide.SELL, 10, 160.0)
        
        # Position should be removed
        assert "AAPL" not in self.account.positions
        assert len(self.account.closed_trades) == 1
    
    def test_sell_without_position(self):
        """Test selling without holding position."""
        success, msg = self.account.execute_trade("AAPL", OrderSide.SELL, 10, 150.0)
        
        assert success is False
        assert "No position" in msg
    
    def test_sell_more_than_owned(self):
        """Test selling more shares than owned."""
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        
        success, msg = self.account.execute_trade("AAPL", OrderSide.SELL, 15, 160.0)
        
        assert success is False
        assert "Insufficient shares" in msg
    
    def test_update_prices(self):
        """Test updating position prices."""
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        self.account.execute_trade("MSFT", OrderSide.BUY, 5, 300.0)
        
        # Update prices
        self.account.update_prices({
            "AAPL": 155.0,
            "MSFT": 310.0
        })
        
        assert self.account.positions["AAPL"].current_price == 155.0
        assert self.account.positions["MSFT"].current_price == 310.0
    
    def test_get_total_value(self):
        """Test total account value calculation."""
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        self.account.update_prices({"AAPL": 160.0})
        
        # Cash: 100000 - (10*150) - 1 = 98499
        # Position value: 10 * 160 = 1600
        # Total: 98499 + 1600 = 100099
        total = self.account.get_total_value()
        assert abs(total - 100099.0) < 0.01
    
    def test_get_total_pnl(self):
        """Test total unrealized P&L."""
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        self.account.update_prices({"AAPL": 160.0})
        
        # Unrealized P&L: (160 - 150) * 10 = 100
        pnl = self.account.get_total_pnl()
        assert pnl == 100.0
    
    def test_get_total_return(self):
        """Test total return percentage."""
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        self.account.update_prices({"AAPL": 160.0})
        
        # Started with 100000, now have 100099
        # Return: (100099 - 100000) / 100000 * 100 = 0.099%
        ret = self.account.get_total_return()
        assert abs(ret - 0.099) < 0.01
    
    def test_get_metrics(self):
        """Test performance metrics calculation."""
        # Execute some trades
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        self.account.execute_trade("AAPL", OrderSide.SELL, 5, 160.0)  # Winner
        
        self.account.execute_trade("MSFT", OrderSide.BUY, 5, 300.0)
        self.account.execute_trade("MSFT", OrderSide.SELL, 5, 290.0)  # Loser
        
        metrics = self.account.get_metrics()
        
        assert metrics["total_trades"] == 2
        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 1
        assert metrics["win_rate"] == 50.0
        assert metrics["avg_win"] > 0
        assert metrics["avg_loss"] < 0
    
    def test_save_and_load(self):
        """Test saving and loading account state."""
        # Execute some trades
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        self.account.execute_trade("MSFT", OrderSide.BUY, 5, 300.0)
        
        # Save is automatic, now load into new account
        new_account = PaperAccount(account_file=self.temp_file.name)
        
        assert new_account.cash == self.account.cash
        assert len(new_account.positions) == 2
        assert "AAPL" in new_account.positions
        assert "MSFT" in new_account.positions
        assert new_account.positions["AAPL"].quantity == 10
        assert len(new_account.trades) == 2
    
    def test_reset(self):
        """Test account reset."""
        # Execute trades
        self.account.execute_trade("AAPL", OrderSide.BUY, 10, 150.0)
        self.account.execute_trade("AAPL", OrderSide.SELL, 5, 160.0)
        
        # Reset
        self.account.reset()
        
        assert self.account.cash == self.account.initial_balance
        assert len(self.account.positions) == 0
        assert len(self.account.trades) == 0
        assert len(self.account.closed_trades) == 0
    
    def test_invalid_quantity(self):
        """Test invalid quantity."""
        success, msg = self.account.execute_trade("AAPL", OrderSide.BUY, -10, 150.0)
        assert success is False
        assert "positive" in msg.lower()
    
    def test_invalid_price(self):
        """Test invalid price."""
        success, msg = self.account.execute_trade("AAPL", OrderSide.BUY, 10, -150.0)
        assert success is False
        assert "positive" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
