"""
Risk Management Module for AI Trader.

Provides:
- Position sizing based on risk parameters
- Portfolio risk limits and checks
- Value at Risk (VaR) calculations
- Drawdown monitoring
- Risk-adjusted position recommendations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Risk limit parameters."""
    
    # Position limits
    max_position_size_pct: float = 0.20       # Max 20% of portfolio in single position
    max_sector_exposure_pct: float = 0.40     # Max 40% in single sector
    max_correlated_exposure_pct: float = 0.50 # Max 50% in correlated assets
    
    # Loss limits
    max_daily_loss_pct: float = 0.03          # Max 3% daily loss
    max_weekly_loss_pct: float = 0.07         # Max 7% weekly loss
    max_drawdown_pct: float = 0.15            # Max 15% drawdown
    
    # Trade limits
    risk_per_trade_pct: float = 0.02          # Max 2% risk per trade
    max_open_positions: int = 10              # Max number of positions
    min_position_size: float = 100.00         # Minimum position value
    
    # Volatility limits
    max_portfolio_volatility: float = 0.25    # Max 25% annualized volatility
    max_position_volatility: float = 0.50     # Max 50% volatility for single stock
    
    # Leverage limits
    max_leverage: float = 1.0                 # No leverage by default
    margin_buffer_pct: float = 0.25           # 25% margin buffer


@dataclass
class RiskMetrics:
    """Current risk metrics for portfolio."""
    
    # Value metrics
    portfolio_value: float = 0.0
    cash_available: float = 0.0
    margin_used: float = 0.0
    
    # Loss metrics
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    weekly_pnl: float = 0.0
    weekly_pnl_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # Risk metrics
    portfolio_var_95: float = 0.0            # 95% VaR
    portfolio_var_99: float = 0.0            # 99% VaR
    portfolio_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Exposure metrics
    total_exposure: float = 0.0
    largest_position_pct: float = 0.0
    position_count: int = 0
    
    # Risk level
    overall_risk_level: RiskLevel = RiskLevel.LOW
    warnings: List[str] = field(default_factory=list)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    
    recommended_shares: int
    recommended_value: float
    risk_amount: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    position_risk_pct: float
    approved: bool
    rejection_reasons: List[str] = field(default_factory=list)


class PositionSizer:
    """
    Calculate optimal position sizes based on risk parameters.
    
    Implements multiple position sizing methods:
    - Fixed percentage risk
    - Kelly criterion
    - Volatility-adjusted sizing
    - ATR-based sizing
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.limits = risk_limits or RiskLimits()
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: Optional[float] = None,
        win_rate: float = 0.5,
        volatility: Optional[float] = None,
        method: str = "fixed_risk"
    ) -> PositionSizeResult:
        """
        Calculate recommended position size.
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            take_profit_price: Optional take profit price
            win_rate: Historical win rate for Kelly criterion
            volatility: Stock volatility for volatility-adjusted sizing
            method: Sizing method ('fixed_risk', 'kelly', 'volatility', 'atr')
        
        Returns:
            PositionSizeResult with recommended size and details
        """
        rejection_reasons = []
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share <= 0:
            return PositionSizeResult(
                recommended_shares=0,
                recommended_value=0,
                risk_amount=0,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price or entry_price,
                risk_reward_ratio=0,
                position_risk_pct=0,
                approved=False,
                rejection_reasons=["Invalid stop loss: must create positive risk"]
            )
        
        # Calculate max risk amount
        max_risk_amount = portfolio_value * self.limits.risk_per_trade_pct
        
        # Calculate position size based on method
        if method == "fixed_risk":
            shares = self._fixed_risk_sizing(max_risk_amount, risk_per_share)
        elif method == "kelly":
            shares = self._kelly_sizing(
                portfolio_value, entry_price, risk_per_share, win_rate, take_profit_price
            )
        elif method == "volatility":
            shares = self._volatility_sizing(
                portfolio_value, entry_price, volatility or 0.3
            )
        else:
            shares = self._fixed_risk_sizing(max_risk_amount, risk_per_share)
        
        # Round down to whole shares
        shares = int(shares)
        position_value = shares * entry_price
        risk_amount = shares * risk_per_share
        
        # Check position limits
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_pct > self.limits.max_position_size_pct:
            max_value = portfolio_value * self.limits.max_position_size_pct
            shares = int(max_value / entry_price)
            position_value = shares * entry_price
            risk_amount = shares * risk_per_share
            rejection_reasons.append(
                f"Position size reduced: exceeds {self.limits.max_position_size_pct:.0%} limit"
            )
        
        # Check minimum position size
        if position_value < self.limits.min_position_size:
            if position_value > 0:
                rejection_reasons.append(
                    f"Position below minimum size: ${self.limits.min_position_size}"
                )
            approved = False
        else:
            approved = len(rejection_reasons) == 0 or all(
                "reduced" in r for r in rejection_reasons
            )
        
        # Calculate risk-reward ratio
        if take_profit_price:
            reward_per_share = abs(take_profit_price - entry_price)
            risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        else:
            risk_reward_ratio = 0
        
        position_risk_pct = (risk_amount / portfolio_value * 100) if portfolio_value > 0 else 0
        
        return PositionSizeResult(
            recommended_shares=shares,
            recommended_value=position_value,
            risk_amount=risk_amount,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price or entry_price,
            risk_reward_ratio=risk_reward_ratio,
            position_risk_pct=position_risk_pct,
            approved=approved,
            rejection_reasons=rejection_reasons
        )
    
    def _fixed_risk_sizing(self, max_risk: float, risk_per_share: float) -> float:
        """Fixed percentage risk position sizing."""
        if risk_per_share <= 0:
            return 0
        return max_risk / risk_per_share
    
    def _kelly_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        risk_per_share: float,
        win_rate: float,
        take_profit_price: Optional[float]
    ) -> float:
        """Kelly criterion position sizing."""
        if not take_profit_price or risk_per_share <= 0:
            return self._fixed_risk_sizing(
                portfolio_value * self.limits.risk_per_trade_pct,
                risk_per_share
            )
        
        # Calculate odds
        reward = abs(take_profit_price - entry_price)
        odds = reward / risk_per_share
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win_rate, q = 1 - p
        kelly_fraction = (odds * win_rate - (1 - win_rate)) / odds
        
        # Use half-Kelly for safety
        kelly_fraction = max(0, kelly_fraction * 0.5)
        
        # Cap at max position size
        kelly_fraction = min(kelly_fraction, self.limits.max_position_size_pct)
        
        position_value = portfolio_value * kelly_fraction
        return position_value / entry_price
    
    def _volatility_sizing(
        self,
        portfolio_value: float,
        entry_price: float,
        volatility: float
    ) -> float:
        """Volatility-adjusted position sizing."""
        # Target volatility contribution
        target_vol_contribution = 0.02  # 2% volatility contribution
        
        if volatility <= 0:
            volatility = 0.3  # Default 30% volatility
        
        # Position size inversely proportional to volatility
        position_pct = target_vol_contribution / volatility
        position_pct = min(position_pct, self.limits.max_position_size_pct)
        
        position_value = portfolio_value * position_pct
        return position_value / entry_price


class RiskManager:
    """
    Comprehensive risk management for trading portfolio.
    
    Monitors and enforces risk limits, calculates risk metrics,
    and provides risk-adjusted recommendations.
    """
    
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_value: float = 100000
    ):
        self.limits = limits or RiskLimits()
        self.position_sizer = PositionSizer(self.limits)
        
        # Historical tracking
        self.initial_value = initial_value
        self.peak_value = initial_value
        self.daily_values: List[Tuple[datetime, float]] = []
        self.trade_history: List[Dict] = []
        
        # Current state
        self.current_positions: Dict[str, Dict] = {}
        self.current_cash = initial_value
        self.is_trading_halted = False
        self.halt_reason: Optional[str] = None
    
    def update_portfolio_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Update portfolio value and track history."""
        timestamp = timestamp or datetime.now()
        self.daily_values.append((timestamp, value))
        
        # Update peak for drawdown calculation
        if value > self.peak_value:
            self.peak_value = value
        
        # Keep only last 365 days
        cutoff = timestamp - timedelta(days=365)
        self.daily_values = [
            (ts, val) for ts, val in self.daily_values if ts > cutoff
        ]
    
    def check_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        portfolio_value: Optional[float] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if a trade is allowed based on risk limits.
        
        Returns:
            Tuple of (approved, list of warning/rejection messages)
        """
        messages = []
        portfolio_value = portfolio_value or self._get_portfolio_value()
        
        # Check if trading is halted
        if self.is_trading_halted:
            return False, [f"Trading halted: {self.halt_reason}"]
        
        # Check max positions
        if action.upper() == "BUY":
            if len(self.current_positions) >= self.limits.max_open_positions:
                return False, [f"Max positions ({self.limits.max_open_positions}) reached"]
        
        # Check position size limit
        position_value = quantity * price
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_pct > self.limits.max_position_size_pct:
            messages.append(
                f"Position size {position_pct:.1%} exceeds limit {self.limits.max_position_size_pct:.0%}"
            )
            return False, messages
        
        # Check minimum position size
        if position_value < self.limits.min_position_size:
            messages.append(
                f"Position ${position_value:.2f} below minimum ${self.limits.min_position_size}"
            )
            return False, messages
        
        # Check risk per trade (if stop loss provided)
        if stop_loss and action.upper() == "BUY":
            risk_amount = quantity * abs(price - stop_loss)
            risk_pct = risk_amount / portfolio_value
            
            if risk_pct > self.limits.risk_per_trade_pct:
                messages.append(
                    f"Trade risk {risk_pct:.1%} exceeds limit {self.limits.risk_per_trade_pct:.0%}"
                )
                return False, messages
        
        # Check daily loss limit
        metrics = self.get_risk_metrics(portfolio_value)
        if metrics.daily_pnl_pct < -self.limits.max_daily_loss_pct:
            messages.append(
                f"Daily loss {metrics.daily_pnl_pct:.1%} exceeds limit"
            )
            return False, messages
        
        # Check drawdown limit
        if metrics.current_drawdown_pct > self.limits.max_drawdown_pct:
            messages.append(
                f"Drawdown {metrics.current_drawdown_pct:.1%} exceeds limit"
            )
            self._halt_trading("Maximum drawdown exceeded")
            return False, messages
        
        return True, messages
    
    def get_risk_metrics(self, portfolio_value: Optional[float] = None) -> RiskMetrics:
        """Calculate current risk metrics."""
        portfolio_value = portfolio_value or self._get_portfolio_value()
        
        warnings = []
        
        # Calculate drawdown
        current_drawdown = 0
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        # Calculate daily/weekly P&L
        daily_pnl = 0
        daily_pnl_pct = 0
        weekly_pnl = 0
        weekly_pnl_pct = 0
        
        now = datetime.now()
        
        if self.daily_values:
            # Daily P&L
            yesterday_values = [
                val for ts, val in self.daily_values 
                if ts.date() == (now - timedelta(days=1)).date()
            ]
            if yesterday_values:
                yesterday_value = yesterday_values[-1]
                daily_pnl = portfolio_value - yesterday_value
                daily_pnl_pct = daily_pnl / yesterday_value if yesterday_value > 0 else 0
            
            # Weekly P&L
            week_ago = now - timedelta(days=7)
            week_values = [
                val for ts, val in self.daily_values 
                if ts.date() == week_ago.date()
            ]
            if week_values:
                week_value = week_values[-1]
                weekly_pnl = portfolio_value - week_value
                weekly_pnl_pct = weekly_pnl / week_value if week_value > 0 else 0
        
        # Calculate VaR (simplified historical VaR)
        var_95 = 0
        var_99 = 0
        volatility = 0
        
        if len(self.daily_values) >= 20:
            values = [val for _, val in self.daily_values]
            returns = np.diff(values) / values[:-1]
            
            if len(returns) > 0:
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                var_95 = np.percentile(returns, 5) * portfolio_value
                var_99 = np.percentile(returns, 1) * portfolio_value
        
        # Calculate largest position
        largest_position_pct = 0
        for pos in self.current_positions.values():
            pos_value = pos.get('value', 0)
            pos_pct = pos_value / portfolio_value if portfolio_value > 0 else 0
            largest_position_pct = max(largest_position_pct, pos_pct)
        
        # Determine risk level
        risk_level = RiskLevel.LOW
        
        if current_drawdown > self.limits.max_drawdown_pct * 0.5:
            risk_level = RiskLevel.MEDIUM
            warnings.append("Drawdown approaching limit")
        
        if current_drawdown > self.limits.max_drawdown_pct * 0.75:
            risk_level = RiskLevel.HIGH
            warnings.append("Drawdown near critical level")
        
        if current_drawdown > self.limits.max_drawdown_pct:
            risk_level = RiskLevel.CRITICAL
            warnings.append("Maximum drawdown exceeded!")
        
        if abs(daily_pnl_pct) > self.limits.max_daily_loss_pct * 0.75:
            if risk_level.value < RiskLevel.MEDIUM.value:
                risk_level = RiskLevel.MEDIUM
            warnings.append("Daily loss approaching limit")
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            cash_available=self.current_cash,
            margin_used=0,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl=weekly_pnl,
            weekly_pnl_pct=weekly_pnl_pct,
            current_drawdown_pct=current_drawdown,
            max_drawdown_pct=current_drawdown,  # Would need history for true max
            portfolio_var_95=float(var_95) if var_95 >= 0 else -float(var_95),
            portfolio_var_99=float(var_99) if var_99 >= 0 else -float(var_99),
            portfolio_volatility=volatility,
            sharpe_ratio=0,  # Would need risk-free rate
            sortino_ratio=0,
            total_exposure=portfolio_value - self.current_cash,
            largest_position_pct=largest_position_pct,
            position_count=len(self.current_positions),
            overall_risk_level=risk_level,
            warnings=warnings
        )
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: Optional[float] = None,
        portfolio_value: Optional[float] = None,
        method: str = "fixed_risk"
    ) -> PositionSizeResult:
        """Calculate recommended position size."""
        portfolio_value = portfolio_value or self._get_portfolio_value()
        
        return self.position_sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            method=method
        )
    
    def add_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> None:
        """Add a position to tracking."""
        self.current_positions[symbol] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': entry_price,
            'value': quantity * entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now()
        }
        self.current_cash -= quantity * entry_price
    
    def remove_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """Remove a position and return P&L."""
        if symbol not in self.current_positions:
            return None
        
        pos = self.current_positions[symbol]
        pnl = (exit_price - pos['entry_price']) * pos['quantity']
        
        self.current_cash += pos['quantity'] * exit_price
        
        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'quantity': pos['quantity'],
            'pnl': pnl,
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now()
        })
        
        del self.current_positions[symbol]
        return pnl
    
    def update_position_price(self, symbol: str, current_price: float) -> None:
        """Update current price of a position."""
        if symbol in self.current_positions:
            pos = self.current_positions[symbol]
            pos['current_price'] = current_price
            pos['value'] = pos['quantity'] * current_price
    
    def check_stop_loss_take_profit(self) -> List[Dict]:
        """Check positions for stop loss / take profit hits."""
        triggers = []
        
        for symbol, pos in self.current_positions.items():
            current_price = pos['current_price']
            
            if pos['stop_loss'] and current_price <= pos['stop_loss']:
                triggers.append({
                    'symbol': symbol,
                    'type': 'stop_loss',
                    'trigger_price': pos['stop_loss'],
                    'current_price': current_price
                })
            
            if pos['take_profit'] and current_price >= pos['take_profit']:
                triggers.append({
                    'symbol': symbol,
                    'type': 'take_profit',
                    'trigger_price': pos['take_profit'],
                    'current_price': current_price
                })
        
        return triggers
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        positions_value = sum(
            pos['value'] for pos in self.current_positions.values()
        )
        return self.current_cash + positions_value
    
    def _halt_trading(self, reason: str) -> None:
        """Halt trading due to risk limit breach."""
        self.is_trading_halted = True
        self.halt_reason = reason
    
    def resume_trading(self) -> None:
        """Resume trading after halt."""
        self.is_trading_halted = False
        self.halt_reason = None
    
    def get_risk_report(self) -> str:
        """Generate a risk report summary."""
        metrics = self.get_risk_metrics()
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║                    RISK REPORT                            ║
╠══════════════════════════════════════════════════════════╣
║  Portfolio Value:     ${metrics.portfolio_value:>15,.2f}      ║
║  Cash Available:      ${metrics.cash_available:>15,.2f}      ║
║  Total Exposure:      ${metrics.total_exposure:>15,.2f}      ║
╠══════════════════════════════════════════════════════════╣
║  Daily P&L:           ${metrics.daily_pnl:>+15,.2f} ({metrics.daily_pnl_pct:>+.2%})  ║
║  Weekly P&L:          ${metrics.weekly_pnl:>+15,.2f} ({metrics.weekly_pnl_pct:>+.2%})  ║
║  Current Drawdown:    {metrics.current_drawdown_pct:>15.2%}          ║
╠══════════════════════════════════════════════════════════╣
║  VaR (95%):           ${metrics.portfolio_var_95:>15,.2f}      ║
║  VaR (99%):           ${metrics.portfolio_var_99:>15,.2f}      ║
║  Portfolio Volatility: {metrics.portfolio_volatility:>14.2%}          ║
╠══════════════════════════════════════════════════════════╣
║  Open Positions:      {metrics.position_count:>15}           ║
║  Largest Position:    {metrics.largest_position_pct:>14.1%}          ║
║  Risk Level:          {metrics.overall_risk_level.value:>15}           ║
╚══════════════════════════════════════════════════════════╝
"""
        
        if metrics.warnings:
            report += "\n⚠️  WARNINGS:\n"
            for warning in metrics.warnings:
                report += f"   • {warning}\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Create risk manager with custom limits
    limits = RiskLimits(
        max_position_size_pct=0.15,
        risk_per_trade_pct=0.02,
        max_drawdown_pct=0.10
    )
    
    risk_manager = RiskManager(limits=limits, initial_value=100000)
    
    # Calculate position size
    result = risk_manager.calculate_position_size(
        entry_price=150.00,
        stop_loss_price=142.50,
        take_profit_price=165.00
    )
    
    print(f"Recommended shares: {result.recommended_shares}")
    print(f"Position value: ${result.recommended_value:,.2f}")
    print(f"Risk amount: ${result.risk_amount:,.2f}")
    print(f"Risk/Reward: {result.risk_reward_ratio:.2f}")
    print(f"Approved: {result.approved}")
    
    # Check trade
    approved, messages = risk_manager.check_trade(
        symbol='AAPL',
        action='BUY',
        quantity=result.recommended_shares,
        price=150.00,
        stop_loss=142.50
    )
    
    print(f"\nTrade approved: {approved}")
    if messages:
        for msg in messages:
            print(f"  • {msg}")
    
    # Add position
    if approved:
        risk_manager.add_position(
            symbol='AAPL',
            quantity=result.recommended_shares,
            entry_price=150.00,
            stop_loss=142.50,
            take_profit=165.00
        )
    
    # Print risk report
    print(risk_manager.get_risk_report())
