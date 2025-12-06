"""
Unified Alert Manager for AI Trader.
Manages notifications across multiple channels (Telegram, Discord).
"""

import logging
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from trader.strategies.base import Signal
from .telegram_bot import TelegramAlert
from .discord_webhook import DiscordAlert

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts."""
    SIGNAL = "signal"
    TRADE_EXECUTION = "trade_execution"
    PORTFOLIO_UPDATE = "portfolio_update"
    DAILY_SUMMARY = "daily_summary"
    BACKTEST_RESULT = "backtest_result"
    ERROR = "error"
    INFO = "info"
    WARNING = "warning"


@dataclass
class AlertConfig:
    """Alert channel configuration."""
    telegram_enabled: bool = True
    discord_enabled: bool = True
    min_confidence: float = 0.6  # Minimum signal confidence to alert
    quiet_hours_start: Optional[int] = None  # Hour to start quiet mode (0-23)
    quiet_hours_end: Optional[int] = None    # Hour to end quiet mode (0-23)
    alert_on_hold: bool = False  # Whether to alert on HOLD signals


class AlertManager:
    """
    Unified alert manager for sending notifications.
    
    Features:
    - Multi-channel support (Telegram, Discord)
    - Configurable alert filtering
    - Quiet hours support
    - Alert history tracking
    """
    
    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        discord_webhook_url: Optional[str] = None,
        config: Optional[AlertConfig] = None,
    ):
        """
        Initialize alert manager.
        
        Args:
            telegram_token: Telegram bot token
            telegram_chat_id: Telegram chat ID
            discord_webhook_url: Discord webhook URL
            config: Alert configuration
        """
        self.telegram = TelegramAlert(
            bot_token=telegram_token,
            chat_id=telegram_chat_id,
        )
        
        self.discord = DiscordAlert(
            webhook_url=discord_webhook_url,
        )
        
        self.config = config or AlertConfig()
        self.alert_history: List[Dict[str, Any]] = []
    
    @property
    def channels_available(self) -> Dict[str, bool]:
        """Get available notification channels."""
        return {
            "telegram": self.telegram.is_configured,
            "discord": self.discord.is_configured,
        }
    
    def _is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        if self.config.quiet_hours_start is None or self.config.quiet_hours_end is None:
            return False
        
        current_hour = datetime.now().hour
        
        if self.config.quiet_hours_start <= self.config.quiet_hours_end:
            # Simple range (e.g., 22 to 8)
            return self.config.quiet_hours_start <= current_hour < self.config.quiet_hours_end
        else:
            # Overnight range (e.g., 22 to 8)
            return current_hour >= self.config.quiet_hours_start or current_hour < self.config.quiet_hours_end
    
    def _record_alert(
        self,
        alert_type: AlertType,
        success: bool,
        details: Dict[str, Any],
    ):
        """Record alert in history."""
        self.alert_history.append({
            "timestamp": datetime.now(),
            "type": alert_type.value,
            "success": success,
            "details": details,
        })
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def send_signal(self, signal: Signal) -> Dict[str, bool]:
        """
        Send signal alert to all configured channels.
        
        Args:
            signal: Trading signal to send
            
        Returns:
            Dictionary with channel names and success status
        """
        results = {}
        
        # Check if we should send this alert
        if signal.confidence < self.config.min_confidence:
            logger.debug(f"Signal confidence {signal.confidence} below threshold {self.config.min_confidence}")
            return results
        
        if not self.config.alert_on_hold and signal.signal_type.value == "hold":
            logger.debug("HOLD signal skipped (alert_on_hold=False)")
            return results
        
        if self._is_quiet_hours():
            logger.debug("Alert skipped - quiet hours active")
            return results
        
        # Send to Telegram
        if self.config.telegram_enabled and self.telegram.is_configured:
            try:
                results["telegram"] = self.telegram.send_signal_alert(signal)
            except Exception as e:
                logger.error(f"Telegram alert failed: {e}")
                results["telegram"] = False
        
        # Send to Discord
        if self.config.discord_enabled and self.discord.is_configured:
            try:
                results["discord"] = self.discord.send_signal_alert(signal)
            except Exception as e:
                logger.error(f"Discord alert failed: {e}")
                results["discord"] = False
        
        self._record_alert(AlertType.SIGNAL, all(results.values()), {"signal": str(signal)})
        
        return results
    
    def send_trade_execution(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        order_id: Optional[str] = None,
        is_paper: bool = True,
    ) -> Dict[str, bool]:
        """
        Send trade execution notification.
        
        Args:
            symbol: Stock symbol
            action: Trade action
            quantity: Number of shares
            price: Execution price
            order_id: Order ID
            is_paper: Whether paper trading
            
        Returns:
            Dictionary with channel names and success status
        """
        results = {}
        
        if self._is_quiet_hours():
            return results
        
        # Send to Telegram
        if self.config.telegram_enabled and self.telegram.is_configured:
            try:
                results["telegram"] = self.telegram.send_trade_execution(
                    symbol, action, quantity, price, order_id
                )
            except Exception as e:
                logger.error(f"Telegram trade alert failed: {e}")
                results["telegram"] = False
        
        # Send to Discord
        if self.config.discord_enabled and self.discord.is_configured:
            try:
                results["discord"] = self.discord.send_trade_execution(
                    symbol, action, quantity, price, order_id, is_paper
                )
            except Exception as e:
                logger.error(f"Discord trade alert failed: {e}")
                results["discord"] = False
        
        self._record_alert(
            AlertType.TRADE_EXECUTION,
            all(results.values()),
            {"symbol": symbol, "action": action, "quantity": quantity}
        )
        
        return results
    
    def send_portfolio_update(
        self,
        positions: List[Dict[str, Any]],
        total_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        buying_power: float = 0.0,
    ) -> Dict[str, bool]:
        """
        Send portfolio update notification.
        
        Args:
            positions: List of positions
            total_value: Total portfolio value
            daily_pnl: Daily P/L
            daily_pnl_pct: Daily P/L percentage
            buying_power: Available buying power
            
        Returns:
            Dictionary with channel names and success status
        """
        results = {}
        
        # Send to Telegram
        if self.config.telegram_enabled and self.telegram.is_configured:
            try:
                results["telegram"] = self.telegram.send_portfolio_update(
                    positions, total_value, daily_pnl, daily_pnl_pct
                )
            except Exception as e:
                logger.error(f"Telegram portfolio alert failed: {e}")
                results["telegram"] = False
        
        # Send to Discord
        if self.config.discord_enabled and self.discord.is_configured:
            try:
                results["discord"] = self.discord.send_portfolio_update(
                    positions, total_value, daily_pnl, daily_pnl_pct, buying_power
                )
            except Exception as e:
                logger.error(f"Discord portfolio alert failed: {e}")
                results["discord"] = False
        
        self._record_alert(
            AlertType.PORTFOLIO_UPDATE,
            all(results.values()),
            {"total_value": total_value, "daily_pnl": daily_pnl}
        )
        
        return results
    
    def send_daily_summary(
        self,
        date: datetime,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        portfolio_value: float,
        top_performers: List[Dict[str, Any]],
        worst_performers: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """
        Send daily summary notification.
        
        Args:
            date: Summary date
            total_trades: Total trades
            winning_trades: Winning trades
            losing_trades: Losing trades
            total_pnl: Total P/L
            portfolio_value: Portfolio value
            top_performers: Top performers
            worst_performers: Worst performers
            
        Returns:
            Dictionary with channel names and success status
        """
        results = {}
        
        # Send to Telegram
        if self.config.telegram_enabled and self.telegram.is_configured:
            try:
                results["telegram"] = self.telegram.send_daily_summary(
                    date, total_trades, winning_trades, losing_trades,
                    total_pnl, top_performers, worst_performers
                )
            except Exception as e:
                logger.error(f"Telegram daily summary failed: {e}")
                results["telegram"] = False
        
        # Send to Discord
        if self.config.discord_enabled and self.discord.is_configured:
            try:
                results["discord"] = self.discord.send_daily_summary(
                    date, total_trades, winning_trades, losing_trades,
                    total_pnl, portfolio_value, top_performers, worst_performers
                )
            except Exception as e:
                logger.error(f"Discord daily summary failed: {e}")
                results["discord"] = False
        
        self._record_alert(
            AlertType.DAILY_SUMMARY,
            all(results.values()),
            {"date": date.isoformat(), "total_pnl": total_pnl}
        )
        
        return results
    
    def send_backtest_result(
        self,
        strategy_name: str,
        symbol: str,
        total_return_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        win_rate: float,
        total_trades: int,
        profit_factor: float = 0.0,
        period: str = "",
    ) -> Dict[str, bool]:
        """
        Send backtest result notification.
        
        Args:
            strategy_name: Strategy name
            symbol: Symbol tested
            total_return_pct: Total return %
            sharpe_ratio: Sharpe ratio
            max_drawdown_pct: Max drawdown %
            win_rate: Win rate %
            total_trades: Total trades
            profit_factor: Profit factor
            period: Test period
            
        Returns:
            Dictionary with channel names and success status
        """
        results = {}
        
        # Send to Telegram
        if self.config.telegram_enabled and self.telegram.is_configured:
            try:
                results["telegram"] = self.telegram.send_backtest_result(
                    strategy_name, symbol, total_return_pct, sharpe_ratio,
                    max_drawdown_pct, win_rate, total_trades
                )
            except Exception as e:
                logger.error(f"Telegram backtest result failed: {e}")
                results["telegram"] = False
        
        # Send to Discord
        if self.config.discord_enabled and self.discord.is_configured:
            try:
                results["discord"] = self.discord.send_backtest_result(
                    strategy_name, symbol, period, total_return_pct, sharpe_ratio,
                    max_drawdown_pct, win_rate, total_trades, profit_factor
                )
            except Exception as e:
                logger.error(f"Discord backtest result failed: {e}")
                results["discord"] = False
        
        self._record_alert(
            AlertType.BACKTEST_RESULT,
            all(results.values()),
            {"strategy": strategy_name, "return": total_return_pct}
        )
        
        return results
    
    def send_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[str] = None,
        severity: str = "error",
    ) -> Dict[str, bool]:
        """
        Send error notification.
        
        Args:
            error_type: Error type
            error_message: Error message
            context: Additional context
            severity: Severity level
            
        Returns:
            Dictionary with channel names and success status
        """
        results = {}
        
        # Always send errors, even during quiet hours
        
        # Send to Telegram
        if self.config.telegram_enabled and self.telegram.is_configured:
            try:
                results["telegram"] = self.telegram.send_error_alert(
                    error_type, error_message, context
                )
            except Exception as e:
                logger.error(f"Telegram error alert failed: {e}")
                results["telegram"] = False
        
        # Send to Discord
        if self.config.discord_enabled and self.discord.is_configured:
            try:
                results["discord"] = self.discord.send_error_alert(
                    error_type, error_message, context, severity
                )
            except Exception as e:
                logger.error(f"Discord error alert failed: {e}")
                results["discord"] = False
        
        self._record_alert(
            AlertType.ERROR,
            all(results.values()),
            {"type": error_type, "message": error_message}
        )
        
        return results
    
    def test_all_channels(self) -> Dict[str, bool]:
        """
        Test all configured notification channels.
        
        Returns:
            Dictionary with channel names and success status
        """
        results = {}
        
        if self.telegram.is_configured:
            results["telegram"] = self.telegram.test_connection()
        else:
            results["telegram"] = False
            logger.info("Telegram not configured")
        
        if self.discord.is_configured:
            results["discord"] = self.discord.test_connection()
        else:
            results["discord"] = False
            logger.info("Discord not configured")
        
        return results
    
    def get_alert_history(
        self,
        alert_type: Optional[AlertType] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            alert_type: Filter by alert type
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert records
        """
        history = self.alert_history
        
        if alert_type:
            history = [a for a in history if a["type"] == alert_type.value]
        
        return history[-limit:]


if __name__ == "__main__":
    # Test the alert manager
    from trader.strategies.base import Signal, SignalType
    
    # Create test signal
    signal = Signal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        confidence=0.85,
        price=150.00,
        stop_loss=142.50,
        take_profit=165.00,
        reasons=["RSI oversold at 28", "MACD bullish crossover"],
    )
    
    # Initialize alert manager
    manager = AlertManager()
    
    print("Available channels:", manager.channels_available)
    
    # Test all channels
    print("\nTesting channels...")
    test_results = manager.test_all_channels()
    for channel, success in test_results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {channel}")
    
    # Send test signal if any channel is available
    if any(manager.channels_available.values()):
        print("\nSending test signal...")
        results = manager.send_signal(signal)
        for channel, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {channel}")
