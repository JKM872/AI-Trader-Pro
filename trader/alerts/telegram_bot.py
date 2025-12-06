"""
Telegram Bot integration for trade alerts.
Sends notifications about signals, trades, and portfolio updates.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import requests

from trader.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str
    chat_id: str
    parse_mode: str = "HTML"
    disable_notification: bool = False


class TelegramAlert:
    """
    Telegram bot for sending trade alerts.
    
    Features:
    - Signal notifications (BUY/SELL/HOLD)
    - Trade execution confirmations
    - Portfolio updates
    - Daily summary reports
    - Error alerts
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}/{method}"
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """
        Initialize Telegram alert.
        
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Chat ID to send messages to (can be user or group)
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
        if not self.chat_id:
            logger.warning("Telegram chat ID not configured")
        
        self.config = TelegramConfig(
            bot_token=self.bot_token or "",
            chat_id=self.chat_id or "",
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id)
    
    def _make_request(self, method: str, data: Dict[str, Any]) -> Optional[Dict]:
        """Make request to Telegram API."""
        if not self.is_configured:
            logger.warning("Telegram not configured, skipping notification")
            return None
        
        url = self.BASE_URL.format(token=self.bot_token, method=method)
        
        try:
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Telegram API error: {e}")
            return None
    
    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a text message.
        
        Args:
            text: Message text (supports HTML formatting)
            parse_mode: Message parse mode (HTML or Markdown)
            disable_notification: Send silently
            
        Returns:
            True if message was sent successfully
        """
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }
        
        result = self._make_request("sendMessage", data)
        return result is not None and result.get("ok", False)
    
    def send_signal_alert(self, signal: Signal) -> bool:
        """
        Send trading signal notification.
        
        Args:
            signal: Trading signal to notify about
            
        Returns:
            True if notification was sent successfully
        """
        # Emoji based on signal type
        emoji_map = {
            SignalType.BUY: "🟢",
            SignalType.SELL: "🔴",
            SignalType.HOLD: "🟡",
        }
        emoji = emoji_map.get(signal.signal_type, "⚪")
        
        # Format confidence as percentage
        confidence_pct = signal.confidence * 100
        
        # Build message
        message = f"""
{emoji} <b>TRADING SIGNAL</b> {emoji}

<b>Symbol:</b> {signal.symbol}
<b>Action:</b> {signal.signal_type.value.upper()}
<b>Confidence:</b> {confidence_pct:.1f}%
<b>Price:</b> ${signal.price:.2f}
"""
        
        if signal.stop_loss:
            message += f"<b>Stop Loss:</b> ${signal.stop_loss:.2f}\n"
        
        if signal.take_profit:
            message += f"<b>Take Profit:</b> ${signal.take_profit:.2f}\n"
        
        if signal.reasons:
            message += f"\n<b>Reasons:</b>\n"
            for reason in signal.reasons:
                message += f"  • {reason}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message(message)
    
    def send_trade_execution(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        order_id: Optional[str] = None,
    ) -> bool:
        """
        Send trade execution notification.
        
        Args:
            symbol: Stock symbol
            action: Trade action (buy/sell)
            quantity: Number of shares
            price: Execution price
            order_id: Order ID from broker
            
        Returns:
            True if notification was sent
        """
        emoji = "📈" if action.lower() == "buy" else "📉"
        
        message = f"""
{emoji} <b>TRADE EXECUTED</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Action:</b> {action.upper()}
<b>Quantity:</b> {quantity:.2f} shares
<b>Price:</b> ${price:.2f}
<b>Total:</b> ${quantity * price:,.2f}
"""
        
        if order_id:
            message += f"<b>Order ID:</b> {order_id}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message(message)
    
    def send_portfolio_update(
        self,
        positions: List[Dict[str, Any]],
        total_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
    ) -> bool:
        """
        Send portfolio summary update.
        
        Args:
            positions: List of current positions
            total_value: Total portfolio value
            daily_pnl: Daily profit/loss in dollars
            daily_pnl_pct: Daily profit/loss percentage
            
        Returns:
            True if notification was sent
        """
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        pnl_sign = "+" if daily_pnl >= 0 else ""
        
        message = f"""
💼 <b>PORTFOLIO UPDATE</b> 💼

<b>Total Value:</b> ${total_value:,.2f}
<b>Daily P/L:</b> {pnl_sign}${daily_pnl:,.2f} ({pnl_sign}{daily_pnl_pct:.2f}%) {pnl_emoji}

<b>Positions:</b>
"""
        
        if positions:
            for pos in positions[:10]:  # Limit to 10 positions
                symbol = pos.get('symbol', 'N/A')
                qty = pos.get('quantity', 0)
                pnl = pos.get('unrealized_pnl', 0)
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                message += f"  {pnl_emoji} {symbol}: {qty} shares (${pnl:+,.2f})\n"
            
            if len(positions) > 10:
                message += f"  ... and {len(positions) - 10} more\n"
        else:
            message += "  No open positions\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message(message)
    
    def send_daily_summary(
        self,
        date: datetime,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        top_performers: List[Dict[str, Any]],
        worst_performers: List[Dict[str, Any]],
    ) -> bool:
        """
        Send daily trading summary.
        
        Args:
            date: Summary date
            total_trades: Total trades executed
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            total_pnl: Total profit/loss
            top_performers: Best performing trades
            worst_performers: Worst performing trades
            
        Returns:
            True if notification was sent
        """
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        pnl_emoji = "🎉" if total_pnl >= 0 else "😔"
        pnl_sign = "+" if total_pnl >= 0 else ""
        
        message = f"""
📊 <b>DAILY SUMMARY</b> 📊
{date.strftime('%A, %B %d, %Y')}

<b>Trading Activity:</b>
  Total Trades: {total_trades}
  Winning: {winning_trades} 🟢
  Losing: {losing_trades} 🔴
  Win Rate: {win_rate:.1f}%

<b>Performance:</b> {pnl_sign}${total_pnl:,.2f} {pnl_emoji}
"""
        
        if top_performers:
            message += "\n<b>Top Performers:</b>\n"
            for trade in top_performers[:3]:
                message += f"  🏆 {trade['symbol']}: +${trade['pnl']:.2f}\n"
        
        if worst_performers:
            message += "\n<b>Worst Performers:</b>\n"
            for trade in worst_performers[:3]:
                message += f"  ⚠️ {trade['symbol']}: ${trade['pnl']:.2f}\n"
        
        return self.send_message(message)
    
    def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        context: Optional[str] = None,
    ) -> bool:
        """
        Send error notification.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context
            
        Returns:
            True if notification was sent
        """
        message = f"""
🚨 <b>ERROR ALERT</b> 🚨

<b>Type:</b> {error_type}
<b>Message:</b> {error_message}
"""
        
        if context:
            message += f"<b>Context:</b> {context}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_message(message, disable_notification=False)
    
    def send_backtest_result(
        self,
        strategy_name: str,
        symbol: str,
        total_return_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        win_rate: float,
        total_trades: int,
    ) -> bool:
        """
        Send backtest results notification.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Symbol tested
            total_return_pct: Total return percentage
            sharpe_ratio: Sharpe ratio
            max_drawdown_pct: Maximum drawdown percentage
            win_rate: Win rate percentage
            total_trades: Total number of trades
            
        Returns:
            True if notification was sent
        """
        result_emoji = "✅" if total_return_pct > 0 else "❌"
        
        message = f"""
🔬 <b>BACKTEST RESULTS</b> 🔬

<b>Strategy:</b> {strategy_name}
<b>Symbol:</b> {symbol}

<b>Performance:</b>
  Total Return: {total_return_pct:+.2f}% {result_emoji}
  Sharpe Ratio: {sharpe_ratio:.2f}
  Max Drawdown: -{max_drawdown_pct:.2f}%
  Win Rate: {win_rate:.1f}%
  Total Trades: {total_trades}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        if not self.is_configured:
            return False
        
        return self.send_message("🤖 AI Trader bot is connected and ready!")


if __name__ == "__main__":
    # Test the Telegram alert
    from trader.strategies.base import Signal, SignalType
    
    # Create test signal
    signal = Signal(
        symbol="AAPL",
        signal_type=SignalType.BUY,
        confidence=0.85,
        price=150.00,
        stop_loss=142.50,
        take_profit=165.00,
        reasons=["RSI oversold at 28", "MACD bullish crossover", "Price above SMA200"],
    )
    
    # Initialize Telegram (will use env vars)
    telegram = TelegramAlert()
    
    if telegram.is_configured:
        print("Testing Telegram connection...")
        if telegram.test_connection():
            print("✅ Connection successful!")
            
            print("Sending test signal...")
            if telegram.send_signal_alert(signal):
                print("✅ Signal alert sent!")
            else:
                print("❌ Failed to send signal alert")
        else:
            print("❌ Connection failed")
    else:
        print("⚠️ Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
