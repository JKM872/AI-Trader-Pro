"""
Discord Webhook integration for trade alerts.
Sends rich embed notifications to Discord channels.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timezone
import requests

from trader.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class DiscordEmbed:
    """Discord embed structure."""
    title: Optional[str] = None
    description: Optional[str] = None
    color: int = 0x5865F2  # Discord blurple
    fields: Optional[List[Dict[str, Any]]] = None
    footer: Optional[Dict[str, str]] = None
    timestamp: Optional[str] = None
    thumbnail: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Discord API format."""
        embed = {}
        
        if self.title:
            embed['title'] = self.title
        if self.description:
            embed['description'] = self.description
        if self.color:
            embed['color'] = self.color
        if self.fields:
            embed['fields'] = self.fields
        if self.footer:
            embed['footer'] = self.footer
        if self.timestamp:
            embed['timestamp'] = self.timestamp
        if self.thumbnail:
            embed['thumbnail'] = self.thumbnail
        
        return embed


class DiscordAlert:
    """
    Discord webhook for sending trade alerts.
    
    Features:
    - Rich embed notifications
    - Signal alerts with visual formatting
    - Trade execution notifications
    - Portfolio updates
    - Daily summaries
    - Error alerts
    """
    
    # Colors for different alert types
    COLORS = {
        'buy': 0x00FF00,      # Green
        'sell': 0xFF0000,     # Red
        'hold': 0xFFFF00,     # Yellow
        'info': 0x5865F2,     # Discord Blurple
        'success': 0x00FF00,  # Green
        'warning': 0xFFA500,  # Orange
        'error': 0xFF0000,    # Red
    }
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        username: str = "AI Trader",
        avatar_url: Optional[str] = None,
    ):
        """
        Initialize Discord alert.
        
        Args:
            webhook_url: Discord webhook URL
            username: Bot username to display
            avatar_url: Bot avatar URL
        """
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.username = username
        self.avatar_url = avatar_url
        
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
    
    @property
    def is_configured(self) -> bool:
        """Check if Discord is properly configured."""
        return bool(self.webhook_url)
    
    def _send_webhook(
        self,
        content: Optional[str] = None,
        embeds: Optional[List[DiscordEmbed]] = None,
    ) -> bool:
        """
        Send webhook message.
        
        Args:
            content: Plain text content
            embeds: List of embeds to send
            
        Returns:
            True if message was sent successfully
        """
        if not self.is_configured:
            logger.warning("Discord not configured, skipping notification")
            return False
        
        data: Dict[str, Any] = {
            "username": self.username,
        }
        
        if self.avatar_url:
            data["avatar_url"] = self.avatar_url
        
        if content:
            data["content"] = content
        
        if embeds:
            data["embeds"] = [e.to_dict() for e in embeds]
        
        try:
            response = requests.post(
                self.webhook_url,  # type: ignore
                json=data,
                timeout=10,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Discord webhook error: {e}")
            return False
    
    def send_message(self, content: str) -> bool:
        """
        Send a simple text message.
        
        Args:
            content: Message text
            
        Returns:
            True if message was sent
        """
        return self._send_webhook(content=content)
    
    def send_signal_alert(self, signal: Signal) -> bool:
        """
        Send trading signal notification with rich embed.
        
        Args:
            signal: Trading signal to notify about
            
        Returns:
            True if notification was sent
        """
        # Determine color based on signal type
        color_map = {
            SignalType.BUY: self.COLORS['buy'],
            SignalType.SELL: self.COLORS['sell'],
            SignalType.HOLD: self.COLORS['hold'],
        }
        color = color_map.get(signal.signal_type, self.COLORS['info'])
        
        # Emoji based on signal type
        emoji_map = {
            SignalType.BUY: "🟢",
            SignalType.SELL: "🔴",
            SignalType.HOLD: "🟡",
        }
        emoji = emoji_map.get(signal.signal_type, "⚪")
        
        # Build fields
        fields = [
            {"name": "Symbol", "value": signal.symbol, "inline": True},
            {"name": "Action", "value": f"{emoji} {signal.signal_type.value.upper()}", "inline": True},
            {"name": "Confidence", "value": f"{signal.confidence * 100:.1f}%", "inline": True},
            {"name": "Price", "value": f"${signal.price:.2f}", "inline": True},
        ]
        
        if signal.stop_loss:
            fields.append({"name": "Stop Loss", "value": f"${signal.stop_loss:.2f}", "inline": True})
        
        if signal.take_profit:
            fields.append({"name": "Take Profit", "value": f"${signal.take_profit:.2f}", "inline": True})
        
        if signal.reasons:
            reasons_text = "\n".join([f"• {r}" for r in signal.reasons[:5]])
            fields.append({"name": "Analysis", "value": reasons_text, "inline": False})
        
        embed = DiscordEmbed(
            title=f"{emoji} Trading Signal - {signal.symbol}",
            color=color,
            fields=fields,
            footer={"text": "AI Trader"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        return self._send_webhook(embeds=[embed])
    
    def send_trade_execution(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        order_id: Optional[str] = None,
        is_paper: bool = True,
    ) -> bool:
        """
        Send trade execution notification.
        
        Args:
            symbol: Stock symbol
            action: Trade action (buy/sell)
            quantity: Number of shares
            price: Execution price
            order_id: Order ID from broker
            is_paper: Whether this is paper trading
            
        Returns:
            True if notification was sent
        """
        color = self.COLORS['buy'] if action.lower() == "buy" else self.COLORS['sell']
        emoji = "📈" if action.lower() == "buy" else "📉"
        
        mode_badge = "📄 PAPER" if is_paper else "💵 LIVE"
        
        fields = [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Action", "value": action.upper(), "inline": True},
            {"name": "Mode", "value": mode_badge, "inline": True},
            {"name": "Quantity", "value": f"{quantity:.2f} shares", "inline": True},
            {"name": "Price", "value": f"${price:.2f}", "inline": True},
            {"name": "Total", "value": f"${quantity * price:,.2f}", "inline": True},
        ]
        
        if order_id:
            fields.append({"name": "Order ID", "value": order_id, "inline": False})
        
        embed = DiscordEmbed(
            title=f"{emoji} Trade Executed - {symbol}",
            color=color,
            fields=fields,
            footer={"text": "AI Trader"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        return self._send_webhook(embeds=[embed])
    
    def send_portfolio_update(
        self,
        positions: List[Dict[str, Any]],
        total_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        buying_power: float,
    ) -> bool:
        """
        Send portfolio summary update.
        
        Args:
            positions: List of current positions
            total_value: Total portfolio value
            daily_pnl: Daily profit/loss in dollars
            daily_pnl_pct: Daily profit/loss percentage
            buying_power: Available buying power
            
        Returns:
            True if notification was sent
        """
        color = self.COLORS['success'] if daily_pnl >= 0 else self.COLORS['error']
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        pnl_sign = "+" if daily_pnl >= 0 else ""
        
        # Build positions text
        if positions:
            positions_lines = []
            for pos in positions[:8]:  # Limit to 8 positions
                symbol = pos.get('symbol', 'N/A')
                qty = pos.get('quantity', 0)
                pnl = pos.get('unrealized_pnl', 0)
                pnl_emoji_pos = "🟢" if pnl >= 0 else "🔴"
                positions_lines.append(f"{pnl_emoji_pos} **{symbol}**: {qty} shares (${pnl:+,.2f})")
            
            if len(positions) > 8:
                positions_lines.append(f"... and {len(positions) - 8} more")
            
            positions_text = "\n".join(positions_lines)
        else:
            positions_text = "No open positions"
        
        fields = [
            {"name": "💰 Total Value", "value": f"${total_value:,.2f}", "inline": True},
            {"name": f"{pnl_emoji} Daily P/L", "value": f"{pnl_sign}${daily_pnl:,.2f} ({pnl_sign}{daily_pnl_pct:.2f}%)", "inline": True},
            {"name": "💵 Buying Power", "value": f"${buying_power:,.2f}", "inline": True},
            {"name": "📊 Positions", "value": positions_text, "inline": False},
        ]
        
        embed = DiscordEmbed(
            title="💼 Portfolio Update",
            color=color,
            fields=fields,
            footer={"text": "AI Trader"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        return self._send_webhook(embeds=[embed])
    
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
    ) -> bool:
        """
        Send daily trading summary.
        
        Args:
            date: Summary date
            total_trades: Total trades executed
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            total_pnl: Total profit/loss
            portfolio_value: End of day portfolio value
            top_performers: Best performing trades
            worst_performers: Worst performing trades
            
        Returns:
            True if notification was sent
        """
        color = self.COLORS['success'] if total_pnl >= 0 else self.COLORS['error']
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        pnl_sign = "+" if total_pnl >= 0 else ""
        
        fields = [
            {"name": "📅 Date", "value": date.strftime('%A, %B %d, %Y'), "inline": False},
            {"name": "📊 Total Trades", "value": str(total_trades), "inline": True},
            {"name": "🟢 Winning", "value": str(winning_trades), "inline": True},
            {"name": "🔴 Losing", "value": str(losing_trades), "inline": True},
            {"name": "🎯 Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            {"name": "💰 P/L", "value": f"{pnl_sign}${total_pnl:,.2f}", "inline": True},
            {"name": "💼 Portfolio", "value": f"${portfolio_value:,.2f}", "inline": True},
        ]
        
        if top_performers:
            top_text = "\n".join([f"🏆 {t['symbol']}: +${t['pnl']:.2f}" for t in top_performers[:3]])
            fields.append({"name": "Top Performers", "value": top_text, "inline": True})
        
        if worst_performers:
            worst_text = "\n".join([f"⚠️ {t['symbol']}: ${t['pnl']:.2f}" for t in worst_performers[:3]])
            fields.append({"name": "Worst Performers", "value": worst_text, "inline": True})
        
        embed = DiscordEmbed(
            title="📊 Daily Trading Summary",
            color=color,
            fields=fields,
            footer={"text": "AI Trader"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        return self._send_webhook(embeds=[embed])
    
    def send_error_alert(
        self,
        error_type: str,
        error_message: str,
        context: Optional[str] = None,
        severity: str = "error",
    ) -> bool:
        """
        Send error notification.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context
            severity: Error severity (warning/error)
            
        Returns:
            True if notification was sent
        """
        color = self.COLORS.get(severity, self.COLORS['error'])
        emoji = "⚠️" if severity == "warning" else "🚨"
        
        fields = [
            {"name": "Type", "value": error_type, "inline": True},
            {"name": "Severity", "value": severity.upper(), "inline": True},
            {"name": "Message", "value": error_message[:1024], "inline": False},
        ]
        
        if context:
            fields.append({"name": "Context", "value": context[:1024], "inline": False})
        
        embed = DiscordEmbed(
            title=f"{emoji} Error Alert",
            color=color,
            fields=fields,
            footer={"text": "AI Trader"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        return self._send_webhook(embeds=[embed])
    
    def send_backtest_result(
        self,
        strategy_name: str,
        symbol: str,
        period: str,
        total_return_pct: float,
        sharpe_ratio: float,
        max_drawdown_pct: float,
        win_rate: float,
        total_trades: int,
        profit_factor: float,
    ) -> bool:
        """
        Send backtest results notification.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Symbol tested
            period: Test period
            total_return_pct: Total return percentage
            sharpe_ratio: Sharpe ratio
            max_drawdown_pct: Maximum drawdown percentage
            win_rate: Win rate percentage
            total_trades: Total number of trades
            profit_factor: Profit factor
            
        Returns:
            True if notification was sent
        """
        color = self.COLORS['success'] if total_return_pct > 0 else self.COLORS['error']
        result_emoji = "✅" if total_return_pct > 0 else "❌"
        
        fields = [
            {"name": "Strategy", "value": strategy_name, "inline": True},
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Period", "value": period, "inline": True},
            {"name": f"{result_emoji} Total Return", "value": f"{total_return_pct:+.2f}%", "inline": True},
            {"name": "📈 Sharpe Ratio", "value": f"{sharpe_ratio:.2f}", "inline": True},
            {"name": "📉 Max Drawdown", "value": f"-{max_drawdown_pct:.2f}%", "inline": True},
            {"name": "🎯 Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            {"name": "📊 Total Trades", "value": str(total_trades), "inline": True},
            {"name": "💰 Profit Factor", "value": f"{profit_factor:.2f}", "inline": True},
        ]
        
        embed = DiscordEmbed(
            title="🔬 Backtest Results",
            color=color,
            fields=fields,
            footer={"text": "AI Trader"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        return self._send_webhook(embeds=[embed])
    
    def send_market_alert(
        self,
        title: str,
        description: str,
        alert_type: str = "info",
        fields: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Send custom market alert.
        
        Args:
            title: Alert title
            description: Alert description
            alert_type: Alert type (info/warning/error/success)
            fields: Additional fields
            
        Returns:
            True if notification was sent
        """
        color = self.COLORS.get(alert_type, self.COLORS['info'])
        
        embed = DiscordEmbed(
            title=title,
            description=description,
            color=color,
            fields=fields or [],
            footer={"text": "AI Trader"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        return self._send_webhook(embeds=[embed])
    
    def test_connection(self) -> bool:
        """Test Discord webhook connection."""
        if not self.is_configured:
            return False
        
        embed = DiscordEmbed(
            title="🤖 AI Trader Connected",
            description="Discord notifications are now active!",
            color=self.COLORS['success'],
            footer={"text": "AI Trader"},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        return self._send_webhook(embeds=[embed])


if __name__ == "__main__":
    # Test the Discord alert
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
    
    # Initialize Discord (will use env vars)
    discord = DiscordAlert()
    
    if discord.is_configured:
        print("Testing Discord connection...")
        if discord.test_connection():
            print("✅ Connection successful!")
            
            print("Sending test signal...")
            if discord.send_signal_alert(signal):
                print("✅ Signal alert sent!")
            else:
                print("❌ Failed to send signal alert")
        else:
            print("❌ Connection failed")
    else:
        print("⚠️ Discord not configured. Set DISCORD_WEBHOOK_URL")
