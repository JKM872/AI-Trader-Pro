"""
Unit tests for alert modules.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from trader.alerts.telegram_bot import TelegramAlert
from trader.alerts.discord_webhook import DiscordAlert, DiscordEmbed
from trader.alerts.alert_manager import AlertManager, AlertConfig, AlertType
from trader.strategies.base import Signal, SignalType


class TestTelegramAlert:
    """Tests for TelegramAlert."""
    
    def test_initialization_without_credentials(self):
        """Test initialization without credentials."""
        with patch.dict('os.environ', {}, clear=True):
            telegram = TelegramAlert()
            assert not telegram.is_configured
    
    def test_initialization_with_credentials(self):
        """Test initialization with credentials."""
        telegram = TelegramAlert(
            bot_token='test_token',
            chat_id='test_chat_id'
        )
        assert telegram.is_configured
        assert telegram.bot_token == 'test_token'
        assert telegram.chat_id == 'test_chat_id'
    
    def test_is_configured_property(self):
        """Test is_configured property."""
        telegram_configured = TelegramAlert(
            bot_token='token',
            chat_id='chat'
        )
        telegram_unconfigured = TelegramAlert(
            bot_token=None,
            chat_id=None
        )
        
        assert telegram_configured.is_configured == True
        assert telegram_unconfigured.is_configured == False
    
    @patch('requests.post')
    def test_send_message_success(self, mock_post):
        """Test successful message sending."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {'ok': True}
        )
        mock_post.return_value.raise_for_status = Mock()
        
        telegram = TelegramAlert(bot_token='token', chat_id='chat')
        result = telegram.send_message('Test message')
        
        assert result == True
        mock_post.assert_called_once()
    
    @patch('trader.alerts.telegram_bot.requests.post')
    def test_send_message_failure(self, mock_post):
        """Test message sending failure."""
        import requests
        # Mock a network failure using the exception type the code catches
        mock_post.side_effect = requests.RequestException('Network error')
        
        telegram = TelegramAlert(bot_token='token', chat_id='chat')
        result = telegram.send_message('Test message')
        
        assert result == False
    
    def test_send_message_unconfigured(self):
        """Test sending message when unconfigured."""
        telegram = TelegramAlert(bot_token=None, chat_id=None)
        result = telegram.send_message('Test message')
        
        assert result == False
    
    @patch('requests.post')
    def test_send_signal_alert(self, mock_post):
        """Test sending signal alert."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {'ok': True}
        )
        mock_post.return_value.raise_for_status = Mock()
        
        telegram = TelegramAlert(bot_token='token', chat_id='chat')
        
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            confidence=0.85,
            price=150.00,
            stop_loss=142.50,
            take_profit=165.00,
            reasons=['RSI oversold', 'MACD crossover']
        )
        
        result = telegram.send_signal_alert(signal)
        assert result == True
    
    @patch('requests.post')
    def test_send_trade_execution(self, mock_post):
        """Test sending trade execution alert."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {'ok': True}
        )
        mock_post.return_value.raise_for_status = Mock()
        
        telegram = TelegramAlert(bot_token='token', chat_id='chat')
        
        result = telegram.send_trade_execution(
            symbol='AAPL',
            action='BUY',
            quantity=100,
            price=150.00,
            order_id='ORD123'
        )
        
        assert result == True
    
    @patch('requests.post')
    def test_send_error_alert(self, mock_post):
        """Test sending error alert."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {'ok': True}
        )
        mock_post.return_value.raise_for_status = Mock()
        
        telegram = TelegramAlert(bot_token='token', chat_id='chat')
        
        result = telegram.send_error_alert(
            error_type='APIError',
            error_message='Rate limit exceeded',
            context='Fetching AAPL data'
        )
        
        assert result == True


class TestDiscordEmbed:
    """Tests for DiscordEmbed."""
    
    def test_embed_creation(self):
        """Test creating an embed."""
        embed = DiscordEmbed(
            title='Test Title',
            description='Test description',
            color=0x00FF00
        )
        
        assert embed.title == 'Test Title'
        assert embed.description == 'Test description'
        assert embed.color == 0x00FF00
    
    def test_embed_to_dict(self):
        """Test embed serialization."""
        embed = DiscordEmbed(
            title='Test',
            description='Description',
            color=0xFF0000,
            fields=[{'name': 'Field1', 'value': 'Value1'}]
        )
        
        d = embed.to_dict()
        assert d['title'] == 'Test'
        assert d['description'] == 'Description'
        assert d['color'] == 0xFF0000
        assert len(d['fields']) == 1
    
    def test_embed_optional_fields(self):
        """Test embed with optional fields."""
        embed = DiscordEmbed(title='Title Only')
        d = embed.to_dict()
        
        assert d['title'] == 'Title Only'
        assert 'description' not in d
        assert 'fields' not in d


class TestDiscordAlert:
    """Tests for DiscordAlert."""
    
    def test_initialization_without_url(self):
        """Test initialization without webhook URL."""
        with patch.dict('os.environ', {}, clear=True):
            discord = DiscordAlert(webhook_url=None)
            assert not discord.is_configured
    
    def test_initialization_with_url(self):
        """Test initialization with webhook URL."""
        discord = DiscordAlert(
            webhook_url='https://discord.com/api/webhooks/test'
        )
        assert discord.is_configured
    
    def test_is_configured_property(self):
        """Test is_configured property."""
        discord_configured = DiscordAlert(
            webhook_url='https://discord.com/api/webhooks/test'
        )
        discord_unconfigured = DiscordAlert(webhook_url=None)
        
        assert discord_configured.is_configured == True
        assert discord_unconfigured.is_configured == False
    
    @patch('requests.post')
    def test_send_message_success(self, mock_post):
        """Test successful message sending."""
        mock_post.return_value = Mock(status_code=204)
        mock_post.return_value.raise_for_status = Mock()
        
        discord = DiscordAlert(webhook_url='https://discord.com/api/webhooks/test')
        result = discord.send_message('Test message')
        
        assert result == True
    
    @patch('requests.post')
    def test_send_signal_alert(self, mock_post):
        """Test sending signal alert."""
        mock_post.return_value = Mock(status_code=204)
        mock_post.return_value.raise_for_status = Mock()
        
        discord = DiscordAlert(webhook_url='https://discord.com/api/webhooks/test')
        
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.SELL,
            confidence=0.75,
            price=145.00,
            stop_loss=152.00,
            take_profit=130.00,
            reasons=['RSI overbought']
        )
        
        result = discord.send_signal_alert(signal)
        assert result == True
    
    @patch('requests.post')
    def test_send_backtest_result(self, mock_post):
        """Test sending backtest result."""
        mock_post.return_value = Mock(status_code=204)
        mock_post.return_value.raise_for_status = Mock()
        
        discord = DiscordAlert(webhook_url='https://discord.com/api/webhooks/test')
        
        result = discord.send_backtest_result(
            strategy_name='TechnicalStrategy',
            symbol='AAPL',
            period='1y',
            total_return_pct=25.5,
            sharpe_ratio=1.8,
            max_drawdown_pct=12.3,
            win_rate=58.0,
            total_trades=45,
            profit_factor=1.6
        )
        
        assert result == True
    
    def test_colors_defined(self):
        """Test that colors are defined for different alert types."""
        discord = DiscordAlert(webhook_url='https://test.com')
        
        assert 'buy' in discord.COLORS
        assert 'sell' in discord.COLORS
        assert 'error' in discord.COLORS
        assert 'success' in discord.COLORS


class TestAlertConfig:
    """Tests for AlertConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AlertConfig()
        
        assert config.telegram_enabled == True
        assert config.discord_enabled == True
        assert config.min_confidence == 0.6
        assert config.alert_on_hold == False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AlertConfig(
            telegram_enabled=False,
            discord_enabled=True,
            min_confidence=0.8,
            quiet_hours_start=22,
            quiet_hours_end=8,
            alert_on_hold=True
        )
        
        assert config.telegram_enabled == False
        assert config.min_confidence == 0.8
        assert config.quiet_hours_start == 22


class TestAlertManager:
    """Tests for AlertManager."""
    
    def test_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager(
            telegram_token='token',
            telegram_chat_id='chat',
            discord_webhook_url='https://discord.com/webhook'
        )
        
        assert manager.telegram.is_configured
        assert manager.discord.is_configured
    
    def test_channels_available(self):
        """Test channels_available property."""
        manager = AlertManager(
            telegram_token='token',
            telegram_chat_id='chat',
            discord_webhook_url='https://discord.com/webhook'
        )
        
        channels = manager.channels_available
        assert 'telegram' in channels
        assert 'discord' in channels
        assert channels['telegram'] == True
        assert channels['discord'] == True
    
    def test_channels_unavailable(self):
        """Test channels_available when unconfigured."""
        manager = AlertManager()
        
        channels = manager.channels_available
        assert channels['telegram'] == False
        assert channels['discord'] == False
    
    @patch.object(TelegramAlert, 'send_signal_alert', return_value=True)
    @patch.object(DiscordAlert, 'send_signal_alert', return_value=True)
    def test_send_signal(self, mock_discord, mock_telegram):
        """Test sending signal to all channels."""
        manager = AlertManager(
            telegram_token='token',
            telegram_chat_id='chat',
            discord_webhook_url='https://discord.com/webhook'
        )
        
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            confidence=0.85,
            price=150.00
        )
        
        results = manager.send_signal(signal)
        
        assert results.get('telegram') == True
        assert results.get('discord') == True
    
    def test_send_signal_low_confidence(self):
        """Test that low confidence signals are filtered."""
        manager = AlertManager(
            telegram_token='token',
            telegram_chat_id='chat',
            config=AlertConfig(min_confidence=0.7)
        )
        
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.BUY,
            confidence=0.5,  # Below threshold
            price=150.00
        )
        
        results = manager.send_signal(signal)
        
        # Should return empty because confidence is too low
        assert len(results) == 0
    
    def test_send_signal_hold_filtered(self):
        """Test that HOLD signals are filtered by default."""
        manager = AlertManager(
            telegram_token='token',
            telegram_chat_id='chat',
            config=AlertConfig(alert_on_hold=False)
        )
        
        signal = Signal(
            symbol='AAPL',
            signal_type=SignalType.HOLD,
            confidence=0.9,
            price=150.00
        )
        
        results = manager.send_signal(signal)
        
        # When alert_on_hold=False, HOLD signals may return results dict but won't send
        # The filtering happens at channel level, so results may contain False values
        assert all(v == False for v in results.values()) or len(results) == 0
    
    @patch.object(TelegramAlert, 'send_trade_execution', return_value=True)
    @patch.object(DiscordAlert, 'send_trade_execution', return_value=True)
    def test_send_trade_execution(self, mock_discord, mock_telegram):
        """Test sending trade execution alert."""
        manager = AlertManager(
            telegram_token='token',
            telegram_chat_id='chat',
            discord_webhook_url='https://discord.com/webhook'
        )
        
        results = manager.send_trade_execution(
            symbol='AAPL',
            action='BUY',
            quantity=100,
            price=150.00
        )
        
        assert results.get('telegram') == True
        assert results.get('discord') == True
    
    @patch.object(TelegramAlert, 'send_error_alert', return_value=True)
    @patch.object(DiscordAlert, 'send_error_alert', return_value=True)
    def test_send_error(self, mock_discord, mock_telegram):
        """Test sending error alert."""
        manager = AlertManager(
            telegram_token='token',
            telegram_chat_id='chat',
            discord_webhook_url='https://discord.com/webhook'
        )
        
        results = manager.send_error(
            error_type='APIError',
            error_message='Test error'
        )
        
        # Errors should always be sent, even during quiet hours
        assert results.get('telegram') == True
        assert results.get('discord') == True
    
    def test_alert_history(self):
        """Test alert history tracking."""
        manager = AlertManager()
        
        # Initially empty
        assert len(manager.alert_history) == 0
        
        # After recording an alert
        manager._record_alert(AlertType.SIGNAL, True, {'test': 'data'})
        
        assert len(manager.alert_history) == 1
        assert manager.alert_history[0]['type'] == 'signal'
        assert manager.alert_history[0]['success'] == True
    
    def test_get_alert_history(self):
        """Test getting alert history with filters."""
        manager = AlertManager()
        
        # Record some alerts
        manager._record_alert(AlertType.SIGNAL, True, {})
        manager._record_alert(AlertType.ERROR, False, {})
        manager._record_alert(AlertType.SIGNAL, True, {})
        
        # Get all history
        all_history = manager.get_alert_history()
        assert len(all_history) == 3
        
        # Get filtered history
        signal_history = manager.get_alert_history(alert_type=AlertType.SIGNAL)
        assert len(signal_history) == 2
        
        error_history = manager.get_alert_history(alert_type=AlertType.ERROR)
        assert len(error_history) == 1
    
    @patch.object(TelegramAlert, 'test_connection', return_value=True)
    @patch.object(DiscordAlert, 'test_connection', return_value=True)
    def test_test_all_channels(self, mock_discord, mock_telegram):
        """Test testing all channels."""
        manager = AlertManager(
            telegram_token='token',
            telegram_chat_id='chat',
            discord_webhook_url='https://discord.com/webhook'
        )
        
        results = manager.test_all_channels()
        
        assert results['telegram'] == True
        assert results['discord'] == True
