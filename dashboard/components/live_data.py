"""
Real-time data service for trading dashboard.

Provides live price updates, WebSocket connections, and data streaming.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class PriceData:
    """Real-time price data structure."""
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    bid: float
    ask: float
    high: float
    low: float
    open: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage."""
        if self.bid == 0:
            return 0
        return (self.spread / self.bid) * 100


@dataclass
class MarketStatus:
    """Market status information."""
    is_open: bool
    session: str
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    
    @classmethod
    def get_current(cls) -> 'MarketStatus':
        """Get current market status (US markets)."""
        now = datetime.now()
        
        # Simple market hours check (EST, 9:30 AM - 4:00 PM, Mon-Fri)
        if now.weekday() >= 5:  # Weekend
            return cls(is_open=False, session="Closed (Weekend)")
        
        # Simplified - real implementation would use pytz for EST
        hour = now.hour
        minute = now.minute
        current_minutes = hour * 60 + minute
        
        market_open = 9 * 60 + 30   # 9:30 AM
        market_close = 16 * 60      # 4:00 PM
        pre_market = 4 * 60         # 4:00 AM
        after_hours = 20 * 60       # 8:00 PM
        
        if market_open <= current_minutes < market_close:
            return cls(is_open=True, session="Regular Hours")
        elif pre_market <= current_minutes < market_open:
            return cls(is_open=False, session="Pre-Market")
        elif market_close <= current_minutes < after_hours:
            return cls(is_open=False, session="After Hours")
        else:
            return cls(is_open=False, session="Closed")


class LiveDataService:
    """
    Service for fetching and streaming live market data.
    
    Uses yfinance for real data with fallback to simulated data.
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize the live data service.
        
        Args:
            symbols: List of symbols to track
        """
        self.symbols = symbols or []
        self._price_cache: Dict[str, PriceData] = {}
        self._callbacks: List[Callable[[str, PriceData], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._update_interval = 5  # seconds
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from tracking."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
        if symbol in self._price_cache:
            del self._price_cache[symbol]
    
    def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get current price for a symbol."""
        if symbol not in self._price_cache:
            self._fetch_price(symbol)
        return self._price_cache.get(symbol)
    
    def get_all_prices(self) -> Dict[str, PriceData]:
        """Get all cached prices."""
        return self._price_cache.copy()
    
    def subscribe(self, callback: Callable[[str, PriceData], None]) -> None:
        """Subscribe to price updates."""
        self._callbacks.append(callback)
    
    def unsubscribe(self, callback: Callable[[str, PriceData], None]) -> None:
        """Unsubscribe from price updates."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def start_streaming(self) -> None:
        """Start background price streaming."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        logger.info("Started live data streaming")
    
    def stop_streaming(self) -> None:
        """Stop background price streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("Stopped live data streaming")
    
    def _stream_loop(self) -> None:
        """Background loop for fetching prices."""
        while self._running:
            for symbol in self.symbols:
                try:
                    self._fetch_price(symbol)
                    price = self._price_cache.get(symbol)
                    if price:
                        for callback in self._callbacks:
                            try:
                                callback(symbol, price)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
            
            time.sleep(self._update_interval)
    
    def _fetch_price(self, symbol: str) -> None:
        """Fetch current price for a symbol."""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            
            # Get today's data
            hist = ticker.history(period='1d', interval='1m')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                open_price = hist['Open'].iloc[0]
                high_price = hist['High'].max()
                low_price = hist['Low'].min()
                volume = int(hist['Volume'].sum())
                
                change = current_price - open_price
                change_pct = (change / open_price) * 100 if open_price > 0 else 0
                
                # Estimate bid/ask from last price
                spread = current_price * 0.001  # 0.1% spread estimate
                
                self._price_cache[symbol] = PriceData(
                    symbol=symbol,
                    price=current_price,
                    change=change,
                    change_pct=change_pct,
                    volume=volume,
                    bid=current_price - spread/2,
                    ask=current_price + spread/2,
                    high=high_price,
                    low=low_price,
                    open=open_price,
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.warning(f"yfinance error for {symbol}, using fallback: {e}")
            self._generate_simulated_price(symbol)
    
    def _generate_simulated_price(self, symbol: str) -> None:
        """Generate simulated price data for testing."""
        import random
        
        # Base prices for common symbols
        base_prices = {
            'AAPL': 175.0, 'MSFT': 370.0, 'GOOGL': 140.0,
            'AMZN': 170.0, 'TSLA': 245.0, 'META': 480.0,
            'NVDA': 850.0, 'SPY': 510.0, 'QQQ': 435.0
        }
        
        base = base_prices.get(symbol, 100.0)
        
        # Add some randomness
        price = base * (1 + random.uniform(-0.02, 0.02))
        change_pct = random.uniform(-3, 3)
        change = price * (change_pct / 100)
        
        self._price_cache[symbol] = PriceData(
            symbol=symbol,
            price=price,
            change=change,
            change_pct=change_pct,
            volume=random.randint(1000000, 50000000),
            bid=price * 0.9995,
            ask=price * 1.0005,
            high=price * 1.02,
            low=price * 0.98,
            open=price - change,
            timestamp=datetime.now()
        )


class TradingSignalQueue:
    """Queue for managing trading signals in real-time."""
    
    def __init__(self, max_size: int = 100):
        """Initialize the signal queue."""
        self.max_size = max_size
        self._signals: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def add_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        price: float,
        strategy: str,
        reasons: Optional[List[str]] = None
    ) -> None:
        """Add a new signal to the queue."""
        with self._lock:
            signal = {
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'price': price,
                'strategy': strategy,
                'reasons': reasons or [],
                'timestamp': datetime.now()
            }
            
            self._signals.insert(0, signal)
            
            # Trim if necessary
            if len(self._signals) > self.max_size:
                self._signals = self._signals[:self.max_size]
    
    def get_signals(
        self,
        limit: int = 10,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent signals with optional filtering."""
        with self._lock:
            signals = self._signals.copy()
        
        if symbol:
            signals = [s for s in signals if s['symbol'] == symbol]
        
        if signal_type:
            signals = [s for s in signals if s['signal_type'].lower() == signal_type.lower()]
        
        return signals[:limit]
    
    def get_latest(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the most recent signal."""
        signals = self.get_signals(limit=1, symbol=symbol)
        return signals[0] if signals else None
    
    def clear(self) -> None:
        """Clear all signals."""
        with self._lock:
            self._signals.clear()
    
    def count_by_type(self) -> Dict[str, int]:
        """Count signals by type."""
        with self._lock:
            counts = {'buy': 0, 'sell': 0, 'hold': 0}
            for signal in self._signals:
                sig_type = signal['signal_type'].lower()
                if sig_type in counts:
                    counts[sig_type] += 1
            return counts


class ActivityLog:
    """Activity log for tracking trades and events."""
    
    def __init__(self, max_entries: int = 500):
        """Initialize the activity log."""
        self.max_entries = max_entries
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        strategy: Optional[str] = None
    ) -> None:
        """Log a trade execution."""
        self._add_entry({
            'type': side.lower(),
            'title': f'{side.upper()} {symbol}',
            'details': f'{quantity} shares @ ${price:.2f}',
            'strategy': strategy,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'time': datetime.now()
        })
    
    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        strategy: str
    ) -> None:
        """Log a signal generation."""
        self._add_entry({
            'type': 'alert',
            'title': f'{signal_type.upper()} Signal: {symbol}',
            'details': f'{strategy} - {confidence:.0%} confidence',
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'strategy': strategy,
            'time': datetime.now()
        })
    
    def log_info(self, title: str, details: str = "") -> None:
        """Log an informational event."""
        self._add_entry({
            'type': 'info',
            'title': title,
            'details': details,
            'time': datetime.now()
        })
    
    def _add_entry(self, entry: Dict[str, Any]) -> None:
        """Add an entry to the log."""
        with self._lock:
            self._entries.insert(0, entry)
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[:self.max_entries]
    
    def get_entries(
        self,
        limit: int = 20,
        entry_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get log entries with optional filtering."""
        with self._lock:
            entries = self._entries.copy()
        
        if entry_type:
            entries = [e for e in entries if e.get('type') == entry_type]
        
        return entries[:limit]
    
    def get_trades_today(self) -> List[Dict[str, Any]]:
        """Get today's trades."""
        today = datetime.now().date()
        with self._lock:
            return [
                e for e in self._entries
                if e.get('type') in ('buy', 'sell') 
                and e.get('time', datetime.min).date() == today
            ]


# Singleton instances for app-wide use
_live_data_service: Optional[LiveDataService] = None
_signal_queue: Optional[TradingSignalQueue] = None
_activity_log: Optional[ActivityLog] = None


def get_live_data_service() -> LiveDataService:
    """Get the global live data service instance."""
    global _live_data_service
    if _live_data_service is None:
        _live_data_service = LiveDataService()
    return _live_data_service


def get_signal_queue() -> TradingSignalQueue:
    """Get the global signal queue instance."""
    global _signal_queue
    if _signal_queue is None:
        _signal_queue = TradingSignalQueue()
    return _signal_queue


def get_activity_log() -> ActivityLog:
    """Get the global activity log instance."""
    global _activity_log
    if _activity_log is None:
        _activity_log = ActivityLog()
    return _activity_log
