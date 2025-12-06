"""
Trading Scheduler for automated signal generation and execution.
Supports scheduling based on market hours and custom intervals.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
import schedule

from trader.data.fetcher import DataFetcher
from trader.strategies.base import Signal, SignalType, TradingStrategy
from trader.strategies.technical import TechnicalStrategy
from trader.strategies.momentum import MomentumStrategy
from trader.strategies.mean_reversion import MeanReversionStrategy
from trader.strategies.breakout import BreakoutStrategy
from trader.execution.paper_trading import PaperTradingExecutor
from trader.alerts.alert_manager import AlertManager

logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """Market session types."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


@dataclass
class MarketHours:
    """Market hours configuration."""
    market_open: str = "09:30"  # EST
    market_close: str = "16:00"  # EST
    pre_market_open: str = "04:00"
    after_hours_close: str = "20:00"
    timezone: str = "America/New_York"
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    
    def is_market_open(self) -> bool:
        """Check if regular market is open."""
        now = datetime.now()
        
        # Check if it's a trading day
        if now.weekday() not in self.trading_days:
            return False
        
        # Parse times
        open_time = datetime.strptime(self.market_open, "%H:%M").time()
        close_time = datetime.strptime(self.market_close, "%H:%M").time()
        current_time = now.time()
        
        return open_time <= current_time <= close_time
    
    def get_session(self) -> MarketSession:
        """Get current market session."""
        now = datetime.now()
        
        # Check if it's a trading day
        if now.weekday() not in self.trading_days:
            return MarketSession.CLOSED
        
        current_time = now.time()
        pre_market = datetime.strptime(self.pre_market_open, "%H:%M").time()
        market_open = datetime.strptime(self.market_open, "%H:%M").time()
        market_close = datetime.strptime(self.market_close, "%H:%M").time()
        after_hours = datetime.strptime(self.after_hours_close, "%H:%M").time()
        
        if pre_market <= current_time < market_open:
            return MarketSession.PRE_MARKET
        elif market_open <= current_time <= market_close:
            return MarketSession.REGULAR
        elif market_close < current_time <= after_hours:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED
    
    def time_until_open(self) -> Optional[timedelta]:
        """Get time until market opens."""
        if self.is_market_open():
            return timedelta(0)
        
        now = datetime.now()
        open_time = datetime.strptime(self.market_open, "%H:%M").time()
        
        # If before market open today
        today_open = datetime.combine(now.date(), open_time)
        if now.time() < open_time and now.weekday() in self.trading_days:
            return today_open - now
        
        # Find next trading day
        days_ahead = 1
        while True:
            next_day = now + timedelta(days=days_ahead)
            if next_day.weekday() in self.trading_days:
                next_open = datetime.combine(next_day.date(), open_time)
                return next_open - now
            days_ahead += 1
            if days_ahead > 7:
                return None


@dataclass
class ScheduleConfig:
    """Scheduler configuration."""
    watchlist: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL'])
    strategy_name: str = 'technical'
    scan_interval_minutes: int = 15
    enable_trading: bool = False  # Safety: disabled by default
    min_confidence: float = 0.7
    max_positions: int = 5
    position_size_pct: float = 0.1  # 10% of portfolio per position
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    only_during_market_hours: bool = True
    send_alerts: bool = True
    paper_trading: bool = True  # Always paper trading


class TradingScheduler:
    """
    Automated trading scheduler.
    
    Features:
    - Scheduled signal scanning
    - Automatic trade execution (paper trading)
    - Market hours awareness
    - Alert notifications
    - Multiple strategy support
    """
    
    STRATEGIES = {
        'technical': TechnicalStrategy,
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'breakout': BreakoutStrategy,
    }
    
    def __init__(
        self,
        config: Optional[ScheduleConfig] = None,
        market_hours: Optional[MarketHours] = None,
    ):
        """
        Initialize trading scheduler.
        
        Args:
            config: Scheduler configuration
            market_hours: Market hours configuration
        """
        self.config = config or ScheduleConfig()
        self.market_hours = market_hours or MarketHours()
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.alert_manager = AlertManager()
        self.executor = PaperTradingExecutor() if self.config.paper_trading else None
        
        # Initialize strategy
        strategy_class = self.STRATEGIES.get(
            self.config.strategy_name.lower(),
            TechnicalStrategy
        )
        self.strategy = strategy_class(
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
        )
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_scan: Optional[datetime] = None
        self._signals_history: List[Dict[str, Any]] = []
        self._callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable[[Signal], None]):
        """Add callback for signal events."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, signal: Signal):
        """Notify all callbacks of a new signal."""
        for callback in self._callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def scan_watchlist(self) -> List[Signal]:
        """
        Scan watchlist for trading signals.
        
        Returns:
            List of signals for stocks in watchlist
        """
        signals = []
        
        # Check market hours if required
        if self.config.only_during_market_hours:
            if not self.market_hours.is_market_open():
                logger.info("Market closed, skipping scan")
                return signals
        
        logger.info(f"Scanning {len(self.config.watchlist)} stocks...")
        
        for symbol in self.config.watchlist:
            try:
                # Fetch data
                df = self.data_fetcher.get_stock_data(symbol, period='3mo')
                
                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Generate signal
                signal = self.strategy.generate_signal(symbol, df)
                signals.append(signal)
                
                # Record in history
                self._signals_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'signal': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'price': signal.price,
                })
                
                # Notify callbacks
                self._notify_callbacks(signal)
                
                # Check if actionable
                if self._is_actionable_signal(signal):
                    logger.info(f"Actionable signal: {symbol} - {signal.signal_type.value} ({signal.confidence:.1%})")
                    
                    # Send alert
                    if self.config.send_alerts:
                        self.alert_manager.send_signal(signal)
                    
                    # Execute trade if enabled
                    if self.config.enable_trading and self.executor:
                        self._execute_signal(signal)
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        self._last_scan = datetime.now()
        logger.info(f"Scan complete. Found {len(signals)} signals.")
        
        return signals
    
    def _is_actionable_signal(self, signal: Signal) -> bool:
        """Check if signal is actionable."""
        if signal.signal_type == SignalType.HOLD:
            return False
        
        if signal.confidence < self.config.min_confidence:
            return False
        
        return True
    
    def _execute_signal(self, signal: Signal):
        """Execute a trading signal."""
        if not self.executor:
            logger.warning("No executor configured")
            return
        
        try:
            # Check position limits
            positions = self.executor.get_positions()
            if len(positions) >= self.config.max_positions:
                logger.info("Maximum positions reached, skipping trade")
                return
            
            # Calculate position size
            account = self.executor.get_account()
            buying_power = float(account.get('buying_power', 0))
            position_value = buying_power * self.config.position_size_pct
            quantity = int(position_value / signal.price)
            
            if quantity <= 0:
                logger.warning("Insufficient funds for trade")
                return
            
            # Submit order
            if signal.signal_type == SignalType.BUY:
                order = self.executor.submit_order(
                    symbol=signal.symbol,
                    qty=quantity,
                    side='buy',
                    order_type='market',
                )
                
                if order:
                    logger.info(f"BUY order submitted: {signal.symbol} x {quantity}")
                    self.alert_manager.send_trade_execution(
                        signal.symbol, 'BUY', quantity, signal.price, 
                        order.id
                    )
            
            elif signal.signal_type == SignalType.SELL:
                # Check if we have position to sell
                for pos in positions:
                    if pos.symbol == signal.symbol:
                        qty = pos.qty
                        if qty > 0:
                            order = self.executor.submit_order(
                                symbol=signal.symbol,
                                qty=qty,
                                side='sell',
                                order_type='market',
                            )
                            
                            if order:
                                logger.info(f"SELL order submitted: {signal.symbol} x {qty}")
                                self.alert_manager.send_trade_execution(
                                    signal.symbol, 'SELL', qty, signal.price,
                                    order.id
                                )
                        break
        
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            self.alert_manager.send_error('TradeError', str(e), signal.symbol)
    
    def _run_scheduler(self):
        """Internal scheduler loop."""
        # Schedule the scan job
        schedule.every(self.config.scan_interval_minutes).minutes.do(self.scan_watchlist)
        
        # Run initial scan
        self.scan_watchlist()
        
        while self._running:
            schedule.run_pending()
            time.sleep(1)
    
    def start(self):
        """Start the scheduler in background thread."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        logger.info("Starting trading scheduler...")
        self._running = True
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        
        logger.info(f"Scheduler started. Scanning every {self.config.scan_interval_minutes} minutes.")
        logger.info(f"Watchlist: {', '.join(self.config.watchlist)}")
        logger.info(f"Strategy: {self.config.strategy_name}")
        logger.info(f"Trading enabled: {self.config.enable_trading}")
    
    def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return
        
        logger.info("Stopping trading scheduler...")
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5)
        
        schedule.clear()
        logger.info("Scheduler stopped.")
    
    def run_once(self) -> List[Signal]:
        """Run a single scan without starting scheduler."""
        return self.scan_watchlist()
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'running': self._running,
            'last_scan': self._last_scan.isoformat() if self._last_scan else None,
            'market_session': self.market_hours.get_session().value,
            'market_open': self.market_hours.is_market_open(),
            'watchlist_count': len(self.config.watchlist),
            'strategy': self.config.strategy_name,
            'trading_enabled': self.config.enable_trading,
            'signals_today': len([s for s in self._signals_history 
                                  if s['timestamp'].date() == datetime.now().date()]),
        }
    
    def get_signals_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent signals history."""
        return self._signals_history[-limit:]
    
    def update_watchlist(self, symbols: List[str]):
        """Update watchlist."""
        self.config.watchlist = symbols
        logger.info(f"Watchlist updated: {', '.join(symbols)}")
    
    def update_strategy(self, strategy_name: str):
        """Update trading strategy."""
        if strategy_name.lower() not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_class = self.STRATEGIES[strategy_name.lower()]
        self.strategy = strategy_class(
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
        )
        self.config.strategy_name = strategy_name.lower()
        logger.info(f"Strategy updated: {strategy_name}")


class SchedulerManager:
    """Manager for multiple schedulers (e.g., different strategies)."""
    
    def __init__(self):
        self.schedulers: Dict[str, TradingScheduler] = {}
    
    def create_scheduler(
        self,
        name: str,
        config: ScheduleConfig,
    ) -> TradingScheduler:
        """Create and register a new scheduler."""
        scheduler = TradingScheduler(config=config)
        self.schedulers[name] = scheduler
        return scheduler
    
    def get_scheduler(self, name: str) -> Optional[TradingScheduler]:
        """Get scheduler by name."""
        return self.schedulers.get(name)
    
    def start_all(self):
        """Start all schedulers."""
        for name, scheduler in self.schedulers.items():
            logger.info(f"Starting scheduler: {name}")
            scheduler.start()
    
    def stop_all(self):
        """Stop all schedulers."""
        for name, scheduler in self.schedulers.items():
            logger.info(f"Stopping scheduler: {name}")
            scheduler.stop()
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all schedulers."""
        return {name: sched.get_status() for name, sched in self.schedulers.items()}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create config
    config = ScheduleConfig(
        watchlist=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        strategy_name='technical',
        scan_interval_minutes=5,
        enable_trading=False,  # Safety: disabled
        min_confidence=0.7,
        send_alerts=True,
    )
    
    # Create scheduler
    scheduler = TradingScheduler(config=config)
    
    # Add callback
    def on_signal(signal: Signal):
        print(f"Signal: {signal.symbol} - {signal.signal_type.value} ({signal.confidence:.1%})")
    
    scheduler.add_callback(on_signal)
    
    # Run single scan
    print("Running single scan...")
    signals = scheduler.run_once()
    
    print(f"\nFound {len(signals)} signals:")
    for signal in signals:
        print(f"  {signal.symbol}: {signal.signal_type.value} - {signal.confidence:.1%}")
    
    # Or start continuous scheduler
    # scheduler.start()
    # time.sleep(60)  # Run for 1 minute
    # scheduler.stop()
