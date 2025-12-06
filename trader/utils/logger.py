"""
Centralized logging configuration for AI Trader.

Provides structured logging with multiple handlers (console, file, JSON),
log rotation, and trading-specific log formatters.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class LogConfig:
    """Logging configuration settings."""
    
    level: str = "INFO"
    log_dir: str = "logs"
    console_enabled: bool = True
    file_enabled: bool = True
    json_enabled: bool = False
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format_string: str = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["data"] = getattr(record, "extra_data")
        
        return json.dumps(log_data)


class TradingFormatter(logging.Formatter):
    """Custom formatter with color support for trading logs."""
    
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET": "\033[0m",
    }
    
    TRADE_COLORS = {
        "BUY": "\033[32m",        # Green
        "SELL": "\033[31m",       # Red
        "HOLD": "\033[33m",       # Yellow
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, 
                 use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
            
            # Colorize trade signals in message
            message = record.getMessage()
            for trade_type, trade_color in self.TRADE_COLORS.items():
                if trade_type in message:
                    message = message.replace(
                        trade_type, 
                        f"{trade_color}{trade_type}{self.COLORS['RESET']}"
                    )
            record.msg = message
            record.args = ()
        
        return super().format(record)


class TradingLogger:
    """
    Centralized logger for AI Trader with trading-specific features.
    
    Features:
    - Multiple output handlers (console, file, JSON)
    - Log rotation
    - Trade signal logging
    - Performance metrics logging
    - Context-aware logging
    """
    
    _instance: Optional['TradingLogger'] = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls, config: Optional[LogConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[LogConfig] = None):
        if self._initialized:
            return
        
        self.config = config or LogConfig()
        self._setup_log_directory()
        self._setup_root_logger()
        self._initialized = True
    
    def _setup_log_directory(self) -> None:
        """Create log directory if it doesn't exist."""
        log_path = Path(self.config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different log types
        (log_path / "trades").mkdir(exist_ok=True)
        (log_path / "errors").mkdir(exist_ok=True)
        (log_path / "performance").mkdir(exist_ok=True)
    
    def _setup_root_logger(self) -> None:
        """Configure the root logger with handlers."""
        root_logger = logging.getLogger("trader")
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        root_logger.handlers.clear()
        
        # Console handler
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_formatter = TradingFormatter(
                fmt=self.config.format_string,
                datefmt=self.config.date_format,
                use_colors=True
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.file_enabled:
            log_file = Path(self.config.log_dir) / "trader.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                fmt=self.config.format_string,
                datefmt=self.config.date_format
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        if self.config.json_enabled:
            json_file = Path(self.config.log_dir) / "trader.json"
            json_handler = logging.handlers.RotatingFileHandler(
                json_file,
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count
            )
            json_handler.setLevel(logging.INFO)
            json_handler.setFormatter(JsonFormatter())
            root_logger.addHandler(json_handler)
        
        # Error file handler
        error_file = Path(self.config.log_dir) / "errors" / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s\n%(exc_info)s",
            datefmt=self.config.date_format
        ))
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a named logger."""
        full_name = f"trader.{name}" if not name.startswith("trader.") else name
        
        if full_name not in self._loggers:
            logger = logging.getLogger(full_name)
            self._loggers[full_name] = logger
        
        return self._loggers[full_name]
    
    def log_trade(self, symbol: str, action: str, quantity: float, 
                  price: float, reason: str = "") -> None:
        """Log a trade execution with dedicated trade log file."""
        trade_logger = self.get_logger("trades")
        
        trade_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "reason": reason
        }
        
        # Log to main logger
        trade_logger.info(
            f"TRADE | {action} {quantity} {symbol} @ ${price:.2f} | "
            f"Value: ${quantity * price:,.2f} | Reason: {reason}"
        )
        
        # Append to dedicated trades log
        trades_file = Path(self.config.log_dir) / "trades" / f"trades_{datetime.now():%Y%m}.jsonl"
        with open(trades_file, "a") as f:
            f.write(json.dumps(trade_data) + "\n")
    
    def log_signal(self, symbol: str, signal_type: str, confidence: float,
                   price: float, reasons: list) -> None:
        """Log a trading signal generation."""
        signal_logger = self.get_logger("signals")
        
        signal_logger.info(
            f"SIGNAL | {signal_type} {symbol} | "
            f"Confidence: {confidence:.1%} | Price: ${price:.2f} | "
            f"Reasons: {', '.join(reasons)}"
        )
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """Log portfolio/strategy performance metrics."""
        perf_logger = self.get_logger("performance")
        
        metrics_str = " | ".join(f"{k}: {v}" for k, v in metrics.items())
        perf_logger.info(f"PERFORMANCE | {metrics_str}")
        
        # Append to dedicated performance log
        perf_file = Path(self.config.log_dir) / "performance" / f"perf_{datetime.now():%Y%m%d}.jsonl"
        with open(perf_file, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                **metrics
            }) + "\n")
    
    def log_api_call(self, service: str, endpoint: str, 
                     status: str, latency_ms: float) -> None:
        """Log external API calls for monitoring."""
        api_logger = self.get_logger("api")
        
        level = logging.INFO if status == "success" else logging.WARNING
        api_logger.log(
            level,
            f"API | {service} | {endpoint} | {status} | {latency_ms:.0f}ms"
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log an error with additional context."""
        error_logger = self.get_logger("errors")
        
        context_str = json.dumps(context, default=str)
        error_logger.error(
            f"ERROR | {type(error).__name__}: {error} | Context: {context_str}",
            exc_info=True
        )


# Module-level functions for convenience
_trading_logger: Optional[TradingLogger] = None


def setup_logging(config: Optional[LogConfig] = None) -> TradingLogger:
    """Initialize the global logging system."""
    global _trading_logger
    _trading_logger = TradingLogger(config)
    return _trading_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger, initializing if necessary."""
    global _trading_logger
    if _trading_logger is None:
        _trading_logger = TradingLogger()
    return _trading_logger.get_logger(name)


def get_trading_logger() -> TradingLogger:
    """Get the global TradingLogger instance."""
    global _trading_logger
    if _trading_logger is None:
        _trading_logger = TradingLogger()
    return _trading_logger


# Example usage and module test
if __name__ == "__main__":
    # Setup logging with custom config
    config = LogConfig(
        level="DEBUG",
        console_enabled=True,
        file_enabled=True,
        json_enabled=True
    )
    setup_logging(config)
    
    # Get various loggers
    main_logger = get_logger("main")
    strategy_logger = get_logger("strategy")
    
    # Test different log levels
    main_logger.debug("Debug message")
    main_logger.info("Info message")
    main_logger.warning("Warning message")
    main_logger.error("Error message")
    
    # Test trading-specific logging
    trading_logger = get_trading_logger()
    
    trading_logger.log_signal(
        symbol="AAPL",
        signal_type="BUY",
        confidence=0.85,
        price=150.00,
        reasons=["RSI oversold", "MACD crossover"]
    )
    
    trading_logger.log_trade(
        symbol="AAPL",
        action="BUY",
        quantity=100,
        price=150.00,
        reason="Strong buy signal"
    )
    
    trading_logger.log_performance({
        "total_return": "15.5%",
        "sharpe_ratio": 1.85,
        "max_drawdown": "-8.2%"
    })
    
    trading_logger.log_api_call(
        service="yfinance",
        endpoint="get_stock_data",
        status="success",
        latency_ms=245.5
    )
    
    print("\n✅ Logging system test complete! Check logs/ directory.")
