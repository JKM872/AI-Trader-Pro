# AI Trader - Copilot Instructions

## Project Overview

Python-based AI trading system with paper trading simulation. Uses Deepseek/Groq AI for market analysis combined with technical indicators for trading decisions. Features backtesting, multi-strategy support, risk management, scheduling, and Telegram/Discord notifications.

## Architecture

```
DataFetcher → Strategies → Signal → RiskManager → PaperTradingExecutor
     ↓            ↓           ↓          ↓              ↓
AIAnalyzer    Backtester   Portfolio  PositionSizer  AlertManager
                  ↓                                   ↓        ↓
            BacktestResult                       Telegram   Discord
                                                      ↓
                                               TradingScheduler
```

## Core Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `DataFetcher` | `trader/data/fetcher.py` | yfinance price data + fundamentals with caching |
| `AIAnalyzer` | `trader/analysis/ai_analyzer.py` | Deepseek/Groq sentiment analysis, JSON responses |
| `TechnicalStrategy` | `trader/strategies/technical.py` | RSI, MACD, Bollinger Bands, Moving Averages |
| `MomentumStrategy` | `trader/strategies/momentum.py` | Multi-timeframe momentum + volume confirmation |
| `MeanReversionStrategy` | `trader/strategies/mean_reversion.py` | Z-score based mean reversion |
| `BreakoutStrategy` | `trader/strategies/breakout.py` | Support/resistance breakout detection |
| `Backtester` | `trader/backtest/backtester.py` | Historical strategy testing with metrics |
| `PaperTradingExecutor` | `trader/execution/paper_trading.py` | Alpaca API or simulation mode |
| `AlertManager` | `trader/alerts/alert_manager.py` | Unified Telegram/Discord notifications |
| `TelegramAlert` | `trader/alerts/telegram_bot.py` | Telegram bot integration |
| `DiscordAlert` | `trader/alerts/discord_webhook.py` | Discord webhook integration |
| `Portfolio` | `trader/portfolio/portfolio.py` | Position tracking, P/L, metrics |
| `RiskManager` | `trader/risk/risk_manager.py` | Risk limits, VaR, position sizing |
| `TradingScheduler` | `trader/scheduler/scheduler.py` | Automated trading with market hours |
| `TradingLogger` | `trader/utils/logger.py` | Structured logging, trade logs |
| `Config` | `trader/config.py` | YAML + .env loader with dot notation |

## Key Patterns

### Signal Generation (ALWAYS follow this pattern)
```python
from trader.strategies.base import Signal, SignalType

# Signals MUST include: symbol, signal_type, confidence, price
# Optional but recommended: stop_loss, take_profit, reasons
signal = Signal(
    symbol='AAPL',
    signal_type=SignalType.BUY,  # BUY | SELL | HOLD
    confidence=0.75,  # 0.0 to 1.0
    price=150.00,
    stop_loss=142.50,  # price * (1 - stop_loss_pct)
    take_profit=165.00,  # price * (1 + take_profit_pct)
    reasons=['RSI oversold at 28', 'MACD bullish crossover']
)
```

### Data Fetching (with caching)
```python
from trader.data.fetcher import DataFetcher

fetcher = DataFetcher()
df = fetcher.get_stock_data('AAPL', period='6mo')  # Cached 1hr
fundamentals = fetcher.get_fundamentals('AAPL')    # Cached 1hr
```

### AI Analysis (JSON responses only)
```python
from trader.analysis.ai_analyzer import AIAnalyzer

analyzer = AIAnalyzer(provider='deepseek')  # or 'groq'
result = analyzer.analyze_sentiment("Apple beats earnings")
# Returns: SentimentResult(sentiment=Sentiment.BULLISH, score=0.8, ...)
```

### Backtesting
```python
from trader.backtest.backtester import Backtester
from trader.strategies.technical import TechnicalStrategy
import yfinance as yf

# Download data
data = yf.Ticker("AAPL").history(period="2y")

# Initialize
backtester = Backtester(
    initial_capital=100000,
    commission=0.001,
    risk_per_trade=0.02,
)

# Run backtest
strategy = TechnicalStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
result = backtester.run(strategy, data, "AAPL")

# Access metrics
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
print(f"Win Rate: {result.metrics.win_rate:.2f}%")
```

### Alerts (Telegram/Discord)
```python
from trader.alerts.alert_manager import AlertManager

# Initialize (uses env vars if not provided)
manager = AlertManager()

# Send signal alert
manager.send_signal(signal)

# Send trade execution
manager.send_trade_execution('AAPL', 'BUY', 100, 150.00)

# Send error alert
manager.send_error('APIError', 'Rate limit exceeded', severity='warning')
```

### Risk Management
```python
from trader.risk.risk_manager import RiskManager, RiskLimits

# Initialize with custom limits
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
print(f"Shares: {result.recommended_shares}, Risk: ${result.risk_amount:.2f}")

# Check if trade is allowed
approved, messages = risk_manager.check_trade(
    symbol='AAPL', action='BUY', quantity=100, 
    price=150.00, stop_loss=142.50
)
```

### Portfolio Management
```python
from trader.portfolio.portfolio import Portfolio

portfolio = Portfolio(initial_cash=100000)

# Open position
portfolio.open_position('AAPL', quantity=100, entry_price=150.00,
                        stop_loss=142.50, take_profit=165.00)

# Update and check
portfolio.update_position_price('AAPL', 155.00)
metrics = portfolio.get_metrics()
print(f"Total Value: ${metrics.total_value:,.2f}")

# Close position
pnl = portfolio.close_position('AAPL', exit_price=160.00)
```

### Scheduler (Automated Trading)
```python
from trader.scheduler.scheduler import TradingScheduler, ScheduleConfig

config = ScheduleConfig(interval_minutes=15, market_hours_only=True)
scheduler = TradingScheduler(
    watchlist=['AAPL', 'MSFT', 'GOOGL'],
    strategy_name='technical',
    config=config
)

scheduler.start()  # Runs until stopped
# scheduler.stop()
```

### Logging
```python
from trader.utils.logger import setup_logging, get_logger, get_trading_logger

# Setup once at startup
setup_logging()

# Get module logger
logger = get_logger('my_module')
logger.info("Processing...")

# Trading-specific logging
trading_logger = get_trading_logger()
trading_logger.log_trade('AAPL', 'BUY', 100, 150.00, 'Strong signal')
trading_logger.log_signal('AAPL', 'BUY', 0.85, 150.00, ['RSI oversold'])
```

### TradingView-Style Indicators
```python
from trader.analysis.indicators import TradingViewIndicators

indicators = TradingViewIndicators()

# Supertrend (trend following)
supertrend, direction = indicators.supertrend(df, period=10, multiplier=3.0)
# direction: 1 = bullish, -1 = bearish

# ADX/DMI (trend strength)
adx, plus_di, minus_di = indicators.adx_dmi(df)
# ADX > 25 = strong trend, ADX > 50 = very strong

# Ichimoku Cloud
ichimoku = indicators.ichimoku_cloud(df)
# Returns: tenkan, kijun, senkou_a, senkou_b, chikou

# Pivot Points (support/resistance)
pivots = indicators.pivot_points(df, pivot_type='fibonacci')
# Returns: pivot, r1, r2, r3, s1, s2, s3
```

### Smart Money Strategy (Institutional Trading)
```python
from trader.strategies.smart_money import SmartMoneyStrategy, MultiTimeframeStrategy

# Smart Money Concepts
smc = SmartMoneyStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
signal = smc.generate_signal('AAPL', df)
# Analyzes: Order Blocks, FVG, Market Structure, Liquidity

# Multi-Timeframe Confluence
mtf = MultiTimeframeStrategy(min_confluence=0.7)
signal = mtf.generate_signal('AAPL', df)
# Aligns HTF, MTF, LTF trends for high-probability entries
```

### Signal Scanner (Watchlist Scanning)
```python
from trader.strategies.scanner import SignalScanner, get_signal_summary

# Scan watchlist
scanner = SignalScanner(
    strategies=['Technical', 'Smart Money', 'Multi-Timeframe'],
    min_score=50
)
results = scanner.scan_watchlist(['AAPL', 'MSFT', 'GOOGL'])

# Get top signals
top_buys = scanner.get_top_signals(['AAPL', 'MSFT'], top_n=5, signal_type=SignalType.BUY)

# Quick summary
summary = get_signal_summary('AAPL')
# Returns: overall_signal, consensus, best_score, signals[]
```

## Trading Strategies

| Strategy | Indicators | Best For |
|----------|------------|----------|
| `TechnicalStrategy` | RSI, MACD, Bollinger, SMA/EMA | Trending markets |
| `MomentumStrategy` | ROC, Volume, Trend strength | Strong trends |
| `MeanReversionStrategy` | Z-score, Bollinger position | Range-bound markets |
| `BreakoutStrategy` | Support/Resistance, Volume, ATR | Volatility breakouts |
| `SmartMoneyStrategy` | Order Blocks, FVG, Structure | Institutional flow |
| `MultiTimeframeStrategy` | HTF/MTF/LTF Confluence | High-probability |

## Advanced Indicators (TradingView-style)

| Indicator | Purpose | Key Levels |
|-----------|---------|------------|
| Supertrend | Trend following + dynamic SL | Direction 1/-1 |
| ADX/DMI | Trend strength | >25 strong, >50 very strong |
| Ichimoku | Complete trading system | Cloud, Tenkan, Kijun |
| VWAP | Fair value (institutional) | Price vs VWAP |
| Squeeze Momentum | Breakout detection | Squeeze ON/OFF |
| Pivot Points | S/R levels | Standard/Fib/Camarilla |

## API Rate Limits (CRITICAL)

| Service | Limit | Implemented In |
|---------|-------|----------------|
| Alpha Vantage | 500/day | `RateLimiter` class in fetcher.py |
| NewsAPI | 100/day | `RateLimiter` class in fetcher.py |
| yfinance | Unlimited | Built-in caching |
| Telegram | 30 msg/sec | N/A (low volume) |
| Discord | 50 req/sec | N/A (low volume) |

## Development Commands

```bash
pip install -r requirements.txt         # Install all deps
python main.py --help                    # Show CLI help
python main.py analyze AAPL              # Analyze stock
python main.py backtest AAPL --period 1y # Run backtest
python main.py portfolio show            # Show portfolio
python main.py schedule AAPL MSFT        # Start scheduler
python main.py dashboard                 # Launch dashboard
python main.py alerts --telegram         # Test alerts
streamlit run dashboard/app.py           # Start dashboard directly
pytest tests/                            # Run all tests
pytest tests/test_strategies.py -v       # Run strategy tests
```

## Dashboard Features

The Streamlit dashboard (`dashboard/app.py`) includes:

1. **Single Stock Analysis**
   - Candlestick charts with SMA overlays
   - RSI, MACD indicators
   - Bollinger Bands visualization
   - Volume profile analysis
   - Fundamental data

2. **Multi-Stock Comparison**
   - Cumulative returns comparison
   - Correlation heatmap
   - Key metrics comparison

3. **Backtesting**
   - Interactive strategy testing
   - Equity curve visualization
   - Drawdown analysis
   - Monthly returns heatmap
   - Trade history

4. **Portfolio Analysis**
   - Portfolio allocation
   - Performance metrics
   - Correlation analysis

## Configuration

- **API Keys**: `.env` file (copy from `.env.example`)
- **Strategy Settings**: `config/config.yaml`
- **Access**: `from trader.config import get_config; config = get_config()`

### Required Environment Variables
```bash
# AI Providers (at least one required)
DEEPSEEK_API_KEY=your_key
GROQ_API_KEY=your_key

# Trading (optional - paper trading works without)
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret

# Alerts (optional)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
DISCORD_WEBHOOK_URL=your_webhook_url

# Data (optional)
ALPHA_VANTAGE_API_KEY=your_key
NEWS_API_KEY=your_key
```

## Constraints (NEVER violate)

1. **Paper Trading Only** - `PaperTradingExecutor` defaults to simulation mode
2. **Rate Limiting** - All external API calls use `RateLimiter` class
3. **Risk Management** - All strategies implement `stop_loss_pct` and `take_profit_pct`
4. **Signal Format** - Use `Signal` dataclass from `trader.strategies.base`
5. **AI Responses** - Always request JSON format, handle parse failures gracefully
6. **Position Sizing** - Max 20% of capital per position, 2% risk per trade

## Testing Strategy Logic

```python
import yfinance as yf
from trader.strategies.technical import TechnicalStrategy

data = yf.Ticker('AAPL').history(period='6mo')
strategy = TechnicalStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
signal = strategy.generate_signal('AAPL', data)

assert signal.confidence >= 0.0 and signal.confidence <= 1.0
assert signal.stop_loss < signal.price < signal.take_profit  # For BUY
```

## Backtest Metrics Reference

| Metric | Description |
|--------|-------------|
| `total_return` | Absolute P/L in dollars |
| `total_return_pct` | Percentage return |
| `sharpe_ratio` | Risk-adjusted return (>1.0 good, >2.0 excellent) |
| `sortino_ratio` | Sharpe using only downside volatility |
| `max_drawdown_pct` | Largest peak-to-trough decline |
| `win_rate` | Percentage of winning trades |
| `profit_factor` | Gross profit / Gross loss (>1.5 good) |
| `calmar_ratio` | Annualized return / Max drawdown |
