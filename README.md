# AI Trader

🤖 Python-based AI trading system with paper trading simulation. Uses Deepseek/Groq AI for market analysis combined with technical indicators for trading decisions.

⚠️ **IMPORTANT**: This is for educational and paper trading purposes only. Never risk real money without understanding the risks.

## Features

- 📊 **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages, ATR
- 🧠 **AI-Powered Analysis**: Sentiment analysis via Deepseek/Groq
- 📈 **4 Trading Strategies**: Technical, Momentum, Mean Reversion, Breakout
- 🔬 **Backtesting**: Historical strategy testing with comprehensive metrics
- 💼 **Portfolio Management**: Position tracking, P/L calculation, risk metrics
- ⚡ **Risk Management**: Position sizing, VaR, drawdown limits
- ⏰ **Automated Scheduling**: Market-hours-aware trading scheduler
- 💰 **Paper Trading**: Safe simulation via Alpaca Paper Trading API
- 📱 **Dashboard**: 4-page Streamlit visualization (Stock, Comparison, Backtest, Portfolio)
- 🔔 **Alerts**: Telegram & Discord notifications

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Required keys:
- `DEEPSEEK_API_KEY` or `GROQ_API_KEY` - For AI analysis (at least one)
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` - For paper trading (optional)
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` - For Telegram alerts (optional)
- `DISCORD_WEBHOOK_URL` - For Discord alerts (optional)

### 3. Run Commands

```bash
# Show help
python main.py --help

# Analyze a stock
python main.py analyze AAPL --strategy technical

# Run backtest
python main.py backtest AAPL --strategy momentum --period 1y

# Show portfolio
python main.py portfolio show

# Start automated scheduler
python main.py schedule AAPL MSFT GOOGL --interval 15

# Launch dashboard
python main.py dashboard

# Test alerts
python main.py alerts --telegram --discord
```

### 4. Generate Trading Signals (Python)

```python
from trader.data.fetcher import DataFetcher
from trader.strategies.technical import TechnicalStrategy

# Fetch data
fetcher = DataFetcher()
data = fetcher.get_stock_data('AAPL', period='6mo')

# Generate signal
strategy = TechnicalStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)
signal = strategy.generate_signal('AAPL', data)

print(f"Signal: {signal.signal_type.value}")
print(f"Confidence: {signal.confidence:.1%}")
print(f"Stop Loss: ${signal.stop_loss:.2f}")
print(f"Take Profit: ${signal.take_profit:.2f}")
```

### 5. Run Backtest

```python
from trader.backtest.backtester import Backtester
from trader.strategies.momentum import MomentumStrategy
import yfinance as yf

data = yf.Ticker("AAPL").history(period="2y")
backtester = Backtester(initial_capital=100000, commission=0.001)
strategy = MomentumStrategy(stop_loss_pct=0.05, take_profit_pct=0.10)

result = backtester.run(strategy, data, "AAPL")

print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
print(f"Win Rate: {result.metrics.win_rate:.1f}%")
```

## Project Structure

```
trader/
├── data/                   # Data fetching
│   └── fetcher.py          # DataFetcher (yfinance + caching)
├── analysis/               # AI analysis
│   └── ai_analyzer.py      # Deepseek/Groq sentiment
├── strategies/             # Trading strategies
│   ├── base.py             # Signal dataclass, TradingStrategy ABC
│   ├── technical.py        # RSI, MACD, Bollinger
│   ├── momentum.py         # ROC, volume, trend strength
│   ├── mean_reversion.py   # Z-score, Bollinger position
│   └── breakout.py         # Support/resistance breakouts
├── backtest/               # Backtesting engine
│   └── backtester.py       # Backtester with metrics
├── portfolio/              # Portfolio management
│   └── portfolio.py        # Position tracking, P/L
├── risk/                   # Risk management
│   └── risk_manager.py     # Position sizing, VaR, limits
├── scheduler/              # Automated trading
│   └── scheduler.py        # Market-hours scheduler
├── alerts/                 # Notifications
│   ├── alert_manager.py    # Unified alert interface
│   ├── telegram_bot.py     # Telegram integration
│   └── discord_webhook.py  # Discord integration
├── execution/              # Trade execution
│   └── paper_trading.py    # Alpaca paper trading
├── utils/                  # Utilities
│   └── logger.py           # Structured logging
├── cli.py                  # CLI interface
└── config.py               # Configuration loader
config/
└── config.yaml             # Strategy configuration
dashboard/
└── app.py                  # Streamlit dashboard (4 pages)
tests/
├── test_strategies.py      # Strategy unit tests
├── test_backtester.py      # Backtest tests
├── test_alerts.py          # Alert tests
├── test_data_fetcher.py    # Data fetcher tests
└── test_integration.py     # End-to-end tests
main.py                     # Entry point
```

## Trading Strategies

| Strategy | Indicators | Best For |
|----------|------------|----------|
| `TechnicalStrategy` | RSI, MACD, Bollinger, SMA/EMA | Trending markets |
| `MomentumStrategy` | ROC, Volume, Trend strength | Strong trends |
| `MeanReversionStrategy` | Z-score, Bollinger position | Range-bound markets |
| `BreakoutStrategy` | Support/Resistance, Volume, ATR | Volatility breakouts |

## Configuration

Edit `config/config.yaml` to customize:

```yaml
trading:
  mode: "paper"  # paper | backtest
  symbols: ["AAPL", "MSFT", "GOOGL"]
  
risk:
  stop_loss_pct: 0.05        # 5%
  take_profit_pct: 0.10      # 10%
  max_position_pct: 0.20     # 20% max per position
  risk_per_trade_pct: 0.02   # 2% risk per trade
  
strategy:
  name: "technical"
  indicators:
    rsi:
      period: 14
      oversold: 30
      overbought: 70
```

## Dashboard Pages

1. **Single Stock Analysis** - Candlestick, RSI, MACD, Bollinger, Volume
2. **Multi-Stock Comparison** - Returns comparison, correlation heatmap
3. **Backtesting** - Strategy testing, equity curve, drawdown, monthly returns
4. **Portfolio Analysis** - Allocation, performance, risk metrics

## API Rate Limits

| Service | Free Tier Limit |
|---------|-----------------|
| Alpha Vantage | 500 requests/day |
| NewsAPI | 100 requests/day |
| yfinance | Unlimited (delayed) |
| Alpaca Paper | Unlimited |
| Telegram | 30 msg/sec |
| Discord | 50 req/sec |

## Development

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_strategies.py -v

# Run with coverage
pytest tests/ --cov=trader

# Run backtest from CLI
python main.py backtest AAPL --strategy technical --period 1y --trades
```

## Risk Management Features

- **Position Sizing**: Fixed risk, Kelly criterion, volatility-adjusted
- **Risk Limits**: Max position size, daily/weekly loss limits, max drawdown
- **VaR Calculation**: 95% and 99% Value at Risk
- **Automatic Trading Halt**: When risk limits are breached

## Legal Disclaimer

This software is for educational purposes only. Trading stocks involves risk of financial loss. The authors are not responsible for any financial decisions made using this software.

Before trading:
1. Always start with paper trading
2. Understand the risks involved
3. Never invest money you cannot afford to lose
4. Check local regulations for automated trading

## License

MIT License - see LICENSE file for details.
