"""
Command Line Interface for AI Trader.

Provides commands for:
- Running trading strategies
- Backtesting
- Portfolio management
- Starting the scheduler
- Launching the dashboard
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Optional
import json

# Defer imports to avoid circular dependencies
def get_config():
    from trader.config import get_config
    return get_config()


def setup_logging_cli(verbose: bool = False):
    """Setup logging for CLI."""
    from trader.utils.logger import setup_logging, LogConfig
    
    config = LogConfig(
        level="DEBUG" if verbose else "INFO",
        console_enabled=True,
        file_enabled=True
    )
    return setup_logging(config)


def cmd_analyze(args):
    """Analyze a stock and generate trading signal."""
    from trader.data.fetcher import DataFetcher
    from trader.strategies.technical import TechnicalStrategy
    from trader.strategies.momentum import MomentumStrategy
    from trader.strategies.mean_reversion import MeanReversionStrategy
    from trader.strategies.breakout import BreakoutStrategy
    from trader.analysis.ai_analyzer import AIAnalyzer
    from trader.utils.logger import get_logger
    
    logger = get_logger("cli.analyze")
    
    # Strategy mapping
    strategies = {
        'technical': TechnicalStrategy,
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'breakout': BreakoutStrategy
    }
    
    if args.strategy not in strategies:
        print(f"❌ Unknown strategy: {args.strategy}")
        print(f"Available strategies: {', '.join(strategies.keys())}")
        return 1
    
    print(f"\n📊 Analyzing {args.symbol} with {args.strategy} strategy...")
    
    try:
        # Fetch data
        fetcher = DataFetcher()
        data = fetcher.get_stock_data(args.symbol, period=args.period)
        
        if data is None or data.empty:
            print(f"❌ Failed to fetch data for {args.symbol}")
            return 1
        
        print(f"   Fetched {len(data)} data points")
        
        # Initialize strategy
        strategy = strategies[args.strategy](
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit
        )
        
        # Generate signal
        signal = strategy.generate_signal(args.symbol, data)
        
        # Display results
        print(f"\n{'='*50}")
        print(f"📈 SIGNAL REPORT: {args.symbol}")
        print(f"{'='*50}")
        print(f"   Signal Type:  {signal.signal_type.value}")
        print(f"   Confidence:   {signal.confidence:.1%}")
        print(f"   Current Price: ${signal.price:.2f}")
        
        if signal.stop_loss:
            print(f"   Stop Loss:    ${signal.stop_loss:.2f}")
        if signal.take_profit:
            print(f"   Take Profit:  ${signal.take_profit:.2f}")
        
        if signal.reasons:
            print(f"\n   Reasons:")
            for reason in signal.reasons:
                print(f"     • {reason}")
        
        # Optional AI analysis
        if args.ai:
            print(f"\n🤖 Running AI analysis...")
            try:
                config = get_config()
                analyzer = AIAnalyzer(provider=args.ai_provider)
                
                # Get news/context for AI
                fundamentals = fetcher.get_fundamentals(args.symbol)
                context = f"{args.symbol} trading at ${signal.price:.2f}. "
                if fundamentals:
                    context += f"P/E: {fundamentals.get('pe_ratio', 'N/A')}, "
                    context += f"Market Cap: {fundamentals.get('market_cap', 'N/A')}"
                
                ai_result = analyzer.analyze_sentiment(context)
                print(f"   AI Sentiment: {ai_result.sentiment.value}")
                print(f"   AI Score:     {ai_result.score:.2f}")
                if ai_result.reasoning:
                    print(f"   AI Reasoning: {ai_result.reasoning}")
            except Exception as e:
                print(f"   ⚠️ AI analysis failed: {e}")
        
        print(f"{'='*50}\n")
        
        # Output JSON if requested
        if args.json:
            print(json.dumps(signal.to_dict(), indent=2))
        
        logger.info(f"Analysis complete for {args.symbol}: {signal.signal_type.value}")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Error: {e}")
        return 1


def cmd_backtest(args):
    """Run backtesting on historical data."""
    from trader.backtest.backtester import Backtester
    from trader.strategies.technical import TechnicalStrategy
    from trader.strategies.momentum import MomentumStrategy
    from trader.strategies.mean_reversion import MeanReversionStrategy
    from trader.strategies.breakout import BreakoutStrategy
    from trader.utils.logger import get_logger
    import yfinance as yf
    
    logger = get_logger("cli.backtest")
    
    strategies = {
        'technical': TechnicalStrategy,
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'breakout': BreakoutStrategy
    }
    
    print(f"\n🔬 Backtesting {args.symbol} with {args.strategy} strategy...")
    print(f"   Period: {args.period}")
    print(f"   Initial Capital: ${args.capital:,.2f}")
    
    try:
        # Fetch historical data
        ticker = yf.Ticker(args.symbol)
        data = ticker.history(period=args.period)
        
        if data.empty:
            print(f"❌ No data available for {args.symbol}")
            return 1
        
        print(f"   Data points: {len(data)}")
        
        # Initialize backtester
        backtester = Backtester(
            initial_capital=args.capital,
            commission=args.commission,
            risk_per_trade=args.risk
        )
        
        # Initialize strategy
        strategy = strategies[args.strategy](
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit
        )
        
        # Run backtest
        result = backtester.run(strategy, data, args.symbol)
        
        # Display results
        metrics = result.metrics
        final_value = backtester.initial_capital + metrics.total_return
        print(f"\n{'='*50}")
        print(f"📊 BACKTEST RESULTS: {args.symbol}")
        print(f"{'='*50}")
        print(f"   Total Return:    ${metrics.total_return:,.2f} ({metrics.total_return_pct:.2f}%)")
        print(f"   Final Value:     ${final_value:,.2f}")
        print(f"   Total Trades:    {metrics.total_trades}")
        print(f"   Win Rate:        {metrics.win_rate:.1f}%")
        print(f"   Profit Factor:   {metrics.profit_factor:.2f}")
        print(f"\n   Risk Metrics:")
        print(f"   Max Drawdown:    {metrics.max_drawdown_pct:.2f}%")
        print(f"   Sharpe Ratio:    {metrics.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio:   {metrics.sortino_ratio:.2f}")
        print(f"   Calmar Ratio:    {metrics.calmar_ratio:.2f}")
        print(f"{'='*50}\n")
        
        # Show trade history if requested
        if args.trades:
            print("📝 Trade History:")
            for i, trade in enumerate(result.trades[-10:], 1):
                trade_date = trade.entry_date.strftime('%Y-%m-%d') if trade.entry_date else 'N/A'
                print(f"   {i}. {trade_date} | {trade.side.upper()} | "
                      f"${trade.entry_price:.2f} | P/L: ${trade.pnl:.2f}")
            
            if len(result.trades) > 10:
                print(f"   ... and {len(result.trades) - 10} more trades")
        
        # Save results if requested
        if args.output:
            output_data = {
                'symbol': args.symbol,
                'strategy': args.strategy,
                'period': args.period,
                'metrics': {
                    'total_return': metrics.total_return,
                    'total_return_pct': metrics.total_return_pct,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown_pct': metrics.max_drawdown_pct,
                    'win_rate': metrics.win_rate,
                    'total_trades': metrics.total_trades
                },
                'trades': result.trades
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"💾 Results saved to {args.output}")
        
        logger.info(f"Backtest complete: {metrics.total_return_pct:.2f}% return")
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"❌ Error: {e}")
        return 1


def cmd_portfolio(args):
    """Manage portfolio positions."""
    from trader.portfolio.portfolio import Portfolio
    from trader.utils.logger import get_logger
    
    logger = get_logger("cli.portfolio")
    portfolio = Portfolio()
    
    if args.action == 'show':
        metrics = portfolio.get_metrics()
        
        print(f"\n{'='*50}")
        print(f"💼 PORTFOLIO SUMMARY")
        print(f"{'='*50}")
        print(f"   Total Value:     ${metrics.total_value:,.2f}")
        print(f"   Cash:            ${metrics.cash_balance:,.2f}")
        print(f"   Positions Value: ${metrics.positions_value:,.2f}")
        print(f"   Unrealized P/L:  ${metrics.unrealized_pnl:,.2f}")
        print(f"   Realized P/L:    ${metrics.realized_pnl:,.2f}")
        print(f"   Daily P/L:       {metrics.daily_pnl_pct:.2%}")
        
        if portfolio.positions:
            print(f"\n📊 Positions:")
            for symbol, pos in portfolio.positions.items():
                current_prices = portfolio.get_current_prices()
                current_price = current_prices.get(symbol, pos.avg_cost)
                pnl = pos.calculate_pnl(current_price)
                pnl_pct = pos.calculate_pnl_pct(current_price)
                print(f"   {symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f} "
                      f"| Current: ${current_price:.2f} "
                      f"| P/L: ${pnl:.2f} ({pnl_pct:.1f}%)")
        else:
            print("\n   No open positions")
        
        print(f"{'='*50}\n")
        
    elif args.action == 'buy':
        if not args.symbol or not args.quantity or not args.price:
            print("❌ --symbol, --quantity, and --price are required for buy")
            return 1
        
        position = portfolio.add_position(
            symbol=args.symbol, 
            quantity=args.quantity, 
            price=args.price,
            stop_loss=args.stop_loss_price,
            take_profit=args.take_profit_price
        )
        
        if position:
            print(f"✅ Bought {args.quantity} {args.symbol} @ ${args.price:.2f}")
        else:
            print(f"❌ Failed to buy {args.symbol}")
            return 1
            
    elif args.action == 'sell':
        if not args.symbol or not args.price:
            print("❌ --symbol and --price are required for sell")
            return 1
        
        quantity = args.quantity  # None means sell all
        pnl = portfolio.close_position(args.symbol, args.price, quantity)
        
        if pnl is not None:
            print(f"✅ Sold {args.symbol} | P/L: ${pnl:.2f}")
        else:
            print(f"❌ Failed to sell {args.symbol}")
            return 1
            
    elif args.action == 'update':
        from trader.data.fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        print("🔄 Updating position prices...")
        for symbol in portfolio.positions:
            try:
                data = fetcher.get_stock_data(symbol, period='1d')
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
                    # Prices are updated via get_current_prices
                    print(f"   {symbol}: ${current_price:.2f}")
            except Exception as e:
                print(f"   ⚠️ {symbol}: Update failed - {e}")
        
        print("✅ Prices updated")
    
    return 0


def cmd_schedule(args):
    """Start the trading scheduler."""
    from trader.scheduler.scheduler import TradingScheduler, ScheduleConfig
    from trader.utils.logger import get_logger
    
    logger = get_logger("cli.scheduler")
    
    print(f"\n⏰ Starting Trading Scheduler...")
    print(f"   Watchlist: {', '.join(args.symbols)}")
    print(f"   Strategy: {args.strategy}")
    print(f"   Interval: {args.interval} minutes")
    
    try:
        config = ScheduleConfig(
            watchlist=list(args.symbols),
            strategy_name=args.strategy,
            scan_interval_minutes=args.interval,
            only_during_market_hours=not args.all_hours
        )
        
        scheduler = TradingScheduler(config=config)
        
        print("\n🚀 Scheduler running. Press Ctrl+C to stop.\n")
        scheduler.start()
        
        # Keep running until interrupted
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping scheduler...")
        scheduler.stop()
        print("✅ Scheduler stopped")
        return 0
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        print(f"❌ Error: {e}")
        return 1


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    import subprocess
    import os
    
    dashboard_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'app.py')
    dashboard_path = os.path.abspath(dashboard_path)
    
    print(f"🚀 Launching dashboard...")
    print(f"   Path: {dashboard_path}")
    print(f"   Port: {args.port}")
    
    try:
        cmd = ['streamlit', 'run', dashboard_path, '--server.port', str(args.port)]
        if args.no_browser:
            cmd.extend(['--server.headless', 'true'])
        
        subprocess.run(cmd)
        return 0
    except FileNotFoundError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return 1
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped")
        return 0


def cmd_alerts_test(args):
    """Test alert system."""
    from trader.alerts.alert_manager import AlertManager
    from trader.strategies.base import Signal, SignalType
    from trader.utils.logger import get_logger
    
    logger = get_logger("cli.alerts")
    
    print(f"\n🔔 Testing Alert System...")
    
    try:
        manager = AlertManager()
        
        # Create test signal
        test_signal = Signal(
            symbol='TEST',
            signal_type=SignalType.BUY,
            confidence=0.85,
            price=100.00,
            stop_loss=95.00,
            take_profit=110.00,
            reasons=['CLI test signal', 'Alert system verification']
        )
        
        if args.telegram:
            print("   Testing Telegram...")
            if manager.telegram.is_configured:
                manager.telegram.send_signal_alert(test_signal)
                print("   ✅ Telegram alert sent")
            else:
                print("   ⚠️ Telegram not configured")
        
        if args.discord:
            print("   Testing Discord...")
            if manager.discord.is_configured:
                manager.discord.send_signal_alert(test_signal)
                print("   ✅ Discord alert sent")
            else:
                print("   ⚠️ Discord not configured")
        
        if not args.telegram and not args.discord:
            print("   Testing all channels...")
            manager.send_signal(test_signal)
            print("   ✅ Alerts sent to all configured channels")
        
        print("\n✅ Alert test complete")
        return 0
        
    except Exception as e:
        logger.error(f"Alert test failed: {e}")
        print(f"❌ Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='trader',
        description='AI Trader - Intelligent Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trader analyze AAPL --strategy technical
  trader backtest AAPL --strategy momentum --period 2y
  trader portfolio show
  trader schedule AAPL MSFT GOOGL --strategy technical
  trader dashboard --port 8501
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--version', action='version', version='AI Trader 1.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze stock and generate signal')
    analyze_parser.add_argument('symbol', help='Stock symbol (e.g., AAPL)')
    analyze_parser.add_argument('-s', '--strategy', default='technical',
                                choices=['technical', 'momentum', 'mean_reversion', 'breakout'],
                                help='Trading strategy to use')
    analyze_parser.add_argument('-p', '--period', default='6mo',
                                help='Data period (e.g., 1mo, 3mo, 6mo, 1y)')
    analyze_parser.add_argument('--stop-loss', type=float, default=0.05,
                                help='Stop loss percentage (default: 0.05)')
    analyze_parser.add_argument('--take-profit', type=float, default=0.10,
                                help='Take profit percentage (default: 0.10)')
    analyze_parser.add_argument('--ai', action='store_true',
                                help='Include AI sentiment analysis')
    analyze_parser.add_argument('--ai-provider', default='deepseek',
                                choices=['deepseek', 'groq'],
                                help='AI provider for sentiment analysis')
    analyze_parser.add_argument('--json', action='store_true',
                                help='Output signal as JSON')
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest on historical data')
    backtest_parser.add_argument('symbol', help='Stock symbol')
    backtest_parser.add_argument('-s', '--strategy', default='technical',
                                 choices=['technical', 'momentum', 'mean_reversion', 'breakout'],
                                 help='Trading strategy to use')
    backtest_parser.add_argument('-p', '--period', default='1y',
                                 help='Historical period (e.g., 6mo, 1y, 2y)')
    backtest_parser.add_argument('-c', '--capital', type=float, default=100000,
                                 help='Initial capital (default: 100000)')
    backtest_parser.add_argument('--commission', type=float, default=0.001,
                                 help='Commission rate (default: 0.001)')
    backtest_parser.add_argument('--risk', type=float, default=0.02,
                                 help='Risk per trade (default: 0.02)')
    backtest_parser.add_argument('--stop-loss', type=float, default=0.05,
                                 help='Stop loss percentage')
    backtest_parser.add_argument('--take-profit', type=float, default=0.10,
                                 help='Take profit percentage')
    backtest_parser.add_argument('--trades', action='store_true',
                                 help='Show trade history')
    backtest_parser.add_argument('-o', '--output', help='Save results to JSON file')
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Manage portfolio')
    portfolio_parser.add_argument('action', choices=['show', 'buy', 'sell', 'update'],
                                  help='Portfolio action')
    portfolio_parser.add_argument('--symbol', help='Stock symbol')
    portfolio_parser.add_argument('--quantity', type=float, help='Number of shares')
    portfolio_parser.add_argument('--price', type=float, help='Price per share')
    portfolio_parser.add_argument('--stop-loss-price', type=float, help='Stop loss price')
    portfolio_parser.add_argument('--take-profit-price', type=float, help='Take profit price')
    portfolio_parser.set_defaults(func=cmd_portfolio)
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Start trading scheduler')
    schedule_parser.add_argument('symbols', nargs='+', help='Stock symbols to watch')
    schedule_parser.add_argument('-s', '--strategy', default='technical',
                                 choices=['technical', 'momentum', 'mean_reversion', 'breakout'],
                                 help='Trading strategy')
    schedule_parser.add_argument('-i', '--interval', type=int, default=15,
                                 help='Check interval in minutes (default: 15)')
    schedule_parser.add_argument('--all-hours', action='store_true',
                                 help='Run outside market hours')
    schedule_parser.set_defaults(func=cmd_schedule)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch web dashboard')
    dashboard_parser.add_argument('-p', '--port', type=int, default=8501,
                                  help='Port number (default: 8501)')
    dashboard_parser.add_argument('--no-browser', action='store_true',
                                  help='Do not open browser automatically')
    dashboard_parser.set_defaults(func=cmd_dashboard)
    
    # Alerts test command
    alerts_parser = subparsers.add_parser('alerts', help='Test alert system')
    alerts_parser.add_argument('--telegram', action='store_true',
                               help='Test Telegram only')
    alerts_parser.add_argument('--discord', action='store_true',
                               help='Test Discord only')
    alerts_parser.set_defaults(func=cmd_alerts_test)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Setup logging
    setup_logging_cli(args.verbose)
    
    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
