"""
AI Trader Pro - Professional Trading Dashboard

A modern, real-time trading dashboard with advanced analytics,
live market data, and automated trading visualization.

Run with: streamlit run dashboard/pro.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dashboard components
from dashboard.components import (
    load_custom_css,
    render_top_navbar,
    render_metric_card,
    render_metric_row,
    render_signal_badge,
    render_positions_table,
    render_activity_feed,
    render_live_trading_panel,
    render_price_ticker,
    create_professional_candlestick,
    create_indicators_chart,
    create_bollinger_bands_chart,
    create_portfolio_donut,
    create_equity_curve,
    MetricData,
    MarketStatus,
    get_live_data_service,
    get_signal_queue,
    get_activity_log
)

# Import trader modules
from trader.data.fetcher import DataFetcher
from trader.strategies.technical import TechnicalStrategy
from trader.strategies.momentum import MomentumStrategy
from trader.strategies.mean_reversion import MeanReversionStrategy
from trader.strategies.breakout import BreakoutStrategy
from trader.strategies.smart_money import SmartMoneyStrategy, MultiTimeframeStrategy
from trader.strategies.scanner import SignalScanner, get_signal_summary
from trader.analysis.indicators import TradingViewIndicators
from trader.analysis.ai_analyzer import AIAnalyzer
from trader.backtest.backtester import Backtester
from trader.portfolio.portfolio import Portfolio
from trader.execution.paper_trading import PaperTradingExecutor


# Page configuration
st.set_page_config(
    page_title="AI Trader Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/trader',
        'Report a bug': 'https://github.com/yourusername/trader/issues',
        'About': "AI Trader Pro - Professional Trading Dashboard"
    }
)

# Load custom CSS
load_custom_css()

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio(initial_capital=100000)

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY']

if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'AAPL'

if 'live_trading' not in st.session_state:
    st.session_state.live_trading = False

if 'activity_log' not in st.session_state:
    st.session_state.activity_log = get_activity_log()


# Cache data fetching
@st.cache_data(ttl=60)
def fetch_stock_data(symbol: str, period: str = '6mo') -> pd.DataFrame:
    """Fetch stock data with caching."""
    fetcher = DataFetcher()
    return fetcher.get_stock_data(symbol, period=period)


@st.cache_data(ttl=3600)
def fetch_fundamentals(symbol: str) -> dict:
    """Fetch fundamentals with caching."""
    fetcher = DataFetcher()
    return fetcher.get_fundamentals(symbol)


@st.cache_data(ttl=60)
def get_watchlist_prices(watchlist: tuple) -> list:
    """Fetch current prices for watchlist."""
    fetcher = DataFetcher()
    prices = []
    
    for symbol in watchlist:
        try:
            df = fetcher.get_stock_data(symbol, period='5d')
            if not df.empty:
                current = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2] if len(df) > 1 else current
                change_pct = ((current - prev) / prev) * 100
                prices.append({
                    'symbol': symbol,
                    'price': current,
                    'change': change_pct
                })
        except Exception:
            prices.append({'symbol': symbol, 'price': 0, 'change': 0})
    
    return prices


def generate_signals(symbol: str, df: pd.DataFrame) -> dict:
    """Generate signals from all strategies."""
    strategies = {
        'Technical': TechnicalStrategy(),
        'Momentum': MomentumStrategy(),
        'Mean Reversion': MeanReversionStrategy(),
        'Breakout': BreakoutStrategy(),
        'Smart Money': SmartMoneyStrategy(),
    }
    
    signals = {}
    for name, strategy in strategies.items():
        try:
            signal = strategy.generate_signal(symbol, df)
            if signal:
                signals[name] = {
                    'type': signal.signal_type.name,
                    'confidence': signal.confidence,
                    'price': signal.price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'reasons': signal.reasons
                }
        except Exception:
            continue
    
    return signals


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Controls")
        
        # Symbol selection
        st.markdown("#### 📊 Symbol Selection")
        symbol = st.selectbox(
            "Active Symbol",
            st.session_state.watchlist,
            index=st.session_state.watchlist.index(st.session_state.selected_symbol)
        )
        st.session_state.selected_symbol = symbol
        
        # Watchlist management
        st.markdown("#### 📋 Watchlist")
        new_symbol = st.text_input("Add Symbol", placeholder="e.g., AMD").upper()
        if st.button("➕ Add to Watchlist", use_container_width=True):
            if new_symbol and new_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol)
                st.rerun()
        
        # Display watchlist with remove buttons
        for sym in st.session_state.watchlist:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📈 {sym}", key=f"watch_{sym}", use_container_width=True):
                    st.session_state.selected_symbol = sym
                    st.rerun()
            with col2:
                if st.button("✕", key=f"remove_{sym}"):
                    st.session_state.watchlist.remove(sym)
                    if st.session_state.selected_symbol == sym:
                        st.session_state.selected_symbol = st.session_state.watchlist[0]
                    st.rerun()
        
        st.divider()
        
        # Chart settings
        st.markdown("#### 📈 Chart Settings")
        period = st.selectbox(
            "Time Period",
            ['1mo', '3mo', '6mo', '1y', '2y'],
            index=2
        )
        
        show_indicators = st.multiselect(
            "Technical Indicators",
            ['SMA 20', 'SMA 50', 'SMA 200', 'RSI', 'MACD', 'Bollinger Bands'],
            default=['SMA 20', 'SMA 50']
        )
        
        st.divider()
        
        # Trading controls
        st.markdown("#### 🤖 Auto Trading")
        
        trading_enabled = st.toggle(
            "Enable Live Trading",
            value=st.session_state.live_trading,
            help="Enable automated trading based on signals"
        )
        st.session_state.live_trading = trading_enabled
        
        if trading_enabled:
            strategy_choice = st.selectbox(
                "Trading Strategy",
                ['Technical', 'Momentum', 'Smart Money', 'Multi-Timeframe']
            )
            
            risk_pct = st.slider(
                "Risk per Trade",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                format="%.1f%%"
            )
            
            min_confidence = st.slider(
                "Min Signal Confidence",
                min_value=0.5,
                max_value=0.95,
                value=0.7,
                step=0.05,
                format="%.0f%%"
            )
        
        st.divider()
        
        # Quick actions
        st.markdown("#### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📊 Run Scanner", use_container_width=True):
            st.session_state.show_scanner = True
        
        if st.button("🧪 Run Backtest", use_container_width=True):
            st.session_state.show_backtest = True
        
        return period, show_indicators


def render_main_content(period: str, show_indicators: list):
    """Render the main dashboard content."""
    symbol = st.session_state.selected_symbol
    market_status = MarketStatus.get_current()
    
    # Top navigation bar
    render_top_navbar(
        is_market_open=market_status.is_open,
        session_name=market_status.session,
        last_update=datetime.now()
    )
    
    # Price ticker
    ticker_data = get_watchlist_prices(tuple(st.session_state.watchlist[:8]))
    render_price_ticker(ticker_data)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fetch data
    df = fetch_stock_data(symbol, period)
    
    if df.empty:
        st.error(f"Unable to fetch data for {symbol}")
        return
    
    # Calculate current metrics
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price > 0 else 0
    
    day_high = df['High'].iloc[-1]
    day_low = df['Low'].iloc[-1]
    volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
    
    # Generate signals
    signals = generate_signals(symbol, df)
    
    # Top metrics row
    metrics = [
        MetricData(
            label="Current Price",
            value=f"${current_price:,.2f}",
            change=f"{price_change_pct:+.2f}%",
            change_positive=price_change_pct >= 0,
            icon="💵"
        ),
        MetricData(
            label="Day Range",
            value=f"${day_low:,.2f} - ${day_high:,.2f}",
            icon="📊"
        ),
        MetricData(
            label="Volume",
            value=f"{volume/1e6:.2f}M",
            change=f"{((volume/avg_volume)-1)*100:+.1f}% vs avg" if avg_volume > 0 else "",
            change_positive=volume >= avg_volume if avg_volume > 0 else True,
            icon="📈"
        ),
        MetricData(
            label="Signals Active",
            value=f"{len([s for s in signals.values() if s['type'] != 'HOLD'])}",
            icon="🎯"
        )
    ]
    render_metric_row(metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content layout
    col_chart, col_panel = st.columns([3, 1])
    
    with col_chart:
        # Tab navigation for chart views
        tab_chart, tab_indicators, tab_signals, tab_backtest, tab_ai = st.tabs([
            "📈 Price Chart", "📊 Indicators", "🎯 Signals", "🧪 Backtest", "🤖 AI Analysis"
        ])
        
        with tab_chart:
            # Determine which MAs to show
            ma_periods = []
            if 'SMA 20' in show_indicators:
                ma_periods.append(20)
            if 'SMA 50' in show_indicators:
                ma_periods.append(50)
            if 'SMA 200' in show_indicators:
                ma_periods.append(200)
            
            fig = create_professional_candlestick(
                df, symbol,
                show_ma=len(ma_periods) > 0,
                ma_periods=ma_periods,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
            
            # Bollinger Bands if selected
            if 'Bollinger Bands' in show_indicators:
                fig_bb = create_bollinger_bands_chart(df, symbol, height=400)
                st.plotly_chart(fig_bb, use_container_width=True, key="bb_chart")
        
        with tab_indicators:
            # Determine which indicators to show
            indicator_list = []
            if 'RSI' in show_indicators:
                indicator_list.append('RSI')
            if 'MACD' in show_indicators:
                indicator_list.append('MACD')
            
            if not indicator_list:
                indicator_list = ['RSI', 'MACD']  # Default
            
            fig_indicators = create_indicators_chart(df, indicators=indicator_list, height=200)
            st.plotly_chart(fig_indicators, use_container_width=True, key="indicators_chart")
            
            # Additional indicator details
            st.markdown("#### 📐 Indicator Values")
            
            # Calculate current indicator values
            indicators_calc = TradingViewIndicators()
            
            col1, col2, col3, col4 = st.columns(4)
            
            # RSI
            delta = df['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            with col1:
                rsi_color = "#ef4444" if rsi > 70 else "#10b981" if rsi < 30 else "#f59e0b"
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">RSI (14)</div>
                    <div class="metric-value" style="color: {rsi_color};">{rsi:.1f}</div>
                    <div style="color: #6b7280; font-size: 12px;">
                        {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = (exp1 - exp2).iloc[-1]
            signal = (exp1 - exp2).ewm(span=9, adjust=False).mean().iloc[-1]
            histogram = macd - signal
            
            with col2:
                macd_color = "#10b981" if histogram > 0 else "#ef4444"
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">MACD</div>
                    <div class="metric-value" style="color: {macd_color};">{macd:.2f}</div>
                    <div style="color: #6b7280; font-size: 12px;">
                        Signal: {signal:.2f}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Supertrend
            try:
                supertrend, direction = indicators_calc.supertrend(df)
                current_direction = direction.iloc[-1]
                trend_text = "Bullish" if current_direction == 1 else "Bearish"
                trend_color = "#10b981" if current_direction == 1 else "#ef4444"
                
                with col3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Supertrend</div>
                        <div class="metric-value" style="color: {trend_color};">{trend_text}</div>
                        <div style="color: #6b7280; font-size: 12px;">
                            Level: ${supertrend.iloc[-1]:.2f}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            except Exception:
                pass
            
            # ADX
            try:
                adx, plus_di, minus_di = indicators_calc.adx_dmi(df)
                adx_value = adx.iloc[-1]
                trend_strength = "Strong" if adx_value > 25 else "Weak"
                
                with col4:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">ADX</div>
                        <div class="metric-value">{adx_value:.1f}</div>
                        <div style="color: #6b7280; font-size: 12px;">
                            {trend_strength} Trend
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            except Exception:
                pass
        
        with tab_signals:
            st.markdown("#### 🎯 Strategy Signals")
            
            if signals:
                for strategy_name, signal_data in signals.items():
                    signal_type = signal_data['type']
                    confidence = signal_data['confidence']
                    
                    # Color based on signal type
                    if signal_type == 'BUY':
                        bg_color = 'rgba(16, 185, 129, 0.1)'
                        border_color = '#10b981'
                        icon = '▲'
                    elif signal_type == 'SELL':
                        bg_color = 'rgba(239, 68, 68, 0.1)'
                        border_color = '#ef4444'
                        icon = '▼'
                    else:
                        bg_color = 'rgba(245, 158, 11, 0.1)'
                        border_color = '#f59e0b'
                        icon = '●'
                    
                    st.markdown(f'''
                    <div style="
                        background: {bg_color};
                        border: 1px solid {border_color};
                        border-radius: 12px;
                        padding: 16px;
                        margin-bottom: 12px;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-weight: 600; color: #ffffff; font-size: 16px;">
                                    {icon} {strategy_name}
                                </div>
                                <div style="color: {border_color}; font-weight: 600; margin-top: 4px;">
                                    {signal_type} @ ${signal_data['price']:.2f}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: #9ca3af; font-size: 12px;">Confidence</div>
                                <div style="font-size: 24px; font-weight: 700; color: {border_color};">
                                    {confidence:.0%}
                                </div>
                            </div>
                        </div>
                        <div style="margin-top: 12px; display: flex; gap: 16px; font-size: 13px;">
                            <span style="color: #9ca3af;">
                                Stop Loss: <span style="color: #ef4444;">${signal_data.get('stop_loss') or 0:.2f}</span>
                            </span>
                            <span style="color: #9ca3af;">
                                Take Profit: <span style="color: #10b981;">${signal_data.get('take_profit') or 0:.2f}</span>
                            </span>
                        </div>
                        <div style="margin-top: 8px; color: #6b7280; font-size: 12px;">
                            {', '.join(signal_data.get('reasons', [])[:3])}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No signals generated. Check if data is available.")
            
            # Signal summary
            st.markdown("#### 📊 Signal Summary")
            
            buy_count = len([s for s in signals.values() if s['type'] == 'BUY'])
            sell_count = len([s for s in signals.values() if s['type'] == 'SELL'])
            hold_count = len([s for s in signals.values() if s['type'] == 'HOLD'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label" style="color: #10b981;">BUY Signals</div>
                    <div class="metric-value" style="color: #10b981;">{buy_count}</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label" style="color: #ef4444;">SELL Signals</div>
                    <div class="metric-value" style="color: #ef4444;">{sell_count}</div>
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label" style="color: #f59e0b;">HOLD Signals</div>
                    <div class="metric-value" style="color: #f59e0b;">{hold_count}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        with tab_backtest:
            st.markdown("#### 🧪 Strategy Backtesting")
            
            col1, col2 = st.columns(2)
            with col1:
                bt_strategy = st.selectbox(
                    "Select Strategy",
                    ['Technical', 'Momentum', 'Mean Reversion', 'Breakout', 'Smart Money']
                )
            with col2:
                bt_period = st.selectbox(
                    "Backtest Period",
                    ['3mo', '6mo', '1y', '2y'],
                    index=1
                )
            
            if st.button("▶️ Run Backtest", use_container_width=True, type="primary"):
                with st.spinner("Running backtest..."):
                    # Fetch data for backtest
                    bt_data = fetch_stock_data(symbol, bt_period)
                    
                    if bt_data.empty:
                        st.error("Unable to fetch data for backtest")
                    else:
                        # Initialize strategy
                        strategy_map = {
                            'Technical': TechnicalStrategy(),
                            'Momentum': MomentumStrategy(),
                            'Mean Reversion': MeanReversionStrategy(),
                            'Breakout': BreakoutStrategy(),
                            'Smart Money': SmartMoneyStrategy()
                        }
                        strategy = strategy_map.get(bt_strategy, TechnicalStrategy())
                        
                        # Run backtest
                        backtester = Backtester(
                            initial_capital=100000,
                            commission=0.001,
                            risk_per_trade=0.02
                        )
                        
                        result = backtester.run(strategy, bt_data, symbol)
                        
                        # Display results
                        st.markdown("##### 📈 Backtest Results")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            total_return = result.metrics.total_return_pct
                            color = "#10b981" if total_return >= 0 else "#ef4444"
                            st.markdown(f'''
                            <div class="metric-card">
                                <div class="metric-label">Total Return</div>
                                <div class="metric-value" style="color: {color};">{total_return:+.2f}%</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f'''
                            <div class="metric-card">
                                <div class="metric-label">Sharpe Ratio</div>
                                <div class="metric-value">{result.metrics.sharpe_ratio:.2f}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f'''
                            <div class="metric-card">
                                <div class="metric-label">Max Drawdown</div>
                                <div class="metric-value" style="color: #ef4444;">{result.metrics.max_drawdown_pct:.2f}%</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f'''
                            <div class="metric-card">
                                <div class="metric-label">Win Rate</div>
                                <div class="metric-value">{result.metrics.win_rate:.1f}%</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Equity curve
                        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
                            fig_equity = create_equity_curve(
                                result.equity_curve,
                                result.drawdown if hasattr(result, 'drawdown') else None,
                                height=350
                            )
                            st.plotly_chart(fig_equity, use_container_width=True, key="equity_curve")
        
        with tab_ai:
            st.markdown("#### 🤖 AI-Powered Analysis")
            
            # Show available AI providers
            providers = AIAnalyzer.get_available_providers()
            active_providers = [p for p, info in providers.items() if info['has_api_key']]
            
            if not active_providers:
                st.warning("""
                ⚠️ No AI API keys configured. Add at least one to .env:
                - `DEEPSEEK_API_KEY` - Most cost-effective
                - `GROQ_API_KEY` - Fastest inference  
                - `XAI_API_KEY` - Grok-2 (excellent reasoning)
                - `OPENAI_API_KEY` - GPT-4o
                """)
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_provider = st.selectbox(
                        "AI Provider",
                        active_providers,
                        format_func=lambda x: f"{x.capitalize()} - {providers[x]['model']}"
                    )
                
                with col2:
                    analysis_type = st.selectbox(
                        "Analysis Type",
                        ['Sentiment Analysis', 'Company Evaluation', 'Trading Insight', 'Trade Plan']
                    )
                
                # Custom news/text input for sentiment
                if analysis_type == 'Sentiment Analysis':
                    news_input = st.text_area(
                        "Enter news or text to analyze",
                        placeholder=f"E.g., '{symbol} reports record Q4 earnings, beats estimates by 15%'",
                        height=100
                    )
                    
                    if st.button("🧠 Analyze Sentiment", use_container_width=True, type="primary"):
                        if news_input:
                            with st.spinner(f"Analyzing with {selected_provider}..."):
                                try:
                                    analyzer = AIAnalyzer(provider=selected_provider)
                                    result = analyzer.analyze_sentiment(news_input)
                                    
                                    # Display results
                                    sentiment_color = "#10b981" if result.score > 0 else "#ef4444" if result.score < 0 else "#f59e0b"
                                    
                                    st.markdown(f'''
                                    <div class="metric-card">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <div>
                                                <div class="metric-label">Sentiment</div>
                                                <div class="metric-value" style="color: {sentiment_color};">{result.sentiment.name}</div>
                                            </div>
                                            <div style="text-align: right;">
                                                <div style="font-size: 36px; color: {sentiment_color};">{result.score:+.2f}</div>
                                                <div style="color: #6b7280; font-size: 12px;">Confidence: {result.confidence:.0%}</div>
                                            </div>
                                        </div>
                                        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(75, 85, 99, 0.4);">
                                            <div style="color: #9ca3af; font-size: 12px;">AI Reasoning</div>
                                            <div style="color: #ffffff; font-size: 14px; margin-top: 8px;">{result.reasoning}</div>
                                        </div>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                    
                                    analyzer.close()
                                except Exception as e:
                                    st.error(f"Analysis failed: {str(e)}")
                        else:
                            st.warning("Please enter text to analyze")
                
                elif analysis_type == 'Company Evaluation':
                    if st.button("📊 Evaluate Company", use_container_width=True, type="primary"):
                        with st.spinner(f"Fetching fundamentals and analyzing with {selected_provider}..."):
                            try:
                                # Fetch fundamentals
                                fundamentals = fetch_fundamentals(symbol)
                                
                                if fundamentals:
                                    analyzer = AIAnalyzer(provider=selected_provider)
                                    evaluation = analyzer.evaluate_company(fundamentals)
                                    
                                    # Display results
                                    rec_color = "#10b981" if evaluation.recommendation == 'BUY' else "#ef4444" if evaluation.recommendation == 'SELL' else "#f59e0b"
                                    
                                    st.markdown(f'''
                                    <div class="metric-card">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                                            <div>
                                                <div style="font-size: 24px; font-weight: 700; color: #ffffff;">{symbol}</div>
                                                <div style="color: #9ca3af;">Fundamental Analysis</div>
                                            </div>
                                            <div style="text-align: right;">
                                                <div style="font-size: 36px; font-weight: 700; color: {rec_color};">{evaluation.recommendation}</div>
                                                <div style="color: #6b7280; font-size: 12px;">Score: {evaluation.overall_score:.1f}/10</div>
                                            </div>
                                        </div>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                    
                                    col_s, col_w = st.columns(2)
                                    
                                    with col_s:
                                        st.markdown("##### ✅ Strengths")
                                        for s in evaluation.strengths[:5]:
                                            st.markdown(f"• {s}")
                                    
                                    with col_w:
                                        st.markdown("##### ⚠️ Weaknesses")
                                        for w in evaluation.weaknesses[:5]:
                                            st.markdown(f"• {w}")
                                    
                                    st.markdown("##### 💡 AI Analysis")
                                    st.markdown(evaluation.reasoning)
                                    
                                    analyzer.close()
                                else:
                                    st.error("Could not fetch company fundamentals")
                            except Exception as e:
                                st.error(f"Evaluation failed: {str(e)}")
                
                elif analysis_type == 'Trading Insight':
                    if st.button("💡 Generate Trading Insight", use_container_width=True, type="primary"):
                        with st.spinner(f"Generating insight with {selected_provider}..."):
                            try:
                                # Prepare price data summary
                                price_data = {
                                    'current_price': float(df['Close'].iloc[-1]),
                                    'change_24h_pct': float(((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100),
                                    'high_52w': float(df['High'].max()),
                                    'low_52w': float(df['Low'].min()),
                                    'avg_volume': float(df['Volume'].mean()),
                                    'volatility_30d': float(df['Close'].pct_change().tail(30).std() * np.sqrt(252) * 100)
                                }
                                
                                fundamentals = fetch_fundamentals(symbol)
                                
                                analyzer = AIAnalyzer(provider=selected_provider)
                                insight = analyzer.generate_trading_insight(
                                    symbol, price_data, fundamentals or {}
                                )
                                
                                # Display insight
                                signal_color = "#10b981" if insight.get('signal') == 'BUY' else "#ef4444" if insight.get('signal') == 'SELL' else "#f59e0b"
                                
                                st.markdown(f'''
                                <div class="metric-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <div style="font-size: 24px; font-weight: 700; color: {signal_color};">{insight.get('signal', 'N/A')}</div>
                                            <div style="color: #9ca3af;">Signal Strength: {insight.get('signal_strength', 0):.0%}</div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="color: #6b7280;">Risk Level</div>
                                            <div style="font-weight: 600; color: #ffffff;">{insight.get('risk_level', 'N/A')}</div>
                                        </div>
                                    </div>
                                </div>
                                ''', unsafe_allow_html=True)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Stop Loss", f"{insight.get('stop_loss_pct', 0):.1f}%")
                                with col2:
                                    st.metric("Take Profit", f"{insight.get('take_profit_pct', 0):.1f}%")
                                with col3:
                                    st.metric("Time Horizon", insight.get('time_horizon', 'N/A'))
                                
                                st.markdown("##### 🔑 Key Factors")
                                for factor in insight.get('key_factors', [])[:5]:
                                    st.markdown(f"• {factor}")
                                
                                st.markdown("##### 💡 Reasoning")
                                st.markdown(insight.get('reasoning', 'No reasoning provided'))
                                
                                analyzer.close()
                            except Exception as e:
                                st.error(f"Insight generation failed: {str(e)}")
                
                elif analysis_type == 'Trade Plan':
                    signal_direction = st.radio("Trade Direction", ['BUY', 'SELL'], horizontal=True)
                    entry_price = st.number_input("Entry Price", value=float(df['Close'].iloc[-1]), step=0.01)
                    
                    if st.button("📋 Generate Trade Plan", use_container_width=True, type="primary"):
                        with st.spinner(f"Creating trade plan with {selected_provider}..."):
                            try:
                                current_data = {
                                    'current_price': float(df['Close'].iloc[-1]),
                                    'day_high': float(df['High'].iloc[-1]),
                                    'day_low': float(df['Low'].iloc[-1]),
                                    'volume': float(df['Volume'].iloc[-1]),
                                    'atr': float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
                                }
                                
                                analyzer = AIAnalyzer(provider=selected_provider)
                                plan = analyzer.generate_trade_plan(
                                    symbol, signal_direction, entry_price, current_data
                                )
                                
                                st.markdown("##### 📍 Entry Zone")
                                entry_zone = plan.get('entry_zone', {})
                                st.markdown(f"""
                                - **Ideal Entry:** ${entry_zone.get('ideal_entry', entry_price):,.2f}
                                - **Range:** ${entry_zone.get('entry_range_low', 0):,.2f} - ${entry_zone.get('entry_range_high', 0):,.2f}
                                """)
                                
                                st.markdown("##### 🛡️ Stop Loss Levels")
                                sl = plan.get('stop_loss', {})
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Tight SL", f"${sl.get('tight', 0):,.2f}")
                                with col2:
                                    st.metric("Standard SL", f"${sl.get('standard', 0):,.2f}")
                                with col3:
                                    st.metric("Wide SL", f"${sl.get('wide', 0):,.2f}")
                                
                                st.markdown("##### 🎯 Take Profit Targets")
                                tp = plan.get('take_profit', {})
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Target 1", f"${tp.get('target_1', 0):,.2f}")
                                with col2:
                                    st.metric("Target 2", f"${tp.get('target_2', 0):,.2f}")
                                with col3:
                                    st.metric("Target 3", f"${tp.get('target_3', 0):,.2f}")
                                
                                st.markdown(f"""
                                ##### 📊 Trade Metrics
                                - **Risk/Reward:** {plan.get('risk_reward_ratio', 0):.2f}
                                - **Success Probability:** {plan.get('success_probability', 0):.0%}
                                - **Time Horizon:** {plan.get('time_horizon', 'N/A')}
                                """)
                                
                                st.markdown("##### ⚠️ Key Levels to Watch")
                                for level in plan.get('key_levels_to_watch', []):
                                    st.markdown(f"• {level}")
                                
                                analyzer.close()
                            except Exception as e:
                                st.error(f"Trade plan generation failed: {str(e)}")
    
    with col_panel:
        # Right panel - Live trading and activity
        st.markdown("#### 🤖 Trading Status")
        render_live_trading_panel(
            is_running=st.session_state.live_trading,
            strategy_name="Multi-Strategy",
            last_signal=signals.get('Technical', {}).get('type', 'N/A') if signals else 'N/A',
            signals_today=len(signals),
            trades_today=0
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Portfolio summary
        st.markdown("#### 💼 Portfolio")
        portfolio = st.session_state.portfolio
        metrics = portfolio.get_metrics()
        
        st.markdown(f'''
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #9ca3af;">Total Value</span>
                <span style="color: #ffffff; font-weight: 600;">${metrics.total_value:,.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #9ca3af;">Cash</span>
                <span style="color: #ffffff;">${metrics.cash_balance:,.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #9ca3af;">Positions</span>
                <span style="color: #ffffff;">${metrics.positions_value:,.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #9ca3af;">P/L</span>
                <span style="color: {'#10b981' if metrics.unrealized_pnl >= 0 else '#ef4444'};">
                    {'+' if metrics.unrealized_pnl >= 0 else ''}${metrics.unrealized_pnl:,.2f}
                </span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Open positions
        st.markdown("#### 📊 Positions")
        positions = portfolio.positions
        
        if positions:
            positions_data = []
            for pos in positions.values():
                current_price = df['Close'].iloc[-1] if pos.symbol == symbol else pos.avg_cost
                pnl = pos.calculate_pnl(current_price)
                pnl_pct = pos.calculate_pnl_pct(current_price)
                positions_data.append({
                    'symbol': pos.symbol,
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.avg_cost,
                    'current_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
            render_positions_table(positions_data)
        else:
            st.markdown('''
            <div style="text-align: center; padding: 20px; color: #6b7280; font-size: 14px;">
                No open positions
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Activity feed
        st.markdown("#### 📜 Recent Activity")
        activity_entries = st.session_state.activity_log.get_entries(limit=5)
        render_activity_feed(activity_entries)


def main():
    """Main application entry point."""
    # Render sidebar and get settings
    period, show_indicators = render_sidebar()
    
    # Render main content
    render_main_content(period, show_indicators)


if __name__ == "__main__":
    main()
