"""
AI Trader Pro - Professional Trading Dashboard (XTB Style)

Premium dark theme trading platform with:
- Real-time market data
- Multi-strategy signals
- Portfolio tracking
- Top traders portfolios
- Advanced charting

Run with: streamlit run dashboard/pro.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader.data.fetcher import DataFetcher
from trader.strategies.technical import TechnicalStrategy
from trader.strategies.momentum import MomentumStrategy
from trader.strategies.mean_reversion import MeanReversionStrategy
from trader.strategies.breakout import BreakoutStrategy
from trader.strategies.smart_money import SmartMoneyStrategy
from trader.backtest.backtester import Backtester
from trader.analysis.indicators import TradingViewIndicators

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="AI Trader Pro | XTB Style",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for XTB-style dark theme
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: rgba(26, 31, 46, 0.95);
        --text-primary: #ffffff;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --accent: #3b82f6;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --border: rgba(75, 85, 99, 0.4);
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0a0e17 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Top header bar */
    .top-header {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.98) 0%, rgba(17, 24, 39, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid var(--border);
        padding: 12px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: -1rem -1rem 1rem -1rem;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .logo {
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
        color: white;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }
    
    .brand-name {
        font-size: 22px;
        font-weight: 700;
        color: white;
        letter-spacing: -0.5px;
    }
    
    .brand-subtitle {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Market status */
    .market-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(17, 24, 39, 0.8);
        border-radius: 20px;
        border: 1px solid var(--border);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-dot.open { background: var(--success); box-shadow: 0 0 10px var(--success); }
    .status-dot.closed { background: var(--danger); }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.2); }
    }
    
    /* Price ticker */
    .price-ticker {
        display: flex;
        gap: 24px;
        padding: 10px 20px;
        background: rgba(17, 24, 39, 0.6);
        border-bottom: 1px solid var(--border);
        overflow-x: auto;
        margin: 0 -1rem;
    }
    
    .ticker-item {
        display: flex;
        align-items: center;
        gap: 12px;
        white-space: nowrap;
    }
    
    .ticker-symbol { font-weight: 600; color: white; }
    .ticker-price { color: var(--text-secondary); }
    .ticker-change.up { color: var(--success); }
    .ticker-change.down { color: var(--danger); }
    
    /* Metric cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 24px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), #8b5cf6);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover::before { opacity: 1; }
    
    .metric-label {
        font-size: 12px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: white;
    }
    
    .metric-change {
        font-size: 14px;
        margin-top: 4px;
    }
    
    .metric-change.positive { color: var(--success); }
    .metric-change.negative { color: var(--danger); }
    
    /* Glass cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
    }
    
    .glass-card-title {
        font-size: 16px;
        font-weight: 600;
        color: white;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Signal badges */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .signal-badge.buy {
        background: rgba(16, 185, 129, 0.15);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .signal-badge.sell {
        background: rgba(239, 68, 68, 0.15);
        color: var(--danger);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .signal-badge.hold {
        background: rgba(245, 158, 11, 0.15);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Top traders table */
    .trader-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s;
    }
    
    .trader-card:hover {
        border-color: var(--accent);
        transform: translateX(4px);
    }
    
    .trader-rank {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: white;
        font-size: 14px;
    }
    
    .trader-name {
        font-weight: 600;
        color: white;
        font-size: 15px;
    }
    
    .trader-return {
        font-size: 18px;
        font-weight: 700;
    }
    
    .trader-return.positive { color: var(--success); }
    .trader-return.negative { color: var(--danger); }
    
    /* Positions table */
    .positions-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
    }
    
    .positions-table th {
        text-align: left;
        padding: 12px 16px;
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .positions-table td {
        padding: 16px;
        background: rgba(26, 31, 46, 0.6);
        font-size: 14px;
        color: white;
    }
    
    .positions-table tr td:first-child { border-radius: 8px 0 0 8px; }
    .positions-table tr td:last-child { border-radius: 0 8px 8px 0; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0a0e17 100%);
        border-right: 1px solid var(--border);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in { animation: fadeIn 0.5s ease-out; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ', 'AMD']

if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'AAPL'

# Data fetching functions
@st.cache_data(ttl=60)
def fetch_stock_data(symbol: str, period: str = '6mo') -> pd.DataFrame:
    """Fetch stock data with caching."""
    try:
        fetcher = DataFetcher()
        return fetcher.get_stock_data(symbol, period=period)
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_fundamentals(symbol: str) -> dict:
    """Fetch company fundamentals."""
    try:
        fetcher = DataFetcher()
        return fetcher.get_fundamentals(symbol)
    except Exception:
        return {}

@st.cache_data(ttl=60)
def get_watchlist_data(symbols: tuple) -> list:
    """Fetch data for all watchlist symbols."""
    data = []
    fetcher = DataFetcher()
    
    for symbol in symbols:
        try:
            df = fetcher.get_stock_data(symbol, period='5d')
            if not df.empty and len(df) >= 2:
                current = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2])
                change_pct = ((current - prev) / prev) * 100
                
                # 7-day data for sparkline
                week_data = df['Close'].tail(7).tolist()
                
                data.append({
                    'symbol': symbol,
                    'price': current,
                    'change_pct': change_pct,
                    'volume': int(df['Volume'].iloc[-1]),
                    'high': float(df['High'].max()),
                    'low': float(df['Low'].min()),
                    'sparkline': week_data
                })
        except Exception:
            data.append({
                'symbol': symbol,
                'price': 0,
                'change_pct': 0,
                'volume': 0,
                'high': 0,
                'low': 0,
                'sparkline': []
            })
    
    return data

# Top traders data (simulated - would come from API in production)
TOP_TRADERS = [
    {'name': 'Warren Buffett', 'style': 'Value Investing', 'return_ytd': 12.5, 'win_rate': 68, 
     'holdings': ['AAPL', 'BAC', 'KO', 'CVX', 'AXP'], 'aum': '785B'},
    {'name': 'Ray Dalio', 'style': 'Macro/All Weather', 'return_ytd': 8.3, 'win_rate': 62,
     'holdings': ['SPY', 'GLD', 'TLT', 'VWO', 'IEF'], 'aum': '150B'},
    {'name': 'Cathie Wood', 'style': 'Disruptive Innovation', 'return_ytd': 45.2, 'win_rate': 55,
     'holdings': ['TSLA', 'COIN', 'ROKU', 'SHOP', 'SQ'], 'aum': '14B'},
    {'name': 'Bill Ackman', 'style': 'Activist Value', 'return_ytd': 18.7, 'win_rate': 71,
     'holdings': ['CMG', 'LOW', 'HLT', 'QSR', 'HHH'], 'aum': '18B'},
    {'name': 'Michael Burry', 'style': 'Contrarian', 'return_ytd': -5.2, 'win_rate': 58,
     'holdings': ['JD', 'BABA', 'GEO', 'CVS', 'WBD'], 'aum': '300M'},
]

def generate_signals(symbol: str, df: pd.DataFrame) -> list:
    """Generate signals from multiple strategies."""
    signals = []
    
    strategies = [
        ('Technical', TechnicalStrategy()),
        ('Momentum', MomentumStrategy()),
        ('Mean Reversion', MeanReversionStrategy()),
        ('Breakout', BreakoutStrategy()),
        ('Smart Money', SmartMoneyStrategy()),
    ]
    
    for name, strategy in strategies:
        try:
            signal = strategy.generate_signal(symbol, df)
            if signal:
                signals.append({
                    'strategy': name,
                    'type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'price': signal.price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'reasons': signal.reasons or []
                })
        except Exception:
            continue
    
    return signals

def create_candlestick_chart(df: pd.DataFrame, symbol: str, height: int = 500) -> go.Figure:
    """Create professional candlestick chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing=dict(line=dict(color='#10b981'), fillcolor='#10b981'),
            decreasing=dict(line=dict(color='#ef4444'), fillcolor='#ef4444')
        ),
        row=1, col=1
    )
    
    # Moving averages
    for period, color in [(20, '#3b82f6'), (50, '#f59e0b')]:
        if len(df) >= period:
            ma = df['Close'].rolling(period).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=ma, name=f'SMA {period}',
                          line=dict(color=color, width=1.5)),
                row=1, col=1
            )
    
    # Volume
    colors = ['#ef4444' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#10b981' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        xaxis=dict(gridcolor='rgba(75,85,99,0.2)'),
        yaxis=dict(gridcolor='rgba(75,85,99,0.2)', side='right'),
        yaxis2=dict(gridcolor='rgba(75,85,99,0.1)', side='right')
    )
    
    return fig

def create_indicators_chart(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Create RSI and MACD indicators chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('RSI (14)', 'MACD'))
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='#8b5cf6', width=2)), row=1, col=1)
    # Add horizontal lines for RSI levels using shapes
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70,
                  line=dict(color="#ef4444", width=1, dash="dash"), opacity=0.5, row=1, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30,
                  line=dict(color="#10b981", width=1, dash="dash"), opacity=0.5, row=1, col=1)
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    
    fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='#3b82f6')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='#f59e0b')), row=2, col=1)
    colors = ['#10b981' if h > 0 else '#ef4444' for h in histogram]
    fig.add_trace(go.Bar(x=df.index, y=histogram, name='Histogram', marker_color=colors), row=2, col=1)
    
    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=50, r=50, t=30, b=30)
    )
    
    return fig

def render_header():
    """Render the top header bar."""
    # Check market status (simplified)
    now = datetime.now()
    is_market_open = 5 <= now.weekday() <= 4 and 9 <= now.hour < 16
    status_class = "open" if is_market_open else "closed"
    status_text = "Market Open" if is_market_open else "Market Closed"
    
    st.markdown(f'''
    <div class="top-header animate-in">
        <div class="logo-section">
            <div class="logo">AT</div>
            <div>
                <div class="brand-name">AI Trader Pro</div>
                <div class="brand-subtitle">Intelligent Trading Platform</div>
            </div>
        </div>
        <div class="market-status">
            <div class="status-dot {status_class}"></div>
            <span style="color: {"#10b981" if is_market_open else "#ef4444"}; font-weight: 600;">{status_text}</span>
            <span style="color: #6b7280; margin-left: 12px;">{now.strftime("%H:%M:%S")}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def render_price_ticker():
    """Render the scrolling price ticker."""
    data = get_watchlist_data(tuple(st.session_state.watchlist[:8]))
    
    ticker_html = '<div class="price-ticker">'
    for item in data:
        change_class = "up" if item['change_pct'] >= 0 else "down"
        sign = "+" if item['change_pct'] >= 0 else ""
        ticker_html += f'''
        <div class="ticker-item">
            <span class="ticker-symbol">{item['symbol']}</span>
            <span class="ticker-price">${item['price']:,.2f}</span>
            <span class="ticker-change {change_class}">{sign}{item['change_pct']:.2f}%</span>
        </div>
        '''
    ticker_html += '</div>'
    
    st.markdown(ticker_html, unsafe_allow_html=True)

def render_top_traders():
    """Render top traders section."""
    st.markdown("### 🏆 Top Traders Portfolios")
    
    for i, trader in enumerate(TOP_TRADERS, 1):
        return_class = "positive" if trader['return_ytd'] >= 0 else "negative"
        sign = "+" if trader['return_ytd'] >= 0 else ""
        
        with st.expander(f"**#{i} {trader['name']}** | {trader['style']} | YTD: {sign}{trader['return_ytd']}%"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">YTD Return</div>
                    <div class="metric-value" style="color: {'#10b981' if trader['return_ytd'] >= 0 else '#ef4444'};">
                        {sign}{trader['return_ytd']}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{trader['win_rate']}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">AUM</div>
                    <div class="metric-value">${trader['aum']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**Top Holdings:**")
            holdings_cols = st.columns(5)
            for j, holding in enumerate(trader['holdings']):
                with holdings_cols[j]:
                    try:
                        df = fetch_stock_data(holding, '5d')
                        if not df.empty:
                            price = df['Close'].iloc[-1]
                            change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                            color = "#10b981" if change >= 0 else "#ef4444"
                            st.markdown(f"""
                            <div style="background: rgba(26,31,46,0.8); padding: 12px; border-radius: 8px; text-align: center;">
                                <div style="font-weight: 600; color: white;">{holding}</div>
                                <div style="color: #9ca3af; font-size: 12px;">${price:.2f}</div>
                                <div style="color: {color}; font-size: 12px;">{change:+.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                    except:
                        st.markdown(f"**{holding}**")

def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("## 🎯 Quick Navigation")
        
        # Symbol selection
        st.markdown("### 📊 Active Symbol")
        symbol = st.selectbox(
            "Select Stock",
            st.session_state.watchlist,
            index=st.session_state.watchlist.index(st.session_state.selected_symbol)
        )
        st.session_state.selected_symbol = symbol
        
        # Add custom symbol
        custom = st.text_input("Add Symbol", placeholder="e.g., AMD").upper()
        if st.button("➕ Add", use_container_width=True):
            if custom and custom not in st.session_state.watchlist:
                st.session_state.watchlist.append(custom)
                st.rerun()
        
        st.divider()
        
        # Time period
        st.markdown("### ⏱️ Time Period")
        period = st.selectbox("Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=2)
        
        st.divider()
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Watchlist
        st.markdown("### 📋 Watchlist")
        for sym in st.session_state.watchlist:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📈 {sym}", key=f"w_{sym}", use_container_width=True):
                    st.session_state.selected_symbol = sym
                    st.rerun()
            with col2:
                if st.button("✕", key=f"r_{sym}"):
                    st.session_state.watchlist.remove(sym)
                    if st.session_state.selected_symbol == sym:
                        st.session_state.selected_symbol = st.session_state.watchlist[0]
                    st.rerun()
        
        return period

def main():
    """Main application."""
    period = render_sidebar()
    
    render_header()
    render_price_ticker()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    symbol = st.session_state.selected_symbol
    
    # Fetch data
    df = fetch_stock_data(symbol, period)
    
    if df.empty:
        st.error(f"Could not load data for {symbol}")
        return
    
    # Top metrics
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price) * 100 if prev_price else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💵 Current Price</div>
            <div class="metric-value">${current_price:,.2f}</div>
            <div class="metric-change {'positive' if change_pct >= 0 else 'negative'}">
                {'↑' if change_pct >= 0 else '↓'} {change_pct:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Day High</div>
            <div class="metric-value">${df['High'].iloc[-1]:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📉 Day Low</div>
            <div class="metric-value">${df['Low'].iloc[-1]:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        volume = df['Volume'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Volume</div>
            <div class="metric-value">{volume/1e6:.1f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ Volatility</div>
            <div class="metric-value">{volatility:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Chart", "🎯 Signals", "🏆 Top Traders", "🔬 Backtest", "📊 Analysis"
    ])
    
    with tab1:
        fig = create_candlestick_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📉 Technical Indicators")
        fig2 = create_indicators_chart(df)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown(f"### 🎯 Trading Signals for {symbol}")
        
        signals = generate_signals(symbol, df)
        
        if signals:
            # Summary
            buy_count = len([s for s in signals if s['type'] == 'BUY'])
            sell_count = len([s for s in signals if s['type'] == 'SELL'])
            hold_count = len([s for s in signals if s['type'] == 'HOLD'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#10b981;">BUY</div><div class="metric-value" style="color:#10b981;">{buy_count}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#ef4444;">SELL</div><div class="metric-value" style="color:#ef4444;">{sell_count}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-label" style="color:#f59e0b;">HOLD</div><div class="metric-value" style="color:#f59e0b;">{hold_count}</div></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Individual signals
            for sig in signals:
                sig_type = sig['type']
                color = '#10b981' if sig_type == 'BUY' else '#ef4444' if sig_type == 'SELL' else '#f59e0b'
                icon = '▲' if sig_type == 'BUY' else '▼' if sig_type == 'SELL' else '●'
                
                st.markdown(f"""
                <div style="background: rgba(26,31,46,0.8); border: 1px solid {color}33; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 18px; font-weight: 600; color: {color};">{icon} {sig['strategy']}</span>
                            <span class="signal-badge {sig_type.lower()}" style="margin-left: 12px;">{sig_type}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #9ca3af; font-size: 12px;">Confidence</div>
                            <div style="font-size: 24px; font-weight: 700; color: {color};">{sig['confidence']:.0%}</div>
                        </div>
                    </div>
                    <div style="margin-top: 12px; display: flex; gap: 20px; font-size: 13px;">
                        <span style="color: #9ca3af;">Price: <span style="color: white;">${sig['price']:.2f}</span></span>
                        <span style="color: #9ca3af;">Stop Loss: <span style="color: #ef4444;">${(sig.get('stop_loss') or 0):.2f}</span></span>
                        <span style="color: #9ca3af;">Take Profit: <span style="color: #10b981;">${(sig.get('take_profit') or 0):.2f}</span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No signals generated. Try refreshing data.")
    
    with tab3:
        render_top_traders()
    
    with tab4:
        st.markdown("### 🔬 Strategy Backtesting")
        
        col1, col2 = st.columns(2)
        with col1:
            bt_strategy = st.selectbox("Strategy", ['Technical', 'Momentum', 'Mean Reversion', 'Breakout', 'Smart Money'])
        with col2:
            bt_period = st.selectbox("Backtest Period", ['3mo', '6mo', '1y', '2y'], index=1)
        
        if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest..."):
                bt_data = fetch_stock_data(symbol, bt_period)
                
                if bt_data.empty:
                    st.error("Could not fetch data")
                else:
                    strategy_map = {
                        'Technical': TechnicalStrategy(),
                        'Momentum': MomentumStrategy(),
                        'Mean Reversion': MeanReversionStrategy(),
                        'Breakout': BreakoutStrategy(),
                        'Smart Money': SmartMoneyStrategy()
                    }
                    
                    strategy = strategy_map[bt_strategy]
                    backtester = Backtester(initial_capital=100000, commission=0.001, risk_per_trade=0.02)
                    
                    try:
                        result = backtester.run(strategy, bt_data, symbol)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            color = "#10b981" if result.metrics.total_return_pct >= 0 else "#ef4444"
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Return</div><div class="metric-value" style="color:{color};">{result.metrics.total_return_pct:+.2f}%</div></div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{result.metrics.sharpe_ratio:.2f}</div></div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-value" style="color:#ef4444;">{result.metrics.max_drawdown_pct:.2f}%</div></div>', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f'<div class="metric-card"><div class="metric-label">Win Rate</div><div class="metric-value">{result.metrics.win_rate:.1f}%</div></div>', unsafe_allow_html=True)
                        
                        # Equity curve
                        st.markdown("### 📈 Equity Curve")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=result.equity_curve.index,
                            y=result.equity_curve.values,
                            fill='tozeroy',
                            line=dict(color='#3b82f6' if result.metrics.total_return >= 0 else '#ef4444')
                        ))
                        fig.update_layout(
                            height=300,
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=50, r=50, t=20, b=30)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
    
    with tab5:
        st.markdown(f"### 📊 Detailed Analysis for {symbol}")
        
        fundamentals = fetch_fundamentals(symbol)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Technical Summary")
            
            # Calculate indicators
            rsi_delta = df['Close'].diff()
            rsi_gain = rsi_delta.clip(lower=0).rolling(14).mean()
            rsi_loss = (-rsi_delta.clip(upper=0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + rsi_gain / rsi_loss))
            current_rsi = rsi.iloc[-1]
            
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            macd = (exp1 - exp2).iloc[-1]
            signal = (exp1 - exp2).ewm(span=9).mean().iloc[-1]
            
            sma20 = df['Close'].rolling(20).mean().iloc[-1]
            sma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma20
            
            rsi_color = "#ef4444" if current_rsi > 70 else "#10b981" if current_rsi < 30 else "#f59e0b"
            rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            
            macd_color = "#10b981" if macd > signal else "#ef4444"
            macd_status = "Bullish" if macd > signal else "Bearish"
            
            trend_color = "#10b981" if current_price > sma20 > sma50 else "#ef4444" if current_price < sma20 < sma50 else "#f59e0b"
            trend_status = "Uptrend" if current_price > sma20 > sma50 else "Downtrend" if current_price < sma20 < sma50 else "Sideways"
            
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(75,85,99,0.3);">
                    <span style="color: #9ca3af;">RSI (14)</span>
                    <span style="color: {rsi_color}; font-weight: 600;">{current_rsi:.1f} ({rsi_status})</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(75,85,99,0.3);">
                    <span style="color: #9ca3af;">MACD</span>
                    <span style="color: {macd_color}; font-weight: 600;">{macd:.2f} ({macd_status})</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(75,85,99,0.3);">
                    <span style="color: #9ca3af;">Trend</span>
                    <span style="color: {trend_color}; font-weight: 600;">{trend_status}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 12px 0;">
                    <span style="color: #9ca3af;">SMA 20 / 50</span>
                    <span style="color: white;">${sma20:.2f} / ${sma50:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 💰 Fundamentals")
            
            if fundamentals:
                pe = fundamentals.get('pe_ratio', 'N/A')
                pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else pe
                
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(75,85,99,0.3);">
                        <span style="color: #9ca3af;">P/E Ratio</span>
                        <span style="color: white; font-weight: 600;">{pe_str}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(75,85,99,0.3);">
                        <span style="color: #9ca3af;">Market Cap</span>
                        <span style="color: white; font-weight: 600;">{fundamentals.get('market_cap', 'N/A')}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(75,85,99,0.3);">
                        <span style="color: #9ca3af;">52W High</span>
                        <span style="color: white;">${df['High'].max():.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 12px 0;">
                        <span style="color: #9ca3af;">52W Low</span>
                        <span style="color: white;">${df['Low'].min():.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Fundamental data not available")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 12px;">
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: Yahoo Finance<br>
        ⚠️ For educational purposes only. Not financial advice.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
