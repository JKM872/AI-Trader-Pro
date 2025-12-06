"""
AI Trader Dashboard - TradingView Style Redesign
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trader.data.fetcher import DataFetcher
from trader.strategies.technical import TechnicalStrategy
from trader.strategies.momentum import MomentumStrategy
from trader.strategies.mean_reversion import MeanReversionStrategy
from trader.strategies.breakout import BreakoutStrategy
from trader.strategies.smart_money import SmartMoneyStrategy, MultiTimeframeStrategy
from trader.strategies.scanner import SignalScanner, get_signal_summary, ScoredSignal
from trader.analysis.indicators import TradingViewIndicators
from trader.backtest.backtester import Backtester
from trader.data.investor_tracker import PortfolioTracker, FAMOUS_INVESTORS, get_investor_summary


# Page configuration
st.set_page_config(
    page_title="AI Trader Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    """Load local CSS file."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load CSS
try:
    local_css("dashboard/style.css")
except Exception as e:
    st.error(f"Failed to load CSS: {e}")

# --- Custom Components ---

def display_ticker_tape():
    """Display animated ticker tape."""
    indices = [
        {"symbol": "SPX", "value": "4,783.45", "change": "+0.45%"},
        {"symbol": "NDX", "value": "16,832.11", "change": "+0.82%"},
        {"symbol": "DJI", "value": "37,440.34", "change": "-0.12%"},
        {"symbol": "BTC", "value": "43,120.50", "change": "+2.30%"},
        {"symbol": "ETH", "value": "2,250.80", "change": "+1.15%"},
        {"symbol": "EURUSD", "value": "1.0945", "change": "-0.05%"},
        {"symbol": "GOLD", "value": "2,050.10", "change": "+0.30%"},
        {"symbol": "OIL", "value": "72.40", "change": "-1.20%"}
    ]
    
    ticker_html = '<div class="ticker-tape"><div class="ticker-content">'
    for item in indices:
        color_class = "ticker-positive" if "+" in item["change"] else "ticker-negative"
        ticker_html += f'<div class="ticker-item">{item["symbol"]} <span class="{color_class}">{item["value"]} ({item["change"]})</span></div>'
    ticker_html += '</div></div>'
    
    st.markdown(ticker_html, unsafe_allow_html=True)

def display_tv_card(title, value, delta=None):
    """Display TradingView style metric card."""
    delta_html = ""
    if delta:
        color_class = "delta-up" if "+" in delta or (not "-" in delta and not "0%" in delta) else "delta-down"
        delta_html = f'<div class="tv-card-delta {color_class}">{delta}</div>'
    
    html = f"""
    <div class="tv-card">
        <div class="tv-card-title">{title}</div>
        <div class="tv-card-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def display_tech_gauge(score):
    """Display technical rating gauge."""
    # Score from 0 to 100
    pos = max(0, min(100, score))
    
    label = "NEUTRAL"
    color = "#999"
    if score >= 80: label, color = "STRONG BUY", "#089981"
    elif score >= 60: label, color = "BUY", "#089981"
    elif score <= 20: label, color = "STRONG SELL", "#f23645"
    elif score <= 40: label, color = "SELL", "#f23645"
        
    html = f"""
    <div style="text-align: center;">
        <div style="color: {color}; font-weight: bold; margin-bottom: 5px;">{label}</div>
        <div class="tech-gauge">
            <div class="gauge-needle" style="left: {pos}%;"></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- Data Functions ---

@st.cache_data(ttl=300)
def fetch_stock_data(symbol: str, period: str = '6mo') -> pd.DataFrame:
    fetcher = DataFetcher()
    return fetcher.get_stock_data(symbol, period=period)

@st.cache_data(ttl=3600)
def fetch_fundamentals(symbol: str) -> dict:
    fetcher = DataFetcher()
    return fetcher.get_fundamentals(symbol)

# --- Chart Functions (Redesigned) ---

def apply_tv_style(fig):
    """Apply TradingView styling to Plotly figure."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            showgrid=True, gridcolor='#2a2e39',
            showline=True, linecolor='#2a2e39',
            zeroline=False,
            showspikes=True, spikemode='across', spikesnap='cursor',
            spikedash='dash', spikecolor='#787b86', spikethickness=1
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#2a2e39',
            showline=True, linecolor='#2a2e39',
            zeroline=False,
            showspikes=True, spikemode='across', spikesnap='cursor',
            spikedash='dash', spikecolor='#787b86', spikethickness=1
        ),
        font=dict(family="sans-serif", size=11, color="#d1d4dc"),
        hovermode='x unified',
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor='rgba(0,0,0,0)'
        )
    )
    return fig

def create_candlestick_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f'{symbol}', 'Vol')
    )
    
    # Candlestick (TV Colors)
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price',
            increasing_line_color='#089981', increasing_fillcolor='#089981',
            decreasing_line_color='#f23645', decreasing_fillcolor='#f23645'
        ),
        row=1, col=1
    )
    
    # MAs
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA 20', line=dict(color='#ff9800', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50', line=dict(color='#2962ff', width=1)), row=1, col=1)
    
    # Volume
    colors = ['#f23645' if c < o else '#089981' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.5),
        row=2, col=1
    )
    
    return apply_tv_style(fig)

def create_indicator_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#b22833')), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#787b86", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#787b86", row=1, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="#2962ff", opacity=0.1, line_width=0, row=1, col=1)

    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    
    fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='#2962ff')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='#ff9800')), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=hist, name='Hist', marker_color=['#089981' if h > 0 else '#f23645' for h in hist]), row=2, col=1)
    
    return apply_tv_style(fig)

def create_bollinger_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA20'] + (df['STD20'] * 2)
    df['Lower'] = df['SMA20'] - (df['STD20'] * 2)
    
    fig = create_candlestick_chart(df, symbol) # Base
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='BB Upper', line=dict(color='#00bcd4', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='BB Lower', line=dict(color='#00bcd4', width=1), fill='tonexty', fillcolor='rgba(0,188,212,0.05)'), row=1, col=1)
    
    return fig

def create_volume_profile(df: pd.DataFrame, symbol: str) -> go.Figure:
    price_range = df['Close'].max() - df['Close'].min()
    bin_size = price_range / 24
    df['PriceBin'] = ((df['Close'] - df['Close'].min()) / bin_size).astype(int)
    vp = df.groupby('PriceBin')['Volume'].sum()
    price_levels = [df['Close'].min() + (i * bin_size) for i in vp.index]
    
    fig = make_subplots(rows=1, cols=2, column_widths=[0.85, 0.15], shared_yaxes=True, horizontal_spacing=0.01)
    
    # Candles
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#089981', decreasing_line_color='#f23645'), row=1, col=1)
    
    # Profile
    fig.add_trace(go.Bar(x=vp.values, y=price_levels, orientation='h', name='Vol Profile', marker_color='#787b86', opacity=0.3), row=1, col=2)
    
    return apply_tv_style(fig)

def create_returns_comparison(symbols: list, period: str) -> go.Figure:
    fig = go.Figure()
    fetcher = DataFetcher()
    for sym in symbols:
        try:
            df = fetcher.get_stock_data(sym, period)
            if not df.empty:
                ret = (1 + df['Close'].pct_change()).cumprod() - 1
                fig.add_trace(go.Scatter(x=df.index, y=ret*100, name=sym))
        except: pass
    return apply_tv_style(fig)

def create_correlation_heatmap(symbols: list, period: str):
    fetcher = DataFetcher()
    prices = {}
    for sym in symbols:
        try:
            df = fetcher.get_stock_data(sym, period)
            prices[sym] = df['Close']
        except: pass
    
    if len(prices) < 2: return None
    
    corr = pd.DataFrame(prices).pct_change().corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale='RdYlGn', zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate='%{text}',
    ))
    return apply_tv_style(fig)

# --- Logic Functions ---

def get_strategy(name, sl, tp):
    strategies = {
        'Technical': TechnicalStrategy, 'Momentum': MomentumStrategy,
        'Mean Reversion': MeanReversionStrategy, 'Breakout': BreakoutStrategy,
        'Smart Money': SmartMoneyStrategy, 'Multi-Timeframe': MultiTimeframeStrategy
    }
    return strategies.get(name, TechnicalStrategy)(stop_loss_pct=sl, take_profit_pct=tp)

def display_signal(signal):
    st_val = signal.signal_type.value
    score = 50
    if st_val == 'BUY': score = 50 + (signal.confidence * 50)
    elif st_val == 'SELL': score = 50 - (signal.confidence * 50)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        display_tech_gauge(score)
    with col2:
        st.markdown(f"### {signal.symbol}")
        st.markdown(f"**Signal:** {st_val} (Conf: {signal.confidence:.0%})")
        st.markdown(f"**Target:** {signal.take_profit:.2f} | **Stop:** {signal.stop_loss:.2f}")

# --- Page Runners ---

def run_single_stock_page(default_symbols):
    # Sidebar
    st.sidebar.markdown("### 🔍 Watchlist")
    symbol = st.sidebar.selectbox("Symbol", default_symbols, label_visibility="collapsed")
    custom = st.sidebar.text_input("Look up symbol", placeholder="e.g. BTC-USD")
    if custom: symbol = custom.upper()
    
    period = st.sidebar.select_slider("Period", options=['1mo', '3mo', '6mo', '1y', '2y', '5y'], value='6mo')
    
    # Data
    df = fetch_stock_data(symbol, period)
    if df.empty:
        st.error(f"No data for {symbol}")
        return
        
    curr = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2]
    chg = curr - prev
    pct = (chg/prev)*100
    
    # Header Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1: display_tv_card("Last Price", f"{curr:.2f}", f"{chg:+.2f} ({pct:+.2f}%)")
    with c2: display_tv_card("High (52W)", f"{df['High'].max():.2f}")
    with c3: display_tv_card("Low (52W)", f"{df['Low'].min():.2f}")
    with c4: display_tv_card("Volume", f"{df['Volume'].mean()/1e6:.1f}M")
    
    # Main Chart Area
    tabs = st.tabs(["Chart", "Technical Indicators", "Fundamentals"])
    
    with tabs[0]:
        fig = create_candlestick_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_indicator_chart(df), use_container_width=True)
        with c2: st.plotly_chart(create_volume_profile(df, symbol), use_container_width=True)
        
    with tabs[2]:
        fund = fetch_fundamentals(symbol)
        if fund:
             st.json(fund)
        else: st.warning("No fundamentals.")

    # Strategy Scanner
    st.markdown("---")
    st.subheader("⚡ Strategy Scanner")
    strat = st.selectbox("Strategy", ['Technical', 'Momentum', 'Smart Money'])
    conf_sl, conf_tp = st.columns(2)
    sl = conf_sl.slider("Stop Loss", 0.01, 0.20, 0.05)
    tp = conf_tp.slider("Take Profit", 0.01, 0.50, 0.10)
    
    if st.button("Run Analysis"):
        s = get_strategy(strat, sl, tp)
        sig = s.generate_signal(symbol, df)
        display_signal(sig)

def run_scanner_page(default_symbols):
    st.title("🎯 Market Scanner")
    symbols = st.multiselect("Watchlist", default_symbols + ['AMD', 'INTC', 'NVDA'], default=default_symbols[:4])
    if st.button("Scan Market"):
        scanner = SignalScanner()
        res = scanner.scan_watchlist(symbols)
        
        # Display as TV list
        for sym, sigs in res.items():
            for s in sigs:
                if s.score > 50:
                    st.markdown(f"""
                    <div class="tv-card">
                        <div style="display:flex; justify-content:space-between;">
                            <span><b>{sym}</b></span>
                            <span class="{'ticker-positive' if s.signal.signal_type.value == 'BUY' else 'ticker-negative'}">
                                {s.signal.signal_type.value} ({s.score})
                            </span>
                        </div>
                        <div style="font-size:0.8rem; color:#787b86;">Strategy: {s.strategy_name}</div>
                    </div>
                    """, unsafe_allow_html=True)

def run_charts_page(default_symbols):
    st.title("📉 Advanced Charts")
    symbol = st.sidebar.selectbox("Symbol", default_symbols)
    period = st.sidebar.selectbox("Period", ['1mo', '6mo', '1y'], index=1)
    
    df = fetch_stock_data(symbol, period)
    if df.empty: return
    
    # TV Indicators
    tvi = TradingViewIndicators()
    
    fig = create_candlestick_chart(df, symbol)
    
    # Supertrend
    if st.checkbox("Supertrend", True):
        line, direction = tvi.supertrend(df)
        fig.add_trace(go.Scatter(x=df.index, y=line, name='Supertrend', line=dict(color='purple')))
        
    st.plotly_chart(fig, use_container_width=True)

def run_portfolio(default_symbols):
    st.title("💼 Portfolio")
    # Simple placeholder redesign
    st.info("Portfolio View - Work in Progress in Redesign Mode")

def run_investors_page(default_symbols):
    """Famous Investor Portfolios page."""
    st.title("🏆 Famous Investor Portfolios")
    st.markdown("Track what legendary investors are holding via SEC 13F filings.")
    
    # Get investor summary
    summary = get_investor_summary()
    
    # Display investor cards
    st.subheader("📊 Tracked Investors")
    
    cols = st.columns(3)
    for i, inv in enumerate(summary["investors"]):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="tv-card">
                <div class="tv-card-title">{inv['fund']}</div>
                <div class="tv-card-value">{inv['name']}</div>
                <div style="font-size:0.8rem; color:#787b86; margin-top:5px;">{inv['strategy']}</div>
                <div style="font-size:0.75rem; color:#565a66; margin-top:3px;">{inv['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check which stocks from watchlist are held by gurus
    st.subheader("🔍 Check Your Watchlist")
    st.markdown("See which of your stocks are held by famous investors.")
    
    selected_symbols = st.multiselect(
        "Select symbols to check",
        default_symbols + ['AMD', 'INTC', 'PLTR', 'COIN', 'ABNB'],
        default=default_symbols[:4]
    )
    
    if st.button("🔎 Check Institutional Ownership"):
        if selected_symbols:
            tracker = PortfolioTracker()
            try:
                for symbol in selected_symbols:
                    with st.expander(f"📈 {symbol}", expanded=True):
                        ownership = tracker.get_stock_institutional_owners(symbol)
                        
                        if "error" in ownership:
                            st.warning(f"Could not fetch data: {ownership['error']}")
                            continue
                        
                        # Major holders summary
                        major = ownership.get("major_holders", {})
                        if major:
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                display_tv_card("Insiders", str(major.get('insiders_pct', 'N/A')))
                            with c2:
                                display_tv_card("Institutions", str(major.get('institutions_pct', 'N/A')))
                            with c3:
                                display_tv_card("# Institutions", str(major.get('institutions_count', 'N/A')))
                        
                        # Famous investors
                        famous = ownership.get("famous_investors", [])
                        if famous:
                            st.success(f"🌟 Held by {len(famous)} famous investor(s)!")
                            for inv in famous:
                                st.markdown(f"- **{inv['investor']}** ({inv['fund']}): {inv.get('shares', 'N/A'):,} shares")
                        else:
                            st.info("No famous investors detected in top holders.")
                        
                        # Top institutional holders
                        inst = ownership.get("institutional_holders", [])
                        if inst:
                            st.markdown("**Top Institutional Holders:**")
                            df = pd.DataFrame(inst[:10])
                            if not df.empty:
                                st.dataframe(df, use_container_width=True)
            finally:
                tracker.close()
        else:
            st.warning("Please select at least one symbol.")

# --- Main App ---

def main():
    display_ticker_tape()
    
    # Custom Sidebar Navigation
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/GitHub_Invertocat_Logo.svg/1200px-GitHub_Invertocat_Logo.svg.png", width=50)
    st.sidebar.title("Pro Terminal")
    
    st.sidebar.markdown("---")
    
    menu_options = [
        "🖥️ Terminal", 
        "⚡ Scanner", 
        "🔎 Screeners", 
        "🏆 Investors",
        "🧪 Backtest", 
        "💼 Portfolio"
    ]
    
    menu = st.sidebar.radio(
        "Navigation", 
        menu_options,
        label_visibility="collapsed"
    )
    
    default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BTC-USD', 'ETH-USD']
    
    if menu == "🖥️ Terminal":
        run_single_stock_page(default_symbols)
    elif menu == "⚡ Scanner":
        run_scanner_page(default_symbols)
    elif menu == "🔎 Screeners":
        run_charts_page(default_symbols)
    elif menu == "🏆 Investors":
        run_investors_page(default_symbols)
    elif menu == "🧪 Backtest":
        st.title("Strategy Tester")
        run_single_stock_page(default_symbols) # Placeholder for demo
    else:
        run_portfolio(default_symbols)

if __name__ == "__main__":
    main()
