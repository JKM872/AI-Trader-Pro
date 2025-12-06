"""
AI Trader Pro - Main Application Entry Point

Professional trading dashboard with multiple views:
- Live Trading Dashboard
- Multi-Stock Scanner
- Portfolio Management
- Strategy Backtesting

Run with: streamlit run dashboard/main.py
"""

import streamlit as st
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="AI Trader Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/trader',
        'Report a bug': 'https://github.com/yourusername/trader/issues',
        'About': """
        # AI Trader Pro
        
        Professional AI-powered trading dashboard for market analysis,
        signal generation, and portfolio management.
        
        **Features:**
        - Real-time market data
        - Multi-strategy signal scanner
        - Advanced technical analysis
        - Portfolio risk management
        - Automated trading visualization
        
        © 2024 AI Trader Pro
        """
    }
)

# Import components after page config
from dashboard.components import load_custom_css

# Load custom styling
load_custom_css()

# Custom landing page styling
st.markdown("""
<style>
    .landing-hero {
        text-align: center;
        padding: 60px 20px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 24px;
        margin-bottom: 40px;
    }
    
    .landing-title {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 16px;
    }
    
    .landing-subtitle {
        font-size: 20px;
        color: #9ca3af;
        margin-bottom: 32px;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 24px;
        margin: 40px 0;
    }
    
    .feature-card {
        background: rgba(26, 31, 46, 0.8);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.15);
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 16px;
    }
    
    .feature-title {
        font-size: 20px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 8px;
    }
    
    .feature-desc {
        font-size: 14px;
        color: #9ca3af;
        line-height: 1.5;
    }
    
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 48px;
        margin: 32px 0;
        flex-wrap: wrap;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 36px;
        font-weight: 700;
        color: #3b82f6;
    }
    
    .stat-label {
        font-size: 14px;
        color: #6b7280;
    }
    
    .quick-start-btn {
        display: inline-block;
        padding: 16px 48px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 12px;
        text-decoration: none;
        transition: all 0.3s ease;
        margin-top: 24px;
    }
    
    .quick-start-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 40px;">🚀</div>
        <div style="font-size: 24px; font-weight: 700; color: #ffffff; margin-top: 8px;">
            AI Trader Pro
        </div>
        <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">
            Professional Trading Suite
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 🧭 Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Home", "📈 Live Dashboard", "🔍 Scanner", "💼 Portfolio"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("### ℹ️ Quick Info")
    
    # Market status
    from dashboard.components.live_data import MarketStatus
    market = MarketStatus.get_current()
    
    status_color = "#10b981" if market.is_open else "#ef4444"
    status_dot = "🟢" if market.is_open else "🔴"
    
    st.markdown(f"""
    <div style="
        background: rgba(26, 31, 46, 0.8);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 12px;
        padding: 16px;
    ">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <span>{status_dot}</span>
            <span style="color: {status_color}; font-weight: 600;">Market {market.session}</span>
        </div>
        <div style="font-size: 12px; color: #6b7280;">
            Powered by yfinance
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content based on selection
if page == "🏠 Home":
    # Landing page
    st.markdown("""
    <div class="landing-hero">
        <div class="landing-title">AI Trader Pro</div>
        <div class="landing-subtitle">
            Professional AI-powered trading dashboard for intelligent market analysis
        </div>
        
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-value">6+</div>
                <div class="stat-label">Trading Strategies</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">20+</div>
                <div class="stat-label">Technical Indicators</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">Real-time</div>
                <div class="stat-label">Market Data</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("### 🎯 Core Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Live Dashboard</div>
            <div class="feature-desc">
                Real-time price charts, technical indicators, and multi-strategy signal generation
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Signal Scanner</div>
            <div class="feature-desc">
                Scan multiple stocks simultaneously with heat maps and opportunity detection
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💼</div>
            <div class="feature-title">Portfolio</div>
            <div class="feature-desc">
                Track positions, analyze P/L, manage risk limits and monitor performance
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧪</div>
            <div class="feature-title">Backtesting</div>
            <div class="feature-desc">
                Test strategies on historical data with detailed performance metrics
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Trading strategies
    st.markdown("### 🎲 Trading Strategies")
    
    strategies = [
        {
            "name": "Technical Analysis",
            "icon": "📊",
            "desc": "RSI, MACD, Bollinger Bands, Moving Averages",
            "best_for": "Trending Markets"
        },
        {
            "name": "Momentum",
            "icon": "🚀",
            "desc": "Multi-timeframe momentum with volume confirmation",
            "best_for": "Strong Trends"
        },
        {
            "name": "Mean Reversion",
            "icon": "↩️",
            "desc": "Z-score based reversal detection",
            "best_for": "Range-bound Markets"
        },
        {
            "name": "Breakout",
            "icon": "💥",
            "desc": "Support/resistance breakout with ATR",
            "best_for": "Volatility Breakouts"
        },
        {
            "name": "Smart Money",
            "icon": "🏦",
            "desc": "Order blocks, FVG, market structure analysis",
            "best_for": "Institutional Flow"
        },
        {
            "name": "Multi-Timeframe",
            "icon": "⏰",
            "desc": "HTF/MTF/LTF confluence for high-probability entries",
            "best_for": "High Probability"
        }
    ]
    
    cols = st.columns(3)
    for i, strategy in enumerate(strategies):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background: rgba(26, 31, 46, 0.6);
                border: 1px solid rgba(75, 85, 99, 0.3);
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 16px;
            ">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 24px;">{strategy['icon']}</span>
                    <span style="font-size: 16px; font-weight: 600; color: #ffffff;">{strategy['name']}</span>
                </div>
                <div style="font-size: 13px; color: #9ca3af; margin-bottom: 8px;">
                    {strategy['desc']}
                </div>
                <div style="font-size: 11px; color: #3b82f6;">
                    Best for: {strategy['best_for']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start")
    
    st.markdown("""
    <div style="
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 24px;
    ">
        <div style="font-size: 18px; font-weight: 600; color: #ffffff; margin-bottom: 16px;">
            Get Started in 3 Steps
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
            <div>
                <div style="
                    width: 40px; height: 40px;
                    background: #3b82f6;
                    border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-weight: 700; color: white; margin-bottom: 12px;
                ">1</div>
                <div style="color: #ffffff; font-weight: 600; margin-bottom: 4px;">Select Symbols</div>
                <div style="color: #9ca3af; font-size: 13px;">
                    Choose stocks from watchlist or add your own symbols
                </div>
            </div>
            <div>
                <div style="
                    width: 40px; height: 40px;
                    background: #8b5cf6;
                    border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-weight: 700; color: white; margin-bottom: 12px;
                ">2</div>
                <div style="color: #ffffff; font-weight: 600; margin-bottom: 4px;">Analyze Signals</div>
                <div style="color: #9ca3af; font-size: 13px;">
                    View multi-strategy signals and technical analysis
                </div>
            </div>
            <div>
                <div style="
                    width: 40px; height: 40px;
                    background: #10b981;
                    border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-weight: 700; color: white; margin-bottom: 12px;
                ">3</div>
                <div style="color: #ffffff; font-weight: 600; margin-bottom: 4px;">Take Action</div>
                <div style="color: #9ca3af; font-size: 13px;">
                    Execute trades or let the bot trade automatically
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # CTA buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Open Live Dashboard", use_container_width=True, type="primary"):
            st.switch_page("pages/1_📈_Live_Dashboard.py") if Path("pages/1_📈_Live_Dashboard.py").exists() else st.info("Navigate using sidebar")

elif page == "📈 Live Dashboard":
    # Import and run the pro dashboard
    exec(open("dashboard/pro.py").read())

elif page == "🔍 Scanner":
    # Import and run the scanner
    exec(open("dashboard/scanner.py").read())

elif page == "💼 Portfolio":
    # Import and run the portfolio view
    exec(open("dashboard/portfolio_view.py").read())

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 20px; border-top: 1px solid rgba(75, 85, 99, 0.3);">
    <div style="color: #6b7280; font-size: 12px;">
        AI Trader Pro © 2024 | Paper Trading Only | Not Financial Advice
    </div>
</div>
""", unsafe_allow_html=True)
