"""
AI Trader Pro - Portfolio Management Dashboard

Advanced portfolio tracking, risk analytics, and performance monitoring.

Run with: streamlit run dashboard/portfolio_view.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.components import (
    load_custom_css,
    render_top_navbar,
    render_metric_row,
    render_positions_table,
    render_activity_feed,
    render_progress_bar,
    create_portfolio_donut,
    create_equity_curve,
    create_heatmap_calendar,
    MetricData,
    MarketStatus,
    get_activity_log,
    CHART_THEME
)

from trader.data.fetcher import DataFetcher
from trader.portfolio.portfolio import Portfolio, PortfolioMetrics
from trader.risk.risk_manager import RiskManager, RiskLimits


# Page configuration
st.set_page_config(
    page_title="AI Trader - Portfolio",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio(initial_capital=100000)

if 'risk_manager' not in st.session_state:
    limits = RiskLimits(
        max_position_size_pct=0.15,
        risk_per_trade_pct=0.02,
        max_drawdown_pct=0.10
    )
    st.session_state.risk_manager = RiskManager(limits=limits, initial_value=100000)

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []


@st.cache_data(ttl=60)
def fetch_current_price(symbol: str) -> float:
    """Fetch current price for a symbol."""
    try:
        fetcher = DataFetcher()
        df = fetcher.get_stock_data(symbol, period='5d')
        if not df.empty:
            return df['Close'].iloc[-1]
    except Exception:
        pass
    return 0.0


@st.cache_data(ttl=300)
def fetch_historical_data(symbol: str, period: str = '1y') -> pd.DataFrame:
    """Fetch historical data."""
    fetcher = DataFetcher()
    return fetcher.get_stock_data(symbol, period=period)


def update_position_prices(portfolio: Portfolio) -> Dict[str, float]:
    """Update all position prices."""
    prices = {}
    for symbol in portfolio.positions.keys():
        price = fetch_current_price(symbol)
        if price > 0:
            prices[symbol] = price
    return prices


def create_performance_chart(dates: pd.DatetimeIndex, values: pd.Series, benchmark: Optional[pd.Series] = None) -> go.Figure:
    """Create performance comparison chart."""
    fig = go.Figure()
    
    # Portfolio performance
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        name='Portfolio',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    # Benchmark if provided
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark,
            name='S&P 500',
            line=dict(color='#f59e0b', width=2, dash='dot')
        ))
    
    fig.update_layout(
        paper_bgcolor=CHART_THEME['bg_color'],
        plot_bgcolor=CHART_THEME['plot_bg'],
        height=400,
        margin=dict(l=50, r=20, t=40, b=50),
        title=dict(
            text='📈 Performance vs Benchmark',
            font=dict(color='#ffffff', size=16),
            x=0.02
        ),
        xaxis=dict(
            gridcolor='rgba(75, 85, 99, 0.2)',
            tickfont=dict(color='#9ca3af')
        ),
        yaxis=dict(
            title='Return %',
            gridcolor='rgba(75, 85, 99, 0.2)',
            tickfont=dict(color='#9ca3af'),
            titlefont=dict(color='#9ca3af')
        ),
        legend=dict(
            bgcolor='rgba(17, 24, 39, 0.8)',
            bordercolor='rgba(75, 85, 99, 0.4)',
            font=dict(color='#9ca3af')
        ),
        hovermode='x unified'
    )
    
    return fig


def create_risk_gauge(value: float, max_value: float, title: str, color: str) -> go.Figure:
    """Create a gauge chart for risk metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': '#9ca3af', 'size': 14}},
        number={'font': {'color': '#ffffff', 'size': 28}, 'suffix': '%'},
        gauge={
            'axis': {
                'range': [0, max_value],
                'tickwidth': 1,
                'tickcolor': '#6b7280',
                'tickfont': {'color': '#6b7280'}
            },
            'bar': {'color': color},
            'bgcolor': '#1a1f2e',
            'borderwidth': 0,
            'steps': [
                {'range': [0, max_value * 0.5], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [max_value * 0.5, max_value * 0.75], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [max_value * 0.75, max_value], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': '#ef4444', 'width': 2},
                'thickness': 0.75,
                'value': max_value * 0.75
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_allocation_treemap(positions: Dict[str, float]) -> go.Figure:
    """Create a treemap of portfolio allocation."""
    if not positions:
        return go.Figure()
    
    symbols = list(positions.keys())
    values = list(positions.values())
    
    # Create color scale based on value
    colors = px.colors.qualitative.Set2[:len(symbols)]
    
    fig = go.Figure(go.Treemap(
        labels=symbols,
        parents=[''] * len(symbols),
        values=values,
        textinfo='label+value+percent parent',
        textfont=dict(size=14, color='white'),
        marker=dict(
            colors=colors,
            line=dict(width=2, color=CHART_THEME['bg_color'])
        ),
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>%{percentParent:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor=CHART_THEME['bg_color'],
        plot_bgcolor=CHART_THEME['plot_bg'],
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text='📊 Portfolio Allocation',
            font=dict(color='#ffffff', size=16),
            x=0.02
        )
    )
    
    return fig


def create_position_pnl_chart(positions_data: List[Dict]) -> go.Figure:
    """Create P/L bar chart for positions."""
    if not positions_data:
        return go.Figure()
    
    symbols = [p['symbol'] for p in positions_data]
    pnl_values = [p['pnl'] for p in positions_data]
    pnl_pct = [p['pnl_pct'] for p in positions_data]
    colors = ['#10b981' if p >= 0 else '#ef4444' for p in pnl_values]
    
    fig = go.Figure(data=go.Bar(
        x=symbols,
        y=pnl_values,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f'{p:+.1f}%' for p in pnl_pct],
        textposition='outside',
        textfont=dict(color='#9ca3af'),
        hovertemplate='<b>%{x}</b><br>P/L: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor=CHART_THEME['bg_color'],
        plot_bgcolor=CHART_THEME['plot_bg'],
        height=300,
        margin=dict(l=50, r=20, t=40, b=50),
        title=dict(
            text='💰 Position P/L',
            font=dict(color='#ffffff', size=16),
            x=0.02
        ),
        xaxis=dict(
            tickfont=dict(color='#9ca3af'),
            gridcolor='rgba(75, 85, 99, 0.2)'
        ),
        yaxis=dict(
            title='P/L ($)',
            tickfont=dict(color='#9ca3af'),
            titlefont=dict(color='#9ca3af'),
            gridcolor='rgba(75, 85, 99, 0.2)'
        )
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    
    return fig


def render_trade_form():
    """Render the trade entry form."""
    st.markdown("### 📝 New Trade")
    
    with st.form("trade_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
            quantity = st.number_input("Quantity", min_value=1, value=100)
            side = st.selectbox("Side", ['BUY', 'SELL'])
        
        with col2:
            price = st.number_input("Price", min_value=0.01, value=100.0, format="%.2f")
            stop_loss = st.number_input("Stop Loss", min_value=0.0, value=0.0, format="%.2f")
            take_profit = st.number_input("Take Profit", min_value=0.0, value=0.0, format="%.2f")
        
        submitted = st.form_submit_button("Execute Trade", use_container_width=True, type="primary")
        
        if submitted and symbol:
            portfolio = st.session_state.portfolio
            
            if side == 'BUY':
                try:
                    portfolio.add_position(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        stop_loss=stop_loss if stop_loss > 0 else None,
                        take_profit=take_profit if take_profit > 0 else None
                    )
                    
                    st.session_state.trade_history.append({
                        'type': 'buy',
                        'title': f'BUY {symbol}',
                        'details': f'{quantity} shares @ ${price:.2f}',
                        'time': datetime.now()
                    })
                    
                    st.success(f"✅ Bought {quantity} shares of {symbol} @ ${price:.2f}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                try:
                    pnl = portfolio.close_position(symbol, price)
                    
                    st.session_state.trade_history.append({
                        'type': 'sell',
                        'title': f'SELL {symbol}',
                        'details': f'{quantity} shares @ ${price:.2f} | P/L: ${pnl:.2f}',
                        'time': datetime.now()
                    })
                    
                    st.success(f"✅ Sold {symbol} @ ${price:.2f} | P/L: ${pnl:+.2f}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


def main():
    """Main application entry point."""
    market_status = MarketStatus.get_current()
    portfolio = st.session_state.portfolio
    risk_manager = st.session_state.risk_manager
    
    # Update position prices
    update_position_prices(portfolio)
    
    # Get metrics
    metrics = portfolio.get_metrics()
    
    # Top navigation
    render_top_navbar(
        is_market_open=market_status.is_open,
        session_name=market_status.session,
        last_update=datetime.now()
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 💼 Portfolio Management")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Portfolio Controls")
        
        # Account summary
        st.markdown("#### 💰 Account Summary")
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
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #9ca3af;">Invested</span>
                <span style="color: #ffffff;">${metrics.positions_value:,.2f}</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.divider()
        
        # Risk limits
        st.markdown("#### ⚠️ Risk Settings")
        
        max_position = st.slider(
            "Max Position Size",
            5, 30, int(risk_manager.limits.max_position_size_pct * 100),
            format="%d%%"
        )
        
        risk_per_trade = st.slider(
            "Risk per Trade",
            0.5, 5.0, risk_manager.limits.risk_per_trade_pct * 100,
            step=0.5,
            format="%.1f%%"
        )
        
        max_drawdown = st.slider(
            "Max Drawdown",
            5, 25, int(risk_manager.limits.max_drawdown_pct * 100),
            format="%d%%"
        )
        
        if st.button("Update Risk Limits", use_container_width=True):
            risk_manager.limits.max_position_size_pct = max_position / 100
            risk_manager.limits.risk_per_trade_pct = risk_per_trade / 100
            risk_manager.limits.max_drawdown_pct = max_drawdown / 100
            st.success("Risk limits updated!")
        
        st.divider()
        
        # Quick actions
        st.markdown("#### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Prices", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📊 Export Report", use_container_width=True):
            st.info("Report generation coming soon!")
    
    # Main content - Top metrics
    pnl_color = metrics.unrealized_pnl >= 0
    day_pnl_color = metrics.daily_pnl >= 0 if hasattr(metrics, 'daily_pnl') else True
    
    render_metric_row([
        MetricData(
            label="Total Value",
            value=f"${metrics.total_value:,.2f}",
            icon="💰"
        ),
        MetricData(
            label="Unrealized P/L",
            value=f"${metrics.unrealized_pnl:+,.2f}",
            change=f"{metrics.unrealized_pnl_pct:+.2f}%",
            change_positive=pnl_color,
            icon="📈"
        ),
        MetricData(
            label="Cash Available",
            value=f"${metrics.cash_balance:,.2f}",
            icon="💵"
        ),
        MetricData(
            label="Positions",
            value=str(len(portfolio.positions)),
            icon="📊"
        )
    ])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for different views
    tab_overview, tab_positions, tab_risk, tab_trade = st.tabs([
        "📊 Overview", "📋 Positions", "⚠️ Risk", "📝 Trade"
    ])
    
    with tab_overview:
        col1, col2 = st.columns(2)
        
        with col1:
            # Portfolio allocation
            positions = portfolio.positions
            if positions:
                allocation = {}
                for symbol, pos in positions.items():
                    price = fetch_current_price(symbol) or pos.avg_cost
                    allocation[symbol] = pos.quantity * price
                
                # Add cash
                allocation['Cash'] = metrics.cash_balance
                
                fig_treemap = create_allocation_treemap(allocation)
                st.plotly_chart(fig_treemap, use_container_width=True, key="allocation")
            else:
                st.info("No positions to display. Add some trades to see allocation.")
        
        with col2:
            # Position P/L
            if positions:
                positions_data = []
                for symbol, pos in positions.items():
                    price = fetch_current_price(symbol) or pos.avg_cost
                    pnl = pos.calculate_pnl(price)
                    pnl_pct = pos.calculate_pnl_pct(price)
                    positions_data.append({
                        'symbol': symbol,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
                
                fig_pnl = create_position_pnl_chart(positions_data)
                st.plotly_chart(fig_pnl, use_container_width=True, key="pnl_chart")
            else:
                st.info("No positions to display.")
        
        # Performance chart (simulated for now)
        st.markdown("### 📈 Performance History")
        
        # Generate sample performance data
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, len(dates))
        portfolio_value = 100000 * (1 + returns).cumprod()
        
        # Benchmark (S&P 500 simulation)
        benchmark_returns = np.random.normal(0.0005, 0.015, len(dates))
        benchmark_value = 100000 * (1 + benchmark_returns).cumprod()
        
        # Convert to percentage returns
        portfolio_pct = (portfolio_value / 100000 - 1) * 100
        benchmark_pct = (benchmark_value / 100000 - 1) * 100
        
        fig_perf = create_performance_chart(
            dates, 
            pd.Series(portfolio_pct, index=dates),
            pd.Series(benchmark_pct, index=dates)
        )
        st.plotly_chart(fig_perf, use_container_width=True, key="performance")
        
        # Monthly returns heatmap
        daily_returns = pd.Series(returns, index=dates)
        fig_calendar = create_heatmap_calendar(daily_returns, height=250)
        st.plotly_chart(fig_calendar, use_container_width=True, key="calendar")
    
    with tab_positions:
        st.markdown("### 📋 Open Positions")
        
        positions = portfolio.positions
        
        if positions:
            positions_data = []
            for symbol, pos in positions.items():
                price = fetch_current_price(symbol) or pos.avg_cost
                pnl = pos.calculate_pnl(price)
                pnl_pct = pos.calculate_pnl_pct(price)
                positions_data.append({
                    'symbol': symbol,
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.avg_cost,
                    'current_price': price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                })
            
            render_positions_table(positions_data)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Position details
            st.markdown("### 📊 Position Details")
            
            for pos_data in positions_data:
                with st.expander(f"📈 {pos_data['symbol']} - {pos_data['side'].upper()}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Entry Price</div>
                            <div class="metric-value">${pos_data['entry_price']:.2f}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Current Price</div>
                            <div class="metric-value">${pos_data['current_price']:.2f}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col3:
                        pnl_color = '#10b981' if pos_data['pnl'] >= 0 else '#ef4444'
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">P/L</div>
                            <div class="metric-value" style="color: {pnl_color};">
                                ${pos_data['pnl']:+,.2f} ({pos_data['pnl_pct']:+.2f}%)
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Chart for the position
                    df = fetch_historical_data(pos_data['symbol'], '3mo')
                    if not df.empty:
                        from dashboard.components.charts import create_professional_candlestick
                        fig = create_professional_candlestick(df, pos_data['symbol'], height=350)
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{pos_data['symbol']}")
                    
                    # Close position button
                    if st.button(f"❌ Close Position", key=f"close_{pos_data['symbol']}"):
                        try:
                            pnl = portfolio.close_position(pos_data['symbol'], pos_data['current_price'])
                            st.success(f"Closed {pos_data['symbol']} with P/L: ${pnl:+,.2f}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.markdown('''
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 64px; margin-bottom: 24px;">📭</div>
                <div style="font-size: 24px; font-weight: 600; color: #ffffff; margin-bottom: 12px;">
                    No Open Positions
                </div>
                <div style="color: #9ca3af;">
                    Go to the Trade tab to open new positions.
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    with tab_risk:
        st.markdown("### ⚠️ Risk Analytics")
        
        # Risk gauges
        col1, col2, col3 = st.columns(3)
        
        # Calculate risk metrics
        positions_value = metrics.positions_value
        total_value = metrics.total_value
        exposure_pct = (positions_value / total_value * 100) if total_value > 0 else 0
        
        with col1:
            fig_exposure = create_risk_gauge(
                exposure_pct, 100, "Portfolio Exposure", "#3b82f6"
            )
            st.plotly_chart(fig_exposure, use_container_width=True, key="gauge_exposure")
        
        with col2:
            # Simulated max drawdown
            max_dd = abs(metrics.unrealized_pnl_pct) if metrics.unrealized_pnl < 0 else 0
            fig_drawdown = create_risk_gauge(
                max_dd, risk_manager.limits.max_drawdown_pct * 100, "Current Drawdown", 
                "#ef4444" if max_dd > 5 else "#f59e0b" if max_dd > 2 else "#10b981"
            )
            st.plotly_chart(fig_drawdown, use_container_width=True, key="gauge_drawdown")
        
        with col3:
            # Concentration risk
            positions = portfolio.positions
            max_position_pct = 0
            if positions and total_value > 0:
                for symbol, pos in positions.items():
                    price = fetch_current_price(symbol) or pos.avg_cost
                    pos_value = pos.quantity * price
                    pos_pct = pos_value / total_value * 100
                    max_position_pct = max(max_position_pct, pos_pct)
            
            fig_concentration = create_risk_gauge(
                max_position_pct, risk_manager.limits.max_position_size_pct * 100,
                "Max Position Size", 
                "#ef4444" if max_position_pct > 15 else "#f59e0b" if max_position_pct > 10 else "#10b981"
            )
            st.plotly_chart(fig_concentration, use_container_width=True, key="gauge_concentration")
        
        st.markdown("### 📊 Risk Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('''
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">📋 Risk Limits Status</div>
                </div>
            ''', unsafe_allow_html=True)
            
            # Max position size
            render_progress_bar(
                max_position_pct,
                risk_manager.limits.max_position_size_pct * 100,
                variant='danger' if max_position_pct > 15 else 'primary',
                label='Max Position Size'
            )
            
            # Drawdown
            render_progress_bar(
                max_dd,
                risk_manager.limits.max_drawdown_pct * 100,
                variant='danger' if max_dd > 5 else 'primary',
                label='Drawdown Limit'
            )
            
            # Exposure
            render_progress_bar(
                exposure_pct,
                100,
                variant='primary',
                label='Portfolio Exposure'
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="glass-card">
                <div class="glass-card-header">
                    <div class="glass-card-title">📈 Risk Metrics</div>
                </div>
            ''', unsafe_allow_html=True)
            
            # Simulated risk metrics
            st.markdown(f'''
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div>
                    <div style="color: #9ca3af; font-size: 12px;">Value at Risk (1d, 95%)</div>
                    <div style="color: #ffffff; font-size: 18px; font-weight: 600;">${total_value * 0.02:,.2f}</div>
                </div>
                <div>
                    <div style="color: #9ca3af; font-size: 12px;">Expected Shortfall</div>
                    <div style="color: #ffffff; font-size: 18px; font-weight: 600;">${total_value * 0.03:,.2f}</div>
                </div>
                <div>
                    <div style="color: #9ca3af; font-size: 12px;">Sharpe Ratio</div>
                    <div style="color: #ffffff; font-size: 18px; font-weight: 600;">1.42</div>
                </div>
                <div>
                    <div style="color: #9ca3af; font-size: 12px;">Beta</div>
                    <div style="color: #ffffff; font-size: 18px; font-weight: 600;">1.12</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab_trade:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_trade_form()
        
        with col2:
            st.markdown("### 📜 Recent Trades")
            
            if st.session_state.trade_history:
                render_activity_feed(st.session_state.trade_history[-10:])
            else:
                st.markdown('''
                <div style="text-align: center; padding: 40px; color: #6b7280;">
                    No recent trades
                </div>
                ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
