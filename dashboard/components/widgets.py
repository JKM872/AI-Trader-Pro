"""
UI Components for Professional Trading Dashboard.

Reusable components with consistent styling.
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class SignalDirection(Enum):
    """Signal direction enum."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class MetricData:
    """Data for a metric card."""
    label: str
    value: str
    change: Optional[str] = None
    change_positive: bool = True
    icon: str = ""


def render_top_navbar(
    is_market_open: bool = False,
    session_name: str = "Closed",
    last_update: Optional[datetime] = None
) -> None:
    """Render the top navigation bar."""
    navbar_html = f'''
    <div class="top-navbar">
        <div class="navbar-brand">
            <div class="navbar-logo">AT</div>
            <div>
                <div class="navbar-title">AI Trader Pro</div>
                <div class="navbar-subtitle">Intelligent Trading System</div>
            </div>
        </div>
        <div class="market-status {'open' if is_market_open else 'closed'}">
            <div class="status-dot"></div>
            <span style="color: {'#10b981' if is_market_open else '#ef4444'}; font-weight: 600;">
                Market {session_name}
            </span>
            {f'<span style="color: #6b7280; margin-left: 12px;">Updated: {last_update.strftime("%H:%M:%S")}</span>' if last_update else ''}
        </div>
    </div>
    '''
    st.markdown(navbar_html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    change: Optional[str] = None,
    change_positive: bool = True,
    icon: str = "",
    width: str = "100%"
) -> None:
    """Render a single metric card."""
    change_html = ""
    if change:
        change_class = "positive" if change_positive else "negative"
        arrow = "↑" if change_positive else "↓"
        change_html = f'<div class="metric-change {change_class}">{arrow} {change}</div>'
    
    html = f'''
    <div class="metric-card" style="width: {width};">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def render_metric_row(metrics: List[MetricData]) -> None:
    """Render a row of metric cards."""
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            render_metric_card(
                label=metric.label,
                value=metric.value,
                change=metric.change,
                change_positive=metric.change_positive,
                icon=metric.icon
            )


def render_glass_card(
    title: str,
    content: str,
    icon: str = "",
    actions: Optional[List[Dict[str, str]]] = None
) -> None:
    """Render a glass-morphism card."""
    actions_html = ""
    if actions:
        action_buttons = "".join([
            f'<button class="chart-control-btn">{a.get("label", "")}</button>'
            for a in actions
        ])
        actions_html = f'<div class="chart-controls">{action_buttons}</div>'
    
    html = f'''
    <div class="glass-card animate-fade-in">
        <div class="glass-card-header">
            <div class="glass-card-title">{icon} {title}</div>
            {actions_html}
        </div>
        <div class="glass-card-content">
            {content}
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def render_signal_badge(
    signal_type: Union[str, SignalDirection],
    confidence: Optional[float] = None
) -> str:
    """Generate HTML for signal badge."""
    if isinstance(signal_type, SignalDirection):
        signal_type = signal_type.value
    
    signal_type = signal_type.lower()
    
    icons = {
        'buy': '▲',
        'sell': '▼',
        'hold': '●'
    }
    
    confidence_text = f" ({confidence:.0%})" if confidence else ""
    icon = icons.get(signal_type, '●')
    
    return f'''
    <span class="signal-badge {signal_type}">
        {icon} {signal_type.upper()}{confidence_text}
    </span>
    '''


def render_positions_table(positions: List[Dict[str, Any]]) -> None:
    """Render positions table with professional styling."""
    if not positions:
        st.markdown('''
        <div style="text-align: center; padding: 40px; color: #6b7280;">
            <div style="font-size: 48px; margin-bottom: 16px;">📭</div>
            <div style="font-size: 18px; font-weight: 600;">No Open Positions</div>
            <div style="font-size: 14px; margin-top: 8px;">Start trading to see your positions here</div>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    rows_html = ""
    for pos in positions:
        pnl = pos.get('pnl', 0)
        pnl_pct = pos.get('pnl_pct', 0)
        pnl_class = "positive" if pnl >= 0 else "negative"
        pnl_sign = "+" if pnl >= 0 else ""
        
        rows_html += f'''
        <tr>
            <td style="font-weight: 600;">{pos.get('symbol', 'N/A')}</td>
            <td>{pos.get('side', 'long').upper()}</td>
            <td>{pos.get('quantity', 0):,}</td>
            <td>${pos.get('entry_price', 0):,.2f}</td>
            <td>${pos.get('current_price', 0):,.2f}</td>
            <td class="metric-change {pnl_class}">{pnl_sign}${abs(pnl):,.2f}</td>
            <td class="metric-change {pnl_class}">{pnl_sign}{pnl_pct:.2f}%</td>
        </tr>
        '''
    
    html = f'''
    <table class="positions-table">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Side</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>Current</th>
                <th>P/L</th>
                <th>P/L %</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    '''
    st.markdown(html, unsafe_allow_html=True)


def render_activity_feed(activities: List[Dict[str, Any]], max_items: int = 10) -> None:
    """Render activity feed with recent trades and signals."""
    if not activities:
        st.markdown('''
        <div style="text-align: center; padding: 20px; color: #6b7280;">
            No recent activity
        </div>
        ''', unsafe_allow_html=True)
        return
    
    items_html = ""
    for activity in activities[:max_items]:
        activity_type = activity.get('type', 'info')
        icon_class = activity_type.lower()
        
        icon_map = {
            'buy': '▲',
            'sell': '▼',
            'alert': '⚠',
            'info': 'ℹ'
        }
        icon = icon_map.get(activity_type, 'ℹ')
        
        time_str = ""
        if activity.get('time'):
            time_str = activity['time'].strftime('%H:%M:%S')
        
        items_html += f'''
        <div class="activity-item">
            <div class="activity-icon {icon_class}">{icon}</div>
            <div class="activity-content">
                <div class="activity-title">{activity.get('title', 'Activity')}</div>
                <div class="activity-details">{activity.get('details', '')}</div>
            </div>
            <div class="activity-time">{time_str}</div>
        </div>
        '''
    
    html = f'''
    <div class="activity-feed">
        {items_html}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def render_order_book(bids: List[Dict], asks: List[Dict]) -> None:
    """Render order book with bids and asks."""
    bids_html = "".join([
        f'<div class="order-book-row bid"><span>${b.get("price", 0):.2f}</span><span>{b.get("size", 0):,}</span></div>'
        for b in bids[:5]
    ])
    
    asks_html = "".join([
        f'<div class="order-book-row ask"><span>${a.get("price", 0):.2f}</span><span>{a.get("size", 0):,}</span></div>'
        for a in asks[:5]
    ])
    
    html = f'''
    <div class="order-book">
        <div class="order-book-side">
            <div class="order-book-header">Bids</div>
            {bids_html}
        </div>
        <div class="order-book-side">
            <div class="order-book-header">Asks</div>
            {asks_html}
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def render_live_trading_panel(
    is_running: bool = False,
    strategy_name: str = "",
    last_signal: Optional[str] = None,
    signals_today: int = 0,
    trades_today: int = 0
) -> None:
    """Render live trading status panel."""
    status_indicator = '<div class="trading-status-indicator"></div>' if is_running else ''
    status_text = "LIVE TRADING ACTIVE" if is_running else "TRADING PAUSED"
    status_color = "#10b981" if is_running else "#f59e0b"
    
    html = f'''
    <div class="live-trading-panel">
        <div class="trading-status">
            {status_indicator}
            <span class="trading-status-text" style="color: {status_color};">{status_text}</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-top: 16px;">
            <div>
                <div style="color: #6b7280; font-size: 12px;">Strategy</div>
                <div style="color: #ffffff; font-weight: 600;">{strategy_name or 'Not Set'}</div>
            </div>
            <div>
                <div style="color: #6b7280; font-size: 12px;">Last Signal</div>
                <div style="color: #ffffff; font-weight: 600;">{last_signal or 'N/A'}</div>
            </div>
            <div>
                <div style="color: #6b7280; font-size: 12px;">Signals Today</div>
                <div style="color: #ffffff; font-weight: 600;">{signals_today}</div>
            </div>
            <div>
                <div style="color: #6b7280; font-size: 12px;">Trades Today</div>
                <div style="color: #ffffff; font-weight: 600;">{trades_today}</div>
            </div>
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def render_progress_bar(
    value: float,
    max_value: float = 100,
    variant: str = "primary",
    label: Optional[str] = None,
    show_percentage: bool = True
) -> None:
    """Render a progress bar."""
    percentage = min(100, (value / max_value) * 100) if max_value > 0 else 0
    
    label_html = f'<div style="color: #9ca3af; font-size: 12px; margin-bottom: 4px;">{label}</div>' if label else ""
    percentage_html = f'<span style="color: #9ca3af; font-size: 12px; margin-left: 8px;">{percentage:.1f}%</span>' if show_percentage else ""
    
    html = f'''
    <div style="margin-bottom: 12px;">
        {label_html}
        <div style="display: flex; align-items: center;">
            <div class="progress-bar" style="flex: 1;">
                <div class="progress-bar-fill {variant}" style="width: {percentage}%;"></div>
            </div>
            {percentage_html}
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def render_price_ticker(tickers: List[Dict[str, Any]]) -> None:
    """Render scrolling price ticker."""
    items_html = ""
    for ticker in tickers:
        change = ticker.get('change', 0)
        change_class = "positive" if change >= 0 else "negative"
        change_sign = "+" if change >= 0 else ""
        
        items_html += f'''
        <div class="ticker-item">
            <span class="ticker-symbol">{ticker.get('symbol', '')}</span>
            <span class="ticker-price">${ticker.get('price', 0):,.2f}</span>
            <span class="ticker-change {change_class}">{change_sign}{change:.2f}%</span>
        </div>
        '''
    
    html = f'''
    <div class="price-ticker">
        {items_html}
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def load_custom_css() -> None:
    """Load custom CSS from external file."""
    from pathlib import Path
    
    # Get absolute path to CSS file
    css_path = Path(__file__).parent.parent / 'assets' / 'styles.css'
    
    try:
        css_content = css_path.read_text(encoding='utf-8')
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback to inline critical CSS
        st.markdown('''
        <style>
            .stApp { background: #0a0e17; }
            .metric-card {
                background: rgba(26, 31, 46, 0.8);
                border: 1px solid rgba(75, 85, 99, 0.4);
                border-radius: 12px;
                padding: 20px;
            }
            .signal-badge { padding: 6px 12px; border-radius: 16px; }
            .signal-badge.buy { background: rgba(16, 185, 129, 0.15); color: #10b981; }
            .signal-badge.sell { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
            .signal-badge.hold { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
        </style>
        ''', unsafe_allow_html=True)
