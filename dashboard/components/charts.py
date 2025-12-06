"""
Chart Components for Professional Trading Dashboard.

High-performance interactive charts with TradingView-style aesthetics.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime


# Professional Dark Theme for all charts
CHART_THEME = {
    'bg_color': '#0a0e17',
    'plot_bg': '#111827',
    'grid_color': 'rgba(75, 85, 99, 0.2)',
    'text_color': '#9ca3af',
    'title_color': '#ffffff',
    'up_color': '#10b981',
    'down_color': '#ef4444',
    'volume_up': 'rgba(16, 185, 129, 0.5)',
    'volume_down': 'rgba(239, 68, 68, 0.5)',
    'ma_colors': {
        'sma20': '#3b82f6',
        'sma50': '#8b5cf6',
        'sma200': '#f59e0b',
        'ema9': '#ec4899',
        'ema21': '#06b6d4'
    },
    'indicator_colors': {
        'rsi': '#8b5cf6',
        'macd': '#3b82f6',
        'signal': '#f59e0b',
        'histogram_pos': 'rgba(16, 185, 129, 0.7)',
        'histogram_neg': 'rgba(239, 68, 68, 0.7)',
        'bb_upper': 'rgba(139, 92, 246, 0.4)',
        'bb_lower': 'rgba(139, 92, 246, 0.4)',
        'bb_mid': '#8b5cf6'
    }
}


def create_layout_template() -> Dict[str, Any]:
    """Create base layout template for all charts."""
    return {
        'paper_bgcolor': CHART_THEME['bg_color'],
        'plot_bgcolor': CHART_THEME['plot_bg'],
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            'color': CHART_THEME['text_color'],
            'size': 12
        },
        'title': {
            'font': {
                'color': CHART_THEME['title_color'],
                'size': 16
            }
        },
        'xaxis': {
            'gridcolor': CHART_THEME['grid_color'],
            'zerolinecolor': CHART_THEME['grid_color'],
            'tickfont': {'color': CHART_THEME['text_color']},
            'showgrid': True,
            'rangeslider': {'visible': False}
        },
        'yaxis': {
            'gridcolor': CHART_THEME['grid_color'],
            'zerolinecolor': CHART_THEME['grid_color'],
            'tickfont': {'color': CHART_THEME['text_color']},
            'showgrid': True,
            'side': 'right'
        },
        'legend': {
            'bgcolor': 'rgba(17, 24, 39, 0.8)',
            'bordercolor': 'rgba(75, 85, 99, 0.4)',
            'borderwidth': 1,
            'font': {'color': CHART_THEME['text_color']}
        },
        'margin': {'l': 10, 'r': 60, 't': 40, 'b': 40},
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': '#1a1f2e',
            'bordercolor': '#3b82f6',
            'font': {'color': '#ffffff'}
        }
    }


def create_professional_candlestick(
    df: pd.DataFrame,
    symbol: str,
    show_volume: bool = True,
    show_ma: bool = True,
    ma_periods: List[int] = [20, 50],
    height: int = 600
) -> go.Figure:
    """
    Create professional candlestick chart with volume and moving averages.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        show_volume: Whether to show volume subplot
        show_ma: Whether to show moving averages
        ma_periods: List of MA periods to display
        height: Chart height in pixels
    
    Returns:
        Plotly Figure object
    """
    # Create subplots
    row_heights = [0.75, 0.25] if show_volume else [1]
    rows = 2 if show_volume else 1
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=symbol,
            increasing=dict(line=dict(color=CHART_THEME['up_color'], width=1),
                          fillcolor=CHART_THEME['up_color']),
            decreasing=dict(line=dict(color=CHART_THEME['down_color'], width=1),
                          fillcolor=CHART_THEME['down_color']),
            hoverinfo='x+y'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if show_ma:
        ma_color_map = {
            20: CHART_THEME['ma_colors']['sma20'],
            50: CHART_THEME['ma_colors']['sma50'],
            200: CHART_THEME['ma_colors']['sma200']
        }
        
        for period in ma_periods:
            ma = df['Close'].rolling(window=period).mean()
            color = ma_color_map.get(period, '#ffffff')
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ma,
                    name=f'SMA {period}',
                    line=dict(color=color, width=1.5),
                    hovertemplate=f'SMA {period}: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Volume
    if show_volume:
        colors = [
            CHART_THEME['volume_up'] if df['Close'].iloc[i] >= df['Open'].iloc[i] 
            else CHART_THEME['volume_down']
            for i in range(len(df))
        ]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate='Volume: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Apply theme
    layout = create_layout_template()
    layout.update({
        'height': height,
        'title': {'text': f'📈 {symbol}', 'x': 0.02, 'xanchor': 'left'},
        'showlegend': True,
        'legend': {'x': 0.02, 'y': 0.98, 'xanchor': 'left', 'yanchor': 'top'}
    })
    
    if show_volume:
        layout['yaxis2'] = {
            'gridcolor': CHART_THEME['grid_color'],
            'tickfont': {'color': CHART_THEME['text_color']},
            'showgrid': False,
            'side': 'right'
        }
    
    fig.update_layout(**layout)
    
    # Update x-axis for range selector
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="ALL")
            ]),
            bgcolor='#1a1f2e',
            activecolor='#3b82f6',
            font=dict(color='#ffffff'),
            x=0.02,
            y=1.02
        )
    )
    
    return fig


def create_indicators_chart(
    df: pd.DataFrame,
    indicators: List[str] = ['RSI', 'MACD'],
    height: int = 300
) -> go.Figure:
    """
    Create technical indicators chart.
    
    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicators to show
        height: Chart height per indicator
    
    Returns:
        Plotly Figure object
    """
    n_indicators = len(indicators)
    
    fig = make_subplots(
        rows=n_indicators, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[f'{ind}' for ind in indicators]
    )
    
    row = 1
    
    if 'RSI' in indicators:
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=rsi, name='RSI(14)',
                line=dict(color=CHART_THEME['indicator_colors']['rsi'], width=2)
            ),
            row=row, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", row=row, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0, row=row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(16, 185, 129, 0.1)", line_width=0, row=row, col=1)
        
        row += 1
    
    if 'MACD' in indicators:
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # MACD Line
        fig.add_trace(
            go.Scatter(
                x=df.index, y=macd, name='MACD',
                line=dict(color=CHART_THEME['indicator_colors']['macd'], width=2)
            ),
            row=row, col=1
        )
        
        # Signal Line
        fig.add_trace(
            go.Scatter(
                x=df.index, y=signal, name='Signal',
                line=dict(color=CHART_THEME['indicator_colors']['signal'], width=2)
            ),
            row=row, col=1
        )
        
        # Histogram
        colors = [
            CHART_THEME['indicator_colors']['histogram_pos'] if v >= 0 
            else CHART_THEME['indicator_colors']['histogram_neg']
            for v in histogram
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=histogram, name='Histogram',
                marker=dict(color=colors, line=dict(width=0))
            ),
            row=row, col=1
        )
        
        row += 1
    
    if 'Stochastic' in indicators:
        # Calculate Stochastic
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        k = 100 * ((df['Close'] - low14) / (high14 - low14))
        d = k.rolling(window=3).mean()
        
        fig.add_trace(
            go.Scatter(x=df.index, y=k, name='%K', 
                      line=dict(color='#3b82f6', width=2)),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=d, name='%D',
                      line=dict(color='#f59e0b', width=2)),
            row=row, col=1
        )
        
        fig.add_hline(y=80, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", row=row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", row=row, col=1)
        
        row += 1
    
    # Apply theme
    layout = create_layout_template()
    layout.update({
        'height': height * n_indicators,
        'showlegend': True,
        'legend': {'orientation': 'h', 'y': 1.02}
    })
    
    fig.update_layout(**layout)
    
    return fig


def create_bollinger_bands_chart(
    df: pd.DataFrame,
    symbol: str,
    period: int = 20,
    std_dev: float = 2.0,
    height: int = 500
) -> go.Figure:
    """Create Bollinger Bands chart with squeeze indicator."""
    # Calculate Bollinger Bands
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    # Calculate squeeze (bandwidth)
    bandwidth = (upper - lower) / sma * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name=symbol,
            increasing=dict(fillcolor=CHART_THEME['up_color'], line=dict(color=CHART_THEME['up_color'])),
            decreasing=dict(fillcolor=CHART_THEME['down_color'], line=dict(color=CHART_THEME['down_color']))
        ),
        row=1, col=1
    )
    
    # Upper Band
    fig.add_trace(
        go.Scatter(
            x=df.index, y=upper, name='Upper BB',
            line=dict(color=CHART_THEME['indicator_colors']['bb_upper'], dash='dash')
        ),
        row=1, col=1
    )
    
    # Middle Band (SMA)
    fig.add_trace(
        go.Scatter(
            x=df.index, y=sma, name='SMA(20)',
            line=dict(color=CHART_THEME['indicator_colors']['bb_mid'], width=2)
        ),
        row=1, col=1
    )
    
    # Lower Band
    fig.add_trace(
        go.Scatter(
            x=df.index, y=lower, name='Lower BB',
            line=dict(color=CHART_THEME['indicator_colors']['bb_lower'], dash='dash'),
            fill='tonexty',
            fillcolor='rgba(139, 92, 246, 0.1)'
        ),
        row=1, col=1
    )
    
    # Bandwidth
    fig.add_trace(
        go.Scatter(
            x=df.index, y=bandwidth, name='Bandwidth %',
            line=dict(color='#f59e0b', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(245, 158, 11, 0.2)'
        ),
        row=2, col=1
    )
    
    layout = create_layout_template()
    layout.update({
        'height': height,
        'title': {'text': f'📊 {symbol} - Bollinger Bands', 'x': 0.02}
    })
    
    fig.update_layout(**layout)
    
    return fig


def create_portfolio_donut(
    positions: Dict[str, float],
    height: int = 350
) -> go.Figure:
    """Create portfolio allocation donut chart."""
    symbols = list(positions.keys())
    values = list(positions.values())
    
    # Color palette
    colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', 
              '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#6366f1']
    
    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=values,
        hole=0.65,
        marker=dict(colors=colors[:len(symbols)]),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(color=CHART_THEME['text_color']),
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>%{percent}<extra></extra>'
    )])
    
    # Add total value annotation in center
    total = sum(values)
    fig.add_annotation(
        text=f'${total:,.0f}',
        x=0.5, y=0.55,
        font=dict(size=24, color=CHART_THEME['title_color'], family='Inter'),
        showarrow=False
    )
    fig.add_annotation(
        text='Total Value',
        x=0.5, y=0.42,
        font=dict(size=12, color=CHART_THEME['text_color']),
        showarrow=False
    )
    
    layout = create_layout_template()
    layout.update({
        'height': height,
        'showlegend': False
    })
    
    fig.update_layout(**layout)
    
    return fig


def create_equity_curve(
    equity: pd.Series,
    drawdown: Optional[pd.Series] = None,
    height: int = 400
) -> go.Figure:
    """Create equity curve with optional drawdown."""
    rows = 2 if drawdown is not None else 1
    row_heights = [0.7, 0.3] if drawdown is not None else [1]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index, y=equity,
            name='Portfolio Value',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ),
        row=1, col=1
    )
    
    # Drawdown
    if drawdown is not None:
        fig.add_trace(
            go.Scatter(
                x=drawdown.index, y=drawdown * 100,
                name='Drawdown %',
                line=dict(color='#ef4444', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.2)'
            ),
            row=2, col=1
        )
    
    layout = create_layout_template()
    layout.update({
        'height': height,
        'title': {'text': '💰 Portfolio Performance', 'x': 0.02}
    })
    
    fig.update_layout(**layout)
    
    return fig


def create_heatmap_calendar(
    daily_returns: pd.Series,
    height: int = 300
) -> go.Figure:
    """Create monthly returns heatmap calendar."""
    # Resample to monthly returns
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Create pivot table (Year x Month)
    df = pd.DataFrame({'return': monthly_returns})
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    pivot = df.pivot_table(values='return', index='year', columns='month', aggfunc='mean')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[
            [0, '#ef4444'],
            [0.5, '#1a1f2e'],
            [1, '#10b981']
        ],
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate='%{text}%',
        textfont=dict(size=11, color='white'),
        hovertemplate='%{y} %{x}: %{z:.2f}%<extra></extra>',
        colorbar=dict(
            title='Return %',
            titleside='right',
            tickfont=dict(color=CHART_THEME['text_color'])
        )
    ))
    
    layout = create_layout_template()
    layout.update({
        'height': height,
        'title': {'text': '📆 Monthly Returns Heatmap', 'x': 0.02}
    })
    
    fig.update_layout(**layout)
    
    return fig


def create_mini_sparkline(
    prices: pd.Series,
    width: int = 100,
    height: int = 40,
    color: Optional[str] = None
) -> go.Figure:
    """Create mini sparkline chart for tables."""
    if color is None:
        color = CHART_THEME['up_color'] if prices.iloc[-1] >= prices.iloc[0] else CHART_THEME['down_color']
    
    fig = go.Figure(data=go.Scatter(
        x=list(range(len(prices))),
        y=prices,
        mode='lines',
        line=dict(color=color, width=1.5),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}'
    ))
    
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    
    return fig
