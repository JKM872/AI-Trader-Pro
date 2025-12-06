"""
AI Trader Pro - Multi-Stock Scanner Dashboard

Advanced real-time scanner with multi-strategy analysis,
heat maps, and comprehensive market overview.

Run with: streamlit run dashboard/scanner.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.components import (
    load_custom_css,
    render_top_navbar,
    render_metric_row,
    render_signal_badge,
    render_price_ticker,
    create_professional_candlestick,
    create_portfolio_donut,
    MetricData,
    MarketStatus,
    CHART_THEME
)

from trader.data.fetcher import DataFetcher
from trader.strategies.technical import TechnicalStrategy
from trader.strategies.momentum import MomentumStrategy
from trader.strategies.mean_reversion import MeanReversionStrategy
from trader.strategies.breakout import BreakoutStrategy
from trader.strategies.smart_money import SmartMoneyStrategy
from trader.strategies.scanner import SignalScanner, get_signal_summary, ScoredSignal
from trader.strategies.base import SignalType


# Page configuration
st.set_page_config(
    page_title="AI Trader - Scanner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# Comprehensive watchlists - all major stocks
WATCHLISTS = {
    # Major indices & ETFs
    'Market ETFs': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'VGT', 'XLK', 'XLF', 'XLE', 'XLV', 'ARKK'],
    
    # Tech - FAANG+ & Semiconductors
    'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ORCL', 'IBM', 'CSCO', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX'],
    
    # AI & Cloud
    'AI & Cloud': ['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'CRM', 'NOW', 'SNOW', 'PLTR', 'AI', 'PATH', 'DDOG', 'MDB', 'NET', 'ZS', 'CRWD', 'PANW', 'OKTA', 'SPLK', 'TEAM'],
    
    # S&P 500 Top 50 by market cap
    'S&P 500 Top 50': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'XOM', 'JPM', 'WMT', 'MA', 'PG', 'HD', 'CVX', 'MRK', 'ABBV',
        'KO', 'PEP', 'LLY', 'COST', 'AVGO', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT',
        'DHR', 'NKE', 'TXN', 'NEE', 'PM', 'UNP', 'ORCL', 'COP', 'HON', 'QCOM',
        'LOW', 'AMGN', 'INTU', 'BA', 'IBM', 'GE', 'AMD', 'CAT', 'SBUX', 'PLD'
    ],
    
    # Growth stocks
    'High Growth': ['NVDA', 'AMD', 'CRM', 'NOW', 'SNOW', 'DDOG', 'NET', 'ZS', 'MDB', 'PLTR', 'CRWD', 'PANW', 'SHOP', 'SQ', 'COIN', 'SOFI', 'ROKU', 'SPOT', 'UBER', 'LYFT'],
    
    # Dividend aristocrats
    'Dividend Kings': ['JNJ', 'PG', 'KO', 'PEP', 'MCD', 'WMT', 'HD', 'COST', 'ABBV', 'MRK', 'MMM', 'CL', 'GPC', 'EMR', 'SWK', 'PPG', 'SHW', 'ADP', 'ITW', 'BDX'],
    
    # Financials
    'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA', 'COF', 'USB', 'TFC', 'PNC', 'BK', 'STT', 'CME', 'ICE', 'SPGI'],
    
    # Healthcare & Pharma
    'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD', 'ISRG', 'VRTX', 'REGN', 'MDT', 'SYK', 'ZTS', 'BDX', 'HCA'],
    
    # Energy
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL', 'PXD', 'DVN', 'HES', 'FANG', 'KMI', 'WMB', 'OKE', 'BKR', 'TRGP', 'LNG'],
    
    # Consumer
    'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'ROST', 'DG', 'DLTR', 'COST', 'WMT', 'BBY', 'CMG', 'DPZ', 'YUM', 'LULU', 'GPS'],
    
    # Industrial & Defense
    'Industrial': ['CAT', 'DE', 'HON', 'GE', 'BA', 'LMT', 'RTX', 'NOC', 'GD', 'UNP', 'UPS', 'FDX', 'CSX', 'NSC', 'WM', 'RSG', 'MMM', 'EMR', 'ROK', 'ITW'],
    
    # Communication Services
    'Media & Telecom': ['GOOGL', 'META', 'NFLX', 'DIS', 'T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'PARA', 'WBD', 'FOX', 'FOXA', 'OMC', 'IPG', 'TTWO', 'EA', 'ATVI', 'RBLX', 'MTCH'],
    
    # Real Estate
    'REITs': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'SPG', 'WELL', 'DLR', 'AVB', 'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'BXP', 'SLG', 'VNO', 'KIM', 'REG'],
    
    # All S&P 500 (grouped by sector) - Full list
    'S&P 500 Full': [
        # Technology
        'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE', 'ACN', 'CSCO', 'INTC', 'IBM', 'TXN', 'QCOM', 'NOW', 'INTU', 'AMAT', 'ADI', 'MU', 'LRCX',
        'SNPS', 'KLAC', 'APH', 'CDNS', 'ROP', 'NXPI', 'FTNT', 'MCHP', 'MSI', 'HPQ', 'TEL', 'ANSS', 'KEYS', 'ON', 'FSLR', 'MPWR', 'TYL', 'EPAM', 'PTC', 'CTSH',
        # Communication Services
        'GOOGL', 'META', 'NFLX', 'TMUS', 'CMCSA', 'T', 'VZ', 'DIS', 'CHTR', 'EA', 'WBD', 'OMC', 'TTWO', 'PARA', 'IPG', 'MTCH', 'LYV', 'FOXA', 'FOX', 'NWS',
        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG', 'MAR', 'ROST', 'GM', 'F', 'ORLY', 'AZO', 'YUM', 'DHI', 'LEN', 'HLT',
        'EBAY', 'LVS', 'WYNN', 'MGM', 'RCL', 'CCL', 'NCLH', 'EXPE', 'DRI', 'POOL', 'BBY', 'ULTA', 'TGT', 'DG', 'DLTR', 'KMX', 'APTV', 'BWA', 'GRMN', 'PHM',
        # Consumer Staples
        'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'EL', 'KMB', 'GIS', 'KDP', 'SYY', 'HSY', 'K', 'STZ', 'CAG', 'CHD', 'MKC',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL', 'PXD', 'DVN', 'HES', 'FANG', 'KMI', 'WMB', 'OKE', 'BKR', 'TRGP', 'CTRA',
        # Financials
        'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'AXP', 'SPGI', 'CME', 'PGR', 'ICE', 'AON', 'MMC', 'USB', 'CB',
        'TFC', 'PNC', 'COF', 'MET', 'AFL', 'PRU', 'AIG', 'TRV', 'ALL', 'AMP', 'BK', 'STT', 'MSCI', 'FIS', 'NDAQ', 'MCO', 'FITB', 'HIG', 'CFG', 'RF',
        # Healthcare
        'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD', 'ISRG', 'ELV', 'VRTX', 'CVS', 'REGN', 'MDT', 'SYK', 'ZTS',
        'CI', 'MCK', 'HCA', 'BDX', 'BSX', 'EW', 'A', 'IQV', 'DXCM', 'IDXX', 'MTD', 'RMD', 'ILMN', 'ALGN', 'CAH', 'BIIB', 'HOLX', 'BAX', 'COO', 'MRNA',
        # Industrials
        'CAT', 'HON', 'UNP', 'UPS', 'GE', 'RTX', 'BA', 'DE', 'LMT', 'ADP', 'NOC', 'GD', 'MMM', 'ITW', 'CSX', 'NSC', 'WM', 'EMR', 'FDX', 'ETN',
        'JCI', 'PH', 'CTAS', 'PCAR', 'TT', 'CARR', 'CMI', 'ROK', 'FAST', 'OTIS', 'AME', 'RSG', 'VRSK', 'GWW', 'PWR', 'CPRT', 'ODFL', 'IR', 'XYL', 'DOV',
        # Materials
        'LIN', 'APD', 'SHW', 'FCX', 'NUE', 'ECL', 'DD', 'DOW', 'CTVA', 'NEM', 'PPG', 'VMC', 'MLM', 'ALB', 'FMC', 'IFF', 'CE', 'EMN', 'CF', 'MOS',
        # Real Estate
        'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'SPG', 'WELL', 'DLR', 'AVB', 'EQR', 'VTR', 'ARE', 'MAA', 'UDR', 'BXP', 'HST', 'PEAK', 'KIM', 'REG',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'PCG', 'ED', 'WEC', 'ES', 'AWK', 'EIX', 'DTE', 'PPL', 'FE', 'ETR', 'AEE', 'CMS'
    ],
    
    # Popular trading stocks (high volume, volatility)
    'Day Trading Favorites': ['TSLA', 'NVDA', 'AMD', 'AAPL', 'SPY', 'QQQ', 'META', 'AMZN', 'GOOGL', 'MSFT', 'COIN', 'PLTR', 'SOFI', 'NIO', 'RIVN', 'LCID', 'GME', 'AMC', 'BB', 'BBBY'],
    
    # Crypto-related
    'Crypto Stocks': ['COIN', 'MSTR', 'RIOT', 'MARA', 'HUT', 'BITF', 'CLSK', 'SQ', 'PYPL', 'HOOD'],
    
    # EV & Clean Energy
    'EV & Clean Energy': ['TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'F', 'GM', 'PLUG', 'FCEL', 'BE', 'ENPH', 'SEDG', 'FSLR', 'RUN', 'NEE', 'AES', 'VST', 'CEG', 'CHPT'],
    
    # ============ POLISH STOCKS (GPW - Warsaw Stock Exchange) ============
    # WIG20 - Top 20 Polish blue chips
    'WIG20 (Poland)': [
        'PKO.WA',       # PKO Bank Polski
        'PKN.WA',       # PKN Orlen
        'PZU.WA',       # PZU
        'PEO.WA',       # Bank Pekao
        'KGH.WA',       # KGHM Polska Miedź
        'SPL.WA',       # Santander Bank Polska
        'DNP.WA',       # Dino Polska
        'ALE.WA',       # Allegro
        'CDR.WA',       # CD Projekt
        'LPP.WA',       # LPP
        'CCC.WA',       # CCC
        'MBK.WA',       # mBank
        'PGE.WA',       # PGE
        'OPL.WA',       # Orange Polska
        'ALR.WA',       # Alior Bank
        'BDX.WA',       # Budimex
        'KRU.WA',       # Kruk
        'KTY.WA',       # Grupa Kęty
        'PCO.WA',       # Pepco Group
        'ZAB.WA',       # Żabka
    ],
    
    # mWIG40 - Mid-cap Polish stocks (selection)
    'mWIG40 (Poland)': [
        'ACP.WA',       # Asseco Poland
        'ASE.WA',       # Asseco SEE
        'BFT.WA',       # Benefit Systems
        'CPS.WA',       # Cyfrowy Polsat
        'ENA.WA',       # Enea
        'EUR.WA',       # Eurocash
        'GPW.WA',       # GPW
        'ING.WA',       # ING Bank Śląski
        'JSW.WA',       # JSW
        'LWB.WA',       # Bogdanka
        'MIL.WA',       # Millennium Bank
        'MRC.WA',       # Mercator Medical
        'NEU.WA',       # Neuca
        'PLY.WA',       # Playway
        'SNT.WA',       # Synektik
        'TEN.WA',       # Ten Square Games
        'VRG.WA',       # VRG
        'XTB.WA',       # XTB
        'ZEP.WA',       # Zepak
        'AMC.WA',       # Amica
    ],
    
    # sWIG80 - Small-cap Polish stocks (selection)
    'sWIG80 (Poland)': [
        '11B.WA',       # 11 Bit Studios
        'BIO.WA',       # Bioton
        'CAR.WA',       # Inter Cars
        'CIG.WA',       # CI Games
        'DOM.WA',       # Dom Development
        'EAT.WA',       # AmRest
        'ECH.WA',       # Echo Investment
        'FMF.WA',       # Famur
        'GTN.WA',       # Getin Noble Bank
        'HUG.WA',       # Huuuge
        'KER.WA',       # Kernel
        'KRK.WA',       # Krakchemia
        'MAB.WA',       # Mabion
        'MOL.WA',       # MOL
        'OAT.WA',       # Oat
        'PCE.WA',       # PCE
        'PKP.WA',       # PKP Cargo
        'RFK.WA',       # Rafako
        'TXT.WA',       # Text (dawniej LiveChat)
        'WPL.WA',       # Wirtualna Polska
    ],
    
    # Polish Banks
    'Polish Banks': [
        'PKO.WA',       # PKO Bank Polski
        'PEO.WA',       # Bank Pekao
        'SPL.WA',       # Santander Bank Polska
        'MBK.WA',       # mBank
        'ALR.WA',       # Alior Bank
        'MIL.WA',       # Millennium Bank
        'ING.WA',       # ING Bank Śląski
        'BNP.WA',       # BNP Paribas Bank Polska
    ],
    
    # Polish Energy & Mining
    'Polish Energy & Mining': [
        'PKN.WA',       # PKN Orlen
        'PGE.WA',       # PGE
        'KGH.WA',       # KGHM Polska Miedź
        'JSW.WA',       # JSW
        'LWB.WA',       # Bogdanka
        'ENA.WA',       # Enea
        'TPE.WA',       # Tauron
        'ZEP.WA',       # Zepak
    ],
    
    # Polish Gaming & Tech
    'Polish Gaming & Tech': [
        'CDR.WA',       # CD Projekt
        'PLY.WA',       # Playway
        '11B.WA',       # 11 Bit Studios
        'TEN.WA',       # Ten Square Games
        'CIG.WA',       # CI Games
        'HUG.WA',       # Huuuge
        'TXT.WA',       # Text (LiveChat)
        'ACP.WA',       # Asseco Poland
        'ASE.WA',       # Asseco SEE
        'COM.WA',       # Comarch
    ],
    
    # Polish Retail
    'Polish Retail': [
        'DNP.WA',       # Dino Polska
        'LPP.WA',       # LPP
        'CCC.WA',       # CCC
        'ALE.WA',       # Allegro
        'PCO.WA',       # Pepco Group
        'ZAB.WA',       # Żabka
        'EUR.WA',       # Eurocash
        'AMC.WA',       # Amica
    ],
}

# Initialize session state
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = None

if 'selected_watchlist' not in st.session_state:
    st.session_state.selected_watchlist = 'Tech Giants'


@st.cache_data(ttl=120)
def fetch_stock_data(symbol: str, period: str = '3mo') -> pd.DataFrame:
    """Fetch stock data with caching."""
    fetcher = DataFetcher()
    return fetcher.get_stock_data(symbol, period=period)


@st.cache_data(ttl=120)
def scan_single_stock(symbol: str, strategies: list) -> dict:
    """Scan a single stock with multiple strategies."""
    try:
        df = fetch_stock_data(symbol, '3mo')
        
        if df.empty:
            return {'symbol': symbol, 'error': 'No data'}
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        change_pct = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0
        
        # Volume analysis
        volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        # Volatility
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Technical levels
        high_52w = df['High'].tail(252).max() if len(df) >= 252 else df['High'].max()
        low_52w = df['Low'].tail(252).min() if len(df) >= 252 else df['Low'].min()
        
        # Generate signals
        signals = {}
        strategy_map = {
            'Technical': TechnicalStrategy(),
            'Momentum': MomentumStrategy(),
            'Mean Reversion': MeanReversionStrategy(),
            'Breakout': BreakoutStrategy(),
            'Smart Money': SmartMoneyStrategy()
        }
        
        for strat_name in strategies:
            if strat_name in strategy_map:
                try:
                    strategy = strategy_map[strat_name]
                    signal = strategy.generate_signal(symbol, df)
                    if signal:
                        signals[strat_name] = {
                            'type': signal.signal_type.name,
                            'confidence': signal.confidence,
                            'reasons': signal.reasons[:2] if signal.reasons else []
                        }
                except Exception:
                    continue
        
        # Calculate overall score
        buy_count = len([s for s in signals.values() if s['type'] == 'BUY'])
        sell_count = len([s for s in signals.values() if s['type'] == 'SELL'])
        total_confidence = sum(s['confidence'] for s in signals.values()) / len(signals) if signals else 0
        
        if buy_count > sell_count:
            overall_signal = 'BUY'
            score = total_confidence * (buy_count / len(signals)) * 100 if signals else 50
        elif sell_count > buy_count:
            overall_signal = 'SELL'
            score = (1 - total_confidence) * (sell_count / len(signals)) * 100 if signals else 50
        else:
            overall_signal = 'HOLD'
            score = 50
        
        return {
            'symbol': symbol,
            'price': current_price,
            'change_pct': change_pct,
            'volume': volume,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'signals': signals,
            'overall_signal': overall_signal,
            'score': score,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'avg_confidence': total_confidence
        }
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}


def run_scanner(symbols: list, strategies: list) -> list:
    """Run scanner on multiple symbols."""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scan_single_stock, symbol, strategies): symbol for symbol in symbols}
        
        for i, future in enumerate(as_completed(futures)):
            symbol = futures[future]
            status_text.text(f"Scanning {symbol}...")
            
            try:
                result = future.result()
                if 'error' not in result:
                    results.append(result)
            except Exception as e:
                pass
            
            progress_bar.progress((i + 1) / len(symbols))
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort by score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return results


def create_sector_heatmap(results: list) -> go.Figure:
    """Create sector performance heatmap."""
    # Create a simple grid heatmap
    symbols = [r['symbol'] for r in results]
    changes = [r['change_pct'] for r in results]
    
    # Calculate grid dimensions
    n = len(symbols)
    cols = min(7, n)
    rows = (n + cols - 1) // cols
    
    # Pad data to fill grid
    while len(symbols) < rows * cols:
        symbols.append('')
        changes.append(0)
    
    # Reshape into grid
    z = np.array(changes).reshape(rows, cols)
    text = np.array(symbols).reshape(rows, cols)
    
    # Create custom colorscale
    colorscale = [
        [0.0, '#ef4444'],
        [0.4, '#f97316'],
        [0.5, '#1a1f2e'],
        [0.6, '#22c55e'],
        [1.0, '#10b981']
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        text=text,
        texttemplate='<b>%{text}</b><br>%{z:+.2f}%',
        textfont=dict(size=11, color='white'),
        colorscale=colorscale,
        zmid=0,
        showscale=True,
        colorbar=dict(
            title='Change %',
            titleside='right',
            tickfont=dict(color='#9ca3af')
        ),
        hovertemplate='%{text}: %{z:+.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor=CHART_THEME['bg_color'],
        plot_bgcolor=CHART_THEME['plot_bg'],
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text='📊 Market Heatmap',
            font=dict(color='#ffffff', size=16),
            x=0.02
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig


def create_signal_distribution(results: list) -> go.Figure:
    """Create signal distribution chart."""
    buy_count = sum(1 for r in results if r['overall_signal'] == 'BUY')
    sell_count = sum(1 for r in results if r['overall_signal'] == 'SELL')
    hold_count = sum(1 for r in results if r['overall_signal'] == 'HOLD')
    
    fig = go.Figure(data=[go.Pie(
        labels=['BUY', 'SELL', 'HOLD'],
        values=[buy_count, sell_count, hold_count],
        hole=0.6,
        marker=dict(colors=['#10b981', '#ef4444', '#f59e0b']),
        textinfo='label+value',
        textposition='outside',
        textfont=dict(color='#9ca3af')
    )])
    
    fig.add_annotation(
        text=f'{len(results)}',
        x=0.5, y=0.55,
        font=dict(size=32, color='#ffffff'),
        showarrow=False
    )
    fig.add_annotation(
        text='Stocks',
        x=0.5, y=0.4,
        font=dict(size=14, color='#9ca3af'),
        showarrow=False
    )
    
    fig.update_layout(
        paper_bgcolor=CHART_THEME['bg_color'],
        plot_bgcolor=CHART_THEME['plot_bg'],
        height=300,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text='🎯 Signal Distribution',
            font=dict(color='#ffffff', size=16),
            x=0.02
        )
    )
    
    return fig


def create_volatility_scatter(results: list) -> go.Figure:
    """Create volatility vs performance scatter plot."""
    df = pd.DataFrame(results)
    
    # Create color based on signal
    color_map = {'BUY': '#10b981', 'SELL': '#ef4444', 'HOLD': '#f59e0b'}
    colors = [color_map.get(r['overall_signal'], '#9ca3af') for r in results]
    
    fig = go.Figure(data=go.Scatter(
        x=df['volatility'],
        y=df['change_pct'],
        mode='markers+text',
        text=df['symbol'],
        textposition='top center',
        textfont=dict(color='#9ca3af', size=10),
        marker=dict(
            size=df['volume_ratio'] * 10,
            color=colors,
            line=dict(width=1, color='rgba(255,255,255,0.3)')
        ),
        hovertemplate='<b>%{text}</b><br>Change: %{y:+.2f}%<br>Volatility: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor=CHART_THEME['bg_color'],
        plot_bgcolor=CHART_THEME['plot_bg'],
        height=350,
        margin=dict(l=50, r=20, t=40, b=50),
        title=dict(
            text='📈 Risk/Return Profile',
            font=dict(color='#ffffff', size=16),
            x=0.02
        ),
        xaxis=dict(
            title='Volatility %',
            gridcolor='rgba(75, 85, 99, 0.2)',
            tickfont=dict(color='#9ca3af'),
            titlefont=dict(color='#9ca3af')
        ),
        yaxis=dict(
            title='Daily Change %',
            gridcolor='rgba(75, 85, 99, 0.2)',
            tickfont=dict(color='#9ca3af'),
            titlefont=dict(color='#9ca3af')
        )
    )
    
    # Add quadrant lines
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    
    return fig


def create_top_movers_chart(results: list, top_n: int = 10) -> go.Figure:
    """Create top movers bar chart."""
    # Sort by absolute change
    sorted_results = sorted(results, key=lambda x: abs(x['change_pct']), reverse=True)[:top_n]
    
    symbols = [r['symbol'] for r in sorted_results]
    changes = [r['change_pct'] for r in sorted_results]
    colors = ['#10b981' if c >= 0 else '#ef4444' for c in changes]
    
    fig = go.Figure(data=go.Bar(
        x=symbols,
        y=changes,
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f'{c:+.2f}%' for c in changes],
        textposition='outside',
        textfont=dict(color='#9ca3af'),
        hovertemplate='<b>%{x}</b><br>Change: %{y:+.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        paper_bgcolor=CHART_THEME['bg_color'],
        plot_bgcolor=CHART_THEME['plot_bg'],
        height=300,
        margin=dict(l=50, r=20, t=40, b=50),
        title=dict(
            text='🔥 Top Movers',
            font=dict(color='#ffffff', size=16),
            x=0.02
        ),
        xaxis=dict(
            tickfont=dict(color='#9ca3af'),
            gridcolor='rgba(75, 85, 99, 0.2)'
        ),
        yaxis=dict(
            title='Change %',
            tickfont=dict(color='#9ca3af'),
            titlefont=dict(color='#9ca3af'),
            gridcolor='rgba(75, 85, 99, 0.2)'
        )
    )
    
    return fig


def render_results_table(results: list):
    """Render scanner results as interactive table."""
    st.markdown("""
    <style>
        .scanner-table { width: 100%; border-collapse: separate; border-spacing: 0 4px; }
        .scanner-table th { 
            text-align: left; padding: 12px 16px; font-size: 11px; 
            color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;
            border-bottom: 1px solid rgba(75, 85, 99, 0.4);
        }
        .scanner-table td { 
            padding: 12px 16px; background: #1a1f2e; font-size: 13px;
        }
        .scanner-table tr td:first-child { border-radius: 8px 0 0 8px; }
        .scanner-table tr td:last-child { border-radius: 0 8px 8px 0; }
        .scanner-table tr:hover td { background: rgba(59, 130, 246, 0.1); }
    </style>
    """, unsafe_allow_html=True)
    
    rows = ""
    for r in results:
        signal_color = '#10b981' if r['overall_signal'] == 'BUY' else '#ef4444' if r['overall_signal'] == 'SELL' else '#f59e0b'
        change_color = '#10b981' if r['change_pct'] >= 0 else '#ef4444'
        
        # Strategy signals
        signals_html = ""
        for strat, sig in r.get('signals', {}).items():
            sig_color = '#10b981' if sig['type'] == 'BUY' else '#ef4444' if sig['type'] == 'SELL' else '#f59e0b'
            signals_html += f'<span style="color: {sig_color}; font-size: 11px; margin-right: 8px;">{strat[:3].upper()}</span>'
        
        rows += f'''
        <tr>
            <td style="font-weight: 600; color: #ffffff;">{r['symbol']}</td>
            <td style="color: #ffffff;">${r['price']:.2f}</td>
            <td style="color: {change_color};">{r['change_pct']:+.2f}%</td>
            <td>{r['volume']/1e6:.2f}M</td>
            <td>{r['volatility']:.1f}%</td>
            <td style="color: {signal_color}; font-weight: 600;">{r['overall_signal']}</td>
            <td>{r['score']:.0f}</td>
            <td>{signals_html}</td>
            <td style="color: #3b82f6;">{r['avg_confidence']:.0%}</td>
        </tr>
        '''
    
    st.markdown(f'''
    <table class="scanner-table">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Price</th>
                <th>Change</th>
                <th>Volume</th>
                <th>Volatility</th>
                <th>Signal</th>
                <th>Score</th>
                <th>Strategies</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    ''', unsafe_allow_html=True)


def main():
    """Main application entry point."""
    market_status = MarketStatus.get_current()
    
    # Top navigation
    render_top_navbar(
        is_market_open=market_status.is_open,
        session_name=market_status.session,
        last_update=datetime.now()
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🔍 Multi-Stock Signal Scanner")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Scanner Settings")
        
        # Watchlist selection
        watchlist_name = st.selectbox(
            "📋 Watchlist",
            list(WATCHLISTS.keys()),
            index=list(WATCHLISTS.keys()).index(st.session_state.selected_watchlist)
        )
        st.session_state.selected_watchlist = watchlist_name
        
        symbols = WATCHLISTS[watchlist_name].copy()
        
        # Custom symbols
        st.markdown("#### Add Custom Symbols")
        custom_symbols = st.text_input(
            "Enter symbols (comma separated)",
            placeholder="e.g., AMD, INTC, UBER"
        )
        if custom_symbols:
            for sym in custom_symbols.upper().split(','):
                sym = sym.strip()
                if sym and sym not in symbols:
                    symbols.append(sym)
        
        st.markdown(f"**{len(symbols)} symbols selected**")
        
        st.divider()
        
        # Strategy selection
        st.markdown("#### 🎯 Strategies")
        strategies = st.multiselect(
            "Select strategies to analyze",
            ['Technical', 'Momentum', 'Mean Reversion', 'Breakout', 'Smart Money'],
            default=['Technical', 'Momentum', 'Smart Money']
        )
        
        st.divider()
        
        # Filters
        st.markdown("#### 🔧 Filters")
        
        min_score = st.slider(
            "Minimum Score",
            0, 100, 0,
            help="Filter results by minimum signal score"
        )
        
        signal_filter = st.multiselect(
            "Signal Type",
            ['BUY', 'SELL', 'HOLD'],
            default=['BUY', 'SELL', 'HOLD']
        )
        
        st.divider()
        
        # Run scanner button
        if st.button("🚀 Run Scanner", use_container_width=True, type="primary"):
            if not strategies:
                st.error("Please select at least one strategy")
            else:
                with st.spinner("Scanning markets..."):
                    results = run_scanner(symbols, strategies)
                    st.session_state.scanner_results = results
    
    # Main content
    results = st.session_state.scanner_results
    
    if results is None:
        # Empty state
        st.markdown('''
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 64px; margin-bottom: 24px;">🔍</div>
            <div style="font-size: 24px; font-weight: 600; color: #ffffff; margin-bottom: 12px;">
                Ready to Scan
            </div>
            <div style="color: #9ca3af; max-width: 400px; margin: 0 auto;">
                Select a watchlist and strategies from the sidebar, then click "Run Scanner" to analyze the market.
            </div>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    # Apply filters
    filtered_results = [
        r for r in results
        if r['score'] >= min_score and r['overall_signal'] in signal_filter
    ]
    
    # Summary metrics
    total = len(filtered_results)
    buy_count = len([r for r in filtered_results if r['overall_signal'] == 'BUY'])
    sell_count = len([r for r in filtered_results if r['overall_signal'] == 'SELL'])
    avg_change = np.mean([r['change_pct'] for r in filtered_results]) if filtered_results else 0
    
    metrics = [
        MetricData("Stocks Analyzed", str(total), icon="📊"),
        MetricData("BUY Signals", str(buy_count), icon="🟢"),
        MetricData("SELL Signals", str(sell_count), icon="🔴"),
        MetricData("Avg Change", f"{avg_change:+.2f}%", change_positive=avg_change >= 0, icon="📈")
    ]
    render_metric_row(metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        fig_heatmap = create_sector_heatmap(filtered_results)
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")
    
    with col2:
        fig_dist = create_signal_distribution(filtered_results)
        st.plotly_chart(fig_dist, use_container_width=True, key="distribution")
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig_scatter = create_volatility_scatter(filtered_results)
        st.plotly_chart(fig_scatter, use_container_width=True, key="scatter")
    
    with col4:
        fig_movers = create_top_movers_chart(filtered_results)
        st.plotly_chart(fig_movers, use_container_width=True, key="movers")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results table
    st.markdown("### 📋 Detailed Results")
    
    # Sort options
    col1, col2 = st.columns([3, 1])
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ['Score', 'Change %', 'Volume', 'Volatility', 'Symbol'],
            label_visibility="collapsed"
        )
    
    # Apply sorting
    sort_map = {
        'Score': 'score',
        'Change %': 'change_pct',
        'Volume': 'volume',
        'Volatility': 'volatility',
        'Symbol': 'symbol'
    }
    sorted_results = sorted(
        filtered_results,
        key=lambda x: x.get(sort_map[sort_by], 0),
        reverse=(sort_by != 'Symbol')
    )
    
    render_results_table(sorted_results)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top opportunities
    st.markdown("### 🏆 Top Opportunities")
    
    top_buys = [r for r in sorted_results if r['overall_signal'] == 'BUY'][:5]
    
    if top_buys:
        cols = st.columns(min(5, len(top_buys)))
        for i, (col, result) in enumerate(zip(cols, top_buys)):
            with col:
                st.markdown(f'''
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-size: 18px; font-weight: 700; color: #ffffff;">{result['symbol']}</span>
                        <span style="color: #10b981; font-size: 12px; font-weight: 600;">BUY</span>
                    </div>
                    <div style="font-size: 24px; color: #ffffff; margin-bottom: 4px;">
                        ${result['price']:.2f}
                    </div>
                    <div style="color: {'#10b981' if result['change_pct'] >= 0 else '#ef4444'}; font-size: 14px;">
                        {result['change_pct']:+.2f}%
                    </div>
                    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(75, 85, 99, 0.4);">
                        <div style="color: #9ca3af; font-size: 12px;">Score</div>
                        <div style="font-size: 20px; font-weight: 600; color: #3b82f6;">{result['score']:.0f}</div>
                    </div>
                    <div style="margin-top: 8px;">
                        <div style="color: #9ca3af; font-size: 12px;">Confidence</div>
                        <div style="font-size: 16px; color: #8b5cf6;">{result['avg_confidence']:.0%}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    else:
        st.info("No BUY signals found with current filters.")


if __name__ == "__main__":
    main()
