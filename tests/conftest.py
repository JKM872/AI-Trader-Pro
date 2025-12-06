"""
Pytest configuration and fixtures for AI Trader tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Generate realistic price data
    initial_price = 150.0
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = initial_price * np.cumprod(1 + returns)
    
    # Generate OHLCV
    data = {
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'High': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'Low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure High is highest and Low is lowest
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) * 1.001
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) * 0.999
    
    return df


@pytest.fixture
def trending_up_data():
    """Generate trending up OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Strong uptrend
    initial_price = 100.0
    trend = np.linspace(0, 0.5, len(dates))  # 50% increase
    noise = np.random.normal(0, 0.01, len(dates))
    prices = initial_price * (1 + trend + noise)
    
    data = {
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        'High': prices * (1 + np.random.uniform(0.005, 0.015, len(dates))),
        'Low': prices * (1 - np.random.uniform(0.005, 0.015, len(dates))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
    }
    
    df = pd.DataFrame(data, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) * 1.001
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) * 0.999
    
    return df


@pytest.fixture
def trending_down_data():
    """Generate trending down OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Strong downtrend
    initial_price = 150.0
    trend = np.linspace(0, -0.4, len(dates))  # 40% decrease
    noise = np.random.normal(0, 0.01, len(dates))
    prices = initial_price * (1 + trend + noise)
    
    data = {
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        'High': prices * (1 + np.random.uniform(0.005, 0.015, len(dates))),
        'Low': prices * (1 - np.random.uniform(0.005, 0.015, len(dates))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
    }
    
    df = pd.DataFrame(data, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) * 1.001
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) * 0.999
    
    return df


@pytest.fixture
def range_bound_data():
    """Generate range-bound/sideways OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # Sideways movement with oscillation
    initial_price = 100.0
    oscillation = np.sin(np.linspace(0, 8 * np.pi, len(dates))) * 0.1
    noise = np.random.normal(0, 0.01, len(dates))
    prices = initial_price * (1 + oscillation + noise)
    
    data = {
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        'High': prices * (1 + np.random.uniform(0.005, 0.015, len(dates))),
        'Low': prices * (1 - np.random.uniform(0.005, 0.015, len(dates))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
    }
    
    df = pd.DataFrame(data, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) * 1.001
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) * 0.999
    
    return df


@pytest.fixture
def volatile_data():
    """Generate highly volatile OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    # High volatility
    initial_price = 100.0
    returns = np.random.normal(0, 0.05, len(dates))  # 5% daily volatility
    prices = initial_price * np.cumprod(1 + returns)
    
    data = {
        'Open': prices * (1 + np.random.uniform(-0.02, 0.02, len(dates))),
        'High': prices * (1 + np.random.uniform(0.02, 0.05, len(dates))),
        'Low': prices * (1 - np.random.uniform(0.02, 0.05, len(dates))),
        'Close': prices,
        'Volume': np.random.randint(5000000, 20000000, len(dates)),
    }
    
    df = pd.DataFrame(data, index=dates)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) * 1.001
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) * 0.999
    
    return df
