"""
AI Trader - Main Package
Python-based AI trading system with paper trading simulation.

Modules:
- core: Unified Trading Engine (main entry point)
- ensemble: Multi-AI ensemble analysis
- market_analysis: Market regime detection, liquidity mapping
- ml: Machine learning price prediction
- journal: Trade journaling and pattern analysis
- risk: Dynamic risk management
- strategies: Trading strategies
- data: Data fetching and processing
- alerts: Notification system
- execution: Order execution
- portfolio: Portfolio management
"""

__version__ = "0.2.0"
__author__ = "AI Trader Team"

# Core Trading Engine - Main Entry Point
from trader.core import (
    TradingEngine,
    TradingEngineConfig,
    TradingMode,
    create_trading_engine
)

__all__ = [
    # Core
    'TradingEngine',
    'TradingEngineConfig', 
    'TradingMode',
    'create_trading_engine',
]
