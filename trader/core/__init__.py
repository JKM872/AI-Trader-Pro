"""
Core Trading Module - Unified Trading Engine.

This module provides the main entry point for the AI trading system,
integrating all components into a unified workflow.
"""

from trader.core.trading_engine import (
    TradingEngine,
    TradingEngineConfig,
    TradingMode,
    SignalStrength,
    AnalysisResult,
    TradeExecution,
    create_trading_engine
)

__all__ = [
    'TradingEngine',
    'TradingEngineConfig',
    'TradingMode',
    'SignalStrength',
    'AnalysisResult',
    'TradeExecution',
    'create_trading_engine'
]
