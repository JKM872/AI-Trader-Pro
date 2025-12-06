"""
Alternative Data Module - Non-traditional data sources for trading signals.

This module provides access to:
- SEC Filings (insider trading, institutional holdings)
- Economic indicators
- Options flow analysis
- Alternative data manager
"""

from trader.alternative_data.sec_filings import (
    SECFilingsTracker,
    InsiderTrade,
    InstitutionalHolding,
    InsiderSentiment,
    FilingType,
    TransactionType
)
from trader.alternative_data.economic_data import (
    EconomicIndicators,
    EconomicEvent,
    EconomicRelease,
    MarketImpact,
    IndicatorType,
    ImpactLevel
)
from trader.alternative_data.options_flow import (
    OptionsFlowAnalyzer,
    OptionsFlow,
    UnusualActivity,
    OptionsFlowSummary,
    FlowType,
    TradeType
)
from trader.alternative_data.data_manager import (
    AlternativeDataManager,
    AlternativeDataSummary,
    AlternativeSignal,
    InsiderSentiment as ManagerInsiderSentiment,
    InstitutionalSentiment,
    EarningsEvent
)

__all__ = [
    # SEC Filings
    'SECFilingsTracker',
    'InsiderTrade',
    'InstitutionalHolding',
    'InsiderSentiment',
    'FilingType',
    'TransactionType',
    # Economic data
    'EconomicIndicators',
    'EconomicEvent',
    'EconomicRelease',
    'MarketImpact',
    'IndicatorType',
    'ImpactLevel',
    # Options flow
    'OptionsFlowAnalyzer',
    'OptionsFlow',
    'UnusualActivity',
    'OptionsFlowSummary',
    'FlowType',
    'TradeType',
    # Data Manager
    'AlternativeDataManager',
    'AlternativeDataSummary',
    'AlternativeSignal',
    'InstitutionalSentiment',
    'EarningsEvent',
]
