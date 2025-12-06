"""
Tests for Alternative Data Module.

Tests:
- SEC Filings tracking
- Economic indicators
- Options flow analysis
- Alternative data manager
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from trader.alternative_data import (
    # SEC Filings
    SECFilingsTracker,
    InsiderTrade,
    InstitutionalHolding,
    InsiderSentiment,
    FilingType,
    TransactionType,
    # Economic Data
    EconomicIndicators,
    EconomicRelease,
    EconomicEvent,
    MarketImpact,
    IndicatorType,
    ImpactLevel,
    # Options Flow
    OptionsFlowAnalyzer,
    OptionsFlow,
    UnusualActivity,
    OptionsFlowSummary,
    FlowType,
    TradeType,
    # Data Manager
    AlternativeDataManager,
    AlternativeDataSummary,
    AlternativeSignal,
    InstitutionalSentiment,
    EarningsEvent,
)


# ============================================================================
# SEC Filings Tests
# ============================================================================

class TestSECFilingsTracker:
    """Tests for SEC filings tracking."""
    
    def test_init(self):
        """Test tracker initialization."""
        tracker = SECFilingsTracker()
        assert tracker is not None
    
    def test_add_insider_trade(self):
        """Test adding insider trade."""
        tracker = SECFilingsTracker()
        
        trade = InsiderTrade(
            symbol='AAPL',
            insider_name='Tim Cook',
            insider_title='CEO',
            transaction_type=TransactionType.PURCHASE,
            transaction_date=datetime.now(timezone.utc),
            shares=10000,
            price=150.0,
            shares_owned_after=100000,
            filing_date=datetime.now(timezone.utc)
        )
        
        tracker.add_insider_trade(trade)
        trades = tracker.get_recent_trades('AAPL')
        
        assert len(trades) >= 1
        assert trades[0].insider_name == 'Tim Cook'
    
    def test_add_institutional_holding(self):
        """Test adding institutional holding."""
        tracker = SECFilingsTracker()
        
        holding = InstitutionalHolding(
            symbol='AAPL',
            institution_name='Berkshire Hathaway',
            shares=100000000,
            value=15000000000.0,
            shares_change=5000000,
            shares_change_pct=5.0,
            filing_date=datetime.now(timezone.utc)
        )
        
        tracker.add_institutional_holding(holding)
        activity = tracker.get_institutional_activity('AAPL')
        
        assert activity['total_institutions'] >= 1
    
    def test_insider_trade_properties(self):
        """Test insider trade computed properties."""
        trade = InsiderTrade(
            symbol='AAPL',
            insider_name='Test Insider',
            insider_title='CFO',
            transaction_type=TransactionType.PURCHASE,
            transaction_date=datetime.now(timezone.utc),
            shares=1000,
            price=150.0,
            shares_owned_after=11000,
            filing_date=datetime.now(timezone.utc)
        )
        
        assert trade.is_purchase == True
        assert trade.is_sale == False
        assert trade.value == 150000.0
    
    def test_sale_trade(self):
        """Test sale transaction type."""
        trade = InsiderTrade(
            symbol='AAPL',
            insider_name='Test Insider',
            insider_title='CFO',
            transaction_type=TransactionType.SALE,
            transaction_date=datetime.now(timezone.utc),
            shares=1000,
            price=150.0,
            shares_owned_after=9000,
            filing_date=datetime.now(timezone.utc)
        )
        
        assert trade.is_purchase == False
        assert trade.is_sale == True
    
    def test_filing_types(self):
        """Test filing type enum."""
        assert FilingType.FORM_4.value == '4'
        assert FilingType.FORM_13F.value == '13F-HR'
        assert FilingType.FORM_8K.value == '8-K'
    
    def test_get_insider_sentiment(self):
        """Test insider sentiment calculation."""
        tracker = SECFilingsTracker()
        
        # Add purchase trades
        for i in range(5):
            tracker.add_insider_trade(InsiderTrade(
                symbol='AAPL',
                insider_name=f'Insider {i}',
                insider_title='Director',
                transaction_type=TransactionType.PURCHASE,
                transaction_date=datetime.now(timezone.utc),
                shares=1000,
                price=150.0,
                shares_owned_after=10000,
                filing_date=datetime.now(timezone.utc)
            ))
        
        sentiment = tracker.get_insider_sentiment('AAPL')
        
        assert isinstance(sentiment, InsiderSentiment)
        assert sentiment.purchases == 5
        assert sentiment.signal == 'bullish'
    
    def test_cluster_buying_detection(self):
        """Test cluster buying detection."""
        tracker = SECFilingsTracker()
        
        # Add multiple insider purchases
        for i in range(5):
            tracker.add_insider_trade(InsiderTrade(
                symbol='AAPL',
                insider_name=f'Insider {i}',
                insider_title='Director',
                transaction_type=TransactionType.PURCHASE,
                transaction_date=datetime.now(timezone.utc),
                shares=1000,
                price=150.0,
                shares_owned_after=10000,
                filing_date=datetime.now(timezone.utc)
            ))
        
        clusters = tracker.get_cluster_buying(min_insiders=3)
        
        assert len(clusters) >= 1
        assert clusters[0]['symbol'] == 'AAPL'


class TestInstitutionalHolding:
    """Tests for institutional holding dataclass."""
    
    def test_holding_properties(self):
        """Test holding computed properties."""
        holding = InstitutionalHolding(
            symbol='AAPL',
            institution_name='Test Fund',
            shares=100000,
            value=15000000.0,
            shares_change=100000,  # All new
            shares_change_pct=100.0
        )
        
        assert holding.is_new_position == True
        assert holding.is_increased == True
    
    def test_decreased_position(self):
        """Test decreased position detection."""
        holding = InstitutionalHolding(
            symbol='AAPL',
            institution_name='Test Fund',
            shares=50000,
            value=7500000.0,
            shares_change=-50000,
            shares_change_pct=-50.0
        )
        
        assert holding.is_decreased == True


# ============================================================================
# Economic Indicators Tests
# ============================================================================

class TestEconomicIndicators:
    """Tests for economic indicators tracking."""
    
    def test_init(self):
        """Test initialization."""
        indicators = EconomicIndicators()
        assert indicators is not None
    
    def test_add_release(self):
        """Test adding economic release."""
        indicators = EconomicIndicators()
        
        release = EconomicRelease(
            indicator=IndicatorType.CPI,
            name='Consumer Price Index',
            actual=3.2,
            forecast=3.0,
            previous=3.1,
            impact=ImpactLevel.CRITICAL
        )
        
        indicators.add_release(release)
        latest = indicators.get_latest(IndicatorType.CPI)
        
        assert latest is not None
        assert latest.actual == 3.2
    
    def test_release_surprise(self):
        """Test surprise calculation."""
        release = EconomicRelease(
            indicator=IndicatorType.NFP,
            name='Nonfarm Payrolls',
            actual=250000,
            forecast=200000,
            previous=180000
        )
        
        assert release.surprise == 50000
        assert release.surprise_pct == 25.0
    
    def test_beat_expectations_positive(self):
        """Test beat expectations for positive indicator."""
        release = EconomicRelease(
            indicator=IndicatorType.GDP,
            name='GDP',
            actual=3.0,
            forecast=2.5
        )
        
        assert release.beat_expectations == True
    
    def test_beat_expectations_unemployment(self):
        """Test beat expectations for unemployment (lower is better)."""
        release = EconomicRelease(
            indicator=IndicatorType.UNEMPLOYMENT,
            name='Unemployment Rate',
            actual=3.5,
            forecast=3.8
        )
        
        # Lower unemployment is better, so beating means actual < forecast
        assert release.beat_expectations == True
    
    def test_analyze_release(self):
        """Test market impact analysis."""
        indicators = EconomicIndicators()
        
        release = EconomicRelease(
            indicator=IndicatorType.CPI,
            name='CPI',
            actual=3.5,
            forecast=3.0,
            previous=3.0
        )
        
        impact = indicators.analyze_release(release)
        
        assert isinstance(impact, MarketImpact)
        assert impact.indicator == IndicatorType.CPI
    
    def test_add_upcoming_event(self):
        """Test adding upcoming economic event."""
        indicators = EconomicIndicators()
        
        event = EconomicEvent(
            indicator=IndicatorType.NFP,
            name='Nonfarm Payrolls',
            scheduled_time=datetime.now(timezone.utc) + timedelta(days=3),
            forecast=200000,
            impact=ImpactLevel.CRITICAL
        )
        
        indicators.add_upcoming_event(event)
        calendar = indicators.get_calendar(days=7)
        
        assert len(calendar) >= 1
    
    def test_get_economic_outlook(self):
        """Test economic outlook generation."""
        indicators = EconomicIndicators()
        
        # Add some releases
        indicators.add_release(EconomicRelease(
            indicator=IndicatorType.GDP,
            name='GDP',
            actual=3.0,
            forecast=2.5
        ))
        
        outlook = indicators.get_economic_outlook()
        
        assert 'growth' in outlook
        assert 'inflation' in outlook
        assert 'employment' in outlook
        assert 'overall' in outlook


class TestIndicatorTypes:
    """Tests for indicator type enum."""
    
    def test_indicator_types(self):
        """Test indicator type values."""
        assert IndicatorType.GDP.value == 'gdp'
        assert IndicatorType.CPI.value == 'cpi'
        assert IndicatorType.UNEMPLOYMENT.value == 'unemployment'
        assert IndicatorType.NFP.value == 'nonfarm_payrolls'
    
    def test_impact_levels(self):
        """Test impact level enum."""
        assert ImpactLevel.LOW.value == 'low'
        assert ImpactLevel.CRITICAL.value == 'critical'


# ============================================================================
# Options Flow Tests
# ============================================================================

class TestOptionsFlowAnalyzer:
    """Tests for options flow analysis."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = OptionsFlowAnalyzer()
        assert analyzer is not None
    
    def test_add_flow(self):
        """Test adding options flow."""
        analyzer = OptionsFlowAnalyzer()
        
        flow = OptionsFlow(
            symbol='AAPL',
            strike=150.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            contract_type='call',
            premium=50000.0,
            size=100,
            price=5.0,
            flow_type=FlowType.CALL_BUY,
            underlying_price=148.0
        )
        
        analyzer.add_flow(flow)
        summary = analyzer.get_flow_summary('AAPL')
        
        assert summary.total_call_volume >= 100
    
    def test_large_block_detection(self):
        """Test large block trade detection."""
        analyzer = OptionsFlowAnalyzer()
        
        # Large premium trade
        flow = OptionsFlow(
            symbol='AAPL',
            strike=150.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            contract_type='call',
            premium=150000.0,  # $150k
            size=500,
            price=3.0,
            flow_type=FlowType.CALL_BUY
        )
        
        analyzer.add_flow(flow)
        unusual = analyzer.get_unusual_activity(['AAPL'])
        
        # Should detect as unusual
        assert len(unusual) >= 1
        assert unusual[0].alert_type == 'large_block'
    
    def test_options_flow_properties(self):
        """Test options flow computed properties."""
        flow = OptionsFlow(
            symbol='AAPL',
            strike=155.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            contract_type='call',
            premium=5000.0,
            size=10,
            price=5.0,
            underlying_price=150.0,
            open_interest=100,
            daily_volume=200
        )
        
        assert flow.is_otm == True  # Strike 155 > Underlying 150
        assert flow.moneyness == 155.0 / 150.0
        assert flow.volume_to_oi_ratio == 2.0
    
    def test_put_is_otm(self):
        """Test OTM detection for puts."""
        flow = OptionsFlow(
            symbol='AAPL',
            strike=145.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            contract_type='put',
            premium=5000.0,
            size=10,
            price=5.0,
            underlying_price=150.0
        )
        
        assert flow.is_otm == True  # Strike 145 < Underlying 150 for put
    
    def test_classify_flow(self):
        """Test flow classification."""
        analyzer = OptionsFlowAnalyzer()
        
        assert analyzer.classify_flow('call', 'ask') == FlowType.CALL_BUY
        assert analyzer.classify_flow('call', 'bid') == FlowType.CALL_SELL
        assert analyzer.classify_flow('put', 'ask') == FlowType.PUT_BUY
        assert analyzer.classify_flow('put', 'bid') == FlowType.PUT_SELL
    
    def test_get_put_call_ratio(self):
        """Test put/call ratio calculation."""
        analyzer = OptionsFlowAnalyzer()
        
        # Add calls
        for _ in range(3):
            analyzer.add_flow(OptionsFlow(
                symbol='AAPL',
                strike=150.0,
                expiry=datetime.now(timezone.utc) + timedelta(days=30),
                contract_type='call',
                premium=5000.0,
                size=100,
                price=5.0
            ))
        
        # Add puts
        for _ in range(2):
            analyzer.add_flow(OptionsFlow(
                symbol='AAPL',
                strike=145.0,
                expiry=datetime.now(timezone.utc) + timedelta(days=30),
                contract_type='put',
                premium=4000.0,
                size=80,
                price=5.0
            ))
        
        ratios = analyzer.get_put_call_ratio('AAPL')
        
        assert 'volume_ratio' in ratios
        assert 'premium_ratio' in ratios
        assert 'interpretation' in ratios
    
    def test_smart_money_signals(self):
        """Test smart money signal generation."""
        analyzer = OptionsFlowAnalyzer()
        
        # Add bullish flow
        for _ in range(5):
            analyzer.add_flow(OptionsFlow(
                symbol='AAPL',
                strike=150.0,
                expiry=datetime.now(timezone.utc) + timedelta(days=30),
                contract_type='call',
                premium=20000.0,
                size=40,
                price=5.0,
                flow_type=FlowType.CALL_BUY
            ))
        
        signals = analyzer.get_smart_money_signals(['AAPL'])
        
        assert 'AAPL' in signals
        assert 'signal' in signals['AAPL']
        assert 'confidence' in signals['AAPL']


class TestFlowTypes:
    """Tests for flow type enums."""
    
    def test_flow_types(self):
        """Test flow type values."""
        assert FlowType.CALL_BUY.value == 'call_buy'
        assert FlowType.PUT_SELL.value == 'put_sell'
    
    def test_trade_types(self):
        """Test trade type values."""
        assert TradeType.BLOCK.value == 'block'
        assert TradeType.SWEEP.value == 'sweep'


# ============================================================================
# Alternative Data Manager Tests
# ============================================================================

class TestAlternativeDataManager:
    """Tests for unified data manager."""
    
    def test_init(self):
        """Test manager initialization."""
        manager = AlternativeDataManager()
        assert manager is not None
        assert manager.sec_tracker is not None
        assert manager.economic is not None
    
    def test_get_insider_sentiment(self):
        """Test insider sentiment analysis."""
        manager = AlternativeDataManager()
        
        # Add some insider trades
        for i in range(5):
            manager.sec_tracker.add_insider_trade(InsiderTrade(
                symbol='AAPL',
                insider_name=f'Insider {i}',
                insider_title='Director',
                transaction_type=TransactionType.PURCHASE,
                transaction_date=datetime.now(timezone.utc),
                shares=1000,
                price=150.0,
                shares_owned_after=10000,
                filing_date=datetime.now(timezone.utc)
            ))
        
        sentiment = manager.get_insider_sentiment('AAPL')
        
        assert sentiment.buy_count == 5
        assert sentiment.signal == AlternativeSignal.STRONG_BULLISH
    
    def test_insider_sentiment_bearish(self):
        """Test bearish insider sentiment."""
        manager = AlternativeDataManager()
        
        # Add sales
        for i in range(5):
            manager.sec_tracker.add_insider_trade(InsiderTrade(
                symbol='AAPL',
                insider_name=f'Insider {i}',
                insider_title='Director',
                transaction_type=TransactionType.SALE,
                transaction_date=datetime.now(timezone.utc),
                shares=5000,
                price=150.0,
                shares_owned_after=5000,
                filing_date=datetime.now(timezone.utc)
            ))
        
        sentiment = manager.get_insider_sentiment('AAPL')
        
        assert sentiment.sell_count == 5
        assert sentiment.signal in [AlternativeSignal.BEARISH, AlternativeSignal.STRONG_BEARISH]
    
    def test_get_institutional_sentiment(self):
        """Test institutional sentiment analysis."""
        manager = AlternativeDataManager()
        
        # Add holdings
        manager.sec_tracker.add_institutional_holding(InstitutionalHolding(
            symbol='AAPL',
            institution_name='Test Fund',
            shares=1000000,
            value=150000000.0,
            shares_change=200000,
            shares_change_pct=25.0,
            filing_date=datetime.now(timezone.utc)
        ))
        
        sentiment = manager.get_institutional_sentiment('AAPL')
        
        assert isinstance(sentiment, InstitutionalSentiment)
    
    def test_add_earnings_event(self):
        """Test adding earnings event."""
        manager = AlternativeDataManager()
        
        event = EarningsEvent(
            symbol='AAPL',
            company_name='Apple Inc.',
            report_date=datetime.now(timezone.utc) + timedelta(days=7),
            eps_estimate=1.50,
            revenue_estimate=90000000000.0
        )
        
        manager.add_earnings_event(event)
        
        upcoming = manager.get_upcoming_earnings(['AAPL'])
        assert len(upcoming) == 1
        assert upcoming[0].symbol == 'AAPL'
    
    def test_earnings_event_properties(self):
        """Test earnings event computed properties."""
        event = EarningsEvent(
            symbol='AAPL',
            company_name='Apple Inc.',
            report_date=datetime.now(timezone.utc) + timedelta(days=5),
            implied_move=5.0
        )
        
        assert event.days_until == 5
    
    def test_get_data_summary(self):
        """Test comprehensive data summary."""
        manager = AlternativeDataManager()
        
        summary = manager.get_data_summary('AAPL')
        
        assert isinstance(summary, AlternativeDataSummary)
        assert summary.symbol == 'AAPL'
        assert summary.overall_signal in AlternativeSignal
    
    def test_get_high_conviction_signals(self):
        """Test high conviction signal filtering."""
        manager = AlternativeDataManager()
        
        # Add strong insider buying
        for i in range(10):
            manager.sec_tracker.add_insider_trade(InsiderTrade(
                symbol='AAPL',
                insider_name=f'Insider {i}',
                insider_title='Director',
                transaction_type=TransactionType.PURCHASE,
                transaction_date=datetime.now(timezone.utc),
                shares=10000,
                price=150.0,
                shares_owned_after=100000,
                filing_date=datetime.now(timezone.utc)
            ))
        
        signals = manager.get_high_conviction_signals(['AAPL'], min_confidence=0.5)
        
        # Should have some signals
        assert isinstance(signals, dict)
    
    def test_get_economic_calendar(self):
        """Test economic calendar retrieval."""
        manager = AlternativeDataManager()
        
        # Add event
        manager.economic.add_upcoming_event(EconomicEvent(
            indicator=IndicatorType.NFP,
            name='Nonfarm Payrolls',
            scheduled_time=datetime.now(timezone.utc) + timedelta(days=3),
            impact=ImpactLevel.CRITICAL
        ))
        
        calendar = manager.get_economic_calendar(days=7, high_impact_only=True)
        
        assert len(calendar) >= 1
    
    def test_analyze_economic_release(self):
        """Test economic release analysis."""
        manager = AlternativeDataManager()
        
        release = EconomicRelease(
            indicator=IndicatorType.CPI,
            name='CPI',
            actual=3.5,
            forecast=3.0
        )
        
        analysis = manager.analyze_economic_release(release)
        
        assert 'indicator' in analysis
        assert 'surprise' in analysis
        assert 'equity_impact' in analysis


class TestAlternativeSignal:
    """Tests for alternative signal enum."""
    
    def test_signal_values(self):
        """Test signal enum values."""
        assert AlternativeSignal.STRONG_BULLISH.value == 'strong_bullish'
        assert AlternativeSignal.BEARISH.value == 'bearish'
        assert AlternativeSignal.NEUTRAL.value == 'neutral'


# ============================================================================
# Integration Tests
# ============================================================================

class TestAlternativeDataIntegration:
    """Integration tests for alternative data module."""
    
    def test_full_workflow(self):
        """Test complete alternative data workflow."""
        manager = AlternativeDataManager()
        
        # 1. Add insider trades
        manager.sec_tracker.add_insider_trade(InsiderTrade(
            symbol='AAPL',
            insider_name='CEO',
            insider_title='CEO',
            transaction_type=TransactionType.PURCHASE,
            transaction_date=datetime.now(timezone.utc),
            shares=50000,
            price=150.0,
            shares_owned_after=500000,
            filing_date=datetime.now(timezone.utc)
        ))
        
        # 2. Add institutional holdings
        manager.sec_tracker.add_institutional_holding(InstitutionalHolding(
            symbol='AAPL',
            institution_name='Major Fund',
            shares=5000000,
            value=750000000.0,
            shares_change=500000,
            shares_change_pct=11.0,
            filing_date=datetime.now(timezone.utc)
        ))
        
        # 3. Add earnings event
        manager.add_earnings_event(EarningsEvent(
            symbol='AAPL',
            company_name='Apple Inc.',
            report_date=datetime.now(timezone.utc) + timedelta(days=14),
            eps_estimate=1.50
        ))
        
        # 4. Add economic data
        manager.economic.add_release(EconomicRelease(
            indicator=IndicatorType.GDP,
            name='GDP',
            actual=3.0,
            forecast=2.5
        ))
        
        # 5. Get comprehensive summary
        summary = manager.get_data_summary('AAPL')
        
        assert summary.symbol == 'AAPL'
        assert summary.insider_sentiment is not None
        assert summary.institutional_sentiment is not None
        assert len(summary.bullish_factors) > 0 or len(summary.bearish_factors) > 0
    
    def test_multi_symbol_analysis(self):
        """Test analyzing multiple symbols."""
        manager = AlternativeDataManager()
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        for symbol in symbols:
            manager.sec_tracker.add_insider_trade(InsiderTrade(
                symbol=symbol,
                insider_name='Insider',
                insider_title='Director',
                transaction_type=TransactionType.PURCHASE,
                transaction_date=datetime.now(timezone.utc),
                shares=1000,
                price=100.0,
                shares_owned_after=10000,
                filing_date=datetime.now(timezone.utc)
            ))
        
        signals = manager.get_high_conviction_signals(symbols, min_confidence=0.0)
        
        assert len(signals) == 3
        for symbol in symbols:
            assert symbol in signals
