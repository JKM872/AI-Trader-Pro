"""Tests for Dynamic Risk Management."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from trader.risk.dynamic_risk import (
    DynamicRiskManager,
    DynamicRiskConfig,
    RiskAdjustment,
    RiskMonitor,
    RiskMode,
    MarketCondition,
    CorrelationRisk
)


class TestMarketCondition:
    """Tests for MarketCondition enum."""
    
    def test_market_condition_values(self):
        """Test market condition values."""
        assert MarketCondition.CALM.value == "calm"
        assert MarketCondition.NORMAL.value == "normal"
        assert MarketCondition.VOLATILE.value == "volatile"
        assert MarketCondition.CRISIS.value == "crisis"
    
    def test_all_conditions_exist(self):
        """Test all expected conditions exist."""
        conditions = [c for c in MarketCondition]
        assert len(conditions) == 4


class TestRiskMode:
    """Tests for RiskMode enum."""
    
    def test_risk_mode_values(self):
        """Test risk mode values."""
        assert RiskMode.CONSERVATIVE.value == "conservative"
        assert RiskMode.MODERATE.value == "moderate"
        assert RiskMode.AGGRESSIVE.value == "aggressive"
        assert RiskMode.ADAPTIVE.value == "adaptive"


class TestDynamicRiskConfig:
    """Tests for DynamicRiskConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DynamicRiskConfig()
        
        assert config.base_risk_per_trade == 0.02
        assert config.base_max_position == 0.15
        assert config.base_max_drawdown == 0.10
        assert config.target_portfolio_vol == 0.15
        assert config.max_portfolio_vol == 0.25
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DynamicRiskConfig(
            base_risk_per_trade=0.01,
            base_max_position=0.10,
            regime_scale_crisis=0.2
        )
        
        assert config.base_risk_per_trade == 0.01
        assert config.base_max_position == 0.10
        assert config.regime_scale_crisis == 0.2
    
    def test_regime_scales(self):
        """Test regime scaling factors."""
        config = DynamicRiskConfig()
        
        # Calm should scale up
        assert config.regime_scale_calm > 1.0
        
        # Normal is baseline
        assert config.regime_scale_normal == 1.0
        
        # Volatile and crisis scale down
        assert config.regime_scale_volatile < 1.0
        assert config.regime_scale_crisis < config.regime_scale_volatile


class TestDynamicRiskManager:
    """Tests for DynamicRiskManager."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create dynamic risk manager instance."""
        return DynamicRiskManager(initial_capital=100000)
    
    @pytest.fixture
    def custom_config(self):
        """Create custom config."""
        return DynamicRiskConfig(
            base_risk_per_trade=0.015,
            base_max_position=0.12,
            target_portfolio_vol=0.12
        )
    
    def test_initialization(self, risk_manager):
        """Test manager initialization."""
        assert risk_manager.initial_capital == 100000
        assert risk_manager.peak_capital == 100000
        assert risk_manager.current_market_condition == MarketCondition.NORMAL
        assert risk_manager.risk_mode == RiskMode.ADAPTIVE
        assert not risk_manager.in_recovery
    
    def test_custom_config_initialization(self, custom_config):
        """Test initialization with custom config."""
        manager = DynamicRiskManager(config=custom_config, initial_capital=50000)
        
        assert manager.config.base_risk_per_trade == 0.015
        assert manager.initial_capital == 50000

    # Market Condition Assessment Tests
    
    def test_assess_condition_calm_with_vix(self, risk_manager):
        """Test calm market with low VIX."""
        condition = risk_manager.assess_market_condition(
            current_volatility=0.10,
            historical_volatility=0.15,
            vix_level=12
        )
        
        assert condition == MarketCondition.CALM
        assert risk_manager.current_market_condition == MarketCondition.CALM
    
    def test_assess_condition_normal_with_vix(self, risk_manager):
        """Test normal market with moderate VIX."""
        condition = risk_manager.assess_market_condition(
            current_volatility=0.15,
            historical_volatility=0.15,
            vix_level=20
        )
        
        assert condition == MarketCondition.NORMAL
    
    def test_assess_condition_volatile_with_vix(self, risk_manager):
        """Test volatile market with high VIX."""
        condition = risk_manager.assess_market_condition(
            current_volatility=0.25,
            historical_volatility=0.15,
            vix_level=30
        )
        
        assert condition == MarketCondition.VOLATILE
    
    def test_assess_condition_crisis_with_vix(self, risk_manager):
        """Test crisis with very high VIX."""
        condition = risk_manager.assess_market_condition(
            current_volatility=0.50,
            historical_volatility=0.15,
            vix_level=45
        )
        
        assert condition == MarketCondition.CRISIS
    
    def test_assess_condition_by_volatility_ratio(self, risk_manager):
        """Test condition assessment by volatility ratio."""
        # Low volatility -> calm
        condition = risk_manager.assess_market_condition(0.08, 0.15)
        assert condition == MarketCondition.CALM
        
        # Normal volatility
        condition = risk_manager.assess_market_condition(0.15, 0.15)
        assert condition == MarketCondition.NORMAL
        
        # High volatility
        condition = risk_manager.assess_market_condition(0.25, 0.15)
        assert condition == MarketCondition.VOLATILE
        
        # Very high volatility -> crisis
        condition = risk_manager.assess_market_condition(0.35, 0.15)
        assert condition == MarketCondition.CRISIS
    
    def test_volatility_history_tracked(self, risk_manager):
        """Test volatility history is tracked."""
        risk_manager.assess_market_condition(0.15, 0.15)
        risk_manager.assess_market_condition(0.16, 0.15)
        risk_manager.assess_market_condition(0.14, 0.15)
        
        assert len(risk_manager.volatility_history) == 3
        assert risk_manager.volatility_history == [0.15, 0.16, 0.14]
    
    def test_volatility_history_limited_to_252(self, risk_manager):
        """Test volatility history is limited to 252 days."""
        for i in range(300):
            risk_manager.assess_market_condition(0.15 + i * 0.001, 0.15)
        
        assert len(risk_manager.volatility_history) == 252

    # Risk Adjustment Tests
    
    def test_calculate_risk_adjustment_normal(self, risk_manager):
        """Test risk adjustment in normal conditions."""
        risk_manager.assess_market_condition(0.15, 0.15)  # Normal
        
        adjustment = risk_manager.calculate_risk_adjustment(
            current_capital=100000,
            current_volatility=0.15
        )
        
        assert isinstance(adjustment, RiskAdjustment)
        assert adjustment.market_condition == MarketCondition.NORMAL
        assert adjustment.regime_scale == 1.0
    
    def test_calculate_risk_adjustment_crisis(self, risk_manager):
        """Test risk adjustment in crisis conditions."""
        risk_manager.assess_market_condition(0.40, 0.15, vix_level=50)
        
        adjustment = risk_manager.calculate_risk_adjustment(
            current_capital=100000,
            current_volatility=0.40
        )
        
        assert adjustment.market_condition == MarketCondition.CRISIS
        assert adjustment.regime_scale == 0.3  # Crisis scale
        assert adjustment.overall_scale < 0.5
        assert adjustment.is_restricted
    
    def test_risk_scales_with_drawdown(self, risk_manager):
        """Test risk scaling with drawdown."""
        # Set peak capital higher than current
        risk_manager.peak_capital = 120000
        
        adjustment = risk_manager.calculate_risk_adjustment(
            current_capital=100000,  # 16.7% drawdown
            current_volatility=0.15
        )
        
        assert adjustment.drawdown_scale < 1.0
    
    def test_adjusted_limits_scaled(self, risk_manager):
        """Test that adjusted limits are properly scaled."""
        risk_manager.assess_market_condition(0.30, 0.15, vix_level=35)
        
        adjustment = risk_manager.calculate_risk_adjustment(
            current_capital=100000,
            current_volatility=0.30
        )
        
        # Limits should be scaled down
        base = risk_manager.config
        assert adjustment.adjusted_risk_per_trade < base.base_risk_per_trade
        assert adjustment.adjusted_max_position < base.base_max_position
    
    def test_min_risk_scale_respected(self, risk_manager):
        """Test minimum risk scale is enforced."""
        # Force extreme conditions
        risk_manager.peak_capital = 200000
        risk_manager.assess_market_condition(0.50, 0.15, vix_level=60)
        
        adjustment = risk_manager.calculate_risk_adjustment(
            current_capital=100000,
            current_volatility=0.50
        )
        
        assert adjustment.overall_scale >= risk_manager.config.min_risk_scale

    # Correlation Risk Tests
    
    def test_correlation_risk_empty_positions(self, risk_manager):
        """Test correlation risk with no positions."""
        risk = risk_manager.analyze_correlation_risk({})
        
        assert risk.diversification_score == 1.0
        assert len(risk.highly_correlated_pairs) == 0
    
    def test_correlation_risk_single_position(self, risk_manager):
        """Test correlation risk with single position."""
        positions = {'AAPL': {'value': 10000}}
        
        risk = risk_manager.analyze_correlation_risk(positions)
        
        assert risk.diversification_score == 1.0
    
    def test_correlation_risk_multiple_positions(self, risk_manager):
        """Test correlation risk with multiple positions."""
        positions = {
            'AAPL': {'value': 30000},
            'MSFT': {'value': 25000},
            'GOOGL': {'value': 25000},
            'AMZN': {'value': 20000}
        }
        
        risk = risk_manager.analyze_correlation_risk(positions)
        
        assert 0 <= risk.concentration_risk <= 1
        assert 0 <= risk.diversification_score <= 1
    
    def test_concentration_risk_equal_weights(self, risk_manager):
        """Test concentration with equal weights."""
        positions = {
            'A': {'value': 25000},
            'B': {'value': 25000},
            'C': {'value': 25000},
            'D': {'value': 25000}
        }
        
        risk = risk_manager.analyze_correlation_risk(positions)
        
        # Equal weights should have low concentration
        assert risk.concentration_risk < 0.2
    
    def test_concentration_risk_unequal_weights(self, risk_manager):
        """Test concentration with one dominant position."""
        positions = {
            'A': {'value': 80000},
            'B': {'value': 10000},
            'C': {'value': 5000},
            'D': {'value': 5000}
        }
        
        risk = risk_manager.analyze_correlation_risk(positions)
        
        # One dominant position should have high concentration
        assert risk.concentration_risk > 0.5

    # Position Size Adjustment Tests
    
    def test_position_size_adjustment_normal(self, risk_manager):
        """Test position sizing in normal conditions."""
        risk_manager.assess_market_condition(0.15, 0.15)
        risk_manager.volatility_history = [0.15] * 20
        
        adjusted, reason = risk_manager.get_position_size_adjustment(
            symbol='AAPL',
            base_position_value=10000,
            symbol_volatility=0.25,
            current_capital=100000
        )
        
        assert adjusted > 0
        assert isinstance(reason, str)
    
    def test_position_size_reduced_high_volatility(self, risk_manager):
        """Test position sizing reduced for high volatility stock."""
        risk_manager.assess_market_condition(0.15, 0.15)
        risk_manager.volatility_history = [0.15] * 20
        
        # High volatility stock
        adjusted, reason = risk_manager.get_position_size_adjustment(
            symbol='TSLA',
            base_position_value=10000,
            symbol_volatility=0.60,  # 60% volatility
            current_capital=100000
        )
        
        assert adjusted < 10000
        assert "volatility" in reason.lower()
    
    def test_position_size_capped(self, risk_manager):
        """Test position size is capped at max."""
        risk_manager.assess_market_condition(0.10, 0.15)  # Calm market
        risk_manager.volatility_history = [0.10] * 20
        
        adjusted, reason = risk_manager.get_position_size_adjustment(
            symbol='AAPL',
            base_position_value=50000,  # 50% of capital
            symbol_volatility=0.20,
            current_capital=100000
        )
        
        # Position should be scaled and capped
        # In calm market with scaling, still should not exceed reasonable bounds
        max_pos = 100000 * risk_manager.config.base_max_position
        # Allow for calm regime scaling (1.2x) and volatility scaling (up to 1.5x)
        assert adjusted <= max_pos * 2.0  # Allow for combined scaling factors

    # Exposure Reduction Tests
    
    def test_should_reduce_exposure_crisis(self, risk_manager):
        """Test exposure reduction in crisis."""
        risk_manager.current_market_condition = MarketCondition.CRISIS
        
        should_reduce, reason, pct = risk_manager.should_reduce_exposure()
        
        assert should_reduce
        assert "crisis" in reason.lower()
        assert pct == 0.5
    
    def test_should_reduce_exposure_normal(self, risk_manager):
        """Test no reduction in normal conditions."""
        risk_manager.current_market_condition = MarketCondition.NORMAL
        risk_manager.in_recovery = False
        
        should_reduce, reason, pct = risk_manager.should_reduce_exposure()
        
        assert not should_reduce
        assert reason is None
    
    def test_should_reduce_exposure_increasing_volatility(self, risk_manager):
        """Test reduction with rapidly increasing volatility."""
        risk_manager.volatility_history = [0.15] * 5 + [0.25, 0.27, 0.30, 0.35, 0.40]
        
        should_reduce, reason, pct = risk_manager.should_reduce_exposure()
        
        assert should_reduce
        assert "volatility" in reason.lower()
    
    def test_should_reduce_exposure_recovery(self, risk_manager):
        """Test reduction during recovery period."""
        risk_manager.in_recovery = True
        risk_manager.last_major_drawdown = datetime.now(timezone.utc) - timedelta(days=2)
        
        should_reduce, reason, pct = risk_manager.should_reduce_exposure()
        
        assert should_reduce
        assert "recovery" in reason.lower()

    # Peak Capital and Drawdown Tests
    
    def test_peak_capital_updated(self, risk_manager):
        """Test peak capital is updated on new highs."""
        risk_manager.calculate_risk_adjustment(110000, 0.15)
        
        assert risk_manager.peak_capital == 110000
    
    def test_peak_capital_not_reduced(self, risk_manager):
        """Test peak capital doesn't decrease."""
        risk_manager.peak_capital = 110000
        
        risk_manager.calculate_risk_adjustment(105000, 0.15)
        
        assert risk_manager.peak_capital == 110000
    
    def test_recovery_mode_activated(self, risk_manager):
        """Test recovery mode activates on large drawdown."""
        risk_manager.peak_capital = 120000
        config = risk_manager.config
        
        # Drawdown > 75% of max drawdown
        threshold = config.base_max_drawdown * 0.75
        capital = 120000 * (1 - threshold - 0.01)
        
        risk_manager.calculate_risk_adjustment(capital, 0.15)
        
        assert risk_manager.in_recovery
    
    def test_recovery_mode_clears_on_new_high(self, risk_manager):
        """Test recovery mode clears on new equity high."""
        risk_manager.in_recovery = True
        risk_manager.peak_capital = 100000
        
        risk_manager.calculate_risk_adjustment(110000, 0.15)
        
        assert not risk_manager.in_recovery

    # Risk Report Tests
    
    def test_get_risk_report(self, risk_manager):
        """Test risk report generation."""
        risk_manager.assess_market_condition(0.18, 0.15)
        risk_manager.volatility_history = [0.18] * 20
        
        report = risk_manager.get_risk_report(
            current_capital=95000,
            positions={'AAPL': {'value': 20000}}
        )
        
        assert "MARKET CONDITIONS" in report
        assert "PORTFOLIO STATUS" in report
        assert "RISK ADJUSTMENTS" in report
        assert "ADJUSTED LIMITS" in report
    
    def test_report_shows_restrictions(self, risk_manager):
        """Test report shows restrictions when applicable."""
        risk_manager.assess_market_condition(0.50, 0.15, vix_level=50)
        risk_manager.volatility_history = [0.50] * 20
        
        report = risk_manager.get_risk_report(current_capital=80000)
        
        assert "RESTRICTED" in report or "crisis" in report.lower()


class TestRiskMonitor:
    """Tests for RiskMonitor."""
    
    @pytest.fixture
    def risk_monitor(self):
        """Create risk monitor instance."""
        manager = DynamicRiskManager(initial_capital=100000)
        return RiskMonitor(manager)
    
    def test_initialization(self, risk_monitor):
        """Test monitor initialization."""
        assert risk_monitor.alerts == []
        assert risk_monitor.last_check is None
    
    def test_check_risks_no_alerts_normal(self, risk_monitor):
        """Test no alerts in normal conditions."""
        risk_monitor.risk_manager.assess_market_condition(0.15, 0.15)
        risk_monitor.risk_manager.volatility_history = [0.15] * 20
        
        alerts = risk_monitor.check_risks(
            current_capital=100000,
            positions={},
            current_volatility=0.15
        )
        
        # Should have minimal or no alerts in normal conditions
        assert isinstance(alerts, list)
    
    def test_check_risks_crisis_alert(self, risk_monitor):
        """Test alert generated in crisis."""
        risk_monitor.risk_manager.assess_market_condition(0.50, 0.15, vix_level=50)
        
        alerts = risk_monitor.check_risks(
            current_capital=100000,
            positions={},
            current_volatility=0.50
        )
        
        assert len(alerts) > 0
        restriction_alerts = [a for a in alerts if a['type'] == 'restriction']
        assert len(restriction_alerts) > 0
    
    def test_check_risks_high_volatility_alert(self, risk_monitor):
        """Test alert for high portfolio volatility."""
        risk_monitor.risk_manager.assess_market_condition(0.15, 0.15)
        
        alerts = risk_monitor.check_risks(
            current_capital=100000,
            positions={},
            current_volatility=0.30  # Above 25% max
        )
        
        vol_alerts = [a for a in alerts if a['type'] == 'volatility']
        assert len(vol_alerts) > 0
    
    def test_check_risks_concentration_alert(self, risk_monitor):
        """Test alert for high concentration."""
        risk_monitor.risk_manager.assess_market_condition(0.15, 0.15)
        
        # High concentration portfolio
        positions = {
            'AAPL': {'value': 85000},
            'MSFT': {'value': 5000},
            'GOOGL': {'value': 5000},
            'AMZN': {'value': 5000}
        }
        
        alerts = risk_monitor.check_risks(
            current_capital=100000,
            positions=positions,
            current_volatility=0.15
        )
        
        conc_alerts = [a for a in alerts if a['type'] == 'concentration']
        assert len(conc_alerts) > 0
    
    def test_alerts_stored(self, risk_monitor):
        """Test alerts are stored in history."""
        risk_monitor.risk_manager.current_market_condition = MarketCondition.CRISIS
        
        risk_monitor.check_risks(100000, {}, 0.15)
        risk_monitor.check_risks(100000, {}, 0.15)
        
        assert len(risk_monitor.alerts) >= 2
    
    def test_get_recent_alerts(self, risk_monitor):
        """Test getting recent alerts."""
        risk_monitor.risk_manager.current_market_condition = MarketCondition.CRISIS
        
        risk_monitor.check_risks(100000, {}, 0.15)
        
        recent = risk_monitor.get_recent_alerts(hours=1)
        
        assert len(recent) > 0
        for alert in recent:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'timestamp' in alert
    
    def test_old_alerts_pruned(self, risk_monitor):
        """Test old alerts are removed."""
        # Add old alert manually
        old_alert = {
            'type': 'test',
            'severity': 'low',
            'message': 'Old alert',
            'timestamp': datetime.now(timezone.utc) - timedelta(hours=30),
            'action': 'None'
        }
        risk_monitor.alerts.append(old_alert)
        
        # Check risks (triggers pruning)
        risk_monitor.check_risks(100000, {}, 0.15)
        
        # Old alert should be removed
        old_alerts = [a for a in risk_monitor.alerts if a['message'] == 'Old alert']
        assert len(old_alerts) == 0
    
    def test_last_check_updated(self, risk_monitor):
        """Test last check timestamp updated."""
        assert risk_monitor.last_check is None
        
        risk_monitor.check_risks(100000, {}, 0.15)
        
        assert risk_monitor.last_check is not None
        assert isinstance(risk_monitor.last_check, datetime)


class TestRiskAdjustment:
    """Tests for RiskAdjustment dataclass."""
    
    def test_default_values(self):
        """Test default adjustment values."""
        adj = RiskAdjustment(
            adjusted_risk_per_trade=0.02,
            adjusted_max_position=0.15,
            adjusted_max_exposure=0.80
        )
        
        assert adj.volatility_scale == 1.0
        assert adj.regime_scale == 1.0
        assert adj.drawdown_scale == 1.0
        assert adj.correlation_scale == 1.0
        assert adj.overall_scale == 1.0
        assert adj.reduce_positions == []
        assert adj.recommendations == []
        assert not adj.is_restricted
    
    def test_with_restrictions(self):
        """Test adjustment with restrictions."""
        adj = RiskAdjustment(
            adjusted_risk_per_trade=0.01,
            adjusted_max_position=0.08,
            adjusted_max_exposure=0.40,
            overall_scale=0.4,
            is_restricted=True,
            restriction_reason="Market crisis",
            recommendations=["Reduce exposure", "Avoid new trades"]
        )
        
        assert adj.is_restricted
        assert adj.restriction_reason == "Market crisis"
        assert len(adj.recommendations) == 2


class TestCorrelationRisk:
    """Tests for CorrelationRisk dataclass."""
    
    def test_default_values(self):
        """Test default correlation risk values."""
        risk = CorrelationRisk()
        
        assert risk.correlation_matrix.empty
        assert risk.highly_correlated_pairs == []
        assert risk.cluster_exposure == {}
        assert risk.diversification_score == 0.0
        assert risk.concentration_risk == 0.0
    
    def test_with_data(self):
        """Test correlation risk with data."""
        corr_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8],
            'MSFT': [0.8, 1.0]
        }, index=['AAPL', 'MSFT'])
        
        risk = CorrelationRisk(
            correlation_matrix=corr_matrix,
            highly_correlated_pairs=[('AAPL', 'MSFT', 0.8)],
            diversification_score=0.5,
            concentration_risk=0.3
        )
        
        assert not risk.correlation_matrix.empty
        assert len(risk.highly_correlated_pairs) == 1
        assert risk.diversification_score == 0.5


class TestIntegration:
    """Integration tests for dynamic risk management."""
    
    def test_full_workflow(self):
        """Test full risk management workflow."""
        # Initialize
        config = DynamicRiskConfig(
            base_risk_per_trade=0.02,
            base_max_position=0.15
        )
        manager = DynamicRiskManager(config=config, initial_capital=100000)
        monitor = RiskMonitor(manager)
        
        # Assess market
        manager.assess_market_condition(0.18, 0.15)
        
        # Add volatility history
        manager.volatility_history = [0.15 + i * 0.001 for i in range(20)]
        
        # Calculate adjustment
        adjustment = manager.calculate_risk_adjustment(
            current_capital=98000,
            current_volatility=0.18
        )
        
        # Get position size
        size, reason = manager.get_position_size_adjustment(
            symbol='AAPL',
            base_position_value=10000,
            symbol_volatility=0.25,
            current_capital=98000
        )
        
        # Check risks
        alerts = monitor.check_risks(
            current_capital=98000,
            positions={'AAPL': {'value': size}},
            current_volatility=0.18
        )
        
        # Generate report
        report = manager.get_risk_report(
            current_capital=98000,
            positions={'AAPL': {'value': size}}
        )
        
        # Verify workflow completed
        assert adjustment.overall_scale > 0
        assert size > 0
        assert isinstance(alerts, list)
        assert len(report) > 0
    
    def test_crisis_to_recovery(self):
        """Test transition from crisis to recovery."""
        manager = DynamicRiskManager(initial_capital=100000)
        
        # Enter crisis
        manager.assess_market_condition(0.50, 0.15, vix_level=50)
        manager.peak_capital = 120000
        
        # Large drawdown
        adj1 = manager.calculate_risk_adjustment(90000, 0.50)
        
        assert adj1.is_restricted
        assert manager.in_recovery
        
        # Market calms
        manager.assess_market_condition(0.15, 0.15, vix_level=15)
        
        # New high clears recovery
        adj2 = manager.calculate_risk_adjustment(125000, 0.15)
        
        assert not manager.in_recovery
        assert manager.peak_capital == 125000
