"""
Unit tests for OpportunityScorer.

Tests multi-factor scoring system for stock analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from trader.analysis.opportunity_scorer import (
    OpportunityScorer,
    OpportunityScore,
    get_top_opportunities
)


class TestOpportunityScore:
    """Test OpportunityScore dataclass."""
    
    def test_score_creation(self):
        """Test creating an OpportunityScore."""
        score = OpportunityScore(
            symbol="AAPL",
            total_score=75.0,
            recommendation="Buy",
            color="#4caf50"
        )
        
        assert score.symbol == "AAPL"
        assert score.total_score == 75.0
        assert score.recommendation == "Buy"
        assert score.color == "#4caf50"
    
    def test_default_values(self):
        """Test default values are set correctly."""
        score = OpportunityScore(
            symbol="TEST",
            total_score=50.0,
            recommendation="Hold",
            color="#787b86"
        )
        
        assert score.fundamentals_score == 0.0
        assert score.technicals_score == 0.0
        assert score.sentiment_score == 0.0
        assert score.guru_score == 0.0
        assert score.earnings_score == 0.0
        assert score.volatility == 0.0
        assert score.risk_level == "Medium"
        assert score.suggested_position_pct == 5.0


class TestOpportunityScorerRecommendations:
    """Test recommendation logic."""
    
    def setup_method(self):
        self.scorer = OpportunityScorer()
    
    def test_strong_buy_recommendation(self):
        """Test score >= 80 returns Strong Buy."""
        rec, color = self.scorer._get_recommendation(85)
        assert rec == "Strong Buy"
        assert color == "#089981"
    
    def test_buy_recommendation(self):
        """Test score 60-79 returns Buy."""
        rec, color = self.scorer._get_recommendation(70)
        assert rec == "Buy"
        assert color == "#4caf50"
    
    def test_hold_recommendation(self):
        """Test score 40-59 returns Hold."""
        rec, color = self.scorer._get_recommendation(50)
        assert rec == "Hold"
        assert color == "#787b86"
    
    def test_avoid_recommendation(self):
        """Test score 20-39 returns Avoid."""
        rec, color = self.scorer._get_recommendation(30)
        assert rec == "Avoid"
        assert color == "#ff9800"
    
    def test_high_risk_recommendation(self):
        """Test score < 20 returns High Risk."""
        rec, color = self.scorer._get_recommendation(15)
        assert rec == "High Risk"
        assert color == "#f23645"
    
    def test_boundary_values(self):
        """Test boundary values."""
        assert self.scorer._get_recommendation(80)[0] == "Strong Buy"
        assert self.scorer._get_recommendation(79)[0] == "Buy"
        assert self.scorer._get_recommendation(60)[0] == "Buy"
        assert self.scorer._get_recommendation(59)[0] == "Hold"
        assert self.scorer._get_recommendation(40)[0] == "Hold"
        assert self.scorer._get_recommendation(39)[0] == "Avoid"
        assert self.scorer._get_recommendation(20)[0] == "Avoid"
        assert self.scorer._get_recommendation(19)[0] == "High Risk"


class TestFundamentalsScoring:
    """Test fundamentals scoring logic."""
    
    def setup_method(self):
        self.scorer = OpportunityScorer()
    
    def test_empty_fundamentals(self):
        """Test empty fundamentals returns neutral score."""
        score, details = self.scorer._score_fundamentals({})
        assert score == 50.0
        assert "error" in details
    
    def test_low_pe_ratio_boost(self):
        """Test low P/E ratio increases score."""
        fundamentals = {"pe_ratio": 10}
        score, details = self.scorer._score_fundamentals(fundamentals)
        assert score > 50.0
        assert details["pe_ratio"] == 10
    
    def test_high_pe_ratio_penalty(self):
        """Test high P/E ratio decreases score."""
        fundamentals = {"pe_ratio": 100}
        score, details = self.scorer._score_fundamentals(fundamentals)
        assert score < 50.0
    
    def test_negative_pe_penalty(self):
        """Test negative P/E (negative earnings) penalty."""
        fundamentals = {"pe_ratio": -5}
        score, details = self.scorer._score_fundamentals(fundamentals)
        assert score < 50.0
    
    def test_high_revenue_growth_boost(self):
        """Test high revenue growth increases score."""
        fundamentals = {"revenue_growth": 0.25}
        score, details = self.scorer._score_fundamentals(fundamentals)
        assert score > 50.0
    
    def test_high_profit_margin_boost(self):
        """Test high profit margin increases score."""
        fundamentals = {"profit_margin": 0.25}
        score, details = self.scorer._score_fundamentals(fundamentals)
        assert score > 50.0
    
    def test_low_debt_boost(self):
        """Test low debt/equity increases score."""
        fundamentals = {"debt_to_equity": 0.3}
        score, details = self.scorer._score_fundamentals(fundamentals)
        assert score > 50.0
    
    def test_high_debt_penalty(self):
        """Test high debt/equity decreases score."""
        fundamentals = {"debt_to_equity": 3.0}
        score, details = self.scorer._score_fundamentals(fundamentals)
        assert score < 50.0
    
    def test_combined_fundamentals(self):
        """Test combination of all fundamentals."""
        fundamentals = {
            "pe_ratio": 12,
            "revenue_growth": 0.15,
            "profit_margin": 0.18,
            "debt_to_equity": 0.4
        }
        score, details = self.scorer._score_fundamentals(fundamentals)
        # All positive factors should result in high score
        assert score > 70.0


class TestTechnicalsScoring:
    """Test technicals scoring logic."""
    
    def setup_method(self):
        self.scorer = OpportunityScorer()
    
    def create_sample_data(self, days=100, trend="up"):
        """Create sample OHLCV data."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        if trend == "up":
            close = np.linspace(100, 150, days) + np.random.randn(days) * 2
        else:
            close = np.linspace(150, 100, days) + np.random.randn(days) * 2
        
        return pd.DataFrame({
            'Open': close - 1,
            'High': close + 2,
            'Low': close - 2,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)
    
    def test_empty_data(self):
        """Test empty data returns neutral score."""
        score, details = self.scorer._score_technicals(pd.DataFrame())
        assert score == 50.0
        assert "error" in details
    
    def test_insufficient_data(self):
        """Test insufficient data returns neutral score."""
        df = self.create_sample_data(days=10)
        score, details = self.scorer._score_technicals(df)
        assert score == 50.0
    
    def test_uptrend_scoring(self):
        """Test uptrend increases score."""
        df = self.create_sample_data(days=100, trend="up")
        score, details = self.scorer._score_technicals(df)
        # Uptrend should result in positive score
        assert score >= 50.0
        assert "trend" in details
    
    def test_rsi_in_details(self):
        """Test RSI is calculated and included."""
        df = self.create_sample_data(days=100)
        score, details = self.scorer._score_technicals(df)
        assert "rsi" in details
        assert 0 <= details["rsi"] <= 100
    
    def test_macd_in_details(self):
        """Test MACD signal is included."""
        df = self.create_sample_data(days=100)
        score, details = self.scorer._score_technicals(df)
        assert "macd_signal" in details
        assert details["macd_signal"] in ["Bullish", "Bearish"]


class TestRiskAssessment:
    """Test risk assessment logic."""
    
    def setup_method(self):
        self.scorer = OpportunityScorer()
    
    def test_low_volatility_low_risk(self):
        """Test low volatility results in low risk."""
        risk = self.scorer._assess_risk(0.15, {"debt_to_equity": 0.5})
        assert risk == "Low"
    
    def test_high_volatility_high_risk(self):
        """Test high volatility results in high risk."""
        risk = self.scorer._assess_risk(0.60, {"debt_to_equity": 2.5})
        assert risk == "High"
    
    def test_medium_volatility_medium_risk(self):
        """Test medium volatility results in medium risk."""
        risk = self.scorer._assess_risk(0.28, {"debt_to_equity": 0.8})
        assert risk in ["Medium", "Low"]


class TestPositionSizing:
    """Test position size suggestions."""
    
    def setup_method(self):
        self.scorer = OpportunityScorer()
    
    def test_high_score_larger_position(self):
        """Test high score suggests larger position."""
        position = self.scorer._suggest_position_size(90, 0.20)
        assert position > 5.0
    
    def test_low_score_smaller_position(self):
        """Test low score suggests smaller position."""
        position = self.scorer._suggest_position_size(30, 0.20)
        assert position < 5.0
    
    def test_high_volatility_reduces_position(self):
        """Test high volatility reduces position size."""
        normal = self.scorer._suggest_position_size(70, 0.20)
        high_vol = self.scorer._suggest_position_size(70, 0.50)
        assert high_vol < normal
    
    def test_position_bounds(self):
        """Test position size stays within bounds."""
        min_pos = self.scorer._suggest_position_size(0, 1.0)
        max_pos = self.scorer._suggest_position_size(100, 0.10)
        
        assert min_pos >= 1.0
        assert max_pos <= 10.0


class TestVolatilityCalculation:
    """Test volatility calculation."""
    
    def setup_method(self):
        self.scorer = OpportunityScorer()
    
    def test_empty_data_default_volatility(self):
        """Test empty data returns default 25%."""
        vol = self.scorer._calculate_volatility(pd.DataFrame())
        assert vol == 0.25
    
    def test_volatility_calculation(self):
        """Test volatility is calculated correctly."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        close = np.linspace(100, 110, 100) + np.random.randn(100) * 2
        df = pd.DataFrame({'Close': close}, index=dates)
        
        vol = self.scorer._calculate_volatility(df)
        
        # Volatility should be reasonable (not 0, not > 200%)
        assert 0 < vol < 2.0


class TestIntegration:
    """Integration tests for full scoring."""
    
    def setup_method(self):
        self.scorer = OpportunityScorer()
    
    @pytest.mark.slow
    def test_score_real_stock(self):
        """Test scoring a real stock (requires network)."""
        score = self.scorer.score_stock("AAPL")
        
        assert score.symbol == "AAPL"
        assert 0 <= score.total_score <= 100
        assert score.recommendation in ["Strong Buy", "Buy", "Hold", "Avoid", "High Risk"]
        assert score.fundamentals_score >= 0
        assert score.technicals_score >= 0
    
    @pytest.mark.slow
    def test_scan_multiple_stocks(self):
        """Test scanning multiple stocks."""
        symbols = ["AAPL", "MSFT"]
        results = self.scorer.scan_opportunities(symbols, min_score=0)
        
        assert len(results) <= len(symbols)
        for score in results:
            assert score.symbol in symbols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
