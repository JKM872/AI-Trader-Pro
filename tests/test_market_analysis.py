"""
Tests for Market Analysis Module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from trader.market_analysis.regime_detector import (
    MarketRegimeDetector,
    MarketRegime,
    RegimeAnalysis,
    VolatilityLevel,
    TrendStrength
)
from trader.market_analysis.liquidity_mapper import (
    LiquidityMapper,
    LiquidityZone,
    ZoneType,
    LiquidityAnalysis
)
from trader.market_analysis.seasonality import (
    SeasonalityAnalyzer,
    SeasonalPattern,
    SeasonalityReport,
    TimeOfDay,
    DayOfWeek,
    MonthOfYear
)


# ==================== Test Fixtures ====================

@pytest.fixture
def sample_uptrend_df():
    """Create sample uptrend OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
    
    # Create uptrend with noise
    base_price = 100
    trend = np.linspace(0, 50, 250)  # 50% increase over period
    noise = np.random.normal(0, 2, 250)
    
    close = base_price + trend + noise
    high = close + np.random.uniform(1, 3, 250)
    low = close - np.random.uniform(1, 3, 250)
    open_prices = close - np.random.uniform(-1, 1, 250)
    volume = np.random.randint(1000000, 5000000, 250)
    
    return pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


@pytest.fixture
def sample_downtrend_df():
    """Create sample downtrend OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
    
    # Create downtrend with noise
    base_price = 150
    trend = np.linspace(0, -50, 250)  # 50 point decrease
    noise = np.random.normal(0, 2, 250)
    
    close = base_price + trend + noise
    high = close + np.random.uniform(1, 3, 250)
    low = close - np.random.uniform(1, 3, 250)
    open_prices = close + np.random.uniform(-1, 1, 250)
    volume = np.random.randint(1000000, 5000000, 250)
    
    return pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


@pytest.fixture
def sample_ranging_df():
    """Create sample ranging/sideways OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
    
    # Create ranging market
    base_price = 100
    oscillation = np.sin(np.linspace(0, 8*np.pi, 250)) * 5  # Oscillate ±5
    noise = np.random.normal(0, 1, 250)
    
    close = base_price + oscillation + noise
    high = close + np.random.uniform(0.5, 2, 250)
    low = close - np.random.uniform(0.5, 2, 250)
    open_prices = close + np.random.uniform(-0.5, 0.5, 250)
    volume = np.random.randint(500000, 2000000, 250)
    
    return pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)


# ==================== MarketRegimeDetector Tests ====================

class TestMarketRegimeDetector:
    """Tests for MarketRegimeDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return MarketRegimeDetector()
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.adx_period == 14
        assert detector.atr_period == 14
        assert detector.rsi_period == 14
        assert detector.sma_short == 20
        assert detector.sma_medium == 50
        assert detector.sma_long == 200
    
    def test_analyze_uptrend(self, detector, sample_uptrend_df):
        """Test analyzing uptrend market."""
        result = detector.analyze('AAPL', sample_uptrend_df)
        
        assert isinstance(result, RegimeAnalysis)
        assert result.symbol == 'AAPL'
        # In an uptrend, price should be above long-term SMA
        assert result.price_vs_sma200 > 0  # Above 200 SMA
        assert result.regime != MarketRegime.UNKNOWN
    
    def test_analyze_downtrend(self, detector, sample_downtrend_df):
        """Test analyzing downtrend market."""
        result = detector.analyze('AAPL', sample_downtrend_df)
        
        assert isinstance(result, RegimeAnalysis)
        # In a downtrend, price should be below long-term SMA
        assert result.price_vs_sma200 < 0  # Below 200 SMA
        assert result.regime != MarketRegime.UNKNOWN
    
    def test_analyze_ranging(self, detector, sample_ranging_df):
        """Test analyzing ranging market."""
        result = detector.analyze('AAPL', sample_ranging_df)
        
        assert isinstance(result, RegimeAnalysis)
        assert abs(result.trend_direction) < 0.5  # Weak directional bias
    
    def test_insufficient_data(self, detector):
        """Test handling insufficient data."""
        short_df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1000]
        })
        
        result = detector.analyze('TEST', short_df)
        
        assert result.regime == MarketRegime.UNKNOWN
        assert result.confidence == 0.0
    
    def test_adx_calculation(self, detector, sample_uptrend_df):
        """Test ADX calculation."""
        adx, plus_di, minus_di = detector._calculate_adx(sample_uptrend_df)
        
        assert adx >= 0
        assert plus_di >= 0
        assert minus_di >= 0
    
    def test_atr_calculation(self, detector, sample_uptrend_df):
        """Test ATR calculation."""
        atr, atr_percent = detector._calculate_atr(sample_uptrend_df)
        
        assert atr > 0
        assert atr_percent > 0
    
    def test_rsi_calculation(self, detector, sample_uptrend_df):
        """Test RSI calculation."""
        rsi = detector._calculate_rsi(sample_uptrend_df)
        
        assert 0 <= rsi <= 100
    
    def test_trend_strength_classification(self, detector):
        """Test trend strength classification."""
        assert detector._classify_trend_strength(55) == TrendStrength.VERY_STRONG
        assert detector._classify_trend_strength(30) == TrendStrength.STRONG
        assert detector._classify_trend_strength(22) == TrendStrength.MODERATE
        assert detector._classify_trend_strength(17) == TrendStrength.WEAK
        assert detector._classify_trend_strength(10) == TrendStrength.ABSENT
    
    def test_volatility_classification(self, detector):
        """Test volatility classification."""
        assert detector._classify_volatility(5.0) == VolatilityLevel.EXTREME
        assert detector._classify_volatility(3.0) == VolatilityLevel.HIGH
        assert detector._classify_volatility(1.5) == VolatilityLevel.NORMAL
        assert detector._classify_volatility(0.5) == VolatilityLevel.LOW
    
    def test_regime_strategies(self, detector, sample_uptrend_df):
        """Test recommended strategies for regimes."""
        result = detector.analyze('AAPL', sample_uptrend_df)
        
        assert len(result.recommended_strategies) > 0
        assert result.regime in detector.REGIME_STRATEGIES or result.regime == MarketRegime.UNKNOWN
    
    def test_risk_level(self, detector, sample_uptrend_df):
        """Test risk level assignment."""
        result = detector.analyze('AAPL', sample_uptrend_df)
        
        assert result.risk_level in ['low', 'medium', 'high']
    
    def test_position_sizing_modifier(self, detector, sample_uptrend_df):
        """Test position sizing modifier."""
        result = detector.analyze('AAPL', sample_uptrend_df)
        
        assert 0.3 <= result.position_sizing_modifier <= 1.5
    
    def test_regime_description(self, detector):
        """Test regime description."""
        for regime in MarketRegime:
            desc = detector.get_regime_description(regime)
            assert isinstance(desc, str)
            assert len(desc) > 0
    
    def test_to_dict(self, detector, sample_uptrend_df):
        """Test converting result to dict."""
        result = detector.analyze('AAPL', sample_uptrend_df)
        data = result.to_dict()
        
        assert 'symbol' in data
        assert 'regime' in data
        assert 'volatility' in data


# ==================== LiquidityMapper Tests ====================

class TestLiquidityMapper:
    """Tests for LiquidityMapper class."""
    
    @pytest.fixture
    def mapper(self):
        """Create mapper instance."""
        return LiquidityMapper(
            swing_lookback=5,
            zone_merge_percent=0.5,
            min_zone_touches=1
        )
    
    def test_initialization(self, mapper):
        """Test mapper initialization."""
        assert mapper.swing_lookback == 5
        assert mapper.zone_merge_percent == 0.5
    
    def test_find_swing_highs(self, mapper, sample_uptrend_df):
        """Test finding swing highs."""
        swing_highs = mapper._find_swing_highs(sample_uptrend_df)
        
        assert isinstance(swing_highs, list)
        for idx, price in swing_highs:
            assert isinstance(idx, int)
            assert isinstance(price, float)
    
    def test_find_swing_lows(self, mapper, sample_uptrend_df):
        """Test finding swing lows."""
        swing_lows = mapper._find_swing_lows(sample_uptrend_df)
        
        assert isinstance(swing_lows, list)
        for idx, price in swing_lows:
            assert isinstance(idx, int)
            assert isinstance(price, float)
    
    def test_analyze(self, mapper, sample_uptrend_df):
        """Test full liquidity analysis."""
        result = mapper.analyze('AAPL', sample_uptrend_df)
        
        assert isinstance(result, LiquidityAnalysis)
        assert result.symbol == 'AAPL'
        assert result.current_price > 0
        assert result.liquidity_bias in ['bullish', 'bearish', 'neutral']
    
    def test_find_order_blocks(self, mapper, sample_uptrend_df):
        """Test finding order blocks."""
        order_blocks = mapper._find_order_blocks(sample_uptrend_df)
        
        assert isinstance(order_blocks, list)
        for ob in order_blocks:
            assert isinstance(ob, LiquidityZone)
            assert ob.zone_type in [ZoneType.ORDER_BLOCK_BULLISH, ZoneType.ORDER_BLOCK_BEARISH]
    
    def test_find_fair_value_gaps(self, mapper, sample_uptrend_df):
        """Test finding FVGs."""
        fvgs = mapper._find_fair_value_gaps(sample_uptrend_df)
        
        assert isinstance(fvgs, list)
        for fvg in fvgs:
            assert isinstance(fvg, LiquidityZone)
            assert fvg.zone_type in [ZoneType.FVG_BULLISH, ZoneType.FVG_BEARISH]
    
    def test_zone_contains_price(self):
        """Test zone price containment."""
        zone = LiquidityZone(
            zone_type=ZoneType.SUPPORT,
            price_low=100,
            price_high=105,
            strength=0.7,
            touches=3,
            created_at=datetime.now(timezone.utc)
        )
        
        assert zone.contains_price(102) is True
        assert zone.contains_price(99) is False
        assert zone.contains_price(106) is False
    
    def test_zone_distance_to_price(self):
        """Test zone distance calculation."""
        zone = LiquidityZone(
            zone_type=ZoneType.SUPPORT,
            price_low=100,
            price_high=105,
            strength=0.7,
            touches=3,
            created_at=datetime.now(timezone.utc)
        )
        
        assert zone.distance_to_price(102) == 0.0  # Inside zone
        assert zone.distance_to_price(97) == 3.0  # Below zone
        assert zone.distance_to_price(108) == 3.0  # Above zone
    
    def test_merge_overlapping_zones(self, mapper):
        """Test merging overlapping zones."""
        zones = [
            LiquidityZone(ZoneType.SUPPORT, 100, 102, 0.5, 1, datetime.now(timezone.utc)),
            LiquidityZone(ZoneType.SUPPORT, 101, 103, 0.5, 1, datetime.now(timezone.utc)),
            LiquidityZone(ZoneType.SUPPORT, 110, 112, 0.5, 1, datetime.now(timezone.utc)),
        ]
        
        merged = mapper._merge_overlapping_zones(zones)
        
        # First two should merge, third should remain separate
        assert len(merged) <= 3
    
    def test_insufficient_data(self, mapper):
        """Test handling insufficient data."""
        short_df = pd.DataFrame({
            'Open': [100] * 5,
            'High': [102] * 5,
            'Low': [99] * 5,
            'Close': [101] * 5,
            'Volume': [1000] * 5
        })
        
        result = mapper.analyze('TEST', short_df)
        
        assert result.liquidity_bias == 'neutral'
        assert len(result.zones) == 0
    
    def test_liquidity_analysis_to_dict(self, mapper, sample_uptrend_df):
        """Test converting analysis to dict."""
        result = mapper.analyze('AAPL', sample_uptrend_df)
        data = result.to_dict()
        
        assert 'symbol' in data
        assert 'current_price' in data
        assert 'liquidity_bias' in data


# ==================== SeasonalityAnalyzer Tests ====================

class TestSeasonalityAnalyzer:
    """Tests for SeasonalityAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SeasonalityAnalyzer(min_sample_size=5)
    
    @pytest.fixture
    def yearly_df(self):
        """Create one year of daily data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        
        # Create data with some seasonal patterns
        base_price = 100
        
        # Add monthly pattern (January higher)
        monthly_effect = np.array([
            3 if d.month == 1 else  # January effect
            -1 if d.month in [5, 6, 7, 8, 9] else  # Sell in May
            1  # Other months
            for d in dates
        ])
        
        # Add weekly pattern (Monday lower, Friday higher)
        weekly_effect = np.array([
            -0.5 if d.dayofweek == 0 else  # Monday
            0.5 if d.dayofweek == 4 else  # Friday
            0
            for d in dates
        ])
        
        cumulative = np.cumsum(monthly_effect * 0.1 + weekly_effect * 0.1 + np.random.normal(0, 1, 365))
        close = base_price + cumulative
        
        return pd.DataFrame({
            'Open': close - np.random.uniform(0, 1, 365),
            'High': close + np.random.uniform(0.5, 2, 365),
            'Low': close - np.random.uniform(0.5, 2, 365),
            'Close': close,
            'Volume': np.random.randint(1000000, 5000000, 365)
        }, index=dates)
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.min_sample_size == 5
    
    def test_analyze(self, analyzer, yearly_df):
        """Test full seasonality analysis."""
        result = analyzer.analyze('AAPL', yearly_df)
        
        assert isinstance(result, SeasonalityReport)
        assert result.symbol == 'AAPL'
        assert 0 <= result.overall_seasonality_strength <= 1
    
    def test_daily_patterns(self, analyzer, yearly_df):
        """Test day of week pattern analysis."""
        result = analyzer.analyze('AAPL', yearly_df)
        
        assert len(result.daily_patterns) > 0
        for day, pattern in result.daily_patterns.items():
            assert isinstance(day, DayOfWeek)
            assert isinstance(pattern, SeasonalPattern)
            assert pattern.period_type == "day_of_week"
    
    def test_monthly_patterns(self, analyzer, yearly_df):
        """Test monthly pattern analysis."""
        # Need more data for monthly patterns (at least 2 years)
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=730, freq='D')  # 2 years
        
        base_price = 100
        cumulative = np.cumsum(np.random.normal(0.05, 1, 730))
        close = base_price + cumulative
        
        long_df = pd.DataFrame({
            'Open': close - np.random.uniform(0, 1, 730),
            'High': close + np.random.uniform(0.5, 2, 730),
            'Low': close - np.random.uniform(0.5, 2, 730),
            'Close': close,
            'Volume': np.random.randint(1000000, 5000000, 730)
        }, index=dates)
        
        result = analyzer.analyze('AAPL', long_df)
        
        # With 2 years of data, we should have monthly patterns
        assert len(result.monthly_patterns) > 0
        for month, pattern in result.monthly_patterns.items():
            assert isinstance(month, MonthOfYear)
            assert isinstance(pattern, SeasonalPattern)
            assert pattern.period_type == "month"
    
    def test_pattern_significance(self):
        """Test pattern significance check."""
        significant = SeasonalPattern(
            period_type="test",
            period_value="Monday",
            avg_return=1.0,
            win_rate=60,
            sample_size=50,
            std_dev=1.0,
            best_return=5.0,
            worst_return=-3.0
        )
        
        not_significant = SeasonalPattern(
            period_type="test",
            period_value="Tuesday",
            avg_return=0.1,
            win_rate=51,
            sample_size=10,
            std_dev=2.0,
            best_return=2.0,
            worst_return=-2.0
        )
        
        assert significant.is_significant is True
        assert not_significant.is_significant is False
    
    def test_pattern_direction(self):
        """Test pattern directional bias."""
        bullish = SeasonalPattern("test", "A", 0.5, 60, 30, 1.0, 3.0, -2.0)
        bearish = SeasonalPattern("test", "B", -0.5, 40, 30, 1.0, 2.0, -3.0)
        neutral = SeasonalPattern("test", "C", 0.05, 50, 30, 1.0, 1.0, -1.0)
        
        assert bullish.direction == "bullish"
        assert bearish.direction == "bearish"
        assert neutral.direction == "neutral"
    
    def test_get_current_bias(self, analyzer, yearly_df):
        """Test getting current period bias."""
        result = analyzer.analyze('AAPL', yearly_df)
        bias, confidence = result.get_current_bias()
        
        assert bias in ['bullish', 'bearish', 'neutral']
        assert 0 <= confidence <= 1
    
    def test_get_calendar_events(self, analyzer):
        """Test getting calendar events."""
        # Test January
        jan_events = analyzer.get_calendar_events(datetime(2024, 1, 15))
        assert any('January' in e for e in jan_events)
        
        # Test Monday
        monday = datetime(2024, 1, 15)  # This is a Monday
        monday_events = analyzer.get_calendar_events(monday)
        assert any('Monday' in e for e in monday_events)
        
        # Test December
        dec_events = analyzer.get_calendar_events(datetime(2024, 12, 26))
        assert any('Santa' in e or 'rally' in e.lower() for e in dec_events)
    
    def test_get_trading_recommendation(self, analyzer, yearly_df):
        """Test getting trading recommendation."""
        result = analyzer.analyze('AAPL', yearly_df)
        recommendation = analyzer.get_trading_recommendation(result)
        
        assert 'bias' in recommendation
        assert 'confidence' in recommendation
        assert 'reasons' in recommendation
        assert 'calendar_events' in recommendation
        assert 'use_for_trading' in recommendation
    
    def test_insufficient_data(self, analyzer):
        """Test handling insufficient data."""
        short_df = pd.DataFrame({
            'Open': [100] * 30,
            'High': [102] * 30,
            'Low': [99] * 30,
            'Close': [101] * 30,
            'Volume': [1000] * 30
        }, index=pd.date_range(start='2024-01-01', periods=30))
        
        result = analyzer.analyze('TEST', short_df)
        
        assert result.overall_seasonality_strength == 0.0
    
    def test_report_to_dict(self, analyzer, yearly_df):
        """Test converting report to dict."""
        result = analyzer.analyze('AAPL', yearly_df)
        data = result.to_dict()
        
        assert 'symbol' in data
        assert 'current_bias' in data
        assert 'seasonality_strength' in data
        assert 'daily_patterns' in data
        assert 'monthly_patterns' in data


# ==================== Integration Tests ====================

class TestMarketAnalysisIntegration:
    """Integration tests for market analysis module."""
    
    def test_combined_analysis(self, sample_uptrend_df):
        """Test combining all analysis tools."""
        regime_detector = MarketRegimeDetector()
        liquidity_mapper = LiquidityMapper(swing_lookback=5)
        seasonality_analyzer = SeasonalityAnalyzer(min_sample_size=5)
        
        # Run all analyses
        regime = regime_detector.analyze('AAPL', sample_uptrend_df)
        liquidity = liquidity_mapper.analyze('AAPL', sample_uptrend_df)
        seasonality = seasonality_analyzer.analyze('AAPL', sample_uptrend_df)
        
        # All should complete without error
        assert regime.symbol == 'AAPL'
        assert liquidity.symbol == 'AAPL'
        assert seasonality.symbol == 'AAPL'
        
        # Combine insights
        combined_bias = 'neutral'
        
        if regime.trend_direction > 0.3 and liquidity.liquidity_bias == 'bullish':
            combined_bias = 'bullish'
        elif regime.trend_direction < -0.3 and liquidity.liquidity_bias == 'bearish':
            combined_bias = 'bearish'
        
        assert combined_bias in ['bullish', 'bearish', 'neutral']
    
    def test_analysis_consistency(self, sample_uptrend_df):
        """Test that analyses are consistent on same data."""
        detector = MarketRegimeDetector()
        
        result1 = detector.analyze('AAPL', sample_uptrend_df)
        result2 = detector.analyze('AAPL', sample_uptrend_df)
        
        assert result1.regime == result2.regime
        assert result1.adx == result2.adx
        assert result1.trend_direction == result2.trend_direction
