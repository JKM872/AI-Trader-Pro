"""
Economic Indicators - Track and analyze macroeconomic data.

Monitors:
- GDP, CPI, Unemployment
- Interest rates
- Manufacturing data
- Consumer sentiment
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of economic indicators."""
    # Growth
    GDP = "gdp"
    GDP_GROWTH = "gdp_growth"
    
    # Inflation
    CPI = "cpi"
    PPI = "ppi"
    PCE = "pce"
    
    # Employment
    UNEMPLOYMENT = "unemployment"
    NFP = "nonfarm_payrolls"
    JOBLESS_CLAIMS = "jobless_claims"
    
    # Interest Rates
    FED_FUNDS = "fed_funds_rate"
    TREASURY_10Y = "treasury_10y"
    TREASURY_2Y = "treasury_2y"
    
    # Manufacturing
    ISM_MANUFACTURING = "ism_manufacturing"
    ISM_SERVICES = "ism_services"
    INDUSTRIAL_PRODUCTION = "industrial_production"
    
    # Consumer
    CONSUMER_CONFIDENCE = "consumer_confidence"
    RETAIL_SALES = "retail_sales"
    HOUSING_STARTS = "housing_starts"
    
    # Other
    TRADE_BALANCE = "trade_balance"
    DURABLE_GOODS = "durable_goods"


class ImpactLevel(Enum):
    """Impact level of economic releases."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EconomicRelease:
    """Individual economic data release."""
    
    indicator: IndicatorType
    name: str
    
    # Values
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    
    # Timing
    release_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Impact
    impact: ImpactLevel = ImpactLevel.MEDIUM
    
    # Unit
    unit: str = ""
    
    @property
    def surprise(self) -> Optional[float]:
        """Calculate surprise (actual vs forecast)."""
        if self.actual is not None and self.forecast is not None:
            return self.actual - self.forecast
        return None
    
    @property
    def surprise_pct(self) -> Optional[float]:
        """Calculate surprise percentage."""
        if self.actual is not None and self.forecast is not None and self.forecast != 0:
            return ((self.actual - self.forecast) / abs(self.forecast)) * 100
        return None
    
    @property
    def change_from_previous(self) -> Optional[float]:
        """Calculate change from previous."""
        if self.actual is not None and self.previous is not None:
            return self.actual - self.previous
        return None
    
    @property
    def beat_expectations(self) -> Optional[bool]:
        """Check if actual beat forecast."""
        surprise = self.surprise
        if surprise is None:
            return None
        
        # For unemployment, lower is better
        if self.indicator in [IndicatorType.UNEMPLOYMENT, IndicatorType.JOBLESS_CLAIMS]:
            return surprise < 0
        
        return surprise > 0


@dataclass
class EconomicEvent:
    """Scheduled economic event."""
    
    indicator: IndicatorType
    name: str
    scheduled_time: datetime
    
    # Expected values
    forecast: Optional[float] = None
    previous: Optional[float] = None
    
    # Impact
    impact: ImpactLevel = ImpactLevel.MEDIUM
    
    # Country
    country: str = "US"
    
    @property
    def is_upcoming(self) -> bool:
        """Check if event is upcoming."""
        return self.scheduled_time > datetime.now(timezone.utc)
    
    @property
    def time_until(self) -> timedelta:
        """Time until event."""
        return self.scheduled_time - datetime.now(timezone.utc)


@dataclass
class MarketImpact:
    """Expected market impact from economic release."""
    
    indicator: IndicatorType
    release: EconomicRelease
    
    # Impact scores (-1 to 1)
    equity_impact: float = 0.0
    bond_impact: float = 0.0
    dollar_impact: float = 0.0
    
    # Sector impacts
    sector_impacts: Dict[str, float] = field(default_factory=dict)
    
    # Summary
    summary: str = ""
    
    # Trading implications
    bullish_for: List[str] = field(default_factory=list)
    bearish_for: List[str] = field(default_factory=list)


class EconomicIndicators:
    """
    Tracks and analyzes economic indicators.
    
    Features:
    - Real-time indicator tracking
    - Surprise analysis
    - Market impact estimation
    - Economic calendar
    """
    
    # Indicator metadata
    INDICATOR_INFO = {
        IndicatorType.GDP: {
            'name': 'Gross Domestic Product',
            'impact': ImpactLevel.HIGH,
            'frequency': 'quarterly',
            'higher_is_better': True,
        },
        IndicatorType.CPI: {
            'name': 'Consumer Price Index',
            'impact': ImpactLevel.CRITICAL,
            'frequency': 'monthly',
            'higher_is_better': False,  # Higher inflation is bad for stocks
        },
        IndicatorType.UNEMPLOYMENT: {
            'name': 'Unemployment Rate',
            'impact': ImpactLevel.HIGH,
            'frequency': 'monthly',
            'higher_is_better': False,
        },
        IndicatorType.NFP: {
            'name': 'Nonfarm Payrolls',
            'impact': ImpactLevel.CRITICAL,
            'frequency': 'monthly',
            'higher_is_better': True,
        },
        IndicatorType.FED_FUNDS: {
            'name': 'Federal Funds Rate',
            'impact': ImpactLevel.CRITICAL,
            'frequency': 'as_needed',
            'higher_is_better': False,  # Higher rates are bearish
        },
        IndicatorType.ISM_MANUFACTURING: {
            'name': 'ISM Manufacturing PMI',
            'impact': ImpactLevel.HIGH,
            'frequency': 'monthly',
            'higher_is_better': True,
            'expansion_threshold': 50,
        },
        IndicatorType.CONSUMER_CONFIDENCE: {
            'name': 'Consumer Confidence',
            'impact': ImpactLevel.MEDIUM,
            'frequency': 'monthly',
            'higher_is_better': True,
        },
    }
    
    # Sector sensitivity
    SECTOR_SENSITIVITY = {
        IndicatorType.FED_FUNDS: {
            'Financials': 0.8,
            'Real Estate': -0.7,
            'Utilities': -0.5,
            'Technology': -0.3,
        },
        IndicatorType.CPI: {
            'Consumer Staples': -0.3,
            'Consumer Discretionary': -0.5,
            'Financials': 0.3,
        },
        IndicatorType.HOUSING_STARTS: {
            'Real Estate': 0.8,
            'Materials': 0.6,
            'Industrials': 0.4,
        },
    }
    
    def __init__(self):
        """Initialize economic indicators tracker."""
        self.releases: Dict[IndicatorType, List[EconomicRelease]] = {}
        self.upcoming_events: List[EconomicEvent] = []
    
    def add_release(self, release: EconomicRelease):
        """Add an economic release."""
        indicator = release.indicator
        
        if indicator not in self.releases:
            self.releases[indicator] = []
        
        self.releases[indicator].append(release)
        
        # Keep sorted by date
        self.releases[indicator].sort(
            key=lambda r: r.release_date,
            reverse=True
        )
        
        # Keep last 50
        self.releases[indicator] = self.releases[indicator][:50]
        
        logger.info(
            f"Economic release: {release.name} = {release.actual} "
            f"(forecast: {release.forecast}, prev: {release.previous})"
        )
    
    def add_upcoming_event(self, event: EconomicEvent):
        """Add upcoming economic event."""
        self.upcoming_events.append(event)
        
        # Sort by time
        self.upcoming_events.sort(key=lambda e: e.scheduled_time)
        
        # Remove past events
        now = datetime.now(timezone.utc)
        self.upcoming_events = [
            e for e in self.upcoming_events
            if e.scheduled_time > now - timedelta(hours=1)
        ]
    
    def get_latest(self, indicator: IndicatorType) -> Optional[EconomicRelease]:
        """Get latest release for an indicator."""
        if indicator not in self.releases or not self.releases[indicator]:
            return None
        return self.releases[indicator][0]
    
    def get_history(
        self,
        indicator: IndicatorType,
        periods: int = 12
    ) -> List[EconomicRelease]:
        """Get historical releases for an indicator."""
        if indicator not in self.releases:
            return []
        return self.releases[indicator][:periods]
    
    def analyze_release(self, release: EconomicRelease) -> MarketImpact:
        """
        Analyze market impact of an economic release.
        
        Args:
            release: EconomicRelease to analyze
            
        Returns:
            MarketImpact analysis
        """
        indicator = release.indicator
        info = self.INDICATOR_INFO.get(indicator, {})
        
        surprise = release.surprise or 0
        surprise_pct = release.surprise_pct or 0
        higher_is_better = info.get('higher_is_better', True)
        
        # Base impact from surprise direction
        if higher_is_better:
            base_impact = 1 if surprise > 0 else -1 if surprise < 0 else 0
        else:
            base_impact = -1 if surprise > 0 else 1 if surprise < 0 else 0
        
        # Scale by surprise magnitude
        impact_scale = min(abs(surprise_pct) / 5, 1.0)  # 5% surprise = max impact
        base_impact *= impact_scale
        
        # Equity impact
        equity_impact = base_impact * 0.8
        
        # Bond impact (usually inverse to equity for growth data)
        if indicator in [IndicatorType.CPI, IndicatorType.FED_FUNDS]:
            bond_impact = -base_impact  # Higher inflation/rates = lower bond prices
        else:
            bond_impact = -base_impact * 0.5
        
        # Dollar impact
        if indicator in [IndicatorType.NFP, IndicatorType.GDP]:
            dollar_impact = base_impact * 0.6
        elif indicator in [IndicatorType.FED_FUNDS]:
            dollar_impact = base_impact  # Higher rates = stronger dollar
        else:
            dollar_impact = base_impact * 0.3
        
        # Sector impacts
        sector_impacts = {}
        if indicator in self.SECTOR_SENSITIVITY:
            for sector, sensitivity in self.SECTOR_SENSITIVITY[indicator].items():
                sector_impacts[sector] = base_impact * sensitivity
        
        # Generate summary
        beat = release.beat_expectations
        if beat is True:
            summary = f"{release.name} beat expectations"
        elif beat is False:
            summary = f"{release.name} missed expectations"
        else:
            summary = f"{release.name} in line with expectations"
        
        # Trading implications
        bullish_for = []
        bearish_for = []
        
        for sector, impact in sector_impacts.items():
            if impact > 0.3:
                bullish_for.append(sector)
            elif impact < -0.3:
                bearish_for.append(sector)
        
        return MarketImpact(
            indicator=indicator,
            release=release,
            equity_impact=equity_impact,
            bond_impact=bond_impact,
            dollar_impact=dollar_impact,
            sector_impacts=sector_impacts,
            summary=summary,
            bullish_for=bullish_for,
            bearish_for=bearish_for
        )
    
    def get_calendar(
        self,
        days: int = 7,
        impact_filter: Optional[ImpactLevel] = None
    ) -> List[EconomicEvent]:
        """
        Get economic calendar.
        
        Args:
            days: Days to look ahead
            impact_filter: Minimum impact level
            
        Returns:
            List of upcoming events
        """
        cutoff = datetime.now(timezone.utc) + timedelta(days=days)
        
        events = [
            e for e in self.upcoming_events
            if e.is_upcoming and e.scheduled_time <= cutoff
        ]
        
        if impact_filter:
            impact_order = [
                ImpactLevel.LOW,
                ImpactLevel.MEDIUM,
                ImpactLevel.HIGH,
                ImpactLevel.CRITICAL
            ]
            min_index = impact_order.index(impact_filter)
            events = [
                e for e in events
                if impact_order.index(e.impact) >= min_index
            ]
        
        return events
    
    def get_economic_outlook(self) -> Dict[str, Any]:
        """
        Get overall economic outlook based on indicators.
        
        Returns:
            Dict with economic outlook
        """
        outlook = {
            'growth': 'neutral',
            'inflation': 'neutral',
            'employment': 'neutral',
            'overall': 'neutral',
            'indicators': {}
        }
        
        # Check GDP
        gdp = self.get_latest(IndicatorType.GDP)
        if gdp and gdp.actual is not None:
            outlook['indicators']['gdp'] = gdp.actual
            if gdp.actual > 2.5:
                outlook['growth'] = 'strong'
            elif gdp.actual > 0:
                outlook['growth'] = 'moderate'
            else:
                outlook['growth'] = 'weak'
        
        # Check CPI
        cpi = self.get_latest(IndicatorType.CPI)
        if cpi and cpi.actual is not None:
            outlook['indicators']['cpi'] = cpi.actual
            if cpi.actual > 4:
                outlook['inflation'] = 'high'
            elif cpi.actual > 2:
                outlook['inflation'] = 'moderate'
            else:
                outlook['inflation'] = 'low'
        
        # Check unemployment
        unemployment = self.get_latest(IndicatorType.UNEMPLOYMENT)
        if unemployment and unemployment.actual is not None:
            outlook['indicators']['unemployment'] = unemployment.actual
            if unemployment.actual < 4:
                outlook['employment'] = 'strong'
            elif unemployment.actual < 6:
                outlook['employment'] = 'moderate'
            else:
                outlook['employment'] = 'weak'
        
        # Overall outlook
        growth_score = {'strong': 2, 'moderate': 1, 'neutral': 0, 'weak': -1}
        employment_score = {'strong': 1, 'moderate': 0.5, 'neutral': 0, 'weak': -1}
        inflation_penalty = {'high': -1, 'moderate': 0, 'low': 0.5, 'neutral': 0}
        
        total = (
            growth_score.get(outlook['growth'], 0) +
            employment_score.get(outlook['employment'], 0) +
            inflation_penalty.get(outlook['inflation'], 0)
        )
        
        if total >= 2:
            outlook['overall'] = 'bullish'
        elif total >= 0:
            outlook['overall'] = 'neutral'
        else:
            outlook['overall'] = 'bearish'
        
        return outlook
