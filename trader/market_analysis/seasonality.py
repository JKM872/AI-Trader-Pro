"""
Seasonality Analyzer - Identifies time-based patterns in price action.

Features:
- Time of day analysis
- Day of week patterns
- Monthly seasonality
- Pre/post market behavior
- Earnings season patterns
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, time
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TimeOfDay(Enum):
    """Trading session time periods."""
    PRE_MARKET = "pre_market"       # 4:00 - 9:30 ET
    MARKET_OPEN = "market_open"     # 9:30 - 10:30 ET
    MORNING = "morning"             # 10:30 - 12:00 ET
    LUNCH = "lunch"                 # 12:00 - 14:00 ET
    AFTERNOON = "afternoon"         # 14:00 - 15:30 ET
    MARKET_CLOSE = "market_close"   # 15:30 - 16:00 ET
    AFTER_HOURS = "after_hours"     # 16:00 - 20:00 ET


class DayOfWeek(Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4


class MonthOfYear(Enum):
    """Months of the year."""
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12


@dataclass
class SeasonalPattern:
    """A seasonal pattern observation."""
    period_type: str  # "day_of_week", "month", "time_of_day"
    period_value: str  # e.g., "Monday", "January", "market_open"
    avg_return: float
    win_rate: float
    sample_size: int
    std_dev: float
    best_return: float
    worst_return: float
    
    @property
    def sharpe_like(self) -> float:
        """Simple Sharpe-like metric."""
        if self.std_dev == 0:
            return 0.0
        return self.avg_return / self.std_dev
    
    @property
    def is_significant(self) -> bool:
        """Check if pattern is statistically significant."""
        return self.sample_size >= 20 and abs(self.sharpe_like) > 0.5
    
    @property
    def direction(self) -> str:
        """Get directional bias."""
        if self.avg_return > 0.1:
            return "bullish"
        elif self.avg_return < -0.1:
            return "bearish"
        else:
            return "neutral"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'period_type': self.period_type,
            'period_value': self.period_value,
            'avg_return': self.avg_return,
            'win_rate': self.win_rate,
            'sample_size': self.sample_size,
            'std_dev': self.std_dev,
            'sharpe_like': self.sharpe_like,
            'is_significant': self.is_significant,
            'direction': self.direction
        }


@dataclass
class SeasonalityReport:
    """Complete seasonality analysis report."""
    symbol: str
    daily_patterns: dict[DayOfWeek, SeasonalPattern]
    monthly_patterns: dict[MonthOfYear, SeasonalPattern]
    current_day_bias: Optional[SeasonalPattern]
    current_month_bias: Optional[SeasonalPattern]
    strongest_bullish_period: Optional[SeasonalPattern]
    strongest_bearish_period: Optional[SeasonalPattern]
    overall_seasonality_strength: float  # 0-1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_current_bias(self) -> tuple[str, float]:
        """Get current period's bias and confidence."""
        biases = []
        
        if self.current_day_bias and self.current_day_bias.is_significant:
            biases.append((self.current_day_bias.direction, 
                          abs(self.current_day_bias.sharpe_like)))
        
        if self.current_month_bias and self.current_month_bias.is_significant:
            biases.append((self.current_month_bias.direction,
                          abs(self.current_month_bias.sharpe_like)))
        
        if not biases:
            return "neutral", 0.0
        
        # Weighted average of biases
        bullish_score = sum(conf for dir, conf in biases if dir == "bullish")
        bearish_score = sum(conf for dir, conf in biases if dir == "bearish")
        
        if bullish_score > bearish_score:
            return "bullish", bullish_score / (bullish_score + bearish_score + 1e-10)
        elif bearish_score > bullish_score:
            return "bearish", bearish_score / (bullish_score + bearish_score + 1e-10)
        else:
            return "neutral", 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'current_bias': self.get_current_bias(),
            'seasonality_strength': self.overall_seasonality_strength,
            'strongest_bullish': self.strongest_bullish_period.to_dict() if self.strongest_bullish_period else None,
            'strongest_bearish': self.strongest_bearish_period.to_dict() if self.strongest_bearish_period else None,
            'daily_patterns': {k.name: v.to_dict() for k, v in self.daily_patterns.items()},
            'monthly_patterns': {k.name: v.to_dict() for k, v in self.monthly_patterns.items()}
        }


class SeasonalityAnalyzer:
    """
    Analyzes seasonal patterns in price action.
    
    Examines:
    - Day of week effects
    - Monthly patterns
    - Turn of month effects
    - Holiday effects
    - Options expiration patterns
    """
    
    # Known seasonal tendencies
    KNOWN_PATTERNS = {
        'january_effect': "Small caps tend to outperform in January",
        'sell_in_may': "Market tends to underperform May-October",
        'santa_rally': "Market tends to rally last week of December",
        'monday_effect': "Mondays historically show weaker returns",
        'friday_strength': "Fridays often see positive closes",
        'quad_witching': "Increased volatility on quarterly options expiration"
    }
    
    def __init__(self, min_sample_size: int = 10):
        """
        Initialize Seasonality Analyzer.
        
        Args:
            min_sample_size: Minimum samples for pattern validity
        """
        self.min_sample_size = min_sample_size
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate daily returns."""
        return df['Close'].pct_change() * 100
    
    def _analyze_day_of_week(self, df: pd.DataFrame) -> dict[DayOfWeek, SeasonalPattern]:
        """Analyze day of week patterns."""
        patterns = {}
        returns = self._calculate_returns(df)
        
        # Add day of week column
        if isinstance(df.index, pd.DatetimeIndex):
            df_copy = df.copy()
            df_copy['day_of_week'] = df.index.dayofweek
        else:
            df_copy = df.copy()
            df_copy['day_of_week'] = pd.to_datetime(df_copy.index).dayofweek
        
        df_copy['returns'] = returns
        
        for day in DayOfWeek:
            day_returns = df_copy[df_copy['day_of_week'] == day.value]['returns'].dropna()
            
            if len(day_returns) >= self.min_sample_size:
                patterns[day] = SeasonalPattern(
                    period_type="day_of_week",
                    period_value=day.name,
                    avg_return=day_returns.mean(),
                    win_rate=(day_returns > 0).mean() * 100,
                    sample_size=len(day_returns),
                    std_dev=day_returns.std(),
                    best_return=day_returns.max(),
                    worst_return=day_returns.min()
                )
        
        return patterns
    
    def _analyze_month(self, df: pd.DataFrame) -> dict[MonthOfYear, SeasonalPattern]:
        """Analyze monthly patterns."""
        patterns = {}
        
        # Calculate monthly returns
        if isinstance(df.index, pd.DatetimeIndex):
            monthly = df['Close'].resample('ME').last()
        else:
            df_copy = df.copy()
            df_copy.index = pd.to_datetime(df_copy.index)
            monthly = df_copy['Close'].resample('ME').last()
        
        monthly_returns = monthly.pct_change() * 100
        
        for month in MonthOfYear:
            month_rets = monthly_returns[monthly_returns.index.month == month.value].dropna()
            
            if len(month_rets) >= max(2, self.min_sample_size // 12):
                patterns[month] = SeasonalPattern(
                    period_type="month",
                    period_value=month.name,
                    avg_return=month_rets.mean(),
                    win_rate=(month_rets > 0).mean() * 100,
                    sample_size=len(month_rets),
                    std_dev=month_rets.std() if len(month_rets) > 1 else 0,
                    best_return=month_rets.max(),
                    worst_return=month_rets.min()
                )
        
        return patterns
    
    def _get_current_patterns(
        self,
        daily_patterns: dict[DayOfWeek, SeasonalPattern],
        monthly_patterns: dict[MonthOfYear, SeasonalPattern]
    ) -> tuple[Optional[SeasonalPattern], Optional[SeasonalPattern]]:
        """Get patterns for current day and month."""
        now = datetime.now()
        current_day = DayOfWeek(now.weekday()) if now.weekday() < 5 else None
        current_month = MonthOfYear(now.month)
        
        current_day_pattern = daily_patterns.get(current_day) if current_day else None
        current_month_pattern = monthly_patterns.get(current_month)
        
        return current_day_pattern, current_month_pattern
    
    def _find_strongest_patterns(
        self,
        daily_patterns: dict[DayOfWeek, SeasonalPattern],
        monthly_patterns: dict[MonthOfYear, SeasonalPattern]
    ) -> tuple[Optional[SeasonalPattern], Optional[SeasonalPattern]]:
        """Find strongest bullish and bearish patterns."""
        all_patterns = list(daily_patterns.values()) + list(monthly_patterns.values())
        
        significant_patterns = [p for p in all_patterns if p.is_significant]
        
        if not significant_patterns:
            return None, None
        
        bullish = [p for p in significant_patterns if p.direction == "bullish"]
        bearish = [p for p in significant_patterns if p.direction == "bearish"]
        
        strongest_bullish = max(bullish, key=lambda p: p.sharpe_like) if bullish else None
        strongest_bearish = min(bearish, key=lambda p: p.sharpe_like) if bearish else None
        
        return strongest_bullish, strongest_bearish
    
    def _calculate_seasonality_strength(
        self,
        daily_patterns: dict[DayOfWeek, SeasonalPattern],
        monthly_patterns: dict[MonthOfYear, SeasonalPattern]
    ) -> float:
        """Calculate overall seasonality strength."""
        all_patterns = list(daily_patterns.values()) + list(monthly_patterns.values())
        
        if not all_patterns:
            return 0.0
        
        # Average absolute Sharpe-like ratio of significant patterns
        significant = [p for p in all_patterns if p.is_significant]
        
        if not significant:
            return 0.0
        
        avg_strength = np.mean([abs(p.sharpe_like) for p in significant])
        
        # Normalize to 0-1
        return min(1.0, avg_strength / 2.0)
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> SeasonalityReport:
        """
        Analyze seasonality patterns for a symbol.
        
        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame with datetime index (preferably years of data)
            
        Returns:
            SeasonalityReport with all patterns
        """
        if len(df) < 60:  # Need at least ~3 months of data
            logger.warning(f"Insufficient data for seasonality analysis: {len(df)} bars")
            return SeasonalityReport(
                symbol=symbol,
                daily_patterns={},
                monthly_patterns={},
                current_day_bias=None,
                current_month_bias=None,
                strongest_bullish_period=None,
                strongest_bearish_period=None,
                overall_seasonality_strength=0.0
            )
        
        # Analyze patterns
        daily_patterns = self._analyze_day_of_week(df)
        monthly_patterns = self._analyze_month(df)
        
        # Get current patterns
        current_day, current_month = self._get_current_patterns(
            daily_patterns, monthly_patterns
        )
        
        # Find strongest patterns
        strongest_bull, strongest_bear = self._find_strongest_patterns(
            daily_patterns, monthly_patterns
        )
        
        # Calculate overall strength
        seasonality_strength = self._calculate_seasonality_strength(
            daily_patterns, monthly_patterns
        )
        
        return SeasonalityReport(
            symbol=symbol,
            daily_patterns=daily_patterns,
            monthly_patterns=monthly_patterns,
            current_day_bias=current_day,
            current_month_bias=current_month,
            strongest_bullish_period=strongest_bull,
            strongest_bearish_period=strongest_bear,
            overall_seasonality_strength=seasonality_strength
        )
    
    def get_calendar_events(self, date: Optional[datetime] = None) -> list[str]:
        """
        Get notable calendar events that may affect trading.
        
        Args:
            date: Date to check (default: today)
            
        Returns:
            List of relevant events/patterns
        """
        if date is None:
            date = datetime.now()
        
        events = []
        
        # Check for known patterns
        month = date.month
        day = date.day
        weekday = date.weekday()
        
        # January effect
        if month == 1:
            events.append(self.KNOWN_PATTERNS['january_effect'])
        
        # Sell in May
        if month == 5:
            events.append(self.KNOWN_PATTERNS['sell_in_may'])
        
        # Santa Rally
        if month == 12 and day >= 24:
            events.append(self.KNOWN_PATTERNS['santa_rally'])
        
        # Monday effect
        if weekday == 0:
            events.append(self.KNOWN_PATTERNS['monday_effect'])
        
        # Friday strength
        if weekday == 4:
            events.append(self.KNOWN_PATTERNS['friday_strength'])
        
        # Quad witching (3rd Friday of March, June, September, December)
        if month in [3, 6, 9, 12]:
            # Find third Friday
            first_day = datetime(date.year, month, 1)
            first_friday = (4 - first_day.weekday()) % 7 + 1
            third_friday = first_friday + 14
            
            if day == third_friday:
                events.append(self.KNOWN_PATTERNS['quad_witching'])
        
        # Turn of month effect (last 3 and first 3 days)
        if day <= 3 or day >= 28:
            events.append("Turn of month effect: historically strong period")
        
        return events
    
    def get_trading_recommendation(
        self,
        report: SeasonalityReport
    ) -> dict:
        """
        Get trading recommendation based on seasonality.
        
        Args:
            report: SeasonalityReport to analyze
            
        Returns:
            Dict with bias, confidence, and reasoning
        """
        bias, confidence = report.get_current_bias()
        
        reasons = []
        
        if report.current_day_bias and report.current_day_bias.is_significant:
            reasons.append(
                f"{report.current_day_bias.period_value}: "
                f"{report.current_day_bias.avg_return:.2f}% avg return, "
                f"{report.current_day_bias.win_rate:.0f}% win rate"
            )
        
        if report.current_month_bias and report.current_month_bias.is_significant:
            reasons.append(
                f"{report.current_month_bias.period_value}: "
                f"{report.current_month_bias.avg_return:.2f}% avg return, "
                f"{report.current_month_bias.win_rate:.0f}% win rate"
            )
        
        # Add calendar events
        events = self.get_calendar_events()
        
        return {
            'bias': bias,
            'confidence': confidence,
            'strength': report.overall_seasonality_strength,
            'reasons': reasons,
            'calendar_events': events,
            'use_for_trading': confidence > 0.3 and report.overall_seasonality_strength > 0.3
        }
