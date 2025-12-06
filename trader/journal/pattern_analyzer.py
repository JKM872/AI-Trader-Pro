"""
Pattern Analyzer - Discover trading patterns and behaviors.

Analyzes historical trades to find repeating patterns and optimize strategies.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np

from .trade_journal import TradeJournal, TradeEntry, TradeStatus, TradeType


class PatternType(Enum):
    """Types of trading patterns."""
    TIME_OF_DAY = "time_of_day"
    DAY_OF_WEEK = "day_of_week"
    MARKET_REGIME = "market_regime"
    STRATEGY = "strategy"
    SYMBOL = "symbol"
    CONFIDENCE_LEVEL = "confidence_level"
    TRADE_DURATION = "trade_duration"
    CONSECUTIVE = "consecutive"
    SEASONALITY = "seasonality"


class PatternStrength(Enum):
    """Pattern reliability strength."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TradingPattern:
    """Identified trading pattern."""
    pattern_type: PatternType
    name: str
    description: str
    
    # Statistics
    occurrences: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Strength
    strength: PatternStrength = PatternStrength.WEAK
    confidence: float = 0.0
    
    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class PatternAnalyzer:
    """
    Analyzes trading patterns to find strengths and weaknesses.
    
    Features:
    - Time-based pattern analysis
    - Strategy performance comparison
    - Symbol affinity detection
    - Behavioral pattern recognition
    - Actionable recommendations
    """
    
    def __init__(self, journal: Optional[TradeJournal] = None):
        """Initialize pattern analyzer."""
        self.journal = journal
        self.patterns: List[TradingPattern] = []
    
    def set_journal(self, journal: TradeJournal) -> None:
        """Set or update the journal."""
        self.journal = journal
    
    def _get_pattern_strength(
        self,
        win_rate: float,
        occurrences: int,
        min_trades: int = 10
    ) -> Tuple[PatternStrength, float]:
        """Determine pattern strength based on stats."""
        # Need minimum trades for reliability
        if occurrences < min_trades:
            return PatternStrength.VERY_WEAK, 0.2
        
        # Calculate confidence based on sample size
        sample_confidence = min(1.0, occurrences / 50)
        
        # Win rate scoring
        if win_rate >= 70:
            strength = PatternStrength.VERY_STRONG
            base_confidence = 0.9
        elif win_rate >= 60:
            strength = PatternStrength.STRONG
            base_confidence = 0.75
        elif win_rate >= 50:
            strength = PatternStrength.MODERATE
            base_confidence = 0.6
        elif win_rate >= 40:
            strength = PatternStrength.WEAK
            base_confidence = 0.4
        else:
            strength = PatternStrength.VERY_WEAK
            base_confidence = 0.2
        
        confidence = base_confidence * sample_confidence
        
        return strength, confidence
    
    def analyze_time_of_day(self) -> List[TradingPattern]:
        """Analyze performance by time of day."""
        patterns = []
        
        if not self.journal:
            return patterns
        
        closed_trades = self.journal.get_closed_trades()
        trades_with_time = [t for t in closed_trades if t.entry_time]
        
        if not trades_with_time:
            return patterns
        
        # Group by hour
        hours: Dict[int, List[TradeEntry]] = {}
        for trade in trades_with_time:
            hour = trade.entry_time.hour
            if hour not in hours:
                hours[hour] = []
            hours[hour].append(trade)
        
        # Analyze each hour
        time_periods = {
            'Morning (6-10)': range(6, 10),
            'Mid-Day (10-14)': range(10, 14),
            'Afternoon (14-16)': range(14, 16),
            'Close (16-20)': range(16, 20),
        }
        
        for period_name, hour_range in time_periods.items():
            period_trades = []
            for hour in hour_range:
                period_trades.extend(hours.get(hour, []))
            
            if len(period_trades) < 5:
                continue
            
            winners = [t for t in period_trades if t.realized_pnl and t.realized_pnl > 0]
            total_pnl = sum(t.realized_pnl for t in period_trades if t.realized_pnl)
            
            win_rate = len(winners) / len(period_trades) * 100
            avg_pnl = total_pnl / len(period_trades)
            
            strength, confidence = self._get_pattern_strength(win_rate, len(period_trades))
            
            pattern = TradingPattern(
                pattern_type=PatternType.TIME_OF_DAY,
                name=f"Time Pattern: {period_name}",
                description=f"Trading performance during {period_name}",
                occurrences=len(period_trades),
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                total_pnl=total_pnl,
                strength=strength,
                confidence=confidence,
                details={
                    'period': period_name,
                    'hours': list(hour_range),
                    'winners': len(winners),
                    'losers': len(period_trades) - len(winners),
                }
            )
            
            # Add recommendations
            if win_rate >= 60:
                pattern.recommendations.append(f"Focus more trades during {period_name}")
            elif win_rate <= 40:
                pattern.recommendations.append(f"Consider avoiding trades during {period_name}")
            
            patterns.append(pattern)
        
        return patterns
    
    def analyze_day_of_week(self) -> List[TradingPattern]:
        """Analyze performance by day of week."""
        patterns = []
        
        if not self.journal:
            return patterns
        
        closed_trades = self.journal.get_closed_trades()
        trades_with_time = [t for t in closed_trades if t.entry_time]
        
        if not trades_with_time:
            return patterns
        
        # Group by day of week
        days = {
            0: ('Monday', []),
            1: ('Tuesday', []),
            2: ('Wednesday', []),
            3: ('Thursday', []),
            4: ('Friday', []),
        }
        
        for trade in trades_with_time:
            day = trade.entry_time.weekday()
            if day in days:
                days[day][1].append(trade)
        
        for day_num, (day_name, day_trades) in days.items():
            if len(day_trades) < 5:
                continue
            
            winners = [t for t in day_trades if t.realized_pnl and t.realized_pnl > 0]
            total_pnl = sum(t.realized_pnl for t in day_trades if t.realized_pnl)
            
            win_rate = len(winners) / len(day_trades) * 100
            avg_pnl = total_pnl / len(day_trades)
            
            strength, confidence = self._get_pattern_strength(win_rate, len(day_trades))
            
            pattern = TradingPattern(
                pattern_type=PatternType.DAY_OF_WEEK,
                name=f"Day Pattern: {day_name}",
                description=f"Trading performance on {day_name}s",
                occurrences=len(day_trades),
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                total_pnl=total_pnl,
                strength=strength,
                confidence=confidence,
                details={
                    'day': day_name,
                    'day_number': day_num,
                    'winners': len(winners),
                    'losers': len(day_trades) - len(winners),
                }
            )
            
            if win_rate >= 60:
                pattern.recommendations.append(f"{day_name} shows strong performance - prioritize trades")
            elif win_rate <= 40:
                pattern.recommendations.append(f"{day_name} shows weak performance - reduce exposure")
            
            patterns.append(pattern)
        
        return patterns
    
    def analyze_by_strategy(self) -> List[TradingPattern]:
        """Analyze performance by strategy."""
        patterns = []
        
        if not self.journal:
            return patterns
        
        closed_trades = self.journal.get_closed_trades()
        
        # Group by strategy
        strategies: Dict[str, List[TradeEntry]] = {}
        for trade in closed_trades:
            strategy = trade.strategy_name or "Unknown"
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(trade)
        
        for strategy_name, strategy_trades in strategies.items():
            if len(strategy_trades) < 5:
                continue
            
            winners = [t for t in strategy_trades if t.realized_pnl and t.realized_pnl > 0]
            total_pnl = sum(t.realized_pnl for t in strategy_trades if t.realized_pnl)
            
            win_rate = len(winners) / len(strategy_trades) * 100
            avg_pnl = total_pnl / len(strategy_trades)
            
            # Calculate average confidence
            confidences = [t.signal_confidence for t in strategy_trades if t.signal_confidence > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            strength, confidence = self._get_pattern_strength(win_rate, len(strategy_trades))
            
            pattern = TradingPattern(
                pattern_type=PatternType.STRATEGY,
                name=f"Strategy: {strategy_name}",
                description=f"Performance of {strategy_name} strategy",
                occurrences=len(strategy_trades),
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                total_pnl=total_pnl,
                strength=strength,
                confidence=confidence,
                details={
                    'strategy': strategy_name,
                    'avg_signal_confidence': avg_confidence,
                    'winners': len(winners),
                    'losers': len(strategy_trades) - len(winners),
                }
            )
            
            if win_rate >= 60:
                pattern.recommendations.append(f"Increase allocation to {strategy_name}")
            elif win_rate <= 40:
                pattern.recommendations.append(f"Review or reduce {strategy_name} usage")
            
            patterns.append(pattern)
        
        return patterns
    
    def analyze_by_symbol(self) -> List[TradingPattern]:
        """Analyze performance by symbol."""
        patterns = []
        
        if not self.journal:
            return patterns
        
        closed_trades = self.journal.get_closed_trades()
        
        # Group by symbol
        symbols: Dict[str, List[TradeEntry]] = {}
        for trade in closed_trades:
            symbol = trade.symbol
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(trade)
        
        for symbol, symbol_trades in symbols.items():
            if len(symbol_trades) < 3:
                continue
            
            winners = [t for t in symbol_trades if t.realized_pnl and t.realized_pnl > 0]
            total_pnl = sum(t.realized_pnl for t in symbol_trades if t.realized_pnl)
            
            win_rate = len(winners) / len(symbol_trades) * 100
            avg_pnl = total_pnl / len(symbol_trades)
            
            strength, confidence = self._get_pattern_strength(win_rate, len(symbol_trades), min_trades=5)
            
            pattern = TradingPattern(
                pattern_type=PatternType.SYMBOL,
                name=f"Symbol: {symbol}",
                description=f"Performance trading {symbol}",
                occurrences=len(symbol_trades),
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                total_pnl=total_pnl,
                strength=strength,
                confidence=confidence,
                details={
                    'symbol': symbol,
                    'winners': len(winners),
                    'losers': len(symbol_trades) - len(winners),
                }
            )
            
            if win_rate >= 65:
                pattern.recommendations.append(f"Strong edge on {symbol} - consider increasing position size")
            elif win_rate <= 35:
                pattern.recommendations.append(f"Weak edge on {symbol} - consider removing from watchlist")
            
            patterns.append(pattern)
        
        return patterns
    
    def analyze_by_market_regime(self) -> List[TradingPattern]:
        """Analyze performance by market regime."""
        patterns = []
        
        if not self.journal:
            return patterns
        
        closed_trades = self.journal.get_closed_trades()
        
        # Group by regime
        regimes: Dict[str, List[TradeEntry]] = {}
        for trade in closed_trades:
            regime = trade.market_regime or "Unknown"
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(trade)
        
        for regime_name, regime_trades in regimes.items():
            if len(regime_trades) < 5 or regime_name == "Unknown":
                continue
            
            winners = [t for t in regime_trades if t.realized_pnl and t.realized_pnl > 0]
            total_pnl = sum(t.realized_pnl for t in regime_trades if t.realized_pnl)
            
            win_rate = len(winners) / len(regime_trades) * 100
            avg_pnl = total_pnl / len(regime_trades)
            
            strength, confidence = self._get_pattern_strength(win_rate, len(regime_trades))
            
            pattern = TradingPattern(
                pattern_type=PatternType.MARKET_REGIME,
                name=f"Regime: {regime_name}",
                description=f"Performance in {regime_name} market conditions",
                occurrences=len(regime_trades),
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                total_pnl=total_pnl,
                strength=strength,
                confidence=confidence,
                details={
                    'regime': regime_name,
                    'winners': len(winners),
                    'losers': len(regime_trades) - len(winners),
                }
            )
            
            if win_rate >= 60:
                pattern.recommendations.append(f"Strong performance in {regime_name} - increase activity")
            elif win_rate <= 40:
                pattern.recommendations.append(f"Weak performance in {regime_name} - reduce trading or adjust strategy")
            
            patterns.append(pattern)
        
        return patterns
    
    def analyze_confidence_correlation(self) -> TradingPattern:
        """Analyze correlation between signal confidence and success."""
        if not self.journal:
            return TradingPattern(
                pattern_type=PatternType.CONFIDENCE_LEVEL,
                name="Confidence Analysis",
                description="No data available"
            )
        
        closed_trades = self.journal.get_closed_trades()
        trades_with_confidence = [t for t in closed_trades if t.signal_confidence > 0]
        
        if len(trades_with_confidence) < 10:
            return TradingPattern(
                pattern_type=PatternType.CONFIDENCE_LEVEL,
                name="Confidence Analysis",
                description="Insufficient data for analysis"
            )
        
        # Analyze by confidence buckets
        buckets = {
            'Low (0-40%)': [],
            'Medium (40-60%)': [],
            'High (60-80%)': [],
            'Very High (80-100%)': [],
        }
        
        for trade in trades_with_confidence:
            conf = trade.signal_confidence * 100
            if conf < 40:
                buckets['Low (0-40%)'].append(trade)
            elif conf < 60:
                buckets['Medium (40-60%)'].append(trade)
            elif conf < 80:
                buckets['High (60-80%)'].append(trade)
            else:
                buckets['Very High (80-100%)'].append(trade)
        
        bucket_stats = {}
        for bucket_name, bucket_trades in buckets.items():
            if bucket_trades:
                winners = len([t for t in bucket_trades if t.realized_pnl and t.realized_pnl > 0])
                bucket_stats[bucket_name] = {
                    'trades': len(bucket_trades),
                    'win_rate': winners / len(bucket_trades) * 100,
                    'avg_pnl': sum(t.realized_pnl for t in bucket_trades if t.realized_pnl) / len(bucket_trades)
                }
        
        # Check if higher confidence = better results
        confidence_effective = False
        if 'High (60-80%)' in bucket_stats and 'Low (0-40%)' in bucket_stats:
            if bucket_stats['High (60-80%)']['win_rate'] > bucket_stats['Low (0-40%)']['win_rate']:
                confidence_effective = True
        
        pattern = TradingPattern(
            pattern_type=PatternType.CONFIDENCE_LEVEL,
            name="Confidence-Success Correlation",
            description="Relationship between signal confidence and trade success",
            occurrences=len(trades_with_confidence),
            details={
                'buckets': bucket_stats,
                'confidence_effective': confidence_effective,
            }
        )
        
        if confidence_effective:
            pattern.strength = PatternStrength.STRONG
            pattern.confidence = 0.8
            pattern.recommendations.append("Signal confidence is predictive - prioritize high-confidence signals")
        else:
            pattern.strength = PatternStrength.WEAK
            pattern.confidence = 0.4
            pattern.recommendations.append("Signal confidence shows weak correlation - review signal generation")
        
        return pattern
    
    def analyze_trade_duration(self) -> TradingPattern:
        """Analyze optimal trade duration."""
        if not self.journal:
            return TradingPattern(
                pattern_type=PatternType.TRADE_DURATION,
                name="Duration Analysis",
                description="No data available"
            )
        
        closed_trades = self.journal.get_closed_trades()
        trades_with_duration = [t for t in closed_trades if t.duration()]
        
        if len(trades_with_duration) < 10:
            return TradingPattern(
                pattern_type=PatternType.TRADE_DURATION,
                name="Duration Analysis",
                description="Insufficient data for analysis"
            )
        
        # Categorize by duration
        durations = {
            'Scalp (<1h)': [],
            'Intraday (1h-1d)': [],
            'Swing (1d-1w)': [],
            'Position (>1w)': [],
        }
        
        for trade in trades_with_duration:
            duration = trade.duration()
            hours = duration.total_seconds() / 3600
            
            if hours < 1:
                durations['Scalp (<1h)'].append(trade)
            elif hours < 24:
                durations['Intraday (1h-1d)'].append(trade)
            elif hours < 168:  # 7 days
                durations['Swing (1d-1w)'].append(trade)
            else:
                durations['Position (>1w)'].append(trade)
        
        duration_stats = {}
        best_duration = None
        best_win_rate = 0
        
        for duration_name, duration_trades in durations.items():
            if duration_trades:
                winners = len([t for t in duration_trades if t.realized_pnl and t.realized_pnl > 0])
                win_rate = winners / len(duration_trades) * 100
                
                duration_stats[duration_name] = {
                    'trades': len(duration_trades),
                    'win_rate': win_rate,
                    'avg_pnl': sum(t.realized_pnl for t in duration_trades if t.realized_pnl) / len(duration_trades)
                }
                
                if win_rate > best_win_rate and len(duration_trades) >= 5:
                    best_win_rate = win_rate
                    best_duration = duration_name
        
        pattern = TradingPattern(
            pattern_type=PatternType.TRADE_DURATION,
            name="Optimal Trade Duration",
            description="Analysis of trade duration and success",
            occurrences=len(trades_with_duration),
            details={
                'duration_stats': duration_stats,
                'best_duration': best_duration,
            }
        )
        
        if best_duration and best_win_rate >= 55:
            pattern.strength = PatternStrength.MODERATE
            pattern.confidence = 0.6
            pattern.recommendations.append(f"Best performance with {best_duration} trades - consider focusing on this timeframe")
        
        return pattern
    
    def analyze_consecutive_patterns(self) -> TradingPattern:
        """Analyze patterns in consecutive wins/losses."""
        if not self.journal:
            return TradingPattern(
                pattern_type=PatternType.CONSECUTIVE,
                name="Consecutive Pattern Analysis",
                description="No data available"
            )
        
        closed_trades = self.journal.get_closed_trades()
        sorted_trades = sorted(
            [t for t in closed_trades if t.exit_time],
            key=lambda t: t.exit_time
        )
        
        if len(sorted_trades) < 20:
            return TradingPattern(
                pattern_type=PatternType.CONSECUTIVE,
                name="Consecutive Pattern Analysis",
                description="Insufficient data for analysis"
            )
        
        # Analyze performance after streaks
        after_win_trades = []
        after_loss_trades = []
        after_win_streak = []
        after_loss_streak = []
        
        win_streak = 0
        loss_streak = 0
        
        for i, trade in enumerate(sorted_trades[:-1]):
            next_trade = sorted_trades[i + 1]
            
            if trade.realized_pnl and trade.realized_pnl > 0:
                win_streak += 1
                loss_streak = 0
                after_win_trades.append(next_trade)
                if win_streak >= 2:
                    after_win_streak.append(next_trade)
            elif trade.realized_pnl and trade.realized_pnl < 0:
                loss_streak += 1
                win_streak = 0
                after_loss_trades.append(next_trade)
                if loss_streak >= 2:
                    after_loss_streak.append(next_trade)
            else:
                win_streak = 0
                loss_streak = 0
        
        details = {}
        recommendations = []
        
        # After single win
        if after_win_trades:
            win_rate = len([t for t in after_win_trades if t.realized_pnl and t.realized_pnl > 0]) / len(after_win_trades) * 100
            details['after_win_rate'] = win_rate
            if win_rate < 45:
                recommendations.append("Performance drops after wins - avoid overconfidence")
        
        # After single loss
        if after_loss_trades:
            win_rate = len([t for t in after_loss_trades if t.realized_pnl and t.realized_pnl > 0]) / len(after_loss_trades) * 100
            details['after_loss_rate'] = win_rate
            if win_rate < 45:
                recommendations.append("Performance drops after losses - consider taking breaks")
        
        # After win streak
        if after_win_streak:
            win_rate = len([t for t in after_win_streak if t.realized_pnl and t.realized_pnl > 0]) / len(after_win_streak) * 100
            details['after_win_streak_rate'] = win_rate
            if win_rate < 45:
                recommendations.append("Performance drops after win streaks - reduce size after winning streak")
        
        # After loss streak
        if after_loss_streak:
            win_rate = len([t for t in after_loss_streak if t.realized_pnl and t.realized_pnl > 0]) / len(after_loss_streak) * 100
            details['after_loss_streak_rate'] = win_rate
            if win_rate > 55:
                recommendations.append("Good recovery after loss streaks - maintain discipline")
        
        pattern = TradingPattern(
            pattern_type=PatternType.CONSECUTIVE,
            name="Consecutive Trade Patterns",
            description="How performance changes based on previous results",
            occurrences=len(sorted_trades),
            details=details,
            recommendations=recommendations,
            strength=PatternStrength.MODERATE if recommendations else PatternStrength.WEAK,
            confidence=0.6 if len(sorted_trades) >= 30 else 0.4
        )
        
        return pattern
    
    def run_full_analysis(self) -> List[TradingPattern]:
        """Run all pattern analyses."""
        self.patterns = []
        
        # Time patterns
        self.patterns.extend(self.analyze_time_of_day())
        self.patterns.extend(self.analyze_day_of_week())
        
        # Category patterns
        self.patterns.extend(self.analyze_by_strategy())
        self.patterns.extend(self.analyze_by_symbol())
        self.patterns.extend(self.analyze_by_market_regime())
        
        # Behavioral patterns
        self.patterns.append(self.analyze_confidence_correlation())
        self.patterns.append(self.analyze_trade_duration())
        self.patterns.append(self.analyze_consecutive_patterns())
        
        return self.patterns
    
    def get_strongest_patterns(self, top_n: int = 5) -> List[TradingPattern]:
        """Get the strongest patterns."""
        if not self.patterns:
            self.run_full_analysis()
        
        # Sort by strength and confidence
        strength_order = {
            PatternStrength.VERY_STRONG: 5,
            PatternStrength.STRONG: 4,
            PatternStrength.MODERATE: 3,
            PatternStrength.WEAK: 2,
            PatternStrength.VERY_WEAK: 1,
        }
        
        sorted_patterns = sorted(
            self.patterns,
            key=lambda p: (strength_order[p.strength], p.confidence),
            reverse=True
        )
        
        return sorted_patterns[:top_n]
    
    def get_all_recommendations(self) -> List[str]:
        """Get all actionable recommendations."""
        if not self.patterns:
            self.run_full_analysis()
        
        recommendations = []
        for pattern in self.patterns:
            if pattern.strength in [PatternStrength.STRONG, PatternStrength.VERY_STRONG]:
                recommendations.extend(pattern.recommendations)
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all patterns."""
        if not self.patterns:
            self.run_full_analysis()
        
        summary = {
            'total_patterns': len(self.patterns),
            'patterns_by_type': {},
            'patterns_by_strength': {},
            'top_recommendations': self.get_all_recommendations()[:10],
        }
        
        for pattern in self.patterns:
            ptype = pattern.pattern_type.value
            pstrength = pattern.strength.value
            
            summary['patterns_by_type'][ptype] = summary['patterns_by_type'].get(ptype, 0) + 1
            summary['patterns_by_strength'][pstrength] = summary['patterns_by_strength'].get(pstrength, 0) + 1
        
        return summary
