"""
Sentiment Fusion - Combines sentiment from multiple sources.

Fuses sentiment from:
- News sources
- Social media
- Analyst ratings
- Technical indicators sentiment
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


class SentimentTrend(Enum):
    """Sentiment trend direction."""
    STRONGLY_IMPROVING = "strongly_improving"
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    STRONGLY_DETERIORATING = "strongly_deteriorating"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SentimentAlert:
    """Sentiment-based alert."""
    
    symbol: str
    alert_type: str
    severity: AlertSeverity
    message: str
    
    # Context
    current_sentiment: float
    previous_sentiment: float
    change: float
    
    # Trigger info
    trigger_source: str  # news, social, combined
    triggered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Action suggestion
    suggested_action: Optional[str] = None


@dataclass
class FusedSentiment:
    """
    Combined sentiment from all sources.
    """
    
    symbol: str
    
    # Overall fused sentiment
    fused_score: float  # -1 to 1
    fused_confidence: float
    
    # Component scores
    news_score: float = 0.0
    news_confidence: float = 0.0
    social_score: float = 0.0
    social_confidence: float = 0.0
    analyst_score: float = 0.0
    analyst_confidence: float = 0.0
    
    # Trend
    trend: SentimentTrend = SentimentTrend.STABLE
    trend_strength: float = 0.0
    
    # Divergence detection
    sources_agree: bool = True
    divergence_score: float = 0.0
    
    # Signal generation
    signal_strength: float = 0.0
    signal_direction: str = "neutral"  # bullish, bearish, neutral
    
    # Alerts
    alerts: List[SentimentAlert] = field(default_factory=list)
    
    # Metadata
    sources_used: List[str] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'fused_score': self.fused_score,
            'fused_confidence': self.fused_confidence,
            'news_score': self.news_score,
            'social_score': self.social_score,
            'analyst_score': self.analyst_score,
            'trend': self.trend.value,
            'signal_direction': self.signal_direction,
            'signal_strength': self.signal_strength,
            'sources_agree': self.sources_agree,
            'analyzed_at': self.analyzed_at.isoformat()
        }


class SentimentFusion:
    """
    Fuses sentiment from multiple sources into unified signal.
    
    Features:
    - Weighted source combination
    - Divergence detection
    - Trend analysis
    - Alert generation
    """
    
    def __init__(
        self,
        news_weight: float = 0.35,
        social_weight: float = 0.35,
        analyst_weight: float = 0.30
    ):
        """
        Initialize sentiment fusion.
        
        Args:
            news_weight: Weight for news sentiment
            social_weight: Weight for social sentiment
            analyst_weight: Weight for analyst ratings
        """
        self.news_weight = news_weight
        self.social_weight = social_weight
        self.analyst_weight = analyst_weight
        
        # History tracking
        self.sentiment_history: Dict[str, List[FusedSentiment]] = {}
        
        # Alert thresholds
        self.alert_threshold_change = 0.3  # 30% change triggers alert
        self.alert_threshold_extreme = 0.7  # Extreme sentiment
        
        # Callbacks for alerts
        self.alert_callbacks: List = []
    
    def fuse_sentiment(
        self,
        symbol: str,
        news_sentiment: Optional[Dict] = None,
        social_sentiment: Optional[Dict] = None,
        analyst_sentiment: Optional[Dict] = None
    ) -> FusedSentiment:
        """
        Fuse sentiment from multiple sources.
        
        Args:
            symbol: Stock symbol
            news_sentiment: News sentiment dict with 'score' and 'confidence'
            social_sentiment: Social sentiment dict
            analyst_sentiment: Analyst ratings dict
            
        Returns:
            FusedSentiment result
        """
        sources_used = []
        
        # Extract scores and confidences
        news_score = 0.0
        news_confidence = 0.0
        if news_sentiment:
            news_score = news_sentiment.get('score', 0.0)
            news_confidence = news_sentiment.get('confidence', 0.0)
            sources_used.append('news')
        
        social_score = 0.0
        social_confidence = 0.0
        if social_sentiment:
            social_score = social_sentiment.get('overall_sentiment', 
                          social_sentiment.get('score', 0.0))
            social_confidence = social_sentiment.get('overall_confidence',
                               social_sentiment.get('confidence', 0.0))
            sources_used.append('social')
        
        analyst_score = 0.0
        analyst_confidence = 0.0
        if analyst_sentiment:
            analyst_score = analyst_sentiment.get('score', 0.0)
            analyst_confidence = analyst_sentiment.get('confidence', 0.0)
            sources_used.append('analyst')
        
        # Calculate fused sentiment
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        
        if news_sentiment:
            weighted_score += news_score * self.news_weight * news_confidence
            weighted_confidence += news_confidence * self.news_weight
            total_weight += self.news_weight * news_confidence
        
        if social_sentiment:
            weighted_score += social_score * self.social_weight * social_confidence
            weighted_confidence += social_confidence * self.social_weight
            total_weight += self.social_weight * social_confidence
        
        if analyst_sentiment:
            weighted_score += analyst_score * self.analyst_weight * analyst_confidence
            weighted_confidence += analyst_confidence * self.analyst_weight
            total_weight += self.analyst_weight * analyst_confidence
        
        if total_weight > 0:
            fused_score = weighted_score / total_weight
            fused_confidence = weighted_confidence / (
                self.news_weight + self.social_weight + self.analyst_weight
            )
        else:
            fused_score = 0.0
            fused_confidence = 0.0
        
        # Normalize
        fused_score = max(-1.0, min(1.0, fused_score))
        
        # Detect divergence
        scores = []
        if news_sentiment:
            scores.append(news_score)
        if social_sentiment:
            scores.append(social_score)
        if analyst_sentiment:
            scores.append(analyst_score)
        
        divergence = 0.0
        sources_agree = True
        if len(scores) >= 2:
            divergence = max(scores) - min(scores)
            # Check if signs match
            positive = sum(1 for s in scores if s > 0.1)
            negative = sum(1 for s in scores if s < -0.1)
            sources_agree = not (positive > 0 and negative > 0)
        
        # Calculate trend
        trend, trend_strength = self._calculate_trend(symbol, fused_score)
        
        # Generate signal
        signal_direction, signal_strength = self._generate_signal(
            fused_score, fused_confidence, sources_agree, divergence
        )
        
        result = FusedSentiment(
            symbol=symbol,
            fused_score=fused_score,
            fused_confidence=fused_confidence,
            news_score=news_score,
            news_confidence=news_confidence,
            social_score=social_score,
            social_confidence=social_confidence,
            analyst_score=analyst_score,
            analyst_confidence=analyst_confidence,
            trend=trend,
            trend_strength=trend_strength,
            sources_agree=sources_agree,
            divergence_score=divergence,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            sources_used=sources_used
        )
        
        # Check for alerts
        alerts = self._check_alerts(symbol, result)
        result.alerts = alerts
        
        # Store in history
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = []
        self.sentiment_history[symbol].append(result)
        
        # Keep last 100
        self.sentiment_history[symbol] = self.sentiment_history[symbol][-100:]
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.debug(f"Alert callback failed: {e}")
        
        return result
    
    def _calculate_trend(
        self,
        symbol: str,
        current_score: float
    ) -> tuple:
        """Calculate sentiment trend."""
        if symbol not in self.sentiment_history:
            return SentimentTrend.STABLE, 0.0
        
        history = self.sentiment_history[symbol]
        if len(history) < 3:
            return SentimentTrend.STABLE, 0.0
        
        # Get recent scores
        recent = [h.fused_score for h in history[-10:]]
        
        # Calculate average change
        changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        avg_change = sum(changes) / len(changes) if changes else 0
        
        # Determine trend
        if avg_change > 0.15:
            trend = SentimentTrend.STRONGLY_IMPROVING
        elif avg_change > 0.05:
            trend = SentimentTrend.IMPROVING
        elif avg_change < -0.15:
            trend = SentimentTrend.STRONGLY_DETERIORATING
        elif avg_change < -0.05:
            trend = SentimentTrend.DETERIORATING
        else:
            trend = SentimentTrend.STABLE
        
        trend_strength = min(abs(avg_change) * 2, 1.0)
        
        return trend, trend_strength
    
    def _generate_signal(
        self,
        score: float,
        confidence: float,
        sources_agree: bool,
        divergence: float
    ) -> tuple:
        """Generate trading signal from sentiment."""
        # Base signal on score
        if score > 0.3:
            direction = "bullish"
        elif score < -0.3:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Calculate strength
        strength = abs(score) * confidence
        
        # Reduce strength if sources disagree
        if not sources_agree:
            strength *= 0.5
        
        # Reduce if high divergence
        if divergence > 0.5:
            strength *= 0.7
        
        strength = min(strength, 1.0)
        
        return direction, strength
    
    def _check_alerts(
        self,
        symbol: str,
        current: FusedSentiment
    ) -> List[SentimentAlert]:
        """Check for alert conditions."""
        alerts = []
        
        # Get previous sentiment
        history = self.sentiment_history.get(symbol, [])
        if len(history) < 2:
            previous_score = 0.0
        else:
            previous_score = history[-2].fused_score
        
        change = current.fused_score - previous_score
        
        # Significant change alert
        if abs(change) > self.alert_threshold_change:
            direction = "improved" if change > 0 else "deteriorated"
            alerts.append(SentimentAlert(
                symbol=symbol,
                alert_type="sentiment_shift",
                severity=AlertSeverity.WARNING,
                message=f"Sentiment {direction} by {abs(change):.1%}",
                current_sentiment=current.fused_score,
                previous_sentiment=previous_score,
                change=change,
                trigger_source="combined",
                suggested_action=f"Review {symbol} - significant sentiment change"
            ))
        
        # Extreme sentiment alert
        if abs(current.fused_score) > self.alert_threshold_extreme:
            if current.fused_score > 0:
                alerts.append(SentimentAlert(
                    symbol=symbol,
                    alert_type="extreme_bullish",
                    severity=AlertSeverity.INFO,
                    message=f"Extremely bullish sentiment ({current.fused_score:.2f})",
                    current_sentiment=current.fused_score,
                    previous_sentiment=previous_score,
                    change=change,
                    trigger_source="combined",
                    suggested_action="Consider taking profits or setting stops"
                ))
            else:
                alerts.append(SentimentAlert(
                    symbol=symbol,
                    alert_type="extreme_bearish",
                    severity=AlertSeverity.WARNING,
                    message=f"Extremely bearish sentiment ({current.fused_score:.2f})",
                    current_sentiment=current.fused_score,
                    previous_sentiment=previous_score,
                    change=change,
                    trigger_source="combined",
                    suggested_action="Review position or consider hedging"
                ))
        
        # Divergence alert
        if current.divergence_score > 0.5 and not current.sources_agree:
            alerts.append(SentimentAlert(
                symbol=symbol,
                alert_type="source_divergence",
                severity=AlertSeverity.INFO,
                message="Sentiment sources disagree significantly",
                current_sentiment=current.fused_score,
                previous_sentiment=previous_score,
                change=change,
                trigger_source="combined",
                suggested_action="Wait for consensus before acting"
            ))
        
        return alerts
    
    def get_sentiment_summary(
        self,
        symbol: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get summary of sentiment over time period.
        
        Args:
            symbol: Stock symbol
            hours: Hours to summarize
            
        Returns:
            Summary dict
        """
        if symbol not in self.sentiment_history:
            return {
                'symbol': symbol,
                'data_points': 0,
                'avg_sentiment': 0.0,
                'trend': 'unknown',
                'has_alerts': False
            }
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        history = [
            h for h in self.sentiment_history[symbol]
            if h.analyzed_at > cutoff
        ]
        
        if not history:
            return {
                'symbol': symbol,
                'data_points': 0,
                'avg_sentiment': 0.0,
                'trend': 'unknown',
                'has_alerts': False
            }
        
        scores = [h.fused_score for h in history]
        
        return {
            'symbol': symbol,
            'data_points': len(history),
            'avg_sentiment': sum(scores) / len(scores),
            'min_sentiment': min(scores),
            'max_sentiment': max(scores),
            'current_sentiment': history[-1].fused_score,
            'trend': history[-1].trend.value,
            'signal_direction': history[-1].signal_direction,
            'signal_strength': history[-1].signal_strength,
            'has_alerts': any(h.alerts for h in history),
            'total_alerts': sum(len(h.alerts) for h in history)
        }
    
    def register_alert_callback(self, callback):
        """Register callback for sentiment alerts."""
        self.alert_callbacks.append(callback)
    
    def get_multi_symbol_sentiment(
        self,
        symbols: List[str]
    ) -> Dict[str, FusedSentiment]:
        """
        Get latest sentiment for multiple symbols.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dict mapping symbol to latest FusedSentiment
        """
        results = {}
        
        for symbol in symbols:
            if symbol in self.sentiment_history and self.sentiment_history[symbol]:
                results[symbol] = self.sentiment_history[symbol][-1]
        
        return results
    
    def rank_by_sentiment(
        self,
        symbols: List[str],
        direction: str = "bullish"
    ) -> List[tuple]:
        """
        Rank symbols by sentiment.
        
        Args:
            symbols: List of symbols to rank
            direction: "bullish" or "bearish"
            
        Returns:
            List of (symbol, score) tuples, sorted
        """
        sentiments = self.get_multi_symbol_sentiment(symbols)
        
        rankings = [
            (symbol, s.fused_score)
            for symbol, s in sentiments.items()
        ]
        
        # Sort by score
        reverse = direction == "bullish"
        rankings.sort(key=lambda x: x[1], reverse=reverse)
        
        return rankings
