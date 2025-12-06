"""
Dynamic Risk Management - Adaptive risk based on market conditions.

Features:
- Regime-based position sizing
- Volatility-adjusted risk limits
- Correlation-aware portfolio risk
- Real-time risk monitoring
- Automatic position adjustments
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RiskMode(Enum):
    """Risk management modes."""
    CONSERVATIVE = "conservative"  # Low risk, defensive
    MODERATE = "moderate"          # Balanced approach
    AGGRESSIVE = "aggressive"      # Higher risk, offensive
    ADAPTIVE = "adaptive"          # Dynamically adjusted


class MarketCondition(Enum):
    """Current market condition assessment."""
    CALM = "calm"                  # Low volatility, stable
    NORMAL = "normal"              # Average conditions
    VOLATILE = "volatile"          # High volatility
    CRISIS = "crisis"              # Extreme conditions


@dataclass
class DynamicRiskConfig:
    """Configuration for dynamic risk management."""
    
    # Base risk parameters
    base_risk_per_trade: float = 0.02  # 2% base risk
    base_max_position: float = 0.15     # 15% max position
    base_max_drawdown: float = 0.10     # 10% max drawdown
    
    # Volatility adjustments
    vol_scale_factor: float = 1.0       # Volatility scaling
    target_portfolio_vol: float = 0.15  # 15% target volatility
    max_portfolio_vol: float = 0.25     # 25% max volatility
    
    # Regime adjustments
    regime_scale_calm: float = 1.2      # Scale up in calm
    regime_scale_normal: float = 1.0    # Normal scaling
    regime_scale_volatile: float = 0.6  # Scale down in volatile
    regime_scale_crisis: float = 0.3    # Scale way down in crisis
    
    # Correlation limits
    max_correlation_exposure: float = 0.5  # 50% max correlated
    correlation_threshold: float = 0.7     # Correlation threshold
    
    # Drawdown adjustments
    drawdown_scale_threshold: float = 0.5  # Start scaling at 50% of max DD
    min_risk_scale: float = 0.25           # Minimum risk scaling
    
    # Recovery settings
    recovery_period_days: int = 5          # Days before resuming
    recovery_scale_rate: float = 0.2       # Rate of risk recovery


@dataclass
class RiskAdjustment:
    """Risk adjustment result."""
    
    # Adjusted limits
    adjusted_risk_per_trade: float
    adjusted_max_position: float
    adjusted_max_exposure: float
    
    # Scaling factors
    volatility_scale: float = 1.0
    regime_scale: float = 1.0
    drawdown_scale: float = 1.0
    correlation_scale: float = 1.0
    overall_scale: float = 1.0
    
    # Recommendations
    reduce_positions: List[str] = field(default_factory=list)
    avoid_symbols: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # State
    market_condition: MarketCondition = MarketCondition.NORMAL
    is_restricted: bool = False
    restriction_reason: Optional[str] = None


@dataclass 
class CorrelationRisk:
    """Correlation-based risk analysis."""
    
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    highly_correlated_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    cluster_exposure: Dict[str, float] = field(default_factory=dict)
    diversification_score: float = 0.0
    concentration_risk: float = 0.0


class DynamicRiskManager:
    """
    Dynamic risk manager that adapts to market conditions.
    
    Features:
    - Volatility-based position sizing
    - Regime-aware risk limits
    - Drawdown-based scaling
    - Correlation monitoring
    - Automatic risk reduction
    """
    
    def __init__(
        self,
        config: Optional[DynamicRiskConfig] = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize Dynamic Risk Manager.
        
        Args:
            config: Risk configuration
            initial_capital: Starting capital
        """
        self.config = config or DynamicRiskConfig()
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        
        # State tracking
        self.current_market_condition = MarketCondition.NORMAL
        self.risk_mode = RiskMode.ADAPTIVE
        self.volatility_history: List[float] = []
        self.drawdown_history: List[Tuple[datetime, float]] = []
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.position_correlations: pd.DataFrame = pd.DataFrame()
        
        # Risk events
        self.risk_events: List[Dict] = []
        self.last_major_drawdown: Optional[datetime] = None
        self.in_recovery: bool = False
    
    def assess_market_condition(
        self,
        current_volatility: float,
        historical_volatility: float,
        vix_level: Optional[float] = None,
        market_trend: Optional[str] = None
    ) -> MarketCondition:
        """
        Assess current market condition.
        
        Args:
            current_volatility: Current volatility (annualized)
            historical_volatility: Long-term average volatility
            vix_level: Optional VIX level
            market_trend: Optional trend ("up", "down", "sideways")
            
        Returns:
            MarketCondition enum
        """
        # Volatility ratio
        vol_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1.0
        
        # Determine condition
        if vix_level is not None:
            # Use VIX if available
            if vix_level < 15:
                condition = MarketCondition.CALM
            elif vix_level < 25:
                condition = MarketCondition.NORMAL
            elif vix_level < 35:
                condition = MarketCondition.VOLATILE
            else:
                condition = MarketCondition.CRISIS
        else:
            # Use volatility ratio
            if vol_ratio < 0.7:
                condition = MarketCondition.CALM
            elif vol_ratio < 1.3:
                condition = MarketCondition.NORMAL
            elif vol_ratio < 2.0:
                condition = MarketCondition.VOLATILE
            else:
                condition = MarketCondition.CRISIS
        
        # Store for tracking
        self.current_market_condition = condition
        self.volatility_history.append(current_volatility)
        
        # Keep last 252 days (1 year)
        if len(self.volatility_history) > 252:
            self.volatility_history = self.volatility_history[-252:]
        
        logger.info(f"Market condition assessed as: {condition.value}")
        return condition
    
    def calculate_risk_adjustment(
        self,
        current_capital: float,
        current_volatility: float,
        positions: Optional[Dict[str, Dict]] = None
    ) -> RiskAdjustment:
        """
        Calculate dynamic risk adjustment based on conditions.
        
        Args:
            current_capital: Current portfolio value
            current_volatility: Current portfolio volatility
            positions: Current positions dict
            
        Returns:
            RiskAdjustment with adjusted limits
        """
        positions = positions or self.positions
        recommendations = []
        reduce_positions = []
        avoid_symbols = []
        
        # 1. Volatility scaling
        vol_scale = self._calculate_volatility_scale(current_volatility)
        
        # 2. Regime scaling
        regime_scale = self._get_regime_scale()
        
        # 3. Drawdown scaling
        drawdown = self._calculate_drawdown(current_capital)
        drawdown_scale = self._calculate_drawdown_scale(drawdown)
        
        # 4. Correlation scaling
        corr_scale = 1.0
        if positions:
            corr_risk = self.analyze_correlation_risk(positions)
            if corr_risk.concentration_risk > 0.5:
                corr_scale = 0.8
                recommendations.append("High concentration risk - reduce position sizes")
                reduce_positions = list(positions.keys())[:3]  # Top 3 positions
        
        # Combined scaling
        overall_scale = vol_scale * regime_scale * drawdown_scale * corr_scale
        overall_scale = max(overall_scale, self.config.min_risk_scale)
        
        # Calculate adjusted limits
        adjusted_risk = self.config.base_risk_per_trade * overall_scale
        adjusted_max_pos = self.config.base_max_position * overall_scale
        adjusted_exposure = min(1.0, 0.8 * overall_scale)  # Max 80% exposure scaled
        
        # Determine restrictions
        is_restricted = False
        restriction_reason = None
        
        if self.current_market_condition == MarketCondition.CRISIS:
            is_restricted = True
            restriction_reason = "Market in crisis mode"
            recommendations.append("Consider reducing all positions to 50%")
        
        if drawdown > self.config.base_max_drawdown * 0.9:
            is_restricted = True
            restriction_reason = "Near maximum drawdown"
            recommendations.append("Stop new trades until recovery")
        
        if self.in_recovery:
            recommendations.append(f"In recovery mode - risk scaled to {overall_scale:.0%}")
        
        # Build result
        adjustment = RiskAdjustment(
            adjusted_risk_per_trade=adjusted_risk,
            adjusted_max_position=adjusted_max_pos,
            adjusted_max_exposure=adjusted_exposure,
            volatility_scale=vol_scale,
            regime_scale=regime_scale,
            drawdown_scale=drawdown_scale,
            correlation_scale=corr_scale,
            overall_scale=overall_scale,
            reduce_positions=reduce_positions,
            avoid_symbols=avoid_symbols,
            recommendations=recommendations,
            market_condition=self.current_market_condition,
            is_restricted=is_restricted,
            restriction_reason=restriction_reason
        )
        
        return adjustment
    
    def _calculate_volatility_scale(self, current_vol: float) -> float:
        """Calculate scaling based on volatility."""
        if current_vol <= 0:
            return 1.0
        
        # Scale inversely with volatility relative to target
        target = self.config.target_portfolio_vol
        scale = target / current_vol
        
        # Clamp to reasonable range
        scale = max(0.3, min(1.5, scale))
        
        return scale
    
    def _get_regime_scale(self) -> float:
        """Get scaling factor for current market regime."""
        regime_scales = {
            MarketCondition.CALM: self.config.regime_scale_calm,
            MarketCondition.NORMAL: self.config.regime_scale_normal,
            MarketCondition.VOLATILE: self.config.regime_scale_volatile,
            MarketCondition.CRISIS: self.config.regime_scale_crisis,
        }
        return regime_scales.get(self.current_market_condition, 1.0)
    
    def _calculate_drawdown(self, current_capital: float) -> float:
        """Calculate current drawdown."""
        if self.peak_capital <= 0:
            return 0.0
        
        # Update peak
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
            self.in_recovery = False
        
        drawdown = (self.peak_capital - current_capital) / self.peak_capital
        
        # Track significant drawdowns
        if drawdown > self.config.base_max_drawdown * 0.5:
            self.drawdown_history.append((datetime.now(timezone.utc), drawdown))
            
            if drawdown > self.config.base_max_drawdown * 0.75:
                self.in_recovery = True
                self.last_major_drawdown = datetime.now(timezone.utc)
        
        return drawdown
    
    def _calculate_drawdown_scale(self, drawdown: float) -> float:
        """Calculate scaling based on drawdown."""
        max_dd = self.config.base_max_drawdown
        threshold = max_dd * self.config.drawdown_scale_threshold
        
        if drawdown <= threshold:
            return 1.0
        
        # Linear scaling from threshold to max
        remaining = max_dd - threshold
        excess = drawdown - threshold
        
        scale = 1.0 - (excess / remaining) * (1.0 - self.config.min_risk_scale)
        
        return max(self.config.min_risk_scale, scale)
    
    def analyze_correlation_risk(
        self,
        positions: Dict[str, Dict],
        price_data: Optional[Dict[str, pd.Series]] = None
    ) -> CorrelationRisk:
        """
        Analyze portfolio correlation risk.
        
        Args:
            positions: Current positions
            price_data: Historical price data for each symbol
            
        Returns:
            CorrelationRisk analysis
        """
        if not positions or len(positions) < 2:
            return CorrelationRisk(diversification_score=1.0)
        
        # Build correlation matrix if we have price data
        correlation_matrix = pd.DataFrame()
        highly_correlated = []
        
        if price_data:
            # Create returns DataFrame
            returns_dict = {}
            for symbol, prices in price_data.items():
                if symbol in positions and len(prices) > 20:
                    returns_dict[symbol] = prices.pct_change().dropna()
            
            if len(returns_dict) >= 2:
                returns_df = pd.DataFrame(returns_dict)
                correlation_matrix = returns_df.corr()
                
                # Find highly correlated pairs
                for i, sym1 in enumerate(correlation_matrix.columns):
                    for sym2 in correlation_matrix.columns[i+1:]:
                        corr = correlation_matrix.loc[sym1, sym2]
                        if abs(corr) > self.config.correlation_threshold:
                            highly_correlated.append((sym1, sym2, corr))
        
        # Calculate position weights
        total_value = sum(p.get('value', 0) for p in positions.values())
        weights = {}
        for symbol, pos in positions.items():
            weights[symbol] = pos.get('value', 0) / total_value if total_value > 0 else 0
        
        # Concentration risk (Herfindahl-Hirschman Index)
        hhi = sum(w ** 2 for w in weights.values())
        concentration_risk = (hhi - 1/len(positions)) / (1 - 1/len(positions)) if len(positions) > 1 else 1.0
        
        # Diversification score (inverse of concentration)
        diversification_score = 1.0 - concentration_risk
        
        # Cluster exposure (simplified - group highly correlated)
        cluster_exposure = {}
        for sym1, sym2, corr in highly_correlated:
            cluster_name = f"{sym1}_{sym2[:4]}"
            exposure = weights.get(sym1, 0) + weights.get(sym2, 0)
            cluster_exposure[cluster_name] = exposure
        
        return CorrelationRisk(
            correlation_matrix=correlation_matrix,
            highly_correlated_pairs=highly_correlated,
            cluster_exposure=cluster_exposure,
            diversification_score=diversification_score,
            concentration_risk=concentration_risk
        )
    
    def get_position_size_adjustment(
        self,
        symbol: str,
        base_position_value: float,
        symbol_volatility: float,
        current_capital: float
    ) -> Tuple[float, str]:
        """
        Get adjusted position size for a symbol.
        
        Args:
            symbol: Stock symbol
            base_position_value: Base position value before adjustment
            symbol_volatility: Symbol's volatility
            current_capital: Current portfolio value
            
        Returns:
            Tuple of (adjusted_value, reason)
        """
        # Get overall risk adjustment
        current_vol = np.mean(self.volatility_history[-20:]) if self.volatility_history else 0.15
        adjustment = self.calculate_risk_adjustment(current_capital, current_vol)
        
        # Apply overall scaling
        adjusted_value = base_position_value * adjustment.overall_scale
        
        # Further adjust for symbol volatility
        if symbol_volatility > 0.5:  # High volatility stock
            vol_adj = 0.5 / symbol_volatility
            adjusted_value *= min(1.0, vol_adj)
        
        # Cap at maximum position size
        max_value = current_capital * adjustment.adjusted_max_position
        if adjusted_value > max_value:
            adjusted_value = max_value
        
        # Build reason
        reasons = []
        if adjustment.overall_scale < 1.0:
            reasons.append(f"Risk scaled to {adjustment.overall_scale:.0%}")
        if adjustment.is_restricted:
            reasons.append(adjustment.restriction_reason or "Trading restricted")
        if symbol_volatility > 0.5:
            reasons.append(f"High volatility ({symbol_volatility:.0%})")
        
        reason = "; ".join(reasons) if reasons else "Normal sizing"
        
        return adjusted_value, reason
    
    def should_reduce_exposure(self) -> Tuple[bool, Optional[str], float]:
        """
        Check if portfolio exposure should be reduced.
        
        Returns:
            Tuple of (should_reduce, reason, target_reduction_pct)
        """
        # Check market condition
        if self.current_market_condition == MarketCondition.CRISIS:
            return True, "Market crisis conditions", 0.5
        
        # Check volatility trend
        if len(self.volatility_history) >= 10:
            recent_vol = np.mean(self.volatility_history[-5:])
            prior_vol = np.mean(self.volatility_history[-10:-5])
            
            if recent_vol > prior_vol * 1.5:
                return True, "Rapidly increasing volatility", 0.3
        
        # Check recent drawdowns
        if self.in_recovery:
            days_since = 0
            if self.last_major_drawdown:
                days_since = (datetime.now(timezone.utc) - self.last_major_drawdown).days
            
            if days_since < self.config.recovery_period_days:
                return True, f"In recovery period ({days_since}/{self.config.recovery_period_days} days)", 0.25
        
        return False, None, 0.0
    
    def get_risk_report(
        self,
        current_capital: float,
        positions: Optional[Dict[str, Dict]] = None
    ) -> str:
        """Generate comprehensive risk report."""
        positions = positions or self.positions
        current_vol = np.mean(self.volatility_history[-20:]) if self.volatility_history else 0.15
        
        adjustment = self.calculate_risk_adjustment(current_capital, current_vol, positions)
        drawdown = self._calculate_drawdown(current_capital)
        
        report = []
        report.append("=" * 60)
        report.append("         DYNAMIC RISK MANAGEMENT REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Market Status
        report.append("📊 MARKET CONDITIONS:")
        report.append(f"   Condition: {self.current_market_condition.value.upper()}")
        report.append(f"   Current Volatility: {current_vol:.1%}")
        report.append(f"   Regime Scale: {adjustment.regime_scale:.0%}")
        report.append("")
        
        # Portfolio Status
        report.append("💰 PORTFOLIO STATUS:")
        report.append(f"   Current Capital: ${current_capital:,.2f}")
        report.append(f"   Peak Capital: ${self.peak_capital:,.2f}")
        report.append(f"   Drawdown: {drawdown:.2%}")
        report.append(f"   In Recovery: {'Yes' if self.in_recovery else 'No'}")
        report.append("")
        
        # Risk Scaling
        report.append("⚖️ RISK ADJUSTMENTS:")
        report.append(f"   Volatility Scale: {adjustment.volatility_scale:.0%}")
        report.append(f"   Regime Scale: {adjustment.regime_scale:.0%}")
        report.append(f"   Drawdown Scale: {adjustment.drawdown_scale:.0%}")
        report.append(f"   Correlation Scale: {adjustment.correlation_scale:.0%}")
        report.append(f"   Overall Scale: {adjustment.overall_scale:.0%}")
        report.append("")
        
        # Adjusted Limits
        report.append("📏 ADJUSTED LIMITS:")
        report.append(f"   Risk per Trade: {adjustment.adjusted_risk_per_trade:.2%}")
        report.append(f"   Max Position: {adjustment.adjusted_max_position:.1%}")
        report.append(f"   Max Exposure: {adjustment.adjusted_max_exposure:.0%}")
        report.append("")
        
        # Status
        if adjustment.is_restricted:
            report.append(f"🚨 RESTRICTED: {adjustment.restriction_reason}")
        else:
            report.append("✅ Trading Active")
        
        # Recommendations
        if adjustment.recommendations:
            report.append("")
            report.append("💡 RECOMMENDATIONS:")
            for rec in adjustment.recommendations:
                report.append(f"   • {rec}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


class RiskMonitor:
    """
    Real-time risk monitoring and alerting.
    """
    
    def __init__(self, dynamic_risk_manager: DynamicRiskManager):
        """Initialize risk monitor."""
        self.risk_manager = dynamic_risk_manager
        self.alerts: List[Dict] = []
        self.last_check: Optional[datetime] = None
    
    def check_risks(
        self,
        current_capital: float,
        positions: Dict[str, Dict],
        current_volatility: float
    ) -> List[Dict]:
        """
        Check for risk conditions and generate alerts.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        now = datetime.now(timezone.utc)
        
        # Get risk adjustment
        adjustment = self.risk_manager.calculate_risk_adjustment(
            current_capital, current_volatility, positions
        )
        
        # Check for market condition change
        if self.last_check:
            # Could add logic to detect condition changes
            pass
        
        # Check restrictions
        if adjustment.is_restricted:
            alerts.append({
                'type': 'restriction',
                'severity': 'high',
                'message': adjustment.restriction_reason,
                'timestamp': now,
                'action': 'Consider reducing exposure'
            })
        
        # Check if reduction needed
        should_reduce, reason, pct = self.risk_manager.should_reduce_exposure()
        if should_reduce:
            alerts.append({
                'type': 'exposure',
                'severity': 'medium',
                'message': reason,
                'timestamp': now,
                'action': f'Reduce exposure by {pct:.0%}'
            })
        
        # Check volatility
        if current_volatility > self.risk_manager.config.max_portfolio_vol:
            alerts.append({
                'type': 'volatility',
                'severity': 'high',
                'message': f'Portfolio volatility ({current_volatility:.1%}) exceeds maximum',
                'timestamp': now,
                'action': 'Reduce volatile positions'
            })
        
        # Check position concentration
        if positions:
            corr_risk = self.risk_manager.analyze_correlation_risk(positions)
            if corr_risk.concentration_risk > 0.6:
                alerts.append({
                    'type': 'concentration',
                    'severity': 'medium',
                    'message': f'High concentration risk ({corr_risk.concentration_risk:.0%})',
                    'timestamp': now,
                    'action': 'Diversify positions'
                })
        
        # Store alerts
        self.alerts.extend(alerts)
        self.last_check = now
        
        # Keep only recent alerts (last 24 hours)
        cutoff = now - timedelta(hours=24)
        self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff]
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [a for a in self.alerts if a['timestamp'] > cutoff]
