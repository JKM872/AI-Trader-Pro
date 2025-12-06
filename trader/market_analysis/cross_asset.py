"""
Cross-Asset Correlation Analysis Module.

Analyzes correlations between different asset classes to identify:
- Lead-lag relationships
- Risk-on/risk-off regimes
- Diversification opportunities
- Cross-market signals
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class categories."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    VOLATILITY = "volatility"
    REAL_ESTATE = "real_estate"


class MarketRegime(Enum):
    """Market regime classification."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    TRANSITIONING = "transitioning"
    NEUTRAL = "neutral"


@dataclass
class AssetInfo:
    """Information about an asset."""
    symbol: str
    name: str
    asset_class: AssetClass
    description: str = ""
    benchmark: bool = False


@dataclass
class CorrelationResult:
    """Result of correlation analysis between two assets."""
    asset1: str
    asset2: str
    correlation: float
    rolling_correlation: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    p_value: float = 0.0
    is_significant: bool = True
    lead_lag_days: int = 0
    relationship_strength: str = "moderate"
    
    def __post_init__(self):
        """Classify relationship strength."""
        abs_corr = abs(self.correlation)
        if abs_corr >= 0.8:
            self.relationship_strength = "very_strong"
        elif abs_corr >= 0.6:
            self.relationship_strength = "strong"
        elif abs_corr >= 0.4:
            self.relationship_strength = "moderate"
        elif abs_corr >= 0.2:
            self.relationship_strength = "weak"
        else:
            self.relationship_strength = "negligible"


@dataclass
class LeadLagResult:
    """Result of lead-lag analysis."""
    leader: str
    follower: str
    optimal_lag: int
    max_correlation: float
    all_lags: dict = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class RegimeAnalysis:
    """Market regime analysis result."""
    current_regime: MarketRegime
    regime_probability: float
    regime_duration_days: int
    indicators: dict = field(default_factory=dict)
    regime_history: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))


@dataclass
class DiversificationScore:
    """Portfolio diversification analysis."""
    overall_score: float  # 0-100
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    cluster_assignments: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)


class CrossAssetAnalyzer:
    """
    Analyzes cross-asset correlations and relationships.
    
    Features:
    - Rolling correlation analysis
    - Lead-lag detection
    - Market regime identification
    - Diversification scoring
    - Cross-market signal generation
    """
    
    # Default universe of cross-asset benchmarks
    DEFAULT_UNIVERSE = {
        # Equities
        "SPY": AssetInfo("SPY", "S&P 500 ETF", AssetClass.EQUITY, "US Large Cap", True),
        "QQQ": AssetInfo("QQQ", "Nasdaq 100 ETF", AssetClass.EQUITY, "US Tech"),
        "IWM": AssetInfo("IWM", "Russell 2000 ETF", AssetClass.EQUITY, "US Small Cap"),
        "EFA": AssetInfo("EFA", "EAFE ETF", AssetClass.EQUITY, "Developed Markets ex-US"),
        "EEM": AssetInfo("EEM", "Emerging Markets ETF", AssetClass.EQUITY, "Emerging Markets"),
        
        # Fixed Income
        "TLT": AssetInfo("TLT", "20+ Year Treasury ETF", AssetClass.FIXED_INCOME, "Long-term Treasuries", True),
        "IEF": AssetInfo("IEF", "7-10 Year Treasury ETF", AssetClass.FIXED_INCOME, "Intermediate Treasuries"),
        "HYG": AssetInfo("HYG", "High Yield Bond ETF", AssetClass.FIXED_INCOME, "High Yield Corporates"),
        "LQD": AssetInfo("LQD", "Investment Grade Bond ETF", AssetClass.FIXED_INCOME, "IG Corporates"),
        
        # Commodities
        "GLD": AssetInfo("GLD", "Gold ETF", AssetClass.COMMODITY, "Gold", True),
        "SLV": AssetInfo("SLV", "Silver ETF", AssetClass.COMMODITY, "Silver"),
        "USO": AssetInfo("USO", "Oil ETF", AssetClass.COMMODITY, "Crude Oil"),
        "DBA": AssetInfo("DBA", "Agriculture ETF", AssetClass.COMMODITY, "Agricultural Commodities"),
        
        # Currencies
        "UUP": AssetInfo("UUP", "US Dollar Index ETF", AssetClass.CURRENCY, "US Dollar", True),
        "FXE": AssetInfo("FXE", "Euro ETF", AssetClass.CURRENCY, "Euro"),
        "FXY": AssetInfo("FXY", "Yen ETF", AssetClass.CURRENCY, "Japanese Yen"),
        
        # Volatility
        "VXX": AssetInfo("VXX", "VIX Short-term ETF", AssetClass.VOLATILITY, "VIX Futures", True),
        
        # Real Estate
        "VNQ": AssetInfo("VNQ", "Real Estate ETF", AssetClass.REAL_ESTATE, "US REITs", True),
        
        # Crypto (if available)
        "BTC-USD": AssetInfo("BTC-USD", "Bitcoin", AssetClass.CRYPTO, "Bitcoin"),
        "ETH-USD": AssetInfo("ETH-USD", "Ethereum", AssetClass.CRYPTO, "Ethereum"),
    }
    
    def __init__(
        self,
        lookback_days: int = 252,
        rolling_window: int = 20,
        significance_level: float = 0.05,
        min_correlation_threshold: float = 0.3,
    ):
        """
        Initialize Cross-Asset Analyzer.
        
        Args:
            lookback_days: Days of history to analyze
            rolling_window: Window for rolling calculations
            significance_level: P-value threshold for significance
            min_correlation_threshold: Minimum correlation to report
        """
        self.lookback_days = lookback_days
        self.rolling_window = rolling_window
        self.significance_level = significance_level
        self.min_correlation_threshold = min_correlation_threshold
        self.price_data: dict[str, pd.DataFrame] = {}
        self.returns_data: dict[str, pd.Series] = {}
        
    def load_data(self, symbols: list[str], fetcher=None) -> dict[str, pd.DataFrame]:
        """
        Load price data for analysis.
        
        Args:
            symbols: List of symbols to load
            fetcher: Optional DataFetcher instance
            
        Returns:
            Dictionary of symbol -> price DataFrame
        """
        if fetcher is None:
            try:
                from trader.data.fetcher import DataFetcher
                fetcher = DataFetcher()
            except ImportError:
                logger.error("DataFetcher not available")
                return {}
        
        for symbol in symbols:
            try:
                df = fetcher.get_stock_data(
                    symbol, 
                    period=f"{self.lookback_days}d"
                )
                if df is not None and not df.empty:
                    self.price_data[symbol] = df
                    # Calculate returns
                    if 'Close' in df.columns:
                        self.returns_data[symbol] = df['Close'].pct_change().dropna()
                    elif 'close' in df.columns:
                        self.returns_data[symbol] = df['close'].pct_change().dropna()
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                
        return self.price_data
    
    def set_data(self, price_data: dict[str, pd.DataFrame]) -> None:
        """
        Set price data directly for analysis.
        
        Args:
            price_data: Dictionary of symbol -> price DataFrame
        """
        self.price_data = price_data
        self.returns_data = {}
        
        for symbol, df in price_data.items():
            if 'Close' in df.columns:
                self.returns_data[symbol] = df['Close'].pct_change().dropna()
            elif 'close' in df.columns:
                self.returns_data[symbol] = df['close'].pct_change().dropna()
    
    def calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
        method: str = "pearson",
    ) -> CorrelationResult:
        """
        Calculate correlation between two assets.
        
        Args:
            symbol1: First asset symbol
            symbol2: Second asset symbol
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            CorrelationResult with correlation metrics
        """
        if symbol1 not in self.returns_data or symbol2 not in self.returns_data:
            raise ValueError(f"Data not loaded for {symbol1} or {symbol2}")
        
        returns1 = self.returns_data[symbol1]
        returns2 = self.returns_data[symbol2]
        
        # Align data
        aligned = pd.concat([returns1, returns2], axis=1, join='inner')
        aligned.columns = [symbol1, symbol2]
        aligned = aligned.dropna()
        
        if len(aligned) < 30:
            logger.warning(f"Insufficient data for correlation: {len(aligned)} points")
            return CorrelationResult(
                asset1=symbol1,
                asset2=symbol2,
                correlation=0.0,
                is_significant=False,
            )
        
        # Calculate correlation
        if method == "pearson":
            correlation = aligned[symbol1].corr(aligned[symbol2])
        elif method == "spearman":
            correlation = aligned[symbol1].corr(aligned[symbol2], method='spearman')
        elif method == "kendall":
            correlation = aligned[symbol1].corr(aligned[symbol2], method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Calculate rolling correlation
        rolling_corr = aligned[symbol1].rolling(
            window=self.rolling_window
        ).corr(aligned[symbol2])
        
        # Calculate p-value using t-test approximation
        n = len(aligned)
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2 + 1e-10))
        # Approximate p-value using normal distribution for large n
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
        
        # Check for lead-lag
        lead_lag = self._detect_lead_lag(aligned[symbol1], aligned[symbol2])
        
        return CorrelationResult(
            asset1=symbol1,
            asset2=symbol2,
            correlation=correlation,
            rolling_correlation=rolling_corr,
            p_value=p_value,
            is_significant=p_value < self.significance_level,
            lead_lag_days=lead_lag.optimal_lag,
        )
    
    def _detect_lead_lag(
        self,
        series1: pd.Series,
        series2: pd.Series,
        max_lag: int = 10,
    ) -> LeadLagResult:
        """
        Detect lead-lag relationship between two series.
        
        Args:
            series1: First return series
            series2: Second return series
            max_lag: Maximum lag to test
            
        Returns:
            LeadLagResult with optimal lag information
        """
        correlations = {}
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # series1 leads series2
                corr = series1.iloc[:lag].corr(series2.iloc[-lag:])
            elif lag > 0:
                # series2 leads series1
                corr = series1.iloc[lag:].corr(series2.iloc[:-lag])
            else:
                corr = series1.corr(series2)
            
            if not np.isnan(corr):
                correlations[lag] = corr
        
        if not correlations:
            return LeadLagResult(
                leader=series1.name if hasattr(series1, 'name') else "series1",
                follower=series2.name if hasattr(series2, 'name') else "series2",
                optimal_lag=0,
                max_correlation=0.0,
            )
        
        # Find optimal lag
        optimal_lag = max(correlations, key=lambda x: abs(correlations[x]))
        max_corr = correlations[optimal_lag]
        
        # Determine leader/follower
        if optimal_lag < 0:
            leader = series1.name if hasattr(series1, 'name') else "series1"
            follower = series2.name if hasattr(series2, 'name') else "series2"
        else:
            leader = series2.name if hasattr(series2, 'name') else "series2"
            follower = series1.name if hasattr(series1, 'name') else "series1"
        
        # Calculate confidence based on correlation improvement vs lag=0
        base_corr = correlations.get(0, 0)
        improvement = abs(max_corr) - abs(base_corr)
        confidence = min(1.0, max(0.0, improvement * 10))  # Scale improvement
        
        return LeadLagResult(
            leader=leader,
            follower=follower,
            optimal_lag=abs(optimal_lag),
            max_correlation=max_corr,
            all_lags=correlations,
            confidence=confidence,
        )
    
    def calculate_correlation_matrix(
        self,
        symbols: Optional[list[str]] = None,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Calculate full correlation matrix.
        
        Args:
            symbols: List of symbols (uses all loaded if None)
            method: Correlation method
            
        Returns:
            Correlation matrix as DataFrame
        """
        if symbols is None:
            symbols = list(self.returns_data.keys())
        
        # Build aligned returns DataFrame
        returns_df = pd.DataFrame()
        for symbol in symbols:
            if symbol in self.returns_data:
                returns_df[symbol] = self.returns_data[symbol]
        
        # Calculate correlation matrix
        if method == "pearson":
            corr_matrix = returns_df.corr()
        elif method == "spearman":
            corr_matrix = returns_df.corr(method='spearman')
        elif method == "kendall":
            corr_matrix = returns_df.corr(method='kendall')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return corr_matrix
    
    def detect_market_regime(
        self,
        equity_symbol: str = "SPY",
        bond_symbol: str = "TLT",
        gold_symbol: str = "GLD",
        vix_symbol: str = "VXX",
    ) -> RegimeAnalysis:
        """
        Detect current market regime based on cross-asset behavior.
        
        Risk-On indicators:
        - Stocks up, bonds down
        - VIX declining
        - High yield outperforming
        - Emerging markets outperforming
        
        Risk-Off indicators:
        - Bonds up, stocks down
        - VIX rising
        - Gold outperforming
        - Dollar strengthening
        
        Args:
            equity_symbol: Equity benchmark symbol
            bond_symbol: Bond benchmark symbol
            gold_symbol: Gold symbol
            vix_symbol: Volatility symbol
            
        Returns:
            RegimeAnalysis with current regime classification
        """
        indicators = {}
        regime_scores = []
        
        # Check equity momentum
        if equity_symbol in self.returns_data:
            equity_ret = self.returns_data[equity_symbol]
            equity_momentum = equity_ret.tail(20).sum()
            indicators['equity_momentum'] = equity_momentum
            regime_scores.append(1 if equity_momentum > 0 else -1)
        
        # Check equity-bond correlation
        if equity_symbol in self.returns_data and bond_symbol in self.returns_data:
            try:
                corr = self.calculate_correlation(equity_symbol, bond_symbol)
                indicators['equity_bond_correlation'] = corr.correlation
                # Negative correlation = risk-on (flight to quality not happening)
                regime_scores.append(1 if corr.correlation < -0.1 else -1)
            except Exception:
                pass
        
        # Check gold performance
        if gold_symbol in self.returns_data:
            gold_ret = self.returns_data[gold_symbol]
            gold_momentum = gold_ret.tail(20).sum()
            indicators['gold_momentum'] = gold_momentum
            # Gold up = risk-off
            regime_scores.append(-1 if gold_momentum > 0.02 else 1)
        
        # Check VIX levels and trend
        if vix_symbol in self.price_data:
            vix_df = self.price_data[vix_symbol]
            if 'Close' in vix_df.columns:
                current_vix = vix_df['Close'].iloc[-1]
                vix_ma = vix_df['Close'].tail(20).mean()
                indicators['vix_level'] = current_vix
                indicators['vix_vs_ma'] = current_vix / vix_ma - 1
                # VIX above average = risk-off
                regime_scores.append(-1 if current_vix > vix_ma else 1)
        
        # Calculate overall regime
        if regime_scores:
            avg_score = np.mean(regime_scores)
            if avg_score > 0.3:
                regime = MarketRegime.RISK_ON
                probability = min(1.0, 0.5 + avg_score * 0.5)
            elif avg_score < -0.3:
                regime = MarketRegime.RISK_OFF
                probability = min(1.0, 0.5 - avg_score * 0.5)
            elif abs(avg_score) > 0.1:
                regime = MarketRegime.TRANSITIONING
                probability = 0.5 + abs(avg_score) * 0.3
            else:
                regime = MarketRegime.NEUTRAL
                probability = 0.5
        else:
            regime = MarketRegime.NEUTRAL
            probability = 0.5
        
        # Estimate regime duration
        regime_duration = self._estimate_regime_duration(regime, indicators)
        
        return RegimeAnalysis(
            current_regime=regime,
            regime_probability=probability,
            regime_duration_days=regime_duration,
            indicators=indicators,
        )
    
    def _estimate_regime_duration(
        self,
        current_regime: MarketRegime,
        indicators: dict,
    ) -> int:
        """Estimate how long current regime has been in place."""
        # Simplified estimation based on indicator persistence
        # In production, would analyze historical regime changes
        
        if current_regime in [MarketRegime.RISK_ON, MarketRegime.RISK_OFF]:
            # Check momentum persistence
            equity_momentum = indicators.get('equity_momentum', 0)
            if abs(equity_momentum) > 0.05:
                return 20  # Strong trend suggests ~1 month
            else:
                return 10
        else:
            return 5  # Transitioning/neutral regimes are shorter
    
    def calculate_diversification_score(
        self,
        portfolio_symbols: list[str],
    ) -> DiversificationScore:
        """
        Calculate portfolio diversification score.
        
        Args:
            portfolio_symbols: Symbols in the portfolio
            
        Returns:
            DiversificationScore with diversification metrics
        """
        if len(portfolio_symbols) < 2:
            return DiversificationScore(
                overall_score=0.0,
                recommendations=["Need at least 2 assets for diversification analysis"],
            )
        
        # Calculate correlation matrix for portfolio
        corr_matrix = self.calculate_correlation_matrix(portfolio_symbols)
        
        # Calculate average pairwise correlation (excluding diagonal)
        n = len(portfolio_symbols)
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        if corr_matrix.values.size > 0:
            avg_correlation = np.abs(corr_matrix.values[mask]).mean()
        else:
            avg_correlation = 0.0
        
        # Diversification score: lower average correlation = better diversification
        # Score from 0 (all correlated) to 100 (uncorrelated)
        score = max(0, min(100, (1 - avg_correlation) * 100))
        
        # Cluster assets by correlation
        clusters = self._cluster_assets(corr_matrix)
        
        # Generate recommendations
        recommendations = []
        
        if score < 50:
            recommendations.append(
                "Portfolio is highly correlated. Consider adding uncorrelated assets."
            )
        
        # Check for asset class diversification
        asset_classes_present = set()
        for symbol in portfolio_symbols:
            if symbol in self.DEFAULT_UNIVERSE:
                asset_classes_present.add(self.DEFAULT_UNIVERSE[symbol].asset_class)
        
        missing_classes = set(AssetClass) - asset_classes_present
        if missing_classes and len(missing_classes) < len(AssetClass):
            recs = [c.value for c in list(missing_classes)[:3]]
            recommendations.append(
                f"Consider adding exposure to: {', '.join(recs)}"
            )
        
        # Find highly correlated pairs
        highly_correlated = []
        for i, sym1 in enumerate(portfolio_symbols):
            for j, sym2 in enumerate(portfolio_symbols):
                if i < j:
                    try:
                        corr = corr_matrix.loc[sym1, sym2]
                        if abs(corr) > 0.8:
                            highly_correlated.append((sym1, sym2, corr))
                    except KeyError:
                        pass
        
        if highly_correlated:
            pairs_str = ", ".join([f"{p[0]}/{p[1]}" for p in highly_correlated[:3]])
            recommendations.append(
                f"Highly correlated pairs (consider reducing): {pairs_str}"
            )
        
        return DiversificationScore(
            overall_score=score,
            correlation_matrix=corr_matrix,
            cluster_assignments=clusters,
            recommendations=recommendations,
        )
    
    def _cluster_assets(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.6,
    ) -> dict[str, int]:
        """
        Cluster assets by correlation similarity.
        
        Simple clustering: assets with correlation > threshold are in same cluster.
        """
        symbols = corr_matrix.columns.tolist()
        clusters = {s: -1 for s in symbols}
        current_cluster = 0
        
        for i, sym1 in enumerate(symbols):
            if clusters[sym1] == -1:
                clusters[sym1] = current_cluster
                
                for j, sym2 in enumerate(symbols[i+1:], i+1):
                    try:
                        if corr_matrix.loc[sym1, sym2] > threshold:
                            clusters[sym2] = current_cluster
                    except KeyError:
                        pass
                
                current_cluster += 1
        
        return clusters
    
    def find_hedge_candidates(
        self,
        target_symbol: str,
        min_negative_correlation: float = -0.3,
        candidate_symbols: Optional[list[str]] = None,
    ) -> list[CorrelationResult]:
        """
        Find assets that could hedge the target position.
        
        Args:
            target_symbol: Symbol to hedge
            min_negative_correlation: Minimum negative correlation required
            candidate_symbols: Candidates to consider (uses all if None)
            
        Returns:
            List of negatively correlated assets sorted by correlation
        """
        if candidate_symbols is None:
            candidate_symbols = list(self.returns_data.keys())
        
        hedges = []
        
        for symbol in candidate_symbols:
            if symbol == target_symbol:
                continue
            
            try:
                result = self.calculate_correlation(target_symbol, symbol)
                if result.correlation < min_negative_correlation:
                    hedges.append(result)
            except Exception as e:
                logger.debug(f"Failed to calculate correlation with {symbol}: {e}")
        
        # Sort by most negative correlation
        hedges.sort(key=lambda x: x.correlation)
        
        return hedges
    
    def generate_cross_asset_signals(
        self,
        target_symbol: str,
        related_symbols: Optional[list[str]] = None,
    ) -> dict:
        """
        Generate trading signals based on cross-asset relationships.
        
        Args:
            target_symbol: Symbol to generate signals for
            related_symbols: Related assets to analyze
            
        Returns:
            Dictionary with signal information
        """
        if related_symbols is None:
            # Use default related assets based on asset class
            if target_symbol in self.DEFAULT_UNIVERSE:
                asset_class = self.DEFAULT_UNIVERSE[target_symbol].asset_class
                related_symbols = [
                    s for s, info in self.DEFAULT_UNIVERSE.items()
                    if info.asset_class != asset_class and info.benchmark
                ]
            else:
                related_symbols = ["SPY", "TLT", "GLD", "UUP"]
        
        signals = {
            'symbol': target_symbol,
            'timestamp': datetime.now().isoformat(),
            'lead_lag_signals': [],
            'regime_signal': None,
            'correlation_signals': [],
            'overall_bias': 'neutral',
        }
        
        # Analyze lead-lag relationships
        for related in related_symbols:
            if related in self.returns_data and target_symbol in self.returns_data:
                try:
                    lead_lag = self._detect_lead_lag(
                        self.returns_data[related],
                        self.returns_data[target_symbol],
                    )
                    if lead_lag.optimal_lag > 0 and lead_lag.max_correlation > 0.3:
                        # Related asset leads target
                        leader_return = self.returns_data[related].tail(lead_lag.optimal_lag).sum()
                        signal = {
                            'leader': related,
                            'lag_days': lead_lag.optimal_lag,
                            'correlation': lead_lag.max_correlation,
                            'leader_recent_return': leader_return,
                            'predicted_direction': 'up' if leader_return > 0 else 'down',
                        }
                        signals['lead_lag_signals'].append(signal)
                except Exception as e:
                    logger.debug(f"Lead-lag analysis failed for {related}: {e}")
        
        # Get market regime
        try:
            regime = self.detect_market_regime()
            signals['regime_signal'] = {
                'regime': regime.current_regime.value,
                'probability': regime.regime_probability,
                'implications': self._get_regime_implications(
                    target_symbol, regime.current_regime
                ),
            }
        except Exception as e:
            logger.debug(f"Regime detection failed: {e}")
        
        # Analyze unusual correlation changes
        for related in related_symbols:
            if related in self.returns_data:
                try:
                    corr_result = self.calculate_correlation(target_symbol, related)
                    if not corr_result.rolling_correlation.empty:
                        # Check if recent correlation deviates from historical
                        recent_corr = corr_result.rolling_correlation.tail(5).mean()
                        hist_corr = corr_result.rolling_correlation.mean()
                        
                        if abs(recent_corr - hist_corr) > 0.2:
                            signals['correlation_signals'].append({
                                'related_asset': related,
                                'current_correlation': recent_corr,
                                'historical_correlation': hist_corr,
                                'change': 'strengthening' if recent_corr > hist_corr else 'weakening',
                                'implication': self._interpret_correlation_change(
                                    target_symbol, related, recent_corr, hist_corr
                                ),
                            })
                except Exception as e:
                    logger.debug(f"Correlation analysis failed for {related}: {e}")
        
        # Determine overall bias
        bullish_signals = 0
        bearish_signals = 0
        
        for ll_signal in signals['lead_lag_signals']:
            if ll_signal['predicted_direction'] == 'up':
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        if signals['regime_signal']:
            regime = signals['regime_signal']['regime']
            target_class = self.DEFAULT_UNIVERSE.get(target_symbol, AssetInfo("", "", AssetClass.EQUITY)).asset_class
            
            if regime == 'risk_on' and target_class in [AssetClass.EQUITY, AssetClass.CRYPTO]:
                bullish_signals += 1
            elif regime == 'risk_off' and target_class in [AssetClass.FIXED_INCOME, AssetClass.COMMODITY]:
                bullish_signals += 1
            elif regime == 'risk_off' and target_class in [AssetClass.EQUITY, AssetClass.CRYPTO]:
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            signals['overall_bias'] = 'bullish'
        elif bearish_signals > bullish_signals:
            signals['overall_bias'] = 'bearish'
        else:
            signals['overall_bias'] = 'neutral'
        
        return signals
    
    def _get_regime_implications(
        self,
        symbol: str,
        regime: MarketRegime,
    ) -> str:
        """Get implications of market regime for a symbol."""
        asset_info = self.DEFAULT_UNIVERSE.get(symbol)
        
        if asset_info is None:
            return f"Unknown asset class - regime is {regime.value}"
        
        implications = {
            (AssetClass.EQUITY, MarketRegime.RISK_ON): "Favorable - equities typically outperform in risk-on",
            (AssetClass.EQUITY, MarketRegime.RISK_OFF): "Unfavorable - consider reducing exposure",
            (AssetClass.FIXED_INCOME, MarketRegime.RISK_ON): "Unfavorable - yields may rise",
            (AssetClass.FIXED_INCOME, MarketRegime.RISK_OFF): "Favorable - flight to safety",
            (AssetClass.COMMODITY, MarketRegime.RISK_ON): "Mixed - depends on demand",
            (AssetClass.COMMODITY, MarketRegime.RISK_OFF): "Gold favorable, others mixed",
            (AssetClass.CURRENCY, MarketRegime.RISK_ON): "USD may weaken vs risk currencies",
            (AssetClass.CURRENCY, MarketRegime.RISK_OFF): "USD typically strengthens",
            (AssetClass.CRYPTO, MarketRegime.RISK_ON): "Favorable - risk assets rally",
            (AssetClass.CRYPTO, MarketRegime.RISK_OFF): "Unfavorable - high beta to risk",
            (AssetClass.VOLATILITY, MarketRegime.RISK_ON): "Unfavorable - VIX typically falls",
            (AssetClass.VOLATILITY, MarketRegime.RISK_OFF): "Favorable - VIX typically rises",
            (AssetClass.REAL_ESTATE, MarketRegime.RISK_ON): "Favorable - REITs benefit",
            (AssetClass.REAL_ESTATE, MarketRegime.RISK_OFF): "Mixed - depends on rates",
        }
        
        key = (asset_info.asset_class, regime)
        return implications.get(key, f"Regime is {regime.value}")
    
    def _interpret_correlation_change(
        self,
        target: str,
        related: str,
        current: float,
        historical: float,
    ) -> str:
        """Interpret correlation changes."""
        change = current - historical
        
        if abs(change) < 0.1:
            return "No significant change in relationship"
        
        if change > 0:
            if current > 0:
                return f"Strengthening positive correlation with {related} - may move together more"
            else:
                return f"Correlation becoming less negative - hedge effectiveness reducing"
        else:
            if current < 0:
                return f"Becoming more negatively correlated - better hedge potential"
            else:
                return f"Weakening positive correlation - increasing diversification benefit"


def create_cross_asset_report(
    symbols: list[str],
    fetcher=None,
) -> dict:
    """
    Create a comprehensive cross-asset analysis report.
    
    Args:
        symbols: List of symbols to analyze
        fetcher: Optional DataFetcher instance
        
    Returns:
        Dictionary with complete cross-asset analysis
    """
    analyzer = CrossAssetAnalyzer()
    
    # Load data
    analyzer.load_data(symbols, fetcher)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'symbols_analyzed': symbols,
        'correlation_matrix': None,
        'regime': None,
        'diversification': None,
        'key_relationships': [],
    }
    
    # Correlation matrix
    try:
        corr_matrix = analyzer.calculate_correlation_matrix()
        report['correlation_matrix'] = corr_matrix.to_dict()
    except Exception as e:
        logger.error(f"Failed to calculate correlation matrix: {e}")
    
    # Market regime
    try:
        regime = analyzer.detect_market_regime()
        report['regime'] = {
            'current': regime.current_regime.value,
            'probability': regime.regime_probability,
            'duration_days': regime.regime_duration_days,
            'indicators': regime.indicators,
        }
    except Exception as e:
        logger.error(f"Failed to detect market regime: {e}")
    
    # Diversification score
    try:
        div_score = analyzer.calculate_diversification_score(symbols)
        report['diversification'] = {
            'score': div_score.overall_score,
            'clusters': div_score.cluster_assignments,
            'recommendations': div_score.recommendations,
        }
    except Exception as e:
        logger.error(f"Failed to calculate diversification: {e}")
    
    # Key relationships
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            try:
                corr = analyzer.calculate_correlation(sym1, sym2)
                if abs(corr.correlation) > 0.5:
                    report['key_relationships'].append({
                        'asset1': sym1,
                        'asset2': sym2,
                        'correlation': corr.correlation,
                        'strength': corr.relationship_strength,
                        'lead_lag_days': corr.lead_lag_days,
                    })
            except Exception:
                pass
    
    return report
