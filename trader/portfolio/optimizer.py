"""
Advanced Portfolio Optimization Module.

Implements modern portfolio theory and beyond:
- Mean-Variance Optimization (Markowitz)
- Black-Litterman Model
- Risk Parity
- Maximum Diversification
- Hierarchical Risk Parity (HRP)
- Factor-based optimization
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "minimum_variance"
    MAX_SHARPE = "maximum_sharpe"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hrp"
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"


class RiskMeasure(Enum):
    """Risk measures for optimization."""
    VARIANCE = "variance"
    SEMI_VARIANCE = "semi_variance"
    CVAR = "cvar"  # Conditional Value at Risk
    MAX_DRAWDOWN = "max_drawdown"


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    min_assets: int = 1  # Minimum number of assets
    max_assets: Optional[int] = None  # Maximum number of assets
    sector_constraints: dict = field(default_factory=dict)  # Sector limits
    factor_constraints: dict = field(default_factory=dict)  # Factor exposure limits
    turnover_limit: Optional[float] = None  # Maximum turnover from current
    long_only: bool = True  # Allow short positions


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    method: OptimizationMethod
    diversification_ratio: float = 1.0
    effective_n: float = 1.0  # Effective number of assets
    risk_contributions: dict = field(default_factory=dict)
    optimization_success: bool = True
    message: str = ""


@dataclass
class BlackLittermanViews:
    """Views for Black-Litterman model."""
    view_matrix: np.ndarray  # P matrix - picking matrix
    view_returns: np.ndarray  # Q vector - expected returns from views
    view_confidence: np.ndarray  # Omega - confidence in views
    
    @classmethod
    def from_absolute_views(
        cls,
        symbols: list[str],
        views: dict[str, float],
        confidences: dict[str, float],
    ) -> "BlackLittermanViews":
        """
        Create views from absolute return expectations.
        
        Args:
            symbols: List of all symbols in universe
            views: Dict of symbol -> expected return
            confidences: Dict of symbol -> confidence (0-1)
            
        Returns:
            BlackLittermanViews instance
        """
        n_views = len(views)
        n_assets = len(symbols)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)
        
        for i, (symbol, expected_return) in enumerate(views.items()):
            if symbol in symbols:
                asset_idx = symbols.index(symbol)
                P[i, asset_idx] = 1.0
                Q[i] = expected_return
                # Lower confidence = higher uncertainty (variance)
                conf = confidences.get(symbol, 0.5)
                omega_diag[i] = (1 - conf) * 0.1  # Scale uncertainty
        
        Omega = np.diag(omega_diag)
        
        return cls(
            view_matrix=P,
            view_returns=Q,
            view_confidence=Omega,
        )
    
    @classmethod
    def from_relative_views(
        cls,
        symbols: list[str],
        outperformers: list[tuple[str, str, float, float]],
    ) -> "BlackLittermanViews":
        """
        Create views from relative return expectations.
        
        Args:
            symbols: List of all symbols
            outperformers: List of (winner, loser, outperformance, confidence)
            
        Returns:
            BlackLittermanViews instance
        """
        n_views = len(outperformers)
        n_assets = len(symbols)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega_diag = np.zeros(n_views)
        
        for i, (winner, loser, outperformance, confidence) in enumerate(outperformers):
            if winner in symbols and loser in symbols:
                winner_idx = symbols.index(winner)
                loser_idx = symbols.index(loser)
                P[i, winner_idx] = 1.0
                P[i, loser_idx] = -1.0
                Q[i] = outperformance
                omega_diag[i] = (1 - confidence) * 0.1
        
        Omega = np.diag(omega_diag)
        
        return cls(
            view_matrix=P,
            view_returns=Q,
            view_confidence=Omega,
        )


class PortfolioOptimizer:
    """
    Advanced portfolio optimization engine.
    
    Supports multiple optimization methods:
    - Mean-Variance (Markowitz)
    - Minimum Variance
    - Maximum Sharpe Ratio
    - Risk Parity
    - Maximum Diversification
    - Black-Litterman
    - Hierarchical Risk Parity
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
        min_history_days: int = 60,
    ):
        """
        Initialize Portfolio Optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            trading_days: Trading days per year
            min_history_days: Minimum days of history required
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.min_history_days = min_history_days
        
        # Data storage
        self.returns: Optional[pd.DataFrame] = None
        self.symbols: list[str] = []
        self.mean_returns: Optional[pd.Series] = None
        self.cov_matrix: Optional[pd.DataFrame] = None
        
    def set_returns(self, returns: pd.DataFrame) -> None:
        """
        Set return data for optimization.
        
        Args:
            returns: DataFrame with asset returns (columns = assets)
        """
        self.returns = returns.dropna()
        self.symbols = returns.columns.tolist()
        self.mean_returns = returns.mean() * self.trading_days
        self.cov_matrix = returns.cov() * self.trading_days
        
    def load_data(
        self,
        symbols: list[str],
        period: str = "2y",
        fetcher=None,
    ) -> pd.DataFrame:
        """
        Load price data and calculate returns.
        
        Args:
            symbols: List of symbols to load
            period: Data period
            fetcher: Optional DataFetcher instance
            
        Returns:
            DataFrame of returns
        """
        if fetcher is None:
            try:
                from trader.data.fetcher import DataFetcher
                fetcher = DataFetcher()
            except ImportError:
                raise ImportError("DataFetcher required for data loading")
        
        prices = pd.DataFrame()
        
        for symbol in symbols:
            try:
                df = fetcher.get_stock_data(symbol, period=period)
                if df is not None and not df.empty:
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    prices[symbol] = df[close_col]
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        
        if prices.empty:
            raise ValueError("No price data loaded")
        
        returns = prices.pct_change().dropna()
        self.set_returns(returns)
        
        return returns
    
    def optimize(
        self,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        constraints: Optional[OptimizationConstraints] = None,
        views: Optional[BlackLittermanViews] = None,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.
        
        Args:
            method: Optimization method to use
            constraints: Optimization constraints
            views: Views for Black-Litterman (required if using BL)
            target_return: Target return for efficient frontier
            target_risk: Target risk for efficient frontier
            
        Returns:
            OptimizationResult with optimal weights
        """
        if self.returns is None:
            raise ValueError("No return data set. Call set_returns() first.")
        
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Dispatch to appropriate method
        method_map = {
            OptimizationMethod.MEAN_VARIANCE: self._optimize_mean_variance,
            OptimizationMethod.MIN_VARIANCE: self._optimize_min_variance,
            OptimizationMethod.MAX_SHARPE: self._optimize_max_sharpe,
            OptimizationMethod.RISK_PARITY: self._optimize_risk_parity,
            OptimizationMethod.MAX_DIVERSIFICATION: self._optimize_max_diversification,
            OptimizationMethod.BLACK_LITTERMAN: lambda c: self._optimize_black_litterman(c, views),
            OptimizationMethod.HIERARCHICAL_RISK_PARITY: self._optimize_hrp,
            OptimizationMethod.EQUAL_WEIGHT: self._optimize_equal_weight,
            OptimizationMethod.INVERSE_VOLATILITY: self._optimize_inverse_volatility,
        }
        
        optimizer_func = method_map.get(method)
        if optimizer_func is None:
            raise ValueError(f"Unknown optimization method: {method}")
        
        result = optimizer_func(constraints)
        result.method = method
        
        # Calculate additional metrics
        weights_array = np.array([result.weights.get(s, 0) for s in self.symbols])
        result.diversification_ratio = self._calculate_diversification_ratio(weights_array)
        result.effective_n = self._calculate_effective_n(weights_array)
        result.risk_contributions = self._calculate_risk_contributions(weights_array)
        
        return result
    
    def _optimize_mean_variance(
        self,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """Mean-variance optimization with target return."""
        n_assets = len(self.symbols)
        
        # Initial weights
        init_weights = np.ones(n_assets) / n_assets
        
        # Bounds
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight)
            for _ in range(n_assets)
        )
        
        # Constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Objective: minimize variance
        def objective(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
        
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
        )
        
        weights = result.x
        weights_dict = {s: w for s, w in zip(self.symbols, weights)}
        
        exp_return = float(np.dot(weights, self.mean_returns.values))
        exp_risk = float(np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights))))
        sharpe = (exp_return - self.risk_free_rate) / exp_risk if exp_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_return,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            method=OptimizationMethod.MEAN_VARIANCE,
            optimization_success=result.success,
            message=result.message,
        )
    
    def _optimize_min_variance(
        self,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """Minimum variance portfolio."""
        return self._optimize_mean_variance(constraints)  # Same as MV without return target
    
    def _optimize_max_sharpe(
        self,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """Maximum Sharpe ratio portfolio."""
        n_assets = len(self.symbols)
        init_weights = np.ones(n_assets) / n_assets
        
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight)
            for _ in range(n_assets)
        )
        
        constraint_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Objective: negative Sharpe ratio (minimize)
        def neg_sharpe(weights):
            port_return = np.dot(weights, self.mean_returns.values)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights)))
            if port_vol <= 0:
                return 1e10
            return -(port_return - self.risk_free_rate) / port_vol
        
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
        )
        
        weights = result.x
        weights_dict = {s: w for s, w in zip(self.symbols, weights)}
        
        exp_return = float(np.dot(weights, self.mean_returns.values))
        exp_risk = float(np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights))))
        sharpe = (exp_return - self.risk_free_rate) / exp_risk if exp_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_return,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            method=OptimizationMethod.MAX_SHARPE,
            optimization_success=result.success,
            message=result.message,
        )
    
    def _optimize_risk_parity(
        self,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """
        Risk parity portfolio - equal risk contribution from each asset.
        """
        n_assets = len(self.symbols)
        init_weights = np.ones(n_assets) / n_assets
        
        cov = self.cov_matrix.values
        
        def risk_parity_objective(weights):
            """Minimize deviation from equal risk contribution."""
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            if port_vol <= 0:
                return 1e10
            
            # Marginal risk contribution
            marginal_contrib = np.dot(cov, weights) / port_vol
            # Risk contribution
            risk_contrib = weights * marginal_contrib
            # Target: equal contribution
            target_contrib = port_vol / n_assets
            
            # Sum of squared deviations
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        bounds = tuple(
            (max(0.001, constraints.min_weight), constraints.max_weight)
            for _ in range(n_assets)
        )
        
        constraint_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        result = minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
        )
        
        weights = result.x
        weights_dict = {s: w for s, w in zip(self.symbols, weights)}
        
        exp_return = float(np.dot(weights, self.mean_returns.values))
        exp_risk = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        sharpe = (exp_return - self.risk_free_rate) / exp_risk if exp_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_return,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            method=OptimizationMethod.RISK_PARITY,
            optimization_success=result.success,
            message=result.message,
        )
    
    def _optimize_max_diversification(
        self,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """
        Maximum diversification portfolio.
        Maximizes the diversification ratio: weighted avg volatility / portfolio volatility
        """
        n_assets = len(self.symbols)
        init_weights = np.ones(n_assets) / n_assets
        
        cov = self.cov_matrix.values
        std_devs = np.sqrt(np.diag(cov))
        
        def neg_diversification_ratio(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            weighted_avg_vol = np.dot(weights, std_devs)
            if port_vol <= 0:
                return 0
            return -weighted_avg_vol / port_vol
        
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight)
            for _ in range(n_assets)
        )
        
        constraint_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        result = minimize(
            neg_diversification_ratio,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
        )
        
        weights = result.x
        weights_dict = {s: w for s, w in zip(self.symbols, weights)}
        
        exp_return = float(np.dot(weights, self.mean_returns.values))
        exp_risk = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        sharpe = (exp_return - self.risk_free_rate) / exp_risk if exp_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_return,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            method=OptimizationMethod.MAX_DIVERSIFICATION,
            optimization_success=result.success,
            message=result.message,
        )
    
    def _optimize_black_litterman(
        self,
        constraints: OptimizationConstraints,
        views: Optional[BlackLittermanViews],
    ) -> OptimizationResult:
        """
        Black-Litterman model optimization.
        
        Combines market equilibrium returns with investor views.
        """
        if views is None:
            logger.warning("No views provided, falling back to max Sharpe")
            return self._optimize_max_sharpe(constraints)
        
        n_assets = len(self.symbols)
        cov = self.cov_matrix.values
        
        # Market capitalization weights (approximate with equal weights if not available)
        mkt_weights = np.ones(n_assets) / n_assets
        
        # Risk aversion parameter
        delta = 2.5
        
        # Implied equilibrium returns (Pi)
        pi = delta * np.dot(cov, mkt_weights)
        
        # Scaling factor for uncertainty in prior
        tau = 0.05
        
        # Black-Litterman formula
        P = views.view_matrix
        Q = views.view_returns
        Omega = views.view_confidence
        
        # Check dimensions
        if P.shape[1] != n_assets:
            logger.warning("View matrix dimensions don't match assets")
            return self._optimize_max_sharpe(constraints)
        
        # Combined return estimate
        # M = tau * Sigma
        M = tau * cov
        
        # BL posterior expected returns
        try:
            inv_omega = np.linalg.inv(Omega + 1e-6 * np.eye(Omega.shape[0]))
            inv_m = np.linalg.inv(M + 1e-6 * np.eye(M.shape[0]))
            
            bl_cov = np.linalg.inv(inv_m + np.dot(P.T, np.dot(inv_omega, P)))
            bl_returns = np.dot(bl_cov, np.dot(inv_m, pi) + np.dot(P.T, np.dot(inv_omega, Q)))
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed, using simple adjustment")
            bl_returns = pi + 0.5 * (Q.mean() - pi.mean())
            bl_cov = cov
        
        # Optimize with BL returns
        init_weights = np.ones(n_assets) / n_assets
        
        def neg_sharpe_bl(weights):
            port_return = np.dot(weights, bl_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(bl_cov, weights)))
            if port_vol <= 0:
                return 1e10
            return -(port_return - self.risk_free_rate) / port_vol
        
        bounds = tuple(
            (constraints.min_weight, constraints.max_weight)
            for _ in range(n_assets)
        )
        
        constraint_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        result = minimize(
            neg_sharpe_bl,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
        )
        
        weights = result.x
        weights_dict = {s: w for s, w in zip(self.symbols, weights)}
        
        exp_return = float(np.dot(weights, bl_returns))
        exp_risk = float(np.sqrt(np.dot(weights.T, np.dot(bl_cov, weights))))
        sharpe = (exp_return - self.risk_free_rate) / exp_risk if exp_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_return,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            method=OptimizationMethod.BLACK_LITTERMAN,
            optimization_success=result.success,
            message=result.message,
        )
    
    def _optimize_hrp(
        self,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """
        Hierarchical Risk Parity (HRP) optimization.
        
        Uses hierarchical clustering to determine portfolio weights.
        More robust than traditional optimization, no matrix inversion.
        """
        returns = self.returns
        cov = self.cov_matrix.values
        
        # Step 1: Tree clustering based on correlation distance
        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr))
        
        # Convert to condensed distance matrix
        dist_condensed = squareform(dist.values, checks=False)
        
        # Hierarchical clustering
        link = linkage(dist_condensed, method='single')
        
        # Step 2: Quasi-diagonalization
        sorted_idx = self._get_quasi_diag(link)
        sorted_symbols = [self.symbols[i] for i in sorted_idx]
        
        # Step 3: Recursive bisection
        weights = self._recursive_bisection(cov, sorted_idx)
        
        # Create weights dict
        weights_dict = {self.symbols[i]: w for i, w in enumerate(weights)}
        
        # Apply constraints (simple clipping)
        for symbol in weights_dict:
            weights_dict[symbol] = max(
                constraints.min_weight,
                min(constraints.max_weight, weights_dict[symbol])
            )
        
        # Renormalize
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {s: w/total for s, w in weights_dict.items()}
        
        weights_array = np.array([weights_dict.get(s, 0) for s in self.symbols])
        exp_return = float(np.dot(weights_array, self.mean_returns.values))
        exp_risk = float(np.sqrt(np.dot(weights_array.T, np.dot(cov, weights_array))))
        sharpe = (exp_return - self.risk_free_rate) / exp_risk if exp_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_return,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            method=OptimizationMethod.HIERARCHICAL_RISK_PARITY,
            optimization_success=True,
            message="HRP optimization complete",
        )
    
    def _get_quasi_diag(self, link: np.ndarray) -> list[int]:
        """Get quasi-diagonal order from linkage matrix."""
        link = link.astype(int)
        sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_idx.max() >= num_items:
            sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
            df0 = sort_idx[sort_idx >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_idx[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_idx = pd.concat([sort_idx, df0])
            sort_idx = sort_idx.sort_index()
            sort_idx.index = range(sort_idx.shape[0])
        
        return sort_idx.tolist()
    
    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sorted_idx: list[int],
    ) -> np.ndarray:
        """Recursive bisection for HRP weights."""
        n = len(sorted_idx)
        weights = np.ones(len(self.symbols))
        clusters = [sorted_idx]
        
        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                
                # Split cluster
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]
                
                # Calculate cluster variances
                left_var = self._get_cluster_variance(cov, left)
                right_var = self._get_cluster_variance(cov, right)
                
                # Allocate based on inverse variance
                total_var = left_var + right_var
                if total_var > 0:
                    left_weight = 1 - left_var / total_var
                    right_weight = 1 - right_var / total_var
                else:
                    left_weight = right_weight = 0.5
                
                # Scale weights
                for i in left:
                    weights[i] *= left_weight
                for i in right:
                    weights[i] *= right_weight
                
                new_clusters.extend([left, right])
            
            clusters = [c for c in new_clusters if len(c) > 1]
        
        return weights / weights.sum()
    
    def _get_cluster_variance(
        self,
        cov: np.ndarray,
        cluster_idx: list[int],
    ) -> float:
        """Calculate variance of a cluster using inverse-variance weights."""
        cluster_cov = cov[np.ix_(cluster_idx, cluster_idx)]
        variances = np.diag(cluster_cov)
        
        # Inverse variance weights within cluster
        inv_var = 1 / (variances + 1e-10)
        weights = inv_var / inv_var.sum()
        
        cluster_var = np.dot(weights.T, np.dot(cluster_cov, weights))
        return cluster_var
    
    def _optimize_equal_weight(
        self,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """Equal weight portfolio (1/N)."""
        n_assets = len(self.symbols)
        weights = np.ones(n_assets) / n_assets
        weights_dict = {s: w for s, w in zip(self.symbols, weights)}
        
        cov = self.cov_matrix.values
        exp_return = float(np.dot(weights, self.mean_returns.values))
        exp_risk = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        sharpe = (exp_return - self.risk_free_rate) / exp_risk if exp_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_return,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            method=OptimizationMethod.EQUAL_WEIGHT,
            optimization_success=True,
            message="Equal weight allocation",
        )
    
    def _optimize_inverse_volatility(
        self,
        constraints: OptimizationConstraints,
    ) -> OptimizationResult:
        """Inverse volatility weighting."""
        cov = self.cov_matrix.values
        volatilities = np.sqrt(np.diag(cov))
        
        # Inverse volatility weights
        inv_vol = 1 / (volatilities + 1e-10)
        weights = inv_vol / inv_vol.sum()
        
        # Apply constraints
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        weights = weights / weights.sum()
        
        weights_dict = {s: w for s, w in zip(self.symbols, weights)}
        
        exp_return = float(np.dot(weights, self.mean_returns.values))
        exp_risk = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        sharpe = (exp_return - self.risk_free_rate) / exp_risk if exp_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=exp_return,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            method=OptimizationMethod.INVERSE_VOLATILITY,
            optimization_success=True,
            message="Inverse volatility weighting",
        )
    
    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """Calculate portfolio diversification ratio."""
        cov = self.cov_matrix.values
        std_devs = np.sqrt(np.diag(cov))
        
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        weighted_avg_vol = np.dot(weights, std_devs)
        
        if port_vol <= 0:
            return 1.0
        return weighted_avg_vol / port_vol
    
    def _calculate_effective_n(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets (1 / sum of squared weights)."""
        weights_squared = weights ** 2
        sum_squared = weights_squared.sum()
        
        if sum_squared <= 0:
            return 1.0
        return 1 / sum_squared
    
    def _calculate_risk_contributions(self, weights: np.ndarray) -> dict[str, float]:
        """Calculate risk contribution from each asset."""
        cov = self.cov_matrix.values
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        
        if port_vol <= 0:
            return {s: 0 for s in self.symbols}
        
        # Marginal risk contribution
        marginal_contrib = np.dot(cov, weights) / port_vol
        
        # Risk contribution = weight * marginal contribution
        risk_contrib = weights * marginal_contrib
        
        # Normalize to percentage
        total_contrib = risk_contrib.sum()
        if total_contrib > 0:
            risk_contrib = risk_contrib / total_contrib
        
        return {s: float(rc) for s, rc in zip(self.symbols, risk_contrib)}
    
    def efficient_frontier(
        self,
        n_points: int = 50,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> pd.DataFrame:
        """
        Calculate the efficient frontier.
        
        Args:
            n_points: Number of points on the frontier
            constraints: Optimization constraints
            
        Returns:
            DataFrame with returns, risks, and weights for each point
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Get min and max return portfolios
        min_var = self._optimize_min_variance(constraints)
        max_sharpe = self._optimize_max_sharpe(constraints)
        
        # Define return range
        min_return = min_var.expected_return
        max_return = max_sharpe.expected_return * 1.2  # Allow some buffer
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_data = []
        n_assets = len(self.symbols)
        
        for target_ret in target_returns:
            # Optimize for minimum variance at target return
            init_weights = np.ones(n_assets) / n_assets
            
            bounds = tuple(
                (constraints.min_weight, constraints.max_weight)
                for _ in range(n_assets)
            )
            
            def objective(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
            
            constraint_list = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, tr=target_ret: np.dot(w, self.mean_returns.values) - tr},
            ]
            
            try:
                result = minimize(
                    objective,
                    init_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraint_list,
                )
                
                if result.success:
                    weights = result.x
                    risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights)))
                    ret = np.dot(weights, self.mean_returns.values)
                    sharpe = (ret - self.risk_free_rate) / risk if risk > 0 else 0
                    
                    point = {
                        'return': ret,
                        'risk': risk,
                        'sharpe': sharpe,
                    }
                    point.update({s: w for s, w in zip(self.symbols, weights)})
                    frontier_data.append(point)
            except Exception:
                continue
        
        return pd.DataFrame(frontier_data)
    
    def compare_methods(
        self,
        methods: Optional[list[OptimizationMethod]] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> pd.DataFrame:
        """
        Compare different optimization methods.
        
        Args:
            methods: List of methods to compare (all if None)
            constraints: Optimization constraints
            
        Returns:
            DataFrame comparing method results
        """
        if methods is None:
            methods = [
                OptimizationMethod.MAX_SHARPE,
                OptimizationMethod.MIN_VARIANCE,
                OptimizationMethod.RISK_PARITY,
                OptimizationMethod.MAX_DIVERSIFICATION,
                OptimizationMethod.HIERARCHICAL_RISK_PARITY,
                OptimizationMethod.EQUAL_WEIGHT,
                OptimizationMethod.INVERSE_VOLATILITY,
            ]
        
        results = []
        
        for method in methods:
            try:
                result = self.optimize(method=method, constraints=constraints)
                results.append({
                    'method': method.value,
                    'expected_return': result.expected_return,
                    'expected_risk': result.expected_risk,
                    'sharpe_ratio': result.sharpe_ratio,
                    'diversification_ratio': result.diversification_ratio,
                    'effective_n': result.effective_n,
                    'success': result.optimization_success,
                })
            except Exception as e:
                logger.error(f"Method {method.value} failed: {e}")
                results.append({
                    'method': method.value,
                    'expected_return': np.nan,
                    'expected_risk': np.nan,
                    'sharpe_ratio': np.nan,
                    'diversification_ratio': np.nan,
                    'effective_n': np.nan,
                    'success': False,
                })
        
        return pd.DataFrame(results)


def create_optimal_portfolio(
    symbols: list[str],
    method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
    views: Optional[dict[str, float]] = None,
    confidences: Optional[dict[str, float]] = None,
    fetcher=None,
) -> OptimizationResult:
    """
    Convenience function to create an optimal portfolio.
    
    Args:
        symbols: List of symbols to include
        method: Optimization method
        views: Optional views for Black-Litterman {symbol: expected_return}
        confidences: Optional confidences for views {symbol: confidence}
        fetcher: Optional DataFetcher
        
    Returns:
        OptimizationResult with optimal weights
    """
    optimizer = PortfolioOptimizer()
    optimizer.load_data(symbols, fetcher=fetcher)
    
    bl_views = None
    if views and method == OptimizationMethod.BLACK_LITTERMAN:
        if confidences is None:
            confidences = {s: 0.5 for s in views}
        bl_views = BlackLittermanViews.from_absolute_views(
            symbols, views, confidences
        )
    
    return optimizer.optimize(method=method, views=bl_views)
