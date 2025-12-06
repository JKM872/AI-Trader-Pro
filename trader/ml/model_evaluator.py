"""
Model Evaluator - Evaluates ML model performance on trading data.

Features:
- Walk-forward validation
- Trading-specific metrics
- Strategy backtesting
- Statistical significance tests
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for model evaluation."""
    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Regression metrics
    mse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    
    # Trading metrics
    directional_accuracy: float = 0.0  # % correct direction
    hit_rate: float = 0.0  # % profitable trades
    profit_factor: float = 0.0  # gross profit / gross loss
    
    # Information metrics
    information_coefficient: float = 0.0  # Correlation with actual returns
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'mae': self.mae,
            'r2': self.r2,
            'directional_accuracy': self.directional_accuracy,
            'hit_rate': self.hit_rate,
            'profit_factor': self.profit_factor,
            'information_coefficient': self.information_coefficient
        }


@dataclass
class BacktestResult:
    """Result from model backtest."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    
    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    
    # Comparison to buy-and-hold
    benchmark_return: float = 0.0
    alpha: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'n_trades': self.n_trades,
            'benchmark_return': self.benchmark_return,
            'alpha': self.alpha
        }


class ModelEvaluator:
    """
    Evaluates ML model performance for trading.
    
    Features:
    - Walk-forward validation
    - Trading-specific metrics
    - Statistical significance
    - Backtest simulation
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005  # 0.05% slippage
    ):
        """
        Initialize Model Evaluator.
        
        Args:
            transaction_cost: Cost per trade as fraction
            slippage: Slippage as fraction
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """
        Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            EvaluationMetrics
        """
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        except ImportError:
            logger.error("scikit-learn required for evaluation")
            return EvaluationMetrics()
        
        metrics = EvaluationMetrics()
        
        metrics.accuracy = accuracy_score(y_true, y_pred)
        
        # Handle binary/multiclass
        try:
            metrics.precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            metrics.f1_score = f1_score(y_true, y_pred, average='binary', zero_division=0)
        except ValueError:
            metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Directional accuracy for trading
        metrics.directional_accuracy = metrics.accuracy
        
        return metrics
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> EvaluationMetrics:
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            EvaluationMetrics
        """
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        except ImportError:
            logger.error("scikit-learn required for evaluation")
            return EvaluationMetrics()
        
        metrics = EvaluationMetrics()
        
        metrics.mse = mean_squared_error(y_true, y_pred)
        metrics.mae = mean_absolute_error(y_true, y_pred)
        metrics.r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        metrics.directional_accuracy = np.mean(true_direction == pred_direction)
        
        # Information coefficient (rank correlation)
        if len(y_true) > 2:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(y_true, y_pred)
            metrics.information_coefficient = ic if not np.isnan(ic) else 0.0
        
        return metrics
    
    def walk_forward_validation(
        self,
        predictor: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        train_size: float = 0.6,
        gap: int = 0
    ) -> list[EvaluationMetrics]:
        """
        Perform walk-forward validation.
        
        Args:
            predictor: ML predictor with fit/predict methods
            X: Feature matrix
            y: Target variable
            n_splits: Number of validation splits
            train_size: Fraction for training in each split
            gap: Gap between train and test (prevent leakage)
            
        Returns:
            List of EvaluationMetrics for each split
        """
        results = []
        n_samples = len(X)
        
        # Calculate split sizes
        test_size = (1 - train_size) / n_splits
        
        for i in range(n_splits):
            # Calculate indices
            train_end = int(n_samples * (train_size + i * test_size))
            test_start = train_end + gap
            test_end = int(n_samples * (train_size + (i + 1) * test_size))
            
            if test_start >= test_end or test_end > n_samples:
                continue
            
            # Split data
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Fit and predict
            predictor.fit(X_train, y_train)
            predictions = predictor.predict_batch(X_test)
            
            y_pred = np.array([p.value for p in predictions])
            
            # Evaluate
            if predictor.config.target_type == "direction":
                metrics = self.evaluate_classification(y_test.values, y_pred)
            else:
                metrics = self.evaluate_regression(y_test.values, y_pred)
            
            results.append(metrics)
        
        return results
    
    def backtest_strategy(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series,
        position_threshold: float = 0.0
    ) -> BacktestResult:
        """
        Backtest a strategy based on predictions.
        
        Args:
            predictions: Model predictions (positive = long, negative = short)
            actual_returns: Actual returns for the period
            position_threshold: Minimum prediction for position
            
        Returns:
            BacktestResult with performance metrics
        """
        # Align indices
        common_idx = predictions.index.intersection(actual_returns.index)
        predictions = predictions.loc[common_idx]
        actual_returns = actual_returns.loc[common_idx]
        
        # Generate positions
        positions = np.sign(predictions)
        positions[abs(predictions) < position_threshold] = 0
        
        # Calculate strategy returns
        strategy_returns = positions.shift(1) * actual_returns
        
        # Account for transaction costs
        position_changes = positions.diff().abs()
        costs = position_changes * (self.transaction_cost + self.slippage)
        strategy_returns = strategy_returns - costs
        
        # Remove NaN
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return BacktestResult(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                n_trades=0
            )
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        
        # Annualized return (assuming daily data)
        n_days = len(strategy_returns)
        annualized_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        
        # Sharpe ratio
        if strategy_returns.std() > 0:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Equity curve and drawdown
        equity_curve = (1 + strategy_returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (strategy_returns != 0).sum()
        win_rate = winning_trades / max(total_trades, 1)
        
        # Benchmark (buy and hold)
        benchmark_return = (1 + actual_returns).prod() - 1
        alpha = total_return - benchmark_return
        
        # Number of trades (position changes)
        n_trades = int(position_changes.sum() / 2)  # Divide by 2 for round trips
        
        return BacktestResult(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            n_trades=n_trades,
            equity_curve=equity_curve,
            returns=strategy_returns,
            benchmark_return=float(benchmark_return),
            alpha=float(alpha)
        )
    
    def statistical_significance(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        confidence_level: float = 0.95
    ) -> dict:
        """
        Test statistical significance of strategy outperformance.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            confidence_level: Confidence level for test
            
        Returns:
            Dict with test results
        """
        from scipy import stats
        
        # Align
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_idx]
        benchmark_returns = benchmark_returns.loc[common_idx]
        
        excess_returns = strategy_returns - benchmark_returns
        
        # T-test for mean excess return
        t_stat, p_value = stats.ttest_1samp(excess_returns.dropna(), 0)
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = excess_returns.sample(frac=1, replace=True)
            bootstrap_means.append(sample.mean())
        
        bootstrap_means = np.array(bootstrap_means)
        ci_lower = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)
        
        is_significant = p_value < (1 - confidence_level) and ci_lower > 0
        
        return {
            'mean_excess_return': float(excess_returns.mean()),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'is_significant': is_significant,
            'confidence_level': confidence_level
        }
    
    def generate_report(
        self,
        metrics: EvaluationMetrics,
        backtest: BacktestResult,
        significance: Optional[dict] = None
    ) -> str:
        """Generate a text report of model evaluation."""
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        report.append("\n📊 PREDICTION METRICS:")
        report.append(f"  Directional Accuracy: {metrics.directional_accuracy:.2%}")
        if metrics.accuracy > 0:
            report.append(f"  Classification Accuracy: {metrics.accuracy:.2%}")
            report.append(f"  Precision: {metrics.precision:.2%}")
            report.append(f"  Recall: {metrics.recall:.2%}")
            report.append(f"  F1 Score: {metrics.f1_score:.2%}")
        if metrics.r2 != 0:
            report.append(f"  R² Score: {metrics.r2:.4f}")
            report.append(f"  MAE: {metrics.mae:.4f}")
        if metrics.information_coefficient != 0:
            report.append(f"  Information Coefficient: {metrics.information_coefficient:.4f}")
        
        report.append("\n💰 BACKTEST RESULTS:")
        report.append(f"  Total Return: {backtest.total_return:.2%}")
        report.append(f"  Annualized Return: {backtest.annualized_return:.2%}")
        report.append(f"  Sharpe Ratio: {backtest.sharpe_ratio:.2f}")
        report.append(f"  Max Drawdown: {backtest.max_drawdown:.2%}")
        report.append(f"  Win Rate: {backtest.win_rate:.2%}")
        report.append(f"  Number of Trades: {backtest.n_trades}")
        report.append(f"  Benchmark Return: {backtest.benchmark_return:.2%}")
        report.append(f"  Alpha: {backtest.alpha:.2%}")
        
        if significance:
            report.append("\n📈 STATISTICAL SIGNIFICANCE:")
            report.append(f"  Mean Excess Return: {significance['mean_excess_return']:.4%}")
            report.append(f"  T-Statistic: {significance['t_statistic']:.2f}")
            report.append(f"  P-Value: {significance['p_value']:.4f}")
            report.append(f"  {significance['confidence_level']:.0%} CI: "
                         f"[{significance['ci_lower']:.4%}, {significance['ci_upper']:.4%}]")
            status = "✅ SIGNIFICANT" if significance['is_significant'] else "❌ NOT SIGNIFICANT"
            report.append(f"  Result: {status}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
