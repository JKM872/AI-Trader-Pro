"""
Price Predictor - ML models for price prediction.

Supports:
- Gradient Boosting (XGBoost/LightGBM style using sklearn)
- Random Forest
- Linear models (Ridge, Lasso)
- Simple Neural Network (MLP)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types."""
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    RIDGE = "ridge"
    LASSO = "lasso"
    MLP = "mlp"
    ENSEMBLE = "ensemble"


@dataclass
class PredictorConfig:
    """Configuration for price predictor."""
    model_type: ModelType = ModelType.GRADIENT_BOOSTING
    target_type: str = "direction"  # "returns", "direction", "volatility"
    
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    min_samples_leaf: int = 20
    
    # Training config
    train_size: float = 0.8
    n_splits: int = 5  # For time series CV
    random_state: int = 42
    
    # Feature selection
    select_top_features: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'model_type': self.model_type.value,
            'target_type': self.target_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'train_size': self.train_size
        }


@dataclass
class Prediction:
    """A prediction from the model."""
    value: float  # Predicted value
    probability: Optional[float] = None  # For classification
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # For returns prediction
    expected_return: Optional[float] = None
    
    # For direction prediction
    direction: Optional[str] = None  # "up", "down", "neutral"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'value': self.value,
            'probability': self.probability,
            'confidence': self.confidence,
            'direction': self.direction,
            'expected_return': self.expected_return
        }


class PricePredictor:
    """
    ML-based price predictor.
    
    Features:
    - Multiple model types
    - Time series cross-validation
    - Feature importance analysis
    - Model persistence
    """
    
    def __init__(self, config: Optional[PredictorConfig] = None):
        """
        Initialize Price Predictor.
        
        Args:
            config: Predictor configuration
        """
        self.config = config or PredictorConfig()
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []
        self.feature_importances: dict[str, float] = {}
        self.is_fitted = False
        self.metrics: dict = {}
    
    def _create_model(self) -> Any:
        """Create the ML model based on config."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import Ridge, Lasso
            from sklearn.neural_network import MLPClassifier, MLPRegressor
        except ImportError:
            logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
            raise
        
        is_classifier = self.config.target_type == "direction"
        
        if self.config.model_type == ModelType.GRADIENT_BOOSTING:
            if is_classifier:
                return GradientBoostingClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state
                )
            else:
                return GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state
                )
        
        elif self.config.model_type == ModelType.RANDOM_FOREST:
            if is_classifier:
                return RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state
                )
            else:
                return RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state
                )
        
        elif self.config.model_type == ModelType.RIDGE:
            if is_classifier:
                from sklearn.linear_model import RidgeClassifier
                return RidgeClassifier(random_state=self.config.random_state)
            else:
                return Ridge(random_state=self.config.random_state)
        
        elif self.config.model_type == ModelType.LASSO:
            if is_classifier:
                # Use logistic regression for classification
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(
                    penalty='l1', solver='saga',
                    random_state=self.config.random_state
                )
            else:
                return Lasso(random_state=self.config.random_state)
        
        elif self.config.model_type == ModelType.MLP:
            hidden_layers = (100, 50, 25)
            if is_classifier:
                return MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=500,
                    random_state=self.config.random_state
                )
            else:
                return MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    max_iter=500,
                    random_state=self.config.random_state
                )
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features using StandardScaler."""
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return X
        
        if fit:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        elif self.scaler is not None:
            return self.scaler.transform(X)
        return X
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[list[str]] = None
    ) -> dict:
        """
        Fit the model.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Optional feature names
            
        Returns:
            Dict of training metrics
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        except ImportError:
            logger.error("scikit-learn not installed")
            raise
        
        self.feature_names = feature_names or list(X.columns)
        
        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Handle NaN
        valid_mask = ~(np.isnan(X_array).any(axis=1) | np.isnan(y_array))
        X_array = X_array[valid_mask]
        y_array = y_array[valid_mask]
        
        if len(X_array) < 100:
            logger.warning(f"Very small dataset: {len(X_array)} samples")
        
        # Scale features
        X_scaled = self._scale_features(X_array, fit=True)
        
        # Create model
        self.model = self._create_model()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        if self.config.target_type == "direction":
            scoring = 'accuracy'
        else:
            scoring = 'neg_mean_squared_error'
        
        cv_scores = cross_val_score(self.model, X_scaled, y_array, cv=tscv, scoring=scoring)
        
        # Fit on full data
        self.model.fit(X_scaled, y_array)
        self.is_fitted = True
        
        # Extract feature importances
        self._extract_feature_importances()
        
        # Store metrics
        self.metrics = {
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'cv_scores': cv_scores.tolist(),
            'n_samples': len(X_array),
            'n_features': X_array.shape[1]
        }
        
        logger.info(
            f"Model trained: {self.config.model_type.value}, "
            f"CV score: {self.metrics['cv_mean']:.4f} (+/- {self.metrics['cv_std']:.4f})"
        )
        
        return self.metrics
    
    def _extract_feature_importances(self) -> None:
        """Extract feature importances from model."""
        if not hasattr(self.model, 'feature_importances_') and not hasattr(self.model, 'coef_'):
            return
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            importances = np.abs(self.model.coef_).flatten()
            if len(importances) != len(self.feature_names):
                return
        
        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        
        self.feature_importances = dict(zip(self.feature_names, importances))
    
    def predict(self, X: pd.DataFrame) -> Prediction:
        """
        Make a prediction.
        
        Args:
            X: Feature matrix (single row or multiple rows)
            
        Returns:
            Prediction object
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Ensure 2D
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(1, -1)
        
        # Scale
        X_scaled = self._scale_features(X_array, fit=False)
        
        # Predict
        raw_pred = self.model.predict(X_scaled)
        
        # Get probability for classifiers
        probability = None
        if hasattr(self.model, 'predict_proba') and self.config.target_type == "direction":
            probs = self.model.predict_proba(X_scaled)
            probability = float(np.max(probs[-1]))
        
        # Use last prediction if multiple rows
        value = float(raw_pred[-1])
        
        # Determine direction and confidence
        direction = None
        confidence = 0.5
        expected_return = None
        
        if self.config.target_type == "direction":
            direction = "up" if value == 1 else "down"
            confidence = probability if probability else 0.5
        elif self.config.target_type == "returns":
            expected_return = value
            direction = "up" if value > 0 else "down"
            confidence = min(abs(value) * 10, 1.0)  # Scale confidence by return magnitude
        
        return Prediction(
            value=value,
            probability=probability,
            confidence=confidence,
            direction=direction,
            expected_return=expected_return
        )
    
    def predict_batch(self, X: pd.DataFrame) -> list[Prediction]:
        """Make predictions for multiple samples."""
        predictions = []
        for i in range(len(X)):
            pred = self.predict(X.iloc[[i]])
            predictions.append(pred)
        return predictions
    
    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features."""
        if not self.feature_importances:
            return []
        
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")
        
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances,
            'metrics': self.metrics
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.model = state['model']
        self.scaler = state['scaler']
        self.config = state['config']
        self.feature_names = state['feature_names']
        self.feature_importances = state['feature_importances']
        self.metrics = state['metrics']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")


class EnsemblePredictor:
    """
    Ensemble of multiple predictors for robust predictions.
    """
    
    def __init__(self, model_types: Optional[list[ModelType]] = None):
        """
        Initialize Ensemble Predictor.
        
        Args:
            model_types: List of model types to use
        """
        self.model_types = model_types or [
            ModelType.GRADIENT_BOOSTING,
            ModelType.RANDOM_FOREST,
            ModelType.RIDGE
        ]
        self.predictors: list[PricePredictor] = []
        self.weights: list[float] = []
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[list[str]] = None
    ) -> dict:
        """Fit all models in ensemble."""
        all_metrics = {}
        
        for model_type in self.model_types:
            config = PredictorConfig(model_type=model_type)
            predictor = PricePredictor(config)
            
            metrics = predictor.fit(X, y, feature_names)
            
            self.predictors.append(predictor)
            self.weights.append(max(0.1, metrics['cv_mean']))  # Use CV score as weight
            all_metrics[model_type.value] = metrics
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        return all_metrics
    
    def predict(self, X: pd.DataFrame) -> Prediction:
        """Make weighted ensemble prediction."""
        if not self.predictors:
            raise ValueError("Ensemble not fitted")
        
        predictions = []
        for predictor, weight in zip(self.predictors, self.weights):
            pred = predictor.predict(X)
            predictions.append((pred, weight))
        
        # Weighted average
        weighted_value = sum(p.value * w for p, w in predictions)
        
        # For direction, use majority voting
        direction_votes = {}
        for pred, weight in predictions:
            if pred.direction:
                direction_votes[pred.direction] = direction_votes.get(pred.direction, 0) + weight
        
        direction = max(direction_votes.items(), key=lambda x: x[1])[0] if direction_votes else None
        
        # Average confidence
        avg_confidence = np.mean([p.confidence for p, _ in predictions])
        
        return Prediction(
            value=weighted_value,
            confidence=avg_confidence,
            direction=direction
        )
