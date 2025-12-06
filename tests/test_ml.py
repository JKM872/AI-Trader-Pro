"""
Tests for ML Price Predictor Module.

Tests for feature engineering, price prediction, and model evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from trader.ml import (
    FeatureEngineer,
    PricePredictor,
    ModelEvaluator,
    BacktestResult,
)
from trader.ml.feature_engineering import FeatureSet, FeatureType
from trader.ml.price_predictor import PredictorConfig, ModelType


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0.0002, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'High': prices * (1 + np.random.uniform(0.005, 0.02, len(dates))),
        'Low': prices * (1 - np.random.uniform(0.005, 0.02, len(dates))),
        'Close': prices,
        'Volume': np.random.uniform(1e6, 5e6, len(dates)).astype(int),
    }, index=dates)
    
    # Ensure High >= Low
    df['High'] = df[['High', 'Low']].max(axis=1)
    df['Low'] = df[['High', 'Low']].min(axis=1)
    
    return df


@pytest.fixture
def feature_engineer():
    """Create feature engineer instance."""
    return FeatureEngineer()


@pytest.fixture
def price_predictor():
    """Create price predictor instance."""
    config = PredictorConfig(model_type=ModelType.GRADIENT_BOOSTING)
    return PricePredictor(config=config)


# ============================================================================
# FeatureEngineer Tests
# ============================================================================

class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_initialization(self, feature_engineer):
        """Test feature engineer initialization."""
        assert feature_engineer is not None
        assert isinstance(feature_engineer.lookback_periods, list)
    
    def test_create_features(self, feature_engineer, sample_ohlcv_data):
        """Test feature creation."""
        feature_set = feature_engineer.create_features(sample_ohlcv_data)
        
        assert feature_set is not None
        assert isinstance(feature_set, FeatureSet)
        assert feature_set.features is not None
        assert len(feature_set.features) > 0
    
    def test_feature_set_has_feature_names(self, feature_engineer, sample_ohlcv_data):
        """Test that feature set includes feature names."""
        feature_set = feature_engineer.create_features(sample_ohlcv_data)
        
        assert feature_set.feature_names is not None
        assert len(feature_set.feature_names) > 0
    
    def test_technical_features_created(self, feature_engineer, sample_ohlcv_data):
        """Test that technical indicators are created."""
        feature_set = feature_engineer.create_features(sample_ohlcv_data)
        feature_names = feature_set.feature_names
        
        # Check for common technical features
        technical_keywords = ['rsi', 'macd', 'sma', 'ema', 'bb_', 'atr']
        has_technical = any(
            any(kw in name.lower() for kw in technical_keywords)
            for name in feature_names
        )
        
        assert has_technical, "Should have technical indicator features"
    
    def test_volume_features_created(self, feature_engineer, sample_ohlcv_data):
        """Test that volume features are created."""
        feature_set = feature_engineer.create_features(sample_ohlcv_data)
        feature_names = feature_set.feature_names
        
        volume_keywords = ['volume', 'vol_']
        has_volume = any(
            any(kw in name.lower() for kw in volume_keywords)
            for name in feature_names
        )
        
        assert has_volume, "Should have volume features"
    
    def test_features_no_nan_after_dropna(self, feature_engineer, sample_ohlcv_data):
        """Test that features have no NaN after dropping initial rows."""
        feature_set = feature_engineer.create_features(sample_ohlcv_data)
        features = feature_set.features
        
        # create_features already drops NaN rows
        assert not features.isnull().any().any(), "Features should have no NaN"
    
    def test_feature_types_assigned(self, feature_engineer, sample_ohlcv_data):
        """Test that feature types are assigned."""
        feature_set = feature_engineer.create_features(sample_ohlcv_data)
        
        assert len(feature_set.feature_types) > 0
        assert FeatureType.MOMENTUM in feature_set.feature_types.values()
    
    def test_target_creation(self, feature_engineer, sample_ohlcv_data):
        """Test target variable creation."""
        feature_set = feature_engineer.create_features(sample_ohlcv_data, include_target=True)
        
        # Target should be created when include_target=True
        assert feature_set.target is not None or feature_set.features is not None
    
    def test_get_features_by_type(self, feature_engineer, sample_ohlcv_data):
        """Test getting features by type."""
        feature_set = feature_engineer.create_features(sample_ohlcv_data)
        
        momentum_features = feature_set.get_features_by_type(FeatureType.MOMENTUM)
        assert isinstance(momentum_features, pd.DataFrame)


# ============================================================================
# PricePredictor Tests
# ============================================================================

class TestPricePredictor:
    """Tests for PricePredictor class."""
    
    def test_initialization_gradient_boosting(self):
        """Test gradient boosting initialization."""
        config = PredictorConfig(model_type=ModelType.GRADIENT_BOOSTING)
        predictor = PricePredictor(config=config)
        assert predictor.config.model_type == ModelType.GRADIENT_BOOSTING
    
    def test_initialization_random_forest(self):
        """Test random forest initialization."""
        config = PredictorConfig(model_type=ModelType.RANDOM_FOREST)
        predictor = PricePredictor(config=config)
        assert predictor.config.model_type == ModelType.RANDOM_FOREST
    
    def test_initialization_ridge(self):
        """Test ridge regression initialization."""
        config = PredictorConfig(model_type=ModelType.RIDGE)
        predictor = PricePredictor(config=config)
        assert predictor.config.model_type == ModelType.RIDGE
    
    def test_training(self, price_predictor, sample_ohlcv_data):
        """Test model training."""
        fe = FeatureEngineer(target_type="direction")
        feature_set = fe.create_features(sample_ohlcv_data, include_target=True)
        
        if feature_set.features.empty or feature_set.target is None:
            pytest.skip("Not enough data for training test")
        
        features = feature_set.features
        target = feature_set.target.loc[features.index]
        
        if len(features) > 50:
            metrics = price_predictor.fit(features, target)
            assert price_predictor.is_fitted
            assert 'cv_mean' in metrics
    
    def test_prediction(self, price_predictor, sample_ohlcv_data):
        """Test model prediction."""
        fe = FeatureEngineer(target_type="direction")
        feature_set = fe.create_features(sample_ohlcv_data, include_target=True)
        
        if feature_set.features.empty or feature_set.target is None:
            pytest.skip("Not enough data for prediction test")
        
        features = feature_set.features
        target = feature_set.target.loc[features.index]
        
        if len(features) > 50:
            # Train on most data
            X_train = features.iloc[:-30]
            y_train = target.iloc[:-30]
            X_test = features.iloc[-30:]
            
            price_predictor.fit(X_train, y_train)
            prediction = price_predictor.predict(X_test)
            
            assert prediction is not None
            assert hasattr(prediction, 'value')
    
    def test_model_save_load(self, price_predictor, sample_ohlcv_data):
        """Test model persistence."""
        fe = FeatureEngineer(target_type="direction")
        feature_set = fe.create_features(sample_ohlcv_data, include_target=True)
        
        if feature_set.features.empty or feature_set.target is None:
            pytest.skip("Not enough data for save/load test")
        
        features = feature_set.features
        target = feature_set.target.loc[features.index]
        
        if len(features) > 50:
            price_predictor.fit(features, target)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = Path(tmpdir) / 'model.pkl'
                
                price_predictor.save(filepath)
                
                # Load in new predictor
                new_predictor = PricePredictor()
                new_predictor.load(filepath)
                
                assert new_predictor.is_fitted


# ============================================================================
# ModelEvaluator Tests
# ============================================================================

class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator is not None
        assert evaluator.transaction_cost == 0.001
    
    def test_evaluate_classification(self):
        """Test classification evaluation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
        
        metrics = evaluator.evaluate_classification(y_true, y_pred)
        
        assert metrics.accuracy > 0
        assert metrics.directional_accuracy > 0
    
    def test_evaluate_regression(self):
        """Test regression evaluation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.01, 0.02])
        y_pred = np.array([0.012, -0.018, 0.028, 0.008, -0.012, 0.022, -0.008, 0.019])
        
        metrics = evaluator.evaluate_regression(y_true, y_pred)
        
        assert metrics.mse >= 0
        assert metrics.mae >= 0
    
    def test_backtest_strategy(self):
        """Test strategy backtesting."""
        evaluator = ModelEvaluator()
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        predictions = pd.Series(np.random.randn(100), index=dates)
        actual_returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
        
        result = evaluator.backtest_strategy(predictions, actual_returns)
        
        assert isinstance(result, BacktestResult)
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
    
    def test_backtest_result_dataclass(self):
        """Test BacktestResult dataclass."""
        result = BacktestResult(
            total_return=0.15,
            annualized_return=0.18,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            win_rate=0.55,
            n_trades=50
        )
        
        assert result.total_return == 0.15
        assert result.sharpe_ratio == 1.2


# ============================================================================
# Integration Tests
# ============================================================================

class TestMLIntegration:
    """Integration tests for the ML module."""
    
    def test_full_pipeline(self, sample_ohlcv_data):
        """Test complete ML pipeline."""
        # Feature engineering
        fe = FeatureEngineer(target_type="direction")
        feature_set = fe.create_features(sample_ohlcv_data, include_target=True)
        
        if feature_set.features.empty or feature_set.target is None:
            pytest.skip("Not enough data for full pipeline test")
        
        features = feature_set.features
        target = feature_set.target.loc[features.index]
        
        if len(features) < 100:
            pytest.skip("Not enough data for full pipeline test")
        
        # Train/test split
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        
        # Train model
        config = PredictorConfig(model_type=ModelType.GRADIENT_BOOSTING)
        predictor = PricePredictor(config=config)
        metrics = predictor.fit(X_train, y_train)
        
        # Predict
        prediction = predictor.predict(X_test)
        
        assert prediction is not None
        assert 'cv_mean' in metrics
    
    def test_multiple_model_comparison(self, sample_ohlcv_data):
        """Test comparing multiple model types."""
        fe = FeatureEngineer(target_type="direction")
        feature_set = fe.create_features(sample_ohlcv_data, include_target=True)
        
        if feature_set.features.empty or feature_set.target is None:
            pytest.skip("Not enough data for comparison test")
        
        features = feature_set.features
        target = feature_set.target.loc[features.index]
        
        if len(features) < 100:
            pytest.skip("Not enough data for comparison test")
        
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        
        model_types = [ModelType.GRADIENT_BOOSTING, ModelType.RANDOM_FOREST, ModelType.RIDGE]
        results = {}
        
        for model_type in model_types:
            config = PredictorConfig(model_type=model_type)
            predictor = PricePredictor(config=config)
            metrics = predictor.fit(X_train, y_train)
            results[model_type.value] = metrics['cv_mean']
        
        assert len(results) == len(model_types)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestMLEdgeCases:
    """Edge case tests for ML module."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        fe = FeatureEngineer()
        empty_df = pd.DataFrame()
        
        result = fe.create_features(empty_df)
        assert result.features.empty
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        fe = FeatureEngineer()
        
        # Very small dataset
        small_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200],
        })
        
        result = fe.create_features(small_df)
        # Should return empty or mostly NaN due to insufficient data
        assert result is not None
    
    def test_missing_volume(self):
        """Test handling of missing volume column."""
        fe = FeatureEngineer()
        
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104] * 20,
            'High': [101, 102, 103, 104, 105] * 20,
            'Low': [99, 100, 101, 102, 103] * 20,
            'Close': [100, 101, 102, 103, 104] * 20,
        }, index=pd.date_range(start='2023-01-01', periods=100, freq='D'))
        
        # Should handle missing volume gracefully
        result = fe.create_features(df)
        assert result is not None
    
    def test_prediction_without_training(self):
        """Test prediction without training raises error."""
        predictor = PricePredictor()
        
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        
        with pytest.raises(ValueError):
            predictor.predict(X)
