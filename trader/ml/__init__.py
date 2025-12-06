"""
Machine Learning Module for Price Prediction.
"""

from .feature_engineering import (
    FeatureEngineer,
    FeatureSet,
    FeatureType
)
from .price_predictor import (
    PricePredictor,
    Prediction,
    PredictorConfig,
    ModelType
)
from .model_evaluator import (
    ModelEvaluator,
    EvaluationMetrics,
    BacktestResult
)

# Transformer-based predictor (requires torch)
try:
    from .transformer_predictor import (
        TransformerPredictor,
        TransformerConfig,
        PriceTransformer
    )
except ImportError:
    TransformerPredictor = None
    TransformerConfig = None
    PriceTransformer = None

# Anomaly Detection
from .anomaly_detection import (
    MarketAnomalyDetector,
    Anomaly,
    AnomalyType,
    AnomalySeverity,
    AnomalyAlert,
    RealTimeAnomalyMonitor,
    scan_for_anomalies
)

# Explainable AI
from .explainability import (
    TradingExplainer,
    SHAPExplainer,
    LIMEExplainer,
    Explanation,
    FeatureContribution,
    DecisionAudit,
    ExplanationType,
    create_explainer_for_model
)

__all__ = [
    # Feature Engineering
    'FeatureEngineer',
    'FeatureSet',
    'FeatureType',
    # Price Prediction
    'PricePredictor',
    'Prediction',
    'PredictorConfig',
    'ModelType',
    # Model Evaluation
    'ModelEvaluator',
    'EvaluationMetrics',
    'BacktestResult',
    # Transformer (optional)
    'TransformerPredictor',
    'TransformerConfig',
    'PriceTransformer',
    # Anomaly Detection
    'MarketAnomalyDetector',
    'Anomaly',
    'AnomalyType',
    'AnomalySeverity',
    'AnomalyAlert',
    'RealTimeAnomalyMonitor',
    'scan_for_anomalies',
    # Explainability
    'TradingExplainer',
    'SHAPExplainer',
    'LIMEExplainer',
    'Explanation',
    'FeatureContribution',
    'DecisionAudit',
    'ExplanationType',
    'create_explainer_for_model',
]
