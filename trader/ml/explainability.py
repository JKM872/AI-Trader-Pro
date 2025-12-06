"""
Explainable AI (XAI) Module for Trading.

Provides explanations for AI trading decisions:
- SHAP values for feature importance
- LIME for local explanations
- Attention visualization for transformers
- Rule extraction from models
- Decision audit trails
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Types of explanations."""
    FEATURE_IMPORTANCE = "feature_importance"
    SHAP_VALUES = "shap_values"
    LIME = "lime"
    ATTENTION = "attention"
    RULE = "rule"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class FeatureContribution:
    """Contribution of a feature to a prediction."""
    feature_name: str
    feature_value: Any
    contribution: float
    direction: str  # positive or negative
    importance_rank: int = 0
    confidence: float = 1.0
    
    @property
    def abs_contribution(self) -> float:
        """Absolute contribution value."""
        return abs(self.contribution)


@dataclass
class Explanation:
    """Explanation for a model prediction."""
    prediction: Any
    confidence: float
    explanation_type: ExplanationType
    feature_contributions: list[FeatureContribution]
    base_value: float  # Expected value without any features
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    @property
    def top_features(self) -> list[FeatureContribution]:
        """Get top 5 contributing features."""
        sorted_features = sorted(
            self.feature_contributions,
            key=lambda x: x.abs_contribution,
            reverse=True
        )
        return sorted_features[:5]
    
    @property
    def positive_drivers(self) -> list[FeatureContribution]:
        """Get features that increased the prediction."""
        return [f for f in self.feature_contributions if f.contribution > 0]
    
    @property
    def negative_drivers(self) -> list[FeatureContribution]:
        """Get features that decreased the prediction."""
        return [f for f in self.feature_contributions if f.contribution < 0]


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation - what would need to change."""
    original_prediction: Any
    target_prediction: Any
    changes_needed: list[tuple[str, Any, Any]]  # (feature, from, to)
    feasibility_score: float
    summary: str


@dataclass
class RuleExplanation:
    """Rule-based explanation."""
    rules: list[str]
    coverage: float  # Percentage of data covered
    precision: float  # Accuracy of the rules
    summary: str


@dataclass
class DecisionAudit:
    """Audit trail for trading decisions."""
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    model_used: str
    prediction_value: float
    confidence: float
    explanation: Explanation
    input_data: dict
    final_decision: str
    human_override: bool = False
    override_reason: str = ""


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) based explainer.
    
    Uses TreeExplainer for tree-based models and KernelExplainer for others.
    """
    
    def __init__(self, model: Any = None, background_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP Explainer.
        
        Args:
            model: Trained model to explain
            background_data: Background data for SHAP values
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._shap_available = False
        
        try:
            import shap
            self._shap_available = True
        except ImportError:
            logger.warning("SHAP not available, using fallback method")
    
    def fit(self, model: Any, background_data: np.ndarray) -> None:
        """
        Fit the SHAP explainer.
        
        Args:
            model: Model to explain
            background_data: Background/training data
        """
        self.model = model
        self.background_data = background_data
        
        if not self._shap_available:
            return
        
        import shap
        
        # Detect model type and use appropriate explainer
        model_type = type(model).__name__.lower()
        
        if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type:
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                self.explainer = shap.KernelExplainer(
                    model.predict, 
                    shap.sample(background_data, 100)
                )
        else:
            # Use KernelExplainer for other models
            self.explainer = shap.KernelExplainer(
                model.predict if hasattr(model, 'predict') else model,
                shap.sample(background_data, 100)
            )
    
    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> Explanation:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            X: Input data to explain (single instance)
            feature_names: Names of features
            
        Returns:
            Explanation with SHAP values
        """
        if self.model is None:
            raise ValueError("No model set. Call fit() first.")
        
        # Reshape if necessary
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate SHAP values
        if self._shap_available and self.explainer is not None:
            shap_values = self.explainer.shap_values(X)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]
            
            base_value = float(self.explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
        else:
            # Fallback: use permutation importance approximation
            shap_values = self._fallback_importance(X, feature_names)
            base_value = 0.5
        
        # Create feature contributions
        contributions = []
        for i, (name, value, shap_val) in enumerate(zip(feature_names, X[0], shap_values)):
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=value,
                contribution=float(shap_val),
                direction="positive" if shap_val > 0 else "negative",
                importance_rank=i,
            ))
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: x.abs_contribution, reverse=True)
        for i, c in enumerate(contributions):
            c.importance_rank = i + 1
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        
        # Generate summary
        top_positive = [c for c in contributions[:3] if c.contribution > 0]
        top_negative = [c for c in contributions[:3] if c.contribution < 0]
        
        summary_parts = []
        if top_positive:
            pos_names = ", ".join([c.feature_name for c in top_positive])
            summary_parts.append(f"Increased by: {pos_names}")
        if top_negative:
            neg_names = ", ".join([c.feature_name for c in top_negative])
            summary_parts.append(f"Decreased by: {neg_names}")
        
        summary = ". ".join(summary_parts) if summary_parts else "No dominant features"
        
        return Explanation(
            prediction=prediction,
            confidence=0.9,
            explanation_type=ExplanationType.SHAP_VALUES,
            feature_contributions=contributions,
            base_value=base_value,
            summary=summary,
        )
    
    def _fallback_importance(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        """Fallback importance calculation when SHAP unavailable."""
        if self.model is None:
            return np.zeros(len(feature_names))
        
        # Simple perturbation-based importance
        original_pred = self.model.predict(X)[0]
        importance = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            X_perturbed = X.copy()
            X_perturbed[0, i] = 0  # Zero out feature
            perturbed_pred = self.model.predict(X_perturbed)[0]
            importance[i] = original_pred - perturbed_pred
        
        return importance


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) explainer.
    
    Creates local linear approximations to explain predictions.
    """
    
    def __init__(
        self,
        model: Any = None,
        mode: str = "regression",
    ):
        """
        Initialize LIME Explainer.
        
        Args:
            model: Model to explain
            mode: 'regression' or 'classification'
        """
        self.model = model
        self.mode = mode
        self.explainer = None
        self._lime_available = False
        
        try:
            import lime
            import lime.lime_tabular
            self._lime_available = True
        except ImportError:
            logger.warning("LIME not available, using fallback")
    
    def fit(
        self,
        model: Any,
        training_data: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """
        Fit LIME explainer.
        
        Args:
            model: Model to explain
            training_data: Training data for statistics
            feature_names: Feature names
        """
        self.model = model
        
        if not self._lime_available:
            return
        
        from lime.lime_tabular import LimeTabularExplainer
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(training_data.shape[1])]
        
        self.explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            mode=self.mode,
            random_state=42,
        )
    
    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        num_features: int = 10,
    ) -> Explanation:
        """
        Generate LIME explanation.
        
        Args:
            X: Instance to explain
            feature_names: Feature names
            num_features: Number of features in explanation
            
        Returns:
            Explanation with local feature importance
        """
        if self.model is None:
            raise ValueError("No model set. Call fit() first.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        prediction = self.model.predict(X)[0]
        
        if self._lime_available and self.explainer is not None:
            # Get LIME explanation
            if self.mode == "classification":
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
            
            lime_exp = self.explainer.explain_instance(
                X[0],
                predict_fn,
                num_features=num_features,
            )
            
            # Extract feature contributions
            contributions = []
            for feature, weight in lime_exp.as_list():
                # Parse feature name from LIME format
                name = feature.split()[0] if ' ' in feature else feature
                
                contributions.append(FeatureContribution(
                    feature_name=name,
                    feature_value=self._get_feature_value(X[0], name, feature_names),
                    contribution=weight,
                    direction="positive" if weight > 0 else "negative",
                ))
        else:
            # Fallback to simple linear approximation
            contributions = self._linear_approximation(X, feature_names)
        
        # Sort and rank
        contributions.sort(key=lambda x: x.abs_contribution, reverse=True)
        for i, c in enumerate(contributions):
            c.importance_rank = i + 1
        
        # Generate summary
        top_features = contributions[:3]
        summary = f"Top factors: {', '.join([c.feature_name for c in top_features])}"
        
        return Explanation(
            prediction=prediction,
            confidence=0.85,
            explanation_type=ExplanationType.LIME,
            feature_contributions=contributions,
            base_value=float(prediction),
            summary=summary,
        )
    
    def _get_feature_value(
        self,
        X: np.ndarray,
        feature_name: str,
        feature_names: list[str],
    ) -> Any:
        """Get feature value by name."""
        try:
            idx = feature_names.index(feature_name)
            return X[idx]
        except (ValueError, IndexError):
            return None
    
    def _linear_approximation(
        self,
        X: np.ndarray,
        feature_names: list[str],
    ) -> list[FeatureContribution]:
        """Simple linear approximation fallback."""
        contributions = []
        
        for i, name in enumerate(feature_names):
            # Approximate contribution as feature * small weight
            weight = X[0, i] * 0.01  # Simple heuristic
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=X[0, i],
                contribution=weight,
                direction="positive" if weight > 0 else "negative",
            ))
        
        return contributions


class AttentionExplainer:
    """
    Attention-based explanation for transformer models.
    
    Extracts and visualizes attention weights.
    """
    
    def __init__(self, model: Any = None):
        """
        Initialize Attention Explainer.
        
        Args:
            model: Transformer model with attention
        """
        self.model = model
    
    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
        layer: int = -1,
    ) -> Explanation:
        """
        Generate attention-based explanation.
        
        Args:
            X: Input sequence
            feature_names: Names for sequence positions
            layer: Which attention layer to use (-1 for last)
            
        Returns:
            Explanation with attention weights
        """
        if self.model is None:
            raise ValueError("No model set")
        
        # Try to extract attention weights
        attention_weights = self._extract_attention(X, layer)
        
        if attention_weights is None:
            # Fallback to uniform attention
            seq_len = X.shape[-1] if X.ndim > 1 else len(X)
            attention_weights = np.ones(seq_len) / seq_len
        
        if feature_names is None:
            feature_names = [f"t-{i}" for i in range(len(attention_weights))]
        
        # Create contributions from attention
        contributions = []
        for i, (name, attn) in enumerate(zip(feature_names, attention_weights)):
            contributions.append(FeatureContribution(
                feature_name=name,
                feature_value=X[i] if i < len(X) else 0,
                contribution=float(attn),
                direction="positive",
                importance_rank=i,
            ))
        
        # Sort by attention weight
        contributions.sort(key=lambda x: x.contribution, reverse=True)
        for i, c in enumerate(contributions):
            c.importance_rank = i + 1
        
        # Get prediction
        try:
            prediction = self.model.predict(X.reshape(1, -1) if X.ndim == 1 else X)[0]
        except Exception:
            prediction = 0.0
        
        # Summary
        top_attended = contributions[:3]
        summary = f"Model focused on: {', '.join([c.feature_name for c in top_attended])}"
        
        return Explanation(
            prediction=prediction,
            confidence=0.8,
            explanation_type=ExplanationType.ATTENTION,
            feature_contributions=contributions,
            base_value=0.0,
            summary=summary,
        )
    
    def _extract_attention(
        self,
        X: np.ndarray,
        layer: int,
    ) -> Optional[np.ndarray]:
        """Extract attention weights from model."""
        try:
            # Try PyTorch model
            if hasattr(self.model, 'get_attention_weights'):
                return self.model.get_attention_weights(X, layer)
            
            # Try to run forward pass with attention output
            import torch
            if isinstance(self.model, torch.nn.Module):
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).unsqueeze(0)
                    output = self.model(X_tensor, return_attention=True)
                    if isinstance(output, tuple):
                        _, attention = output
                        return attention[layer].mean(dim=1).squeeze().numpy()
        except Exception as e:
            logger.debug(f"Could not extract attention: {e}")
        
        return None


class TradingExplainer:
    """
    Unified explainer for trading models and decisions.
    
    Combines multiple explanation methods and provides
    trading-specific insights.
    """
    
    def __init__(
        self,
        model: Any = None,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ):
        """
        Initialize Trading Explainer.
        
        Args:
            model: Trading model to explain
            background_data: Background data for explanations
            feature_names: Names of features
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        
        # Initialize sub-explainers
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        
        # Audit trail
        self.decision_audit: list[DecisionAudit] = []
    
    def fit(
        self,
        model: Any,
        training_data: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """
        Fit all explainers.
        
        Args:
            model: Model to explain
            training_data: Training data
            feature_names: Feature names
        """
        self.model = model
        self.background_data = training_data
        self.feature_names = feature_names
        
        # Fit sub-explainers
        self.shap_explainer.fit(model, training_data)
        self.lime_explainer.fit(model, training_data, feature_names)
    
    def explain_prediction(
        self,
        X: np.ndarray,
        method: ExplanationType = ExplanationType.SHAP_VALUES,
    ) -> Explanation:
        """
        Explain a single prediction.
        
        Args:
            X: Input data
            method: Explanation method to use
            
        Returns:
            Explanation for the prediction
        """
        if method == ExplanationType.SHAP_VALUES:
            return self.shap_explainer.explain(X, self.feature_names)
        elif method == ExplanationType.LIME:
            return self.lime_explainer.explain(X, self.feature_names)
        else:
            # Default to SHAP
            return self.shap_explainer.explain(X, self.feature_names)
    
    def explain_trading_decision(
        self,
        symbol: str,
        features: dict,
        prediction: float,
        action: str,
    ) -> tuple[Explanation, DecisionAudit]:
        """
        Explain a trading decision with full audit trail.
        
        Args:
            symbol: Stock symbol
            features: Feature dictionary
            prediction: Model prediction
            action: Trading action (BUY, SELL, HOLD)
            
        Returns:
            Tuple of (Explanation, DecisionAudit)
        """
        # Convert features to array
        if self.feature_names:
            X = np.array([features.get(f, 0) for f in self.feature_names])
        else:
            X = np.array(list(features.values()))
        
        # Get explanation
        explanation = self.explain_prediction(X)
        
        # Create audit record
        audit = DecisionAudit(
            decision_id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            model_used=type(self.model).__name__ if self.model else "unknown",
            prediction_value=prediction,
            confidence=explanation.confidence,
            explanation=explanation,
            input_data=features,
            final_decision=action,
        )
        
        # Store audit
        self.decision_audit.append(audit)
        
        return explanation, audit
    
    def generate_human_readable_explanation(
        self,
        explanation: Explanation,
        symbol: str,
        action: str,
    ) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            explanation: Model explanation
            symbol: Stock symbol
            action: Trading action
            
        Returns:
            Human-readable explanation string
        """
        lines = [
            f"Trading Decision Explanation for {symbol}",
            f"{'=' * 50}",
            f"Recommended Action: {action}",
            f"Model Confidence: {explanation.confidence:.1%}",
            f"",
            "Key Factors:",
        ]
        
        for i, feature in enumerate(explanation.top_features, 1):
            direction = "↑" if feature.contribution > 0 else "↓"
            lines.append(
                f"  {i}. {feature.feature_name}: {feature.feature_value:.4f} "
                f"({direction} impact: {feature.contribution:.4f})"
            )
        
        lines.extend([
            "",
            "Summary:",
            f"  {explanation.summary}",
            "",
            "Positive Drivers:",
        ])
        
        for driver in explanation.positive_drivers[:3]:
            lines.append(f"  • {driver.feature_name}")
        
        lines.append("")
        lines.append("Negative Drivers:")
        
        for driver in explanation.negative_drivers[:3]:
            lines.append(f"  • {driver.feature_name}")
        
        return "\n".join(lines)
    
    def compare_explanations(
        self,
        X: np.ndarray,
    ) -> dict[str, Explanation]:
        """
        Compare explanations from different methods.
        
        Args:
            X: Input data to explain
            
        Returns:
            Dictionary of method -> explanation
        """
        explanations = {}
        
        try:
            explanations['shap'] = self.shap_explainer.explain(X, self.feature_names)
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
        
        try:
            explanations['lime'] = self.lime_explainer.explain(X, self.feature_names)
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
        
        return explanations
    
    def get_feature_importance_summary(
        self,
        n_samples: int = 100,
    ) -> pd.DataFrame:
        """
        Get overall feature importance across multiple samples.
        
        Args:
            n_samples: Number of samples to explain
            
        Returns:
            DataFrame with feature importance statistics
        """
        if self.background_data is None:
            raise ValueError("No background data available")
        
        # Sample data
        n_samples = min(n_samples, len(self.background_data))
        indices = np.random.choice(len(self.background_data), n_samples, replace=False)
        samples = self.background_data[indices]
        
        # Collect importances
        feature_importances: dict[str, list[float]] = {
            name: [] for name in (self.feature_names or [])
        }
        
        for sample in samples:
            try:
                explanation = self.explain_prediction(sample.reshape(1, -1))
                for contrib in explanation.feature_contributions:
                    if contrib.feature_name in feature_importances:
                        feature_importances[contrib.feature_name].append(
                            contrib.abs_contribution
                        )
            except Exception:
                continue
        
        # Create summary DataFrame
        summary_data = []
        for name, importances in feature_importances.items():
            if importances:
                summary_data.append({
                    'feature': name,
                    'mean_importance': np.mean(importances),
                    'std_importance': np.std(importances),
                    'median_importance': np.median(importances),
                    'max_importance': np.max(importances),
                })
        
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values('mean_importance', ascending=False)
        
        return df
    
    def generate_counterfactual(
        self,
        X: np.ndarray,
        target_action: str,
        max_changes: int = 3,
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual explanation.
        
        What would need to change to get a different decision?
        
        Args:
            X: Current input
            target_action: Desired action (BUY, SELL, HOLD)
            max_changes: Maximum number of features to change
            
        Returns:
            CounterfactualExplanation with suggested changes
        """
        if self.model is None:
            raise ValueError("No model available")
        
        X = X.reshape(1, -1) if X.ndim == 1 else X
        
        # Get current prediction
        current_pred = self.model.predict(X)[0]
        
        # Map action to target prediction
        action_targets = {
            'BUY': 1.0,
            'SELL': -1.0,
            'HOLD': 0.0,
        }
        target_pred = action_targets.get(target_action, 0.0)
        
        # Get feature importance to guide search
        explanation = self.explain_prediction(X)
        
        # Sort features by impact
        sorted_features = sorted(
            explanation.feature_contributions,
            key=lambda x: x.abs_contribution,
            reverse=True
        )
        
        # Try modifying top features
        changes = []
        X_modified = X.copy()
        
        for feature in sorted_features[:max_changes]:
            if self.feature_names:
                try:
                    idx = self.feature_names.index(feature.feature_name)
                except ValueError:
                    continue
            else:
                idx = feature.importance_rank - 1
            
            original_value = X[0, idx]
            
            # Determine change direction
            if target_pred > current_pred:
                # Need to increase prediction
                if feature.contribution > 0:
                    new_value = original_value * 1.5
                else:
                    new_value = original_value * 0.5
            else:
                # Need to decrease prediction
                if feature.contribution > 0:
                    new_value = original_value * 0.5
                else:
                    new_value = original_value * 1.5
            
            changes.append((feature.feature_name, original_value, new_value))
            X_modified[0, idx] = new_value
        
        # Check if changes achieve target
        new_pred = self.model.predict(X_modified)[0]
        
        # Calculate feasibility
        feasibility = 1.0 - min(1.0, abs(new_pred - target_pred))
        
        # Generate summary
        if feasibility > 0.7:
            summary = f"To achieve {target_action}, modify: " + ", ".join(
                [f"{c[0]} from {c[1]:.2f} to {c[2]:.2f}" for c in changes]
            )
        else:
            summary = f"Achieving {target_action} would require significant changes"
        
        return CounterfactualExplanation(
            original_prediction=current_pred,
            target_prediction=target_pred,
            changes_needed=changes,
            feasibility_score=feasibility,
            summary=summary,
        )
    
    def get_audit_trail(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[DecisionAudit]:
        """
        Get decision audit trail.
        
        Args:
            symbol: Filter by symbol
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of DecisionAudit records
        """
        audits = self.decision_audit
        
        if symbol:
            audits = [a for a in audits if a.symbol == symbol]
        
        if start_date:
            audits = [a for a in audits if a.timestamp >= start_date]
        
        if end_date:
            audits = [a for a in audits if a.timestamp <= end_date]
        
        return audits
    
    def export_audit_report(
        self,
        output_format: str = "dataframe",
    ) -> Any:
        """
        Export audit trail as report.
        
        Args:
            output_format: 'dataframe', 'dict', or 'json'
            
        Returns:
            Audit report in requested format
        """
        records = []
        
        for audit in self.decision_audit:
            records.append({
                'decision_id': audit.decision_id,
                'timestamp': audit.timestamp.isoformat(),
                'symbol': audit.symbol,
                'action': audit.action,
                'model': audit.model_used,
                'prediction': audit.prediction_value,
                'confidence': audit.confidence,
                'top_features': [f.feature_name for f in audit.explanation.top_features],
                'human_override': audit.human_override,
                'override_reason': audit.override_reason,
            })
        
        if output_format == "dataframe":
            return pd.DataFrame(records)
        elif output_format == "dict":
            return records
        elif output_format == "json":
            import json
            return json.dumps(records, indent=2)
        else:
            return records


def create_explainer_for_model(
    model: Any,
    training_data: np.ndarray,
    feature_names: list[str],
) -> TradingExplainer:
    """
    Convenience function to create explainer for a model.
    
    Args:
        model: Trained model
        training_data: Training data
        feature_names: Feature names
        
    Returns:
        Fitted TradingExplainer
    """
    explainer = TradingExplainer()
    explainer.fit(model, training_data, feature_names)
    return explainer
