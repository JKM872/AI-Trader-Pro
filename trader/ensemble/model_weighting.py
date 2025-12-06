"""
Model Weight Manager - Dynamic weight adjustment based on performance.

Features:
- Track model accuracy over time
- Bayesian weight updates
- Decay for stale performance
- Performance persistence
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import statistics
import math

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track a model's performance metrics."""
    provider_name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    total_profit: float = 0.0  # Cumulative profit/loss from predictions
    avg_confidence: float = 0.5
    avg_response_time_ms: float = 0.0
    last_prediction: Optional[datetime] = None
    weight: float = 1.0
    
    # Track recent accuracy for recency weighting
    recent_correct: int = 0
    recent_total: int = 0
    recent_window: int = 20  # Last N predictions
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy (0-1)."""
        if self.total_predictions == 0:
            return 0.5  # Prior for new models
        return self.correct_predictions / self.total_predictions
    
    @property
    def recent_accuracy(self) -> float:
        """Recent accuracy (0-1)."""
        if self.recent_total == 0:
            return self.accuracy
        return self.recent_correct / self.recent_total
    
    @property
    def profit_per_prediction(self) -> float:
        """Average profit per prediction."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_profit / self.total_predictions
    
    def record_prediction(
        self,
        was_correct: bool,
        profit: float = 0.0,
        confidence: float = 0.5,
        response_time_ms: float = 0.0
    ) -> None:
        """Record a prediction outcome."""
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        
        self.total_profit += profit
        self.last_prediction = datetime.now(timezone.utc)
        
        # Update running averages
        alpha = 0.1  # Exponential smoothing factor
        self.avg_confidence = (1 - alpha) * self.avg_confidence + alpha * confidence
        self.avg_response_time_ms = (1 - alpha) * self.avg_response_time_ms + alpha * response_time_ms
        
        # Update recent stats (sliding window)
        self.recent_total = min(self.recent_total + 1, self.recent_window)
        if was_correct:
            self.recent_correct = min(self.recent_correct + 1, self.recent_window)
        else:
            # Decay recent correct count proportionally
            decay = self.recent_correct / self.recent_window if self.recent_window > 0 else 0
            self.recent_correct = max(0, self.recent_correct - int(not was_correct))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.last_prediction:
            data['last_prediction'] = self.last_prediction.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelPerformance':
        """Create from dictionary."""
        if 'last_prediction' in data and data['last_prediction']:
            data['last_prediction'] = datetime.fromisoformat(data['last_prediction'])
        return cls(**data)


class ModelWeightManager:
    """
    Manages and updates model weights based on performance.
    
    Features:
    - Track accuracy per model
    - Bayesian weight updates
    - Time-decay for stale models
    - Persistence to file
    """
    
    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        decay_days: int = 30,
        prior_weight: float = 1.0,
        learning_rate: float = 0.1
    ):
        """
        Initialize Weight Manager.
        
        Args:
            persistence_path: Path to save/load performance data
            decay_days: Days after which to start decaying weights
            prior_weight: Default weight for new models
            learning_rate: How fast weights adjust to new data
        """
        self.persistence_path = persistence_path
        self.decay_days = decay_days
        self.prior_weight = prior_weight
        self.learning_rate = learning_rate
        
        self.models: dict[str, ModelPerformance] = {}
        
        # Load existing data if available
        if persistence_path and persistence_path.exists():
            self._load()
    
    def _load(self) -> None:
        """Load performance data from file."""
        if not self.persistence_path:
            return
        
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)
            
            for name, perf_data in data.items():
                self.models[name] = ModelPerformance.from_dict(perf_data)
            
            logger.info(f"Loaded performance data for {len(self.models)} models")
        except Exception as e:
            logger.warning(f"Failed to load performance data: {e}")
    
    def _save(self) -> None:
        """Save performance data to file."""
        if not self.persistence_path:
            return
        
        try:
            data = {name: perf.to_dict() for name, perf in self.models.items()}
            
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved performance data for {len(self.models)} models")
        except Exception as e:
            logger.warning(f"Failed to save performance data: {e}")
    
    def get_model(self, provider_name: str) -> ModelPerformance:
        """Get or create model performance tracker."""
        if provider_name not in self.models:
            self.models[provider_name] = ModelPerformance(
                provider_name=provider_name,
                weight=self.prior_weight
            )
        return self.models[provider_name]
    
    def record_outcome(
        self,
        provider_name: str,
        was_correct: bool,
        profit: float = 0.0,
        confidence: float = 0.5,
        response_time_ms: float = 0.0
    ) -> None:
        """
        Record prediction outcome and update weights.
        
        Args:
            provider_name: Name of the provider
            was_correct: Whether prediction was correct
            profit: Profit/loss from following prediction
            confidence: Model's stated confidence
            response_time_ms: Response time
        """
        model = self.get_model(provider_name)
        model.record_prediction(was_correct, profit, confidence, response_time_ms)
        
        # Update weight using Bayesian-inspired approach
        self._update_weight(model, was_correct, confidence)
        
        self._save()
        
        logger.debug(
            f"Recorded {provider_name}: correct={was_correct}, "
            f"accuracy={model.accuracy:.2%}, weight={model.weight:.3f}"
        )
    
    def _update_weight(
        self,
        model: ModelPerformance,
        was_correct: bool,
        confidence: float
    ) -> None:
        """Update model weight based on outcome."""
        # Base update based on correctness
        if was_correct:
            # Reward proportional to confidence (higher reward for confident correct predictions)
            update = self.learning_rate * confidence
        else:
            # Penalty proportional to confidence (higher penalty for confident wrong predictions)
            update = -self.learning_rate * confidence
        
        # Apply update with momentum toward accuracy
        accuracy_pull = (model.recent_accuracy - 0.5) * 0.1
        
        model.weight = max(0.1, min(3.0, model.weight + update + accuracy_pull))
    
    def apply_decay(self) -> None:
        """Apply time-based decay to stale models."""
        now = datetime.now(timezone.utc)
        
        for model in self.models.values():
            if model.last_prediction is None:
                continue
            
            days_since = (now - model.last_prediction).days
            
            if days_since > self.decay_days:
                # Exponential decay toward prior weight
                decay_factor = 0.95 ** (days_since - self.decay_days)
                model.weight = self.prior_weight + (model.weight - self.prior_weight) * decay_factor
        
        self._save()
    
    def get_weights(self) -> dict[str, float]:
        """Get all model weights."""
        self.apply_decay()
        return {name: model.weight for name, model in self.models.items()}
    
    def get_ranking(self) -> list[tuple[str, float, float]]:
        """
        Get models ranked by performance.
        
        Returns:
            List of (provider_name, accuracy, weight) sorted by weight
        """
        self.apply_decay()
        
        ranking = [
            (name, model.accuracy, model.weight)
            for name, model in self.models.items()
            if model.total_predictions > 0
        ]
        
        return sorted(ranking, key=lambda x: x[2], reverse=True)
    
    def get_best_models(self, n: int = 3) -> list[str]:
        """Get top N models by weight."""
        ranking = self.get_ranking()
        return [name for name, _, _ in ranking[:n]]
    
    def reset_model(self, provider_name: str) -> None:
        """Reset a model's performance history."""
        if provider_name in self.models:
            del self.models[provider_name]
            self._save()
    
    def reset_all(self) -> None:
        """Reset all performance history."""
        self.models.clear()
        self._save()
    
    def get_statistics(self) -> dict:
        """Get aggregate statistics."""
        if not self.models:
            return {
                'total_models': 0,
                'total_predictions': 0,
                'avg_accuracy': 0.0,
                'total_profit': 0.0
            }
        
        total_predictions = sum(m.total_predictions for m in self.models.values())
        total_correct = sum(m.correct_predictions for m in self.models.values())
        total_profit = sum(m.total_profit for m in self.models.values())
        
        accuracies = [m.accuracy for m in self.models.values() if m.total_predictions > 0]
        
        return {
            'total_models': len(self.models),
            'total_predictions': total_predictions,
            'avg_accuracy': statistics.mean(accuracies) if accuracies else 0.0,
            'total_profit': total_profit,
            'best_model': self.get_best_models(1)[0] if self.get_best_models(1) else None,
            'weights': self.get_weights()
        }
    
    def calibrate_weights(self) -> None:
        """
        Recalibrate all weights based on performance.
        
        Normalizes weights so average is 1.0 while maintaining relative differences.
        """
        if not self.models:
            return
        
        weights = [m.weight for m in self.models.values()]
        avg_weight = statistics.mean(weights)
        
        if avg_weight > 0:
            for model in self.models.values():
                model.weight = model.weight / avg_weight
        
        self._save()
        logger.info("Calibrated model weights")
