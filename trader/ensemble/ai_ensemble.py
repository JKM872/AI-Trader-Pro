"""
AI Ensemble - Aggregating predictions from multiple AI models for consensus.

Features:
- Parallel querying of multiple AI providers
- Weighted voting based on historical accuracy
- Confidence-based filtering
- Automatic fallback on failures
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from trader.analysis.ai_providers import (
    PROVIDERS,
    create_provider,
    get_all_available_providers
)

logger = logging.getLogger(__name__)


class PredictionType(Enum):
    """Types of predictions from AI models."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    UNCERTAIN = "uncertain"
    
    @classmethod
    def from_text(cls, text: str) -> 'PredictionType':
        """Parse prediction type from text response."""
        text_lower = text.lower()
        
        # Strong signals
        if any(x in text_lower for x in ['strong buy', 'strongly bullish', 'very bullish', 'definitely buy']):
            return cls.STRONG_BUY
        if any(x in text_lower for x in ['strong sell', 'strongly bearish', 'very bearish', 'definitely sell']):
            return cls.STRONG_SELL
        
        # Regular signals
        if any(x in text_lower for x in ['buy', 'bullish', 'long', 'positive', 'upside']):
            return cls.BUY
        if any(x in text_lower for x in ['sell', 'bearish', 'short', 'negative', 'downside']):
            return cls.SELL
        if any(x in text_lower for x in ['hold', 'neutral', 'wait', 'sideways']):
            return cls.HOLD
            
        return cls.UNCERTAIN
    
    def to_score(self) -> float:
        """Convert prediction to numerical score (-1 to 1)."""
        return {
            PredictionType.STRONG_BUY: 1.0,
            PredictionType.BUY: 0.5,
            PredictionType.HOLD: 0.0,
            PredictionType.SELL: -0.5,
            PredictionType.STRONG_SELL: -1.0,
            PredictionType.UNCERTAIN: 0.0
        }[self]


@dataclass
class ModelPrediction:
    """Individual model prediction."""
    provider_name: str
    model_name: str
    prediction: PredictionType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    response_time_ms: float
    raw_response: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if prediction is valid (no error and not uncertain)."""
        return self.error is None and self.prediction != PredictionType.UNCERTAIN
    
    @property
    def weighted_score(self) -> float:
        """Get confidence-weighted score."""
        return self.prediction.to_score() * self.confidence


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    symbol: str
    final_prediction: PredictionType
    confidence: float
    consensus_score: float  # -1 to 1 (bearish to bullish)
    agreement_ratio: float  # 0 to 1 (how many models agree)
    predictions: list[ModelPrediction]
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def valid_predictions(self) -> list[ModelPrediction]:
        """Get only valid predictions."""
        return [p for p in self.predictions if p.is_valid]
    
    @property
    def model_count(self) -> int:
        """Number of models that responded."""
        return len(self.valid_predictions)
    
    def to_signal_confidence(self) -> float:
        """Convert to signal confidence (0 to 1)."""
        return abs(self.consensus_score) * self.confidence * self.agreement_ratio


class AIEnsemble:
    """
    Ensemble of AI models for consensus-based trading predictions.
    
    Features:
    - Query multiple AI providers in parallel
    - Weight models based on historical accuracy
    - Filter by confidence threshold
    - Aggregate predictions into consensus
    """
    
    DEFAULT_PROMPT_TEMPLATE = """
You are an expert financial analyst. Analyze the following data for {symbol} and provide a trading recommendation.

Market Data:
{market_data}

Technical Indicators:
{technical_data}

News/Sentiment:
{news_data}

Based on this analysis, provide:
1. Your recommendation (STRONG BUY, BUY, HOLD, SELL, or STRONG SELL)
2. Confidence level (0-100%)
3. Brief reasoning (2-3 sentences)

Format your response as:
RECOMMENDATION: [your recommendation]
CONFIDENCE: [X%]
REASONING: [your reasoning]
"""

    def __init__(
        self,
        providers: Optional[list[str]] = None,
        model_weights: Optional[dict[str, float]] = None,
        min_confidence: float = 0.5,
        min_models: int = 2,
        timeout_seconds: float = 30.0,
        max_workers: int = 5
    ):
        """
        Initialize AI Ensemble.
        
        Args:
            providers: List of provider names to use. None = use all available.
            model_weights: Dict of provider_name -> weight. None = equal weights.
            min_confidence: Minimum confidence to include prediction.
            min_models: Minimum models required for valid consensus.
            timeout_seconds: Timeout for each model query.
            max_workers: Max parallel model queries.
        """
        self.providers = providers or list(get_all_available_providers().keys())
        self.model_weights = model_weights or {}
        self.min_confidence = min_confidence
        self.min_models = min_models
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        
        # Initialize provider instances
        self._provider_instances: dict[str, Any] = {}
        self._initialize_providers()
        
        logger.info(f"AIEnsemble initialized with {len(self._provider_instances)} providers")
    
    def _initialize_providers(self) -> None:
        """Initialize provider instances."""
        for provider_name in self.providers:
            try:
                provider = create_provider(provider_name)
                if provider:
                    self._provider_instances[provider_name] = provider
                    logger.debug(f"Initialized provider: {provider_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name}: {e}")
    
    def _get_model_weight(self, provider_name: str) -> float:
        """Get weight for a model (default 1.0)."""
        return self.model_weights.get(provider_name, 1.0)
    
    def _parse_response(self, response: str) -> tuple[PredictionType, float, str]:
        """Parse model response into prediction components."""
        prediction = PredictionType.UNCERTAIN
        confidence = 0.5
        reasoning = response
        
        lines = response.upper().split('\n')
        
        for line in lines:
            # Parse recommendation
            if 'RECOMMENDATION:' in line:
                rec_text = line.split('RECOMMENDATION:')[1].strip()
                prediction = PredictionType.from_text(rec_text)
            
            # Parse confidence
            if 'CONFIDENCE:' in line:
                try:
                    conf_text = line.split('CONFIDENCE:')[1].strip()
                    conf_text = conf_text.replace('%', '').strip()
                    confidence = float(conf_text) / 100.0
                except (ValueError, IndexError):
                    pass
            
            # Parse reasoning
            if 'REASONING:' in line.upper():
                idx = response.upper().find('REASONING:')
                if idx != -1:
                    reasoning = response[idx + 10:].strip()
        
        # If we couldn't parse structured format, try to infer
        if prediction == PredictionType.UNCERTAIN:
            prediction = PredictionType.from_text(response)
        
        return prediction, min(max(confidence, 0.0), 1.0), reasoning
    
    def _query_single_provider(
        self,
        provider_name: str,
        prompt: str
    ) -> ModelPrediction:
        """Query a single provider and return prediction."""
        start_time = datetime.now(timezone.utc)
        
        try:
            provider = self._provider_instances.get(provider_name)
            if not provider:
                raise ValueError(f"Provider {provider_name} not initialized")
            
            # Get response from provider
            response = provider.analyze(prompt)
            
            # Calculate response time
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Parse response
            prediction, confidence, reasoning = self._parse_response(response)
            
            # Get model name from provider
            model_name = getattr(provider, 'model', 'unknown')
            
            return ModelPrediction(
                provider_name=provider_name,
                model_name=model_name,
                prediction=prediction,
                confidence=confidence,
                reasoning=reasoning,
                response_time_ms=response_time,
                raw_response=response
            )
            
        except Exception as e:
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.error(f"Error querying {provider_name}: {e}")
            
            return ModelPrediction(
                provider_name=provider_name,
                model_name="error",
                prediction=PredictionType.UNCERTAIN,
                confidence=0.0,
                reasoning="",
                response_time_ms=response_time,
                raw_response="",
                error=str(e)
            )
    
    def _aggregate_predictions(
        self,
        predictions: list[ModelPrediction],
        symbol: str
    ) -> EnsembleResult:
        """Aggregate multiple predictions into consensus."""
        valid_predictions = [
            p for p in predictions
            if p.is_valid and p.confidence >= self.min_confidence
        ]
        
        if len(valid_predictions) < self.min_models:
            # Not enough valid predictions
            return EnsembleResult(
                symbol=symbol,
                final_prediction=PredictionType.UNCERTAIN,
                confidence=0.0,
                consensus_score=0.0,
                agreement_ratio=0.0,
                predictions=predictions,
                reasoning="Insufficient valid predictions for consensus"
            )
        
        # Calculate weighted consensus score
        total_weight = 0.0
        weighted_score = 0.0
        
        for pred in valid_predictions:
            weight = self._get_model_weight(pred.provider_name) * pred.confidence
            weighted_score += pred.prediction.to_score() * weight
            total_weight += weight
        
        consensus_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine final prediction from consensus score
        if consensus_score >= 0.7:
            final_prediction = PredictionType.STRONG_BUY
        elif consensus_score >= 0.3:
            final_prediction = PredictionType.BUY
        elif consensus_score > -0.3:
            final_prediction = PredictionType.HOLD
        elif consensus_score > -0.7:
            final_prediction = PredictionType.SELL
        else:
            final_prediction = PredictionType.STRONG_SELL
        
        # Calculate agreement ratio (how many agree with final prediction)
        agreeing_predictions = [
            p for p in valid_predictions
            if (final_prediction.to_score() * p.prediction.to_score() > 0 or
                (final_prediction == PredictionType.HOLD and 
                 abs(p.prediction.to_score()) < 0.3))
        ]
        agreement_ratio = len(agreeing_predictions) / len(valid_predictions)
        
        # Calculate overall confidence
        confidences = [p.confidence for p in valid_predictions]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        
        # Aggregate reasoning
        reasonings = [f"[{p.provider_name}] {p.reasoning}" for p in valid_predictions[:3]]
        combined_reasoning = " | ".join(reasonings)
        
        return EnsembleResult(
            symbol=symbol,
            final_prediction=final_prediction,
            confidence=avg_confidence * agreement_ratio,
            consensus_score=consensus_score,
            agreement_ratio=agreement_ratio,
            predictions=predictions,
            reasoning=combined_reasoning
        )
    
    def predict(
        self,
        symbol: str,
        market_data: str,
        technical_data: str = "",
        news_data: str = "",
        custom_prompt: Optional[str] = None
    ) -> EnsembleResult:
        """
        Get ensemble prediction for a symbol.
        
        Args:
            symbol: Stock symbol
            market_data: Market data summary
            technical_data: Technical indicators summary
            news_data: News/sentiment summary
            custom_prompt: Custom prompt (overrides template)
            
        Returns:
            EnsembleResult with consensus prediction
        """
        # Build prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self.DEFAULT_PROMPT_TEMPLATE.format(
                symbol=symbol,
                market_data=market_data,
                technical_data=technical_data,
                news_data=news_data
            )
        
        # Query all providers in parallel
        predictions: list[ModelPrediction] = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_provider = {
                executor.submit(
                    self._query_single_provider,
                    provider_name,
                    prompt
                ): provider_name
                for provider_name in self._provider_instances.keys()
            }
            
            for future in as_completed(future_to_provider, timeout=self.timeout_seconds):
                provider_name = future_to_provider[future]
                try:
                    prediction = future.result()
                    predictions.append(prediction)
                    logger.debug(f"Got prediction from {provider_name}: {prediction.prediction}")
                except Exception as e:
                    logger.error(f"Future failed for {provider_name}: {e}")
        
        # Aggregate predictions
        result = self._aggregate_predictions(predictions, symbol)
        
        logger.info(
            f"Ensemble prediction for {symbol}: {result.final_prediction.value} "
            f"(confidence: {result.confidence:.2f}, agreement: {result.agreement_ratio:.2f})"
        )
        
        return result
    
    async def predict_async(
        self,
        symbol: str,
        market_data: str,
        technical_data: str = "",
        news_data: str = "",
        custom_prompt: Optional[str] = None
    ) -> EnsembleResult:
        """Async version of predict."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.predict(symbol, market_data, technical_data, news_data, custom_prompt)
        )
    
    def update_weights(self, performance_data: dict[str, float]) -> None:
        """
        Update model weights based on performance.
        
        Args:
            performance_data: Dict of provider_name -> accuracy score (0-1)
        """
        self.model_weights.update(performance_data)
        logger.info(f"Updated model weights: {self.model_weights}")
    
    def get_available_providers(self) -> list[str]:
        """Get list of currently available providers."""
        return list(self._provider_instances.keys())
    
    def add_provider(self, provider_name: str) -> bool:
        """Add a provider to the ensemble."""
        if provider_name in self._provider_instances:
            return True
            
        try:
            provider = create_provider(provider_name)
            if provider:
                self._provider_instances[provider_name] = provider
                self.providers.append(provider_name)
                return True
        except Exception as e:
            logger.error(f"Failed to add provider {provider_name}: {e}")
        
        return False
    
    def remove_provider(self, provider_name: str) -> bool:
        """Remove a provider from the ensemble."""
        if provider_name in self._provider_instances:
            del self._provider_instances[provider_name]
            if provider_name in self.providers:
                self.providers.remove(provider_name)
            return True
        return False
