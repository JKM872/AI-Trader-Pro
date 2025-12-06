"""
Consensus Engine - Advanced voting mechanisms for ensemble predictions.

Supports multiple voting methods:
- Simple majority
- Weighted voting
- Confidence-weighted
- Bayesian aggregation
- Borda count
- Unanimous consent
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import statistics
import math

from .ai_ensemble import ModelPrediction, PredictionType

logger = logging.getLogger(__name__)


class VotingMethod(Enum):
    """Available voting methods for consensus."""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    BAYESIAN = "bayesian"
    BORDA_COUNT = "borda_count"
    UNANIMOUS = "unanimous"


class ConsensusStrength(Enum):
    """Strength of consensus."""
    STRONG = "strong"       # >80% agreement
    MODERATE = "moderate"   # 60-80% agreement
    WEAK = "weak"           # 50-60% agreement
    SPLIT = "split"         # <50% agreement (no clear majority)


@dataclass
class Vote:
    """Individual vote in consensus."""
    provider: str
    prediction: PredictionType
    weight: float
    confidence: float
    
    @property
    def effective_weight(self) -> float:
        """Weight adjusted by confidence."""
        return self.weight * self.confidence


@dataclass
class ConsensusResult:
    """Result of consensus voting."""
    prediction: PredictionType
    strength: ConsensusStrength
    confidence: float
    vote_distribution: dict[PredictionType, float]  # prediction -> vote share
    method_used: VotingMethod
    votes: list[Vote]
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def majority_share(self) -> float:
        """Get share of votes for winning prediction."""
        return self.vote_distribution.get(self.prediction, 0.0)
    
    @property
    def is_actionable(self) -> bool:
        """Check if consensus is strong enough to act on."""
        return (
            self.strength in (ConsensusStrength.STRONG, ConsensusStrength.MODERATE) and
            self.prediction != PredictionType.UNCERTAIN and
            self.confidence >= 0.5
        )


class ConsensusEngine:
    """
    Advanced consensus engine with multiple voting strategies.
    
    Aggregates model predictions into actionable trading signals
    using configurable voting mechanisms.
    """
    
    def __init__(
        self,
        default_method: VotingMethod = VotingMethod.CONFIDENCE_WEIGHTED,
        model_weights: Optional[dict[str, float]] = None,
        min_confidence: float = 0.3,
        unanimous_threshold: float = 0.9
    ):
        """
        Initialize Consensus Engine.
        
        Args:
            default_method: Default voting method to use
            model_weights: Provider -> weight mapping
            min_confidence: Minimum confidence to include vote
            unanimous_threshold: Agreement threshold for unanimous voting
        """
        self.default_method = default_method
        self.model_weights = model_weights or {}
        self.min_confidence = min_confidence
        self.unanimous_threshold = unanimous_threshold
        
        # Voting method implementations
        self._methods = {
            VotingMethod.SIMPLE_MAJORITY: self._simple_majority,
            VotingMethod.WEIGHTED_MAJORITY: self._weighted_majority,
            VotingMethod.CONFIDENCE_WEIGHTED: self._confidence_weighted,
            VotingMethod.BAYESIAN: self._bayesian,
            VotingMethod.BORDA_COUNT: self._borda_count,
            VotingMethod.UNANIMOUS: self._unanimous
        }
    
    def _predictions_to_votes(
        self,
        predictions: list[ModelPrediction]
    ) -> list[Vote]:
        """Convert model predictions to votes."""
        votes = []
        for pred in predictions:
            if pred.is_valid and pred.confidence >= self.min_confidence:
                weight = self.model_weights.get(pred.provider_name, 1.0)
                votes.append(Vote(
                    provider=pred.provider_name,
                    prediction=pred.prediction,
                    weight=weight,
                    confidence=pred.confidence
                ))
        return votes
    
    def _calculate_strength(self, majority_share: float) -> ConsensusStrength:
        """Determine consensus strength from majority share."""
        if majority_share >= 0.8:
            return ConsensusStrength.STRONG
        elif majority_share >= 0.6:
            return ConsensusStrength.MODERATE
        elif majority_share >= 0.5:
            return ConsensusStrength.WEAK
        else:
            return ConsensusStrength.SPLIT
    
    def _count_votes(
        self,
        votes: list[Vote],
        use_weights: bool = False,
        use_confidence: bool = False
    ) -> dict[PredictionType, float]:
        """Count votes with optional weighting."""
        counts: dict[PredictionType, float] = {}
        
        for vote in votes:
            if use_confidence:
                value = vote.effective_weight if use_weights else vote.confidence
            elif use_weights:
                value = vote.weight
            else:
                value = 1.0
            
            counts[vote.prediction] = counts.get(vote.prediction, 0.0) + value
        
        # Normalize to percentages
        total = sum(counts.values())
        if total > 0:
            counts = {k: v / total for k, v in counts.items()}
        
        return counts
    
    def _simple_majority(self, votes: list[Vote]) -> ConsensusResult:
        """Simple majority voting (one vote per model)."""
        distribution = self._count_votes(votes, use_weights=False, use_confidence=False)
        
        if not distribution:
            return self._no_consensus_result(votes, VotingMethod.SIMPLE_MAJORITY)
        
        winner = max(distribution.items(), key=lambda x: x[1])
        prediction, share = winner
        
        # Average confidence of winning votes
        winning_votes = [v for v in votes if v.prediction == prediction]
        avg_confidence = statistics.mean([v.confidence for v in winning_votes]) if winning_votes else 0.0
        
        return ConsensusResult(
            prediction=prediction,
            strength=self._calculate_strength(share),
            confidence=avg_confidence,
            vote_distribution=distribution,
            method_used=VotingMethod.SIMPLE_MAJORITY,
            votes=votes,
            reasoning=f"Simple majority: {prediction.value} with {share:.1%} of votes"
        )
    
    def _weighted_majority(self, votes: list[Vote]) -> ConsensusResult:
        """Weighted majority voting (uses model weights)."""
        distribution = self._count_votes(votes, use_weights=True, use_confidence=False)
        
        if not distribution:
            return self._no_consensus_result(votes, VotingMethod.WEIGHTED_MAJORITY)
        
        winner = max(distribution.items(), key=lambda x: x[1])
        prediction, share = winner
        
        # Weighted average confidence
        winning_votes = [v for v in votes if v.prediction == prediction]
        total_weight = sum(v.weight for v in winning_votes)
        weighted_conf = sum(v.weight * v.confidence for v in winning_votes) / total_weight if total_weight > 0 else 0.0
        
        return ConsensusResult(
            prediction=prediction,
            strength=self._calculate_strength(share),
            confidence=weighted_conf,
            vote_distribution=distribution,
            method_used=VotingMethod.WEIGHTED_MAJORITY,
            votes=votes,
            reasoning=f"Weighted majority: {prediction.value} with {share:.1%} weighted share"
        )
    
    def _confidence_weighted(self, votes: list[Vote]) -> ConsensusResult:
        """Confidence-weighted voting (uses confidence * weight)."""
        distribution = self._count_votes(votes, use_weights=True, use_confidence=True)
        
        if not distribution:
            return self._no_consensus_result(votes, VotingMethod.CONFIDENCE_WEIGHTED)
        
        winner = max(distribution.items(), key=lambda x: x[1])
        prediction, share = winner
        
        # Calculate consensus confidence
        winning_votes = [v for v in votes if v.prediction == prediction]
        if winning_votes:
            total_eff_weight = sum(v.effective_weight for v in winning_votes)
            all_eff_weight = sum(v.effective_weight for v in votes)
            confidence = total_eff_weight / all_eff_weight if all_eff_weight > 0 else 0.0
        else:
            confidence = 0.0
        
        return ConsensusResult(
            prediction=prediction,
            strength=self._calculate_strength(share),
            confidence=confidence,
            vote_distribution=distribution,
            method_used=VotingMethod.CONFIDENCE_WEIGHTED,
            votes=votes,
            reasoning=f"Confidence-weighted: {prediction.value} with {share:.1%} effective weight"
        )
    
    def _bayesian(self, votes: list[Vote]) -> ConsensusResult:
        """Bayesian aggregation using log-odds."""
        if not votes:
            return self._no_consensus_result(votes, VotingMethod.BAYESIAN)
        
        # Convert predictions to scores and aggregate using log-odds
        # Positive score = bullish, negative = bearish
        log_odds = 0.0
        total_weight = 0.0
        
        for vote in votes:
            score = vote.prediction.to_score()
            if score != 0:
                # Convert confidence to probability-like value
                p = (1 + score * vote.confidence) / 2
                p = max(0.01, min(0.99, p))  # Clamp to avoid log(0)
                
                # Log-odds contribution
                log_odds += vote.weight * math.log(p / (1 - p))
                total_weight += vote.weight
        
        # Convert back to probability
        if total_weight > 0:
            avg_log_odds = log_odds / total_weight
            aggregated_p = 1 / (1 + math.exp(-avg_log_odds))
        else:
            aggregated_p = 0.5
        
        # Convert probability to prediction
        score = (aggregated_p - 0.5) * 2  # -1 to 1
        
        if score >= 0.5:
            prediction = PredictionType.STRONG_BUY
        elif score >= 0.2:
            prediction = PredictionType.BUY
        elif score > -0.2:
            prediction = PredictionType.HOLD
        elif score > -0.5:
            prediction = PredictionType.SELL
        else:
            prediction = PredictionType.STRONG_SELL
        
        # Calculate distribution based on alignment with final score
        distribution: dict[PredictionType, float] = {}
        for vote in votes:
            alignment = 1.0 if vote.prediction.to_score() * score > 0 else 0.0
            distribution[vote.prediction] = distribution.get(vote.prediction, 0.0) + alignment
        
        total_dist = sum(distribution.values())
        if total_dist > 0:
            distribution = {k: v / total_dist for k, v in distribution.items()}
        
        majority_share = max(distribution.values()) if distribution else 0.0
        
        return ConsensusResult(
            prediction=prediction,
            strength=self._calculate_strength(majority_share),
            confidence=abs(score),
            vote_distribution=distribution,
            method_used=VotingMethod.BAYESIAN,
            votes=votes,
            reasoning=f"Bayesian aggregation: {prediction.value} (score: {score:.2f})"
        )
    
    def _borda_count(self, votes: list[Vote]) -> ConsensusResult:
        """Borda count voting (rank-based scoring)."""
        if not votes:
            return self._no_consensus_result(votes, VotingMethod.BORDA_COUNT)
        
        # Assign Borda points based on prediction strength
        # Strong Buy = 4, Buy = 3, Hold = 2, Sell = 1, Strong Sell = 0
        borda_points = {
            PredictionType.STRONG_BUY: 4,
            PredictionType.BUY: 3,
            PredictionType.HOLD: 2,
            PredictionType.SELL: 1,
            PredictionType.STRONG_SELL: 0,
            PredictionType.UNCERTAIN: 2  # Neutral
        }
        
        # Calculate weighted Borda score
        total_score = 0.0
        total_weight = 0.0
        
        for vote in votes:
            points = borda_points.get(vote.prediction, 2)
            total_score += points * vote.effective_weight
            total_weight += vote.effective_weight
        
        avg_score = total_score / total_weight if total_weight > 0 else 2
        
        # Convert average score to prediction
        if avg_score >= 3.5:
            prediction = PredictionType.STRONG_BUY
        elif avg_score >= 2.5:
            prediction = PredictionType.BUY
        elif avg_score >= 1.5:
            prediction = PredictionType.HOLD
        elif avg_score >= 0.5:
            prediction = PredictionType.SELL
        else:
            prediction = PredictionType.STRONG_SELL
        
        # Calculate distribution
        distribution = self._count_votes(votes, use_weights=True, use_confidence=True)
        majority_share = max(distribution.values()) if distribution else 0.0
        
        # Confidence based on score deviation from neutral
        confidence = abs(avg_score - 2) / 2
        
        return ConsensusResult(
            prediction=prediction,
            strength=self._calculate_strength(majority_share),
            confidence=confidence,
            vote_distribution=distribution,
            method_used=VotingMethod.BORDA_COUNT,
            votes=votes,
            reasoning=f"Borda count: {prediction.value} (avg score: {avg_score:.2f}/4)"
        )
    
    def _unanimous(self, votes: list[Vote]) -> ConsensusResult:
        """Unanimous consent (requires high agreement threshold)."""
        if not votes:
            return self._no_consensus_result(votes, VotingMethod.UNANIMOUS)
        
        distribution = self._count_votes(votes, use_weights=True, use_confidence=True)
        
        if not distribution:
            return self._no_consensus_result(votes, VotingMethod.UNANIMOUS)
        
        winner = max(distribution.items(), key=lambda x: x[1])
        prediction, share = winner
        
        # Check if unanimous threshold is met
        if share >= self.unanimous_threshold:
            strength = ConsensusStrength.STRONG
            winning_votes = [v for v in votes if v.prediction == prediction]
            confidence = statistics.mean([v.confidence for v in winning_votes]) if winning_votes else 0.0
            reasoning = f"Unanimous: {prediction.value} with {share:.1%} agreement (threshold: {self.unanimous_threshold:.1%})"
        else:
            # No unanimous consent
            prediction = PredictionType.HOLD
            strength = ConsensusStrength.SPLIT
            confidence = 0.0
            reasoning = f"No unanimous consent: best was {winner[0].value} at {share:.1%} (threshold: {self.unanimous_threshold:.1%})"
        
        return ConsensusResult(
            prediction=prediction,
            strength=strength,
            confidence=confidence,
            vote_distribution=distribution,
            method_used=VotingMethod.UNANIMOUS,
            votes=votes,
            reasoning=reasoning
        )
    
    def _no_consensus_result(
        self,
        votes: list[Vote],
        method: VotingMethod
    ) -> ConsensusResult:
        """Return result when no consensus can be reached."""
        return ConsensusResult(
            prediction=PredictionType.UNCERTAIN,
            strength=ConsensusStrength.SPLIT,
            confidence=0.0,
            vote_distribution={},
            method_used=method,
            votes=votes,
            reasoning="No valid votes for consensus"
        )
    
    def vote(
        self,
        predictions: list[ModelPrediction],
        method: Optional[VotingMethod] = None
    ) -> ConsensusResult:
        """
        Run consensus voting on predictions.
        
        Args:
            predictions: List of model predictions
            method: Voting method to use (None = use default)
            
        Returns:
            ConsensusResult with voting outcome
        """
        method = method or self.default_method
        votes = self._predictions_to_votes(predictions)
        
        if not votes:
            logger.warning("No valid votes for consensus")
            return self._no_consensus_result(votes, method)
        
        voting_func = self._methods.get(method)
        if not voting_func:
            raise ValueError(f"Unknown voting method: {method}")
        
        result = voting_func(votes)
        
        logger.info(
            f"Consensus: {result.prediction.value} ({result.strength.value}) "
            f"via {method.value} with {len(votes)} votes"
        )
        
        return result
    
    def multi_method_vote(
        self,
        predictions: list[ModelPrediction],
        methods: Optional[list[VotingMethod]] = None
    ) -> dict[VotingMethod, ConsensusResult]:
        """
        Run multiple voting methods and return all results.
        
        Args:
            predictions: List of model predictions
            methods: Methods to use (None = all methods)
            
        Returns:
            Dict of method -> ConsensusResult
        """
        methods = methods or list(VotingMethod)
        results = {}
        
        for method in methods:
            results[method] = self.vote(predictions, method)
        
        return results
    
    def meta_consensus(
        self,
        predictions: list[ModelPrediction]
    ) -> ConsensusResult:
        """
        Get meta-consensus from all voting methods.
        
        Runs all methods and aggregates their results.
        """
        all_results = self.multi_method_vote(predictions)
        
        # Create "votes" from each method's result
        method_votes = []
        for method, result in all_results.items():
            if result.prediction != PredictionType.UNCERTAIN:
                method_votes.append(Vote(
                    provider=method.value,
                    prediction=result.prediction,
                    weight=1.0,  # Equal weight for each method
                    confidence=result.confidence
                ))
        
        if not method_votes:
            return self._no_consensus_result([], VotingMethod.CONFIDENCE_WEIGHTED)
        
        # Use confidence-weighted voting on method results
        distribution = self._count_votes(method_votes, use_weights=True, use_confidence=True)
        
        winner = max(distribution.items(), key=lambda x: x[1])
        prediction, share = winner
        
        # Average confidence across agreeing methods
        agreeing = [v for v in method_votes if v.prediction == prediction]
        avg_confidence = statistics.mean([v.confidence for v in agreeing]) if agreeing else 0.0
        
        return ConsensusResult(
            prediction=prediction,
            strength=self._calculate_strength(share),
            confidence=avg_confidence * share,
            vote_distribution=distribution,
            method_used=VotingMethod.CONFIDENCE_WEIGHTED,
            votes=[],  # Original votes not directly used
            reasoning=f"Meta-consensus: {prediction.value} agreed by {len(agreeing)}/{len(all_results)} methods"
        )
