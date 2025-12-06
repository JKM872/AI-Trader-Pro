"""
AI Ensemble Module - Aggregating multiple AI models for consensus trading decisions.
"""

from .ai_ensemble import AIEnsemble, EnsembleResult, ModelPrediction, PredictionType
from .consensus_engine import ConsensusEngine, ConsensusResult, Vote, VotingMethod, ConsensusStrength
from .model_weighting import ModelWeightManager, ModelPerformance

__all__ = [
    'AIEnsemble',
    'EnsembleResult',
    'ModelPrediction',
    'PredictionType',
    'ModelWeightManager',
    'ModelPerformance',
    'ConsensusEngine',
    'ConsensusResult',
    'Vote',
    'VotingMethod',
    'ConsensusStrength'
]
