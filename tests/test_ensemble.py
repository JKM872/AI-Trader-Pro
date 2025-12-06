"""
Tests for Ensemble AI Module.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import statistics

from trader.ensemble.ai_ensemble import (
    AIEnsemble,
    EnsembleResult,
    ModelPrediction,
    PredictionType
)
from trader.ensemble.consensus_engine import (
    ConsensusEngine,
    ConsensusResult,
    Vote,
    VotingMethod,
    ConsensusStrength
)
from trader.ensemble.model_weighting import (
    ModelWeightManager,
    ModelPerformance
)


# ==================== PredictionType Tests ====================

class TestPredictionType:
    """Tests for PredictionType enum."""
    
    def test_from_text_strong_buy(self):
        """Test parsing strong buy signals."""
        assert PredictionType.from_text("STRONG BUY") == PredictionType.STRONG_BUY
        assert PredictionType.from_text("strongly bullish") == PredictionType.STRONG_BUY
        assert PredictionType.from_text("very bullish outlook") == PredictionType.STRONG_BUY
        assert PredictionType.from_text("definitely buy this") == PredictionType.STRONG_BUY
    
    def test_from_text_buy(self):
        """Test parsing buy signals."""
        assert PredictionType.from_text("BUY") == PredictionType.BUY
        assert PredictionType.from_text("bullish") == PredictionType.BUY
        assert PredictionType.from_text("go long") == PredictionType.BUY
        assert PredictionType.from_text("positive outlook") == PredictionType.BUY
    
    def test_from_text_hold(self):
        """Test parsing hold signals."""
        assert PredictionType.from_text("HOLD") == PredictionType.HOLD
        assert PredictionType.from_text("neutral stance") == PredictionType.HOLD
        assert PredictionType.from_text("wait and see") == PredictionType.HOLD
    
    def test_from_text_sell(self):
        """Test parsing sell signals."""
        assert PredictionType.from_text("SELL") == PredictionType.SELL
        assert PredictionType.from_text("bearish") == PredictionType.SELL
        assert PredictionType.from_text("go short") == PredictionType.SELL
    
    def test_from_text_strong_sell(self):
        """Test parsing strong sell signals."""
        assert PredictionType.from_text("STRONG SELL") == PredictionType.STRONG_SELL
        assert PredictionType.from_text("strongly bearish") == PredictionType.STRONG_SELL
        assert PredictionType.from_text("very bearish") == PredictionType.STRONG_SELL
    
    def test_from_text_uncertain(self):
        """Test uncertain when no signal found."""
        assert PredictionType.from_text("I don't know") == PredictionType.UNCERTAIN
        assert PredictionType.from_text("unclear data") == PredictionType.UNCERTAIN
    
    def test_to_score(self):
        """Test score conversion."""
        assert PredictionType.STRONG_BUY.to_score() == 1.0
        assert PredictionType.BUY.to_score() == 0.5
        assert PredictionType.HOLD.to_score() == 0.0
        assert PredictionType.SELL.to_score() == -0.5
        assert PredictionType.STRONG_SELL.to_score() == -1.0
        assert PredictionType.UNCERTAIN.to_score() == 0.0


# ==================== ModelPrediction Tests ====================

class TestModelPrediction:
    """Tests for ModelPrediction dataclass."""
    
    def test_valid_prediction(self):
        """Test valid prediction check."""
        pred = ModelPrediction(
            provider_name="test",
            model_name="model",
            prediction=PredictionType.BUY,
            confidence=0.8,
            reasoning="Test",
            response_time_ms=100,
            raw_response="BUY"
        )
        assert pred.is_valid is True
    
    def test_invalid_prediction_with_error(self):
        """Test invalid prediction with error."""
        pred = ModelPrediction(
            provider_name="test",
            model_name="model",
            prediction=PredictionType.BUY,
            confidence=0.8,
            reasoning="Test",
            response_time_ms=100,
            raw_response="BUY",
            error="API Error"
        )
        assert pred.is_valid is False
    
    def test_invalid_prediction_uncertain(self):
        """Test uncertain prediction is invalid."""
        pred = ModelPrediction(
            provider_name="test",
            model_name="model",
            prediction=PredictionType.UNCERTAIN,
            confidence=0.8,
            reasoning="Test",
            response_time_ms=100,
            raw_response="unclear"
        )
        assert pred.is_valid is False
    
    def test_weighted_score(self):
        """Test weighted score calculation."""
        pred = ModelPrediction(
            provider_name="test",
            model_name="model",
            prediction=PredictionType.BUY,
            confidence=0.8,
            reasoning="Test",
            response_time_ms=100,
            raw_response="BUY"
        )
        assert pred.weighted_score == 0.4  # 0.5 * 0.8


# ==================== EnsembleResult Tests ====================

class TestEnsembleResult:
    """Tests for EnsembleResult dataclass."""
    
    def test_valid_predictions_filter(self):
        """Test filtering valid predictions."""
        predictions = [
            ModelPrediction("p1", "m1", PredictionType.BUY, 0.8, "", 100, ""),
            ModelPrediction("p2", "m2", PredictionType.UNCERTAIN, 0.5, "", 100, ""),
            ModelPrediction("p3", "m3", PredictionType.SELL, 0.7, "", 100, "", error="Error"),
        ]
        
        result = EnsembleResult(
            symbol="TEST",
            final_prediction=PredictionType.BUY,
            confidence=0.8,
            consensus_score=0.5,
            agreement_ratio=0.8,
            predictions=predictions,
            reasoning="Test"
        )
        
        assert len(result.valid_predictions) == 1
        assert result.model_count == 1
    
    def test_to_signal_confidence(self):
        """Test signal confidence calculation."""
        result = EnsembleResult(
            symbol="TEST",
            final_prediction=PredictionType.BUY,
            confidence=0.8,
            consensus_score=0.6,
            agreement_ratio=0.9,
            predictions=[],
            reasoning="Test"
        )
        
        expected = 0.6 * 0.8 * 0.9  # |consensus_score| * confidence * agreement
        assert result.to_signal_confidence() == pytest.approx(expected, rel=0.01)


# ==================== AIEnsemble Tests ====================

class TestAIEnsemble:
    """Tests for AIEnsemble class."""
    
    def test_initialization(self):
        """Test ensemble initialization."""
        with patch('trader.ensemble.ai_ensemble.get_all_available_providers') as mock_providers:
            mock_providers.return_value = {}
            ensemble = AIEnsemble(providers=['groq', 'ollama'])
            assert 'groq' in ensemble.providers
            assert 'ollama' in ensemble.providers
    
    def test_parse_response_structured(self):
        """Test parsing structured response."""
        with patch('trader.ensemble.ai_ensemble.get_all_available_providers') as mock_providers:
            mock_providers.return_value = {}
            ensemble = AIEnsemble()
            
            response = """
            RECOMMENDATION: BUY
            CONFIDENCE: 75%
            REASONING: Strong fundamentals and technical setup.
            """
            
            prediction, confidence, reasoning = ensemble._parse_response(response)
            
            assert prediction == PredictionType.BUY
            assert confidence == 0.75
            assert "Strong fundamentals" in reasoning
    
    def test_parse_response_unstructured(self):
        """Test parsing unstructured response."""
        with patch('trader.ensemble.ai_ensemble.get_all_available_providers') as mock_providers:
            mock_providers.return_value = {}
            ensemble = AIEnsemble()
            
            response = "I think this stock is bullish and you should consider buying."
            
            prediction, confidence, reasoning = ensemble._parse_response(response)
            
            assert prediction == PredictionType.BUY
    
    def test_aggregate_predictions_consensus(self):
        """Test prediction aggregation."""
        with patch('trader.ensemble.ai_ensemble.get_all_available_providers') as mock_providers:
            mock_providers.return_value = {}
            ensemble = AIEnsemble(min_confidence=0.3, min_models=2)
            
            predictions = [
                ModelPrediction("p1", "m1", PredictionType.BUY, 0.8, "reason", 100, ""),
                ModelPrediction("p2", "m2", PredictionType.BUY, 0.7, "reason", 100, ""),
                ModelPrediction("p3", "m3", PredictionType.HOLD, 0.6, "reason", 100, ""),
            ]
            
            result = ensemble._aggregate_predictions(predictions, "TEST")
            
            assert result.final_prediction in [PredictionType.BUY, PredictionType.HOLD]
            assert result.agreement_ratio > 0
            assert result.model_count == 3
    
    def test_aggregate_predictions_insufficient(self):
        """Test aggregation with insufficient predictions."""
        with patch('trader.ensemble.ai_ensemble.get_all_available_providers') as mock_providers:
            mock_providers.return_value = {}
            ensemble = AIEnsemble(min_confidence=0.5, min_models=3)
            
            predictions = [
                ModelPrediction("p1", "m1", PredictionType.BUY, 0.8, "reason", 100, ""),
            ]
            
            result = ensemble._aggregate_predictions(predictions, "TEST")
            
            assert result.final_prediction == PredictionType.UNCERTAIN
            assert result.confidence == 0.0
    
    def test_get_model_weight(self):
        """Test model weight retrieval."""
        with patch('trader.ensemble.ai_ensemble.get_all_available_providers') as mock_providers:
            mock_providers.return_value = {}
            ensemble = AIEnsemble(model_weights={'groq': 1.5, 'ollama': 0.8})
            
            assert ensemble._get_model_weight('groq') == 1.5
            assert ensemble._get_model_weight('ollama') == 0.8
            assert ensemble._get_model_weight('unknown') == 1.0
    
    def test_add_remove_provider(self):
        """Test adding and removing providers."""
        with patch('trader.ensemble.ai_ensemble.get_all_available_providers') as mock_providers:
            mock_providers.return_value = {}
            with patch('trader.ensemble.ai_ensemble.create_provider') as mock_create:
                mock_create.return_value = Mock()
                
                ensemble = AIEnsemble(providers=[])
                
                # Add provider
                assert ensemble.add_provider('test_provider') is True
                assert 'test_provider' in ensemble.providers
                
                # Remove provider
                assert ensemble.remove_provider('test_provider') is True
                assert 'test_provider' not in ensemble.providers
    
    def test_update_weights(self):
        """Test weight updates."""
        with patch('trader.ensemble.ai_ensemble.get_all_available_providers') as mock_providers:
            mock_providers.return_value = {}
            ensemble = AIEnsemble()
            
            ensemble.update_weights({'groq': 1.2, 'ollama': 0.9})
            
            assert ensemble.model_weights['groq'] == 1.2
            assert ensemble.model_weights['ollama'] == 0.9


# ==================== ConsensusEngine Tests ====================

class TestConsensusEngine:
    """Tests for ConsensusEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create ConsensusEngine instance."""
        return ConsensusEngine(
            model_weights={'p1': 1.0, 'p2': 1.5, 'p3': 0.8}
        )
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        return [
            ModelPrediction("p1", "m1", PredictionType.BUY, 0.8, "reason", 100, ""),
            ModelPrediction("p2", "m2", PredictionType.BUY, 0.7, "reason", 100, ""),
            ModelPrediction("p3", "m3", PredictionType.SELL, 0.6, "reason", 100, ""),
        ]
    
    def test_simple_majority(self, engine, sample_predictions):
        """Test simple majority voting."""
        result = engine.vote(sample_predictions, VotingMethod.SIMPLE_MAJORITY)
        
        assert result.prediction == PredictionType.BUY
        assert result.method_used == VotingMethod.SIMPLE_MAJORITY
        assert result.vote_distribution[PredictionType.BUY] == pytest.approx(2/3, rel=0.01)
    
    def test_weighted_majority(self, engine, sample_predictions):
        """Test weighted majority voting."""
        result = engine.vote(sample_predictions, VotingMethod.WEIGHTED_MAJORITY)
        
        assert result.prediction == PredictionType.BUY
        assert result.method_used == VotingMethod.WEIGHTED_MAJORITY
    
    def test_confidence_weighted(self, engine, sample_predictions):
        """Test confidence-weighted voting."""
        result = engine.vote(sample_predictions, VotingMethod.CONFIDENCE_WEIGHTED)
        
        assert result.prediction == PredictionType.BUY
        assert result.method_used == VotingMethod.CONFIDENCE_WEIGHTED
    
    def test_bayesian(self, engine, sample_predictions):
        """Test Bayesian voting."""
        result = engine.vote(sample_predictions, VotingMethod.BAYESIAN)
        
        assert result.method_used == VotingMethod.BAYESIAN
        assert result.prediction in list(PredictionType)
    
    def test_borda_count(self, engine, sample_predictions):
        """Test Borda count voting."""
        result = engine.vote(sample_predictions, VotingMethod.BORDA_COUNT)
        
        assert result.method_used == VotingMethod.BORDA_COUNT
        assert result.prediction in list(PredictionType)
    
    def test_unanimous_success(self, engine):
        """Test unanimous voting with consensus."""
        predictions = [
            ModelPrediction("p1", "m1", PredictionType.BUY, 0.9, "", 100, ""),
            ModelPrediction("p2", "m2", PredictionType.BUY, 0.85, "", 100, ""),
            ModelPrediction("p3", "m3", PredictionType.BUY, 0.8, "", 100, ""),
        ]
        
        result = engine.vote(predictions, VotingMethod.UNANIMOUS)
        
        assert result.prediction == PredictionType.BUY
        assert result.strength == ConsensusStrength.STRONG
    
    def test_unanimous_failure(self, engine, sample_predictions):
        """Test unanimous voting without consensus."""
        engine.unanimous_threshold = 0.95
        result = engine.vote(sample_predictions, VotingMethod.UNANIMOUS)
        
        assert result.prediction == PredictionType.HOLD
        assert result.strength == ConsensusStrength.SPLIT
    
    def test_multi_method_vote(self, engine, sample_predictions):
        """Test running multiple voting methods."""
        results = engine.multi_method_vote(sample_predictions)
        
        assert len(results) == len(VotingMethod)
        for method in VotingMethod:
            assert method in results
    
    def test_meta_consensus(self, engine, sample_predictions):
        """Test meta-consensus from all methods."""
        result = engine.meta_consensus(sample_predictions)
        
        assert result.prediction in list(PredictionType)
        assert 0 <= result.confidence <= 1
    
    def test_consensus_strength_calculation(self, engine):
        """Test consensus strength classification."""
        assert engine._calculate_strength(0.85) == ConsensusStrength.STRONG
        assert engine._calculate_strength(0.7) == ConsensusStrength.MODERATE
        assert engine._calculate_strength(0.55) == ConsensusStrength.WEAK
        assert engine._calculate_strength(0.4) == ConsensusStrength.SPLIT
    
    def test_empty_votes(self, engine):
        """Test handling empty vote list."""
        result = engine.vote([], VotingMethod.SIMPLE_MAJORITY)
        
        assert result.prediction == PredictionType.UNCERTAIN
        assert result.strength == ConsensusStrength.SPLIT
    
    def test_actionable_result(self, engine):
        """Test actionable result check."""
        result = ConsensusResult(
            prediction=PredictionType.BUY,
            strength=ConsensusStrength.STRONG,
            confidence=0.8,
            vote_distribution={PredictionType.BUY: 0.8},
            method_used=VotingMethod.SIMPLE_MAJORITY,
            votes=[],
            reasoning=""
        )
        
        assert result.is_actionable is True
        
        # Non-actionable
        result2 = ConsensusResult(
            prediction=PredictionType.UNCERTAIN,
            strength=ConsensusStrength.SPLIT,
            confidence=0.3,
            vote_distribution={},
            method_used=VotingMethod.SIMPLE_MAJORITY,
            votes=[],
            reasoning=""
        )
        
        assert result2.is_actionable is False


# ==================== ModelWeightManager Tests ====================

class TestModelWeightManager:
    """Tests for ModelWeightManager class."""
    
    @pytest.fixture
    def manager(self, tmp_path):
        """Create ModelWeightManager instance."""
        return ModelWeightManager(
            persistence_path=tmp_path / "weights.json",
            decay_days=30,
            prior_weight=1.0,
            learning_rate=0.1
        )
    
    def test_get_model_creates_new(self, manager):
        """Test getting a new model creates entry."""
        model = manager.get_model("test_provider")
        
        assert model.provider_name == "test_provider"
        assert model.weight == 1.0
        assert model.total_predictions == 0
    
    def test_record_correct_prediction(self, manager):
        """Test recording correct prediction."""
        manager.record_outcome(
            provider_name="test",
            was_correct=True,
            profit=100.0,
            confidence=0.8
        )
        
        model = manager.get_model("test")
        
        assert model.total_predictions == 1
        assert model.correct_predictions == 1
        assert model.total_profit == 100.0
        assert model.weight > 1.0  # Should increase
    
    def test_record_incorrect_prediction(self, manager):
        """Test recording incorrect prediction."""
        # First record a correct to establish baseline
        manager.record_outcome("test", True, 50, 0.7)
        initial_weight = manager.get_model("test").weight
        
        # Record incorrect
        manager.record_outcome("test", False, -50, 0.8)
        
        model = manager.get_model("test")
        
        assert model.total_predictions == 2
        assert model.correct_predictions == 1
        assert model.weight < initial_weight  # Should decrease
    
    def test_accuracy_calculation(self, manager):
        """Test accuracy calculation."""
        for _ in range(7):
            manager.record_outcome("test", True, 10, 0.7)
        for _ in range(3):
            manager.record_outcome("test", False, -10, 0.7)
        
        model = manager.get_model("test")
        
        assert model.accuracy == 0.7
    
    def test_get_weights(self, manager):
        """Test getting all weights."""
        manager.record_outcome("p1", True, 10, 0.7)
        manager.record_outcome("p2", True, 10, 0.8)
        manager.record_outcome("p3", False, -10, 0.6)
        
        weights = manager.get_weights()
        
        assert "p1" in weights
        assert "p2" in weights
        assert "p3" in weights
    
    def test_get_ranking(self, manager):
        """Test model ranking."""
        # Make p2 best performer
        for _ in range(10):
            manager.record_outcome("p1", True, 10, 0.7)
            manager.record_outcome("p2", True, 20, 0.9)
        for _ in range(5):
            manager.record_outcome("p1", False, -10, 0.7)
        
        ranking = manager.get_ranking()
        
        assert len(ranking) == 2
        assert ranking[0][0] == "p2"  # p2 should be ranked first
    
    def test_get_best_models(self, manager):
        """Test getting best models."""
        for _ in range(10):
            manager.record_outcome("p1", True, 10, 0.9)
            manager.record_outcome("p2", True, 5, 0.7)
            manager.record_outcome("p3", False, -5, 0.5)
        
        best = manager.get_best_models(2)
        
        assert len(best) == 2
        assert "p1" in best
    
    def test_reset_model(self, manager):
        """Test resetting a model."""
        manager.record_outcome("test", True, 10, 0.7)
        manager.reset_model("test")
        
        assert "test" not in manager.models
    
    def test_reset_all(self, manager):
        """Test resetting all models."""
        manager.record_outcome("p1", True, 10, 0.7)
        manager.record_outcome("p2", True, 10, 0.7)
        manager.reset_all()
        
        assert len(manager.models) == 0
    
    def test_calibrate_weights(self, manager):
        """Test weight calibration."""
        manager.record_outcome("p1", True, 10, 0.9)
        manager.record_outcome("p2", True, 10, 0.7)
        manager.record_outcome("p3", False, -10, 0.5)
        
        manager.calibrate_weights()
        
        weights = list(manager.get_weights().values())
        avg_weight = statistics.mean(weights)
        
        assert abs(avg_weight - 1.0) < 0.1
    
    def test_get_statistics(self, manager):
        """Test getting aggregate statistics."""
        for _ in range(5):
            manager.record_outcome("p1", True, 10, 0.8)
            manager.record_outcome("p2", False, -5, 0.6)
        
        stats = manager.get_statistics()
        
        assert stats['total_models'] == 2
        assert stats['total_predictions'] == 10
        assert 'avg_accuracy' in stats
        assert stats['total_profit'] == 25  # 50 - 25
    
    def test_persistence(self, tmp_path):
        """Test saving and loading performance data."""
        path = tmp_path / "weights.json"
        
        # Create and save
        manager1 = ModelWeightManager(persistence_path=path)
        manager1.record_outcome("test", True, 100, 0.9)
        
        # Load in new instance
        manager2 = ModelWeightManager(persistence_path=path)
        
        model = manager2.get_model("test")
        assert model.total_predictions == 1
        assert model.total_profit == 100


# ==================== ModelPerformance Tests ====================

class TestModelPerformance:
    """Tests for ModelPerformance dataclass."""
    
    def test_initial_state(self):
        """Test initial state."""
        perf = ModelPerformance(provider_name="test")
        
        assert perf.accuracy == 0.5  # Prior
        assert perf.total_predictions == 0
        assert perf.profit_per_prediction == 0.0
    
    def test_record_prediction(self):
        """Test recording predictions."""
        perf = ModelPerformance(provider_name="test")
        
        perf.record_prediction(True, 100.0, 0.8, 150.0)
        
        assert perf.total_predictions == 1
        assert perf.correct_predictions == 1
        assert perf.total_profit == 100.0
        assert perf.last_prediction is not None
    
    def test_recent_accuracy(self):
        """Test recent accuracy tracking."""
        perf = ModelPerformance(provider_name="test", recent_window=5)
        
        # Record 3 correct, 2 incorrect
        for _ in range(3):
            perf.record_prediction(True, 10, 0.7, 100)
        for _ in range(2):
            perf.record_prediction(False, -10, 0.7, 100)
        
        assert perf.accuracy == 0.6  # 3/5
    
    def test_to_dict_from_dict(self):
        """Test serialization round trip."""
        perf = ModelPerformance(
            provider_name="test",
            total_predictions=10,
            correct_predictions=7,
            weight=1.2
        )
        perf.record_prediction(True, 100, 0.8, 150)
        
        data = perf.to_dict()
        restored = ModelPerformance.from_dict(data)
        
        assert restored.provider_name == perf.provider_name
        assert restored.total_predictions == perf.total_predictions
        assert restored.weight == perf.weight
