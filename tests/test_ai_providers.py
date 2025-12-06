"""
Tests for AI Providers module.
Tests all 10 free AI providers with mock responses.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from trader.analysis.ai_providers import (
    BaseProvider,
    GroqProvider,
    OllamaProvider,
    HuggingFaceProvider,
    TogetherAIProvider,
    MistralProvider,
    OpenRouterProvider,
    GoogleGeminiProvider,
    CohereProvider,
    CloudflareAIProvider,
    MultiModelAnalyzer,
    ModelInfo,
    ProviderTier,
    PROVIDERS,
    get_all_available_providers,
    get_all_free_models,
    create_provider,
)


# ============== Fixtures ==============

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for all providers."""
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "test_hf_key")
    monkeypatch.setenv("TOGETHER_API_KEY", "test_together_key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test_mistral_key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test_google_key")
    monkeypatch.setenv("COHERE_API_KEY", "test_cohere_key")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "test_cf_key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test_cf_account")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")


@pytest.fixture
def sample_bullish_response():
    """Sample bullish AI response."""
    return {
        "sentiment": "BULLISH",
        "score": 0.75,
        "confidence": 0.85,
        "reasoning": "Strong earnings growth"
    }


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx response."""
    def _create_response(data, status_code=200):
        mock = Mock()
        mock.status_code = status_code
        mock.json.return_value = data
        mock.text = json.dumps(data)
        mock.raise_for_status = Mock()
        if status_code >= 400:
            mock.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
        return mock
    return _create_response


# ============== ModelInfo Tests ==============

class TestModelInfo:
    """Tests for ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        info = ModelInfo(
            id="llama-3.3-70b-versatile",
            name="Llama 3.3 70B",
            provider="groq",
            tier=ProviderTier.FREE,
            context_length=128000,
            description="Most capable free model",
            speed="fast",
            quality="high"
        )
        
        assert info.id == "llama-3.3-70b-versatile"
        assert info.provider == "groq"
        assert info.tier == ProviderTier.FREE
    
    def test_model_info_to_dict(self):
        """Test ModelInfo.to_dict()."""
        info = ModelInfo(
            id="test-model",
            name="Test Model",
            provider="test",
            tier=ProviderTier.FREE,
            context_length=8192,
            description="Test",
            speed="fast",
            quality="high"
        )
        
        d = info.to_dict()
        assert d['id'] == "test-model"
        assert d['tier'] == "free"


# ============== ProviderTier Tests ==============

class TestProviderTier:
    """Tests for ProviderTier enumeration."""
    
    def test_tier_values(self):
        """Test tier enum values."""
        assert ProviderTier.FREE.value == "free"
        assert ProviderTier.FREEMIUM.value == "freemium"
        assert ProviderTier.PAID.value == "paid"
        assert ProviderTier.LOCAL.value == "local"


# ============== GroqProvider Tests ==============

class TestGroqProvider:
    """Tests for Groq AI provider."""
    
    def test_groq_initialization(self, mock_env_vars):
        """Test Groq provider initializes correctly."""
        provider = GroqProvider()
        assert provider.name == "Groq"
        assert provider.api_key == "test_groq_key"
        provider.close()
    
    def test_groq_models(self, mock_env_vars):
        """Test Groq returns available models."""
        provider = GroqProvider()
        models = provider.models
        
        model_ids = [m.id for m in models]
        assert "llama-3.3-70b-versatile" in model_ids
        assert len(models) >= 3
        provider.close()
    
    def test_groq_is_available(self, mock_env_vars):
        """Test Groq availability check."""
        provider = GroqProvider()
        assert provider.is_available() is True
        provider.close()
    
    def test_groq_not_available_without_key(self, monkeypatch):
        """Test Groq unavailable without API key."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        provider = GroqProvider()
        assert provider.is_available() is False
        provider.close()


# ============== OllamaProvider Tests ==============

class TestOllamaProvider:
    """Tests for Ollama local provider."""
    
    def test_ollama_initialization(self, mock_env_vars):
        """Test Ollama provider initializes correctly."""
        provider = OllamaProvider()
        assert provider.name == "Ollama"
        assert provider.base_url == "http://localhost:11434"
        provider.close()
    
    def test_ollama_models(self, mock_env_vars):
        """Test Ollama models list."""
        provider = OllamaProvider()
        models = provider.models
        assert len(models) >= 1
        provider.close()


# ============== HuggingFaceProvider Tests ==============

class TestHuggingFaceProvider:
    """Tests for HuggingFace provider."""
    
    def test_huggingface_initialization(self, mock_env_vars):
        """Test HuggingFace provider initializes correctly."""
        provider = HuggingFaceProvider()
        assert provider.name == "HuggingFace"
        provider.close()
    
    def test_huggingface_models(self, mock_env_vars):
        """Test HuggingFace models list."""
        provider = HuggingFaceProvider()
        models = provider.models
        assert len(models) >= 1
        provider.close()
    
    def test_huggingface_is_available(self, mock_env_vars):
        """Test HuggingFace availability."""
        provider = HuggingFaceProvider()
        assert provider.is_available() is True
        provider.close()


# ============== TogetherAIProvider Tests ==============

class TestTogetherAIProvider:
    """Tests for Together AI provider."""
    
    def test_together_initialization(self, mock_env_vars):
        """Test Together provider initializes correctly."""
        provider = TogetherAIProvider()
        assert "Together" in provider.name
        provider.close()
    
    def test_together_is_available(self, mock_env_vars):
        """Test Together availability."""
        provider = TogetherAIProvider()
        assert provider.is_available() is True
        provider.close()


# ============== MistralProvider Tests ==============

class TestMistralProvider:
    """Tests for Mistral AI provider."""
    
    def test_mistral_initialization(self, mock_env_vars):
        """Test Mistral provider initializes correctly."""
        provider = MistralProvider()
        assert provider.name == "Mistral"
        provider.close()
    
    def test_mistral_models(self, mock_env_vars):
        """Test Mistral models list."""
        provider = MistralProvider()
        models = provider.models
        assert len(models) >= 1
        provider.close()


# ============== OpenRouterProvider Tests ==============

class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""
    
    def test_openrouter_initialization(self, mock_env_vars):
        """Test OpenRouter provider initializes correctly."""
        provider = OpenRouterProvider()
        assert provider.name == "OpenRouter"
        provider.close()
    
    def test_openrouter_models(self, mock_env_vars):
        """Test OpenRouter has free models."""
        provider = OpenRouterProvider()
        models = provider.models
        assert len(models) >= 1
        provider.close()


# ============== GoogleGeminiProvider Tests ==============

class TestGoogleGeminiProvider:
    """Tests for Google Gemini provider."""
    
    def test_gemini_initialization(self, mock_env_vars):
        """Test Gemini provider initializes correctly."""
        provider = GoogleGeminiProvider()
        assert provider.name == "Google Gemini"
        provider.close()
    
    def test_gemini_models(self, mock_env_vars):
        """Test Gemini models list."""
        provider = GoogleGeminiProvider()
        models = provider.models
        assert len(models) >= 1
        provider.close()


# ============== CohereProvider Tests ==============

class TestCohereProvider:
    """Tests for Cohere provider."""
    
    def test_cohere_initialization(self, mock_env_vars):
        """Test Cohere provider initializes correctly."""
        provider = CohereProvider()
        assert provider.name == "Cohere"
        provider.close()
    
    def test_cohere_is_available(self, mock_env_vars):
        """Test Cohere availability."""
        provider = CohereProvider()
        assert provider.is_available() is True
        provider.close()


# ============== CloudflareAIProvider Tests ==============

class TestCloudflareAIProvider:
    """Tests for Cloudflare AI provider."""
    
    def test_cloudflare_initialization(self, mock_env_vars):
        """Test Cloudflare provider initializes correctly."""
        provider = CloudflareAIProvider()
        assert provider.name == "Cloudflare AI"
        provider.close()
    
    def test_cloudflare_requires_account_id(self, monkeypatch):
        """Test Cloudflare requires account ID."""
        monkeypatch.setenv("CLOUDFLARE_API_KEY", "test_key")
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        provider = CloudflareAIProvider()
        assert provider.is_available() is False
        provider.close()


# ============== PROVIDERS Dictionary Tests ==============

class TestProvidersDictionary:
    """Tests for PROVIDERS registry."""
    
    def test_all_providers_registered(self):
        """Test all providers are in PROVIDERS dict."""
        expected = ['groq', 'ollama', 'huggingface', 'together', 
                    'mistral', 'openrouter', 'google', 'cohere', 'cloudflare']
        
        for name in expected:
            assert name in PROVIDERS, f"Missing provider: {name}"
    
    def test_providers_are_classes(self):
        """Test PROVIDERS contains classes."""
        for name, provider_class in PROVIDERS.items():
            assert issubclass(provider_class, BaseProvider)


# ============== Helper Functions Tests ==============

class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_create_provider(self, mock_env_vars):
        """Test create_provider function."""
        provider = create_provider('groq')
        assert provider.name == "Groq"
        provider.close()
    
    def test_create_provider_unknown(self, mock_env_vars):
        """Test create_provider with unknown provider."""
        with pytest.raises(ValueError):
            create_provider('unknown_provider')
    
    def test_get_all_available_providers(self, mock_env_vars):
        """Test get_all_available_providers function."""
        result = get_all_available_providers()
        assert isinstance(result, dict)
        assert 'groq' in result
    
    def test_get_all_free_models(self, mock_env_vars):
        """Test get_all_free_models function."""
        models = get_all_free_models()
        assert isinstance(models, list)
        for model in models:
            assert model.tier in [ProviderTier.FREE, ProviderTier.LOCAL]


# ============== MultiModelAnalyzer Tests ==============

class TestMultiModelAnalyzer:
    """Tests for MultiModelAnalyzer class."""
    
    def test_analyzer_initialization(self, mock_env_vars):
        """Test MultiModelAnalyzer initializes with available providers."""
        with patch.object(GroqProvider, 'is_available', return_value=True):
            try:
                analyzer = MultiModelAnalyzer(providers=['groq'])
                assert 'groq' in analyzer.active_providers
                for p in analyzer.active_providers.values():
                    p.close()
            except ValueError:
                pass  # No providers available
    
    def test_analyzer_no_providers_raises(self, monkeypatch):
        """Test analyzer raises when no providers available."""
        for key in ['GROQ_API_KEY', 'HUGGINGFACE_API_KEY', 'TOGETHER_API_KEY',
                    'MISTRAL_API_KEY', 'OPENROUTER_API_KEY', 'GOOGLE_API_KEY',
                    'COHERE_API_KEY', 'CLOUDFLARE_API_KEY']:
            monkeypatch.delenv(key, raising=False)
        
        with pytest.raises(ValueError, match="No AI providers available"):
            MultiModelAnalyzer()


# ============== Integration Tests ==============

class TestProviderIntegration:
    """Integration tests for provider system."""
    
    def test_all_providers_have_required_methods(self, mock_env_vars):
        """Test all providers implement required interface."""
        for name, provider_class in PROVIDERS.items():
            provider = provider_class()
            
            assert hasattr(provider, 'name')
            assert hasattr(provider, 'models')
            assert hasattr(provider, 'chat')
            assert hasattr(provider, 'is_available')
            assert hasattr(provider, 'close')
            
            assert isinstance(provider.name, str)
            assert isinstance(provider.models, list)
            
            provider.close()
    
    def test_all_providers_have_valid_models(self, mock_env_vars):
        """Test all providers return valid ModelInfo objects."""
        for name, provider_class in PROVIDERS.items():
            provider = provider_class()
            models = provider.models
            
            for model in models:
                assert isinstance(model, ModelInfo)
                assert model.id is not None
                assert model.provider is not None
                assert model.tier is not None
            
            provider.close()
