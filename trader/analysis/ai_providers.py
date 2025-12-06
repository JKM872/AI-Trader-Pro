"""
AI Providers - Extended module with all available FREE AI models for trading analysis.

Supported Providers:
- Groq (Free tier: Llama 3.3, Mixtral, Gemma)
- Ollama (Local, unlimited, requires local install)
- HuggingFace (Free API for inference)
- Together AI (Free tier available)
- Mistral AI (Free tier available)
- Cloudflare Workers AI (Free tier)
- Cohere (Free tier)
- OpenRouter (Free models available)
- Perplexity (Free tier)
- Google Gemini (Free tier)
"""

import logging
import os
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


class ProviderTier(Enum):
    """Provider pricing tier."""
    FREE = "free"
    FREEMIUM = "freemium"  # Free tier with limits
    PAID = "paid"
    LOCAL = "local"  # Runs locally


@dataclass
class ModelInfo:
    """Information about an AI model."""
    id: str
    name: str
    provider: str
    tier: ProviderTier
    context_length: int
    description: str
    speed: str  # "fast", "medium", "slow"
    quality: str  # "high", "medium", "low"
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'provider': self.provider,
            'tier': self.tier.value,
            'context_length': self.context_length,
            'description': self.description,
            'speed': self.speed,
            'quality': self.quality
        }


class BaseProvider(ABC):
    """Base class for AI providers."""
    
    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def models(self) -> List[ModelInfo]:
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], model: Optional[str] = None, 
             temperature: float = 0.3) -> str:
        pass
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return bool(self.api_key)
    
    def close(self):
        self.client.close()


class GroqProvider(BaseProvider):
    """Groq API - Free tier with fast inference."""
    
    BASE_URL = "https://api.groq.com/openai/v1"
    ENV_KEY = "GROQ_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv(self.ENV_KEY)
        super().__init__(api_key)
    
    @property
    def name(self) -> str:
        return "Groq"
    
    @property
    def models(self) -> List[ModelInfo]:
        return [
            ModelInfo("llama-3.3-70b-versatile", "Llama 3.3 70B", "groq", 
                     ProviderTier.FREE, 128000, "Most capable free model", "fast", "high"),
            ModelInfo("llama-3.1-8b-instant", "Llama 3.1 8B", "groq",
                     ProviderTier.FREE, 128000, "Fast lightweight model", "very_fast", "medium"),
            ModelInfo("gemma2-9b-it", "Gemma 2 9B", "groq",
                     ProviderTier.FREE, 8192, "Google's efficient model", "fast", "medium"),
            ModelInfo("mixtral-8x7b-32768", "Mixtral 8x7B", "groq",
                     ProviderTier.FREE, 32768, "MoE architecture", "fast", "high"),
        ]
    
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             temperature: float = 0.3) -> str:
        model = model or "llama-3.3-70b-versatile"
        
        response = self.client.post(
            f"{self.BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class OllamaProvider(BaseProvider):
    """Ollama - Local AI models (unlimited, free)."""
    
    DEFAULT_URL = "http://localhost:11434"
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("OLLAMA_HOST", self.DEFAULT_URL)
        super().__init__(api_key="local")
    
    @property
    def name(self) -> str:
        return "Ollama"
    
    @property
    def models(self) -> List[ModelInfo]:
        return [
            ModelInfo("llama3.2", "Llama 3.2", "ollama",
                     ProviderTier.LOCAL, 128000, "Latest Llama model", "medium", "high"),
            ModelInfo("mistral", "Mistral 7B", "ollama",
                     ProviderTier.LOCAL, 32000, "Fast and capable", "fast", "high"),
            ModelInfo("phi3", "Phi-3", "ollama",
                     ProviderTier.LOCAL, 128000, "Microsoft's efficient model", "fast", "medium"),
            ModelInfo("gemma2", "Gemma 2", "ollama",
                     ProviderTier.LOCAL, 8192, "Google's model", "medium", "high"),
            ModelInfo("qwen2.5", "Qwen 2.5", "ollama",
                     ProviderTier.LOCAL, 32000, "Alibaba's model", "medium", "high"),
            ModelInfo("deepseek-r1", "DeepSeek R1", "ollama",
                     ProviderTier.LOCAL, 64000, "Reasoning model", "slow", "very_high"),
        ]
    
    def is_available(self) -> bool:
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except:
            return False
    
    def get_installed_models(self) -> List[str]:
        """Get list of installed Ollama models."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return [m["name"] for m in response.json().get("models", [])]
        except:
            pass
        return []
    
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             temperature: float = 0.3) -> str:
        model = model or "llama3.2"
        
        response = self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Inference API - Free tier available."""
    
    BASE_URL = "https://api-inference.huggingface.co/models"
    ENV_KEY = "HUGGINGFACE_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv(self.ENV_KEY)
        super().__init__(api_key)
    
    @property
    def name(self) -> str:
        return "HuggingFace"
    
    @property
    def models(self) -> List[ModelInfo]:
        return [
            ModelInfo("microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini", "huggingface",
                     ProviderTier.FREE, 4096, "Microsoft's efficient model", "fast", "medium"),
            ModelInfo("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B", "huggingface",
                     ProviderTier.FREE, 32768, "Mistral's base model", "medium", "high"),
            ModelInfo("google/gemma-2-2b-it", "Gemma 2 2B", "huggingface",
                     ProviderTier.FREE, 8192, "Lightweight Gemma", "fast", "medium"),
            ModelInfo("Qwen/Qwen2.5-7B-Instruct", "Qwen 2.5 7B", "huggingface",
                     ProviderTier.FREE, 32000, "Alibaba's model", "medium", "high"),
        ]
    
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             temperature: float = 0.3) -> str:
        model = model or "microsoft/Phi-3-mini-4k-instruct"
        
        # Convert messages to prompt
        prompt = "\n".join([
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ]) + "\nASSISTANT:"
        
        response = self.client.post(
            f"{self.BASE_URL}/{model}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": 1000,
                    "return_full_text": False
                }
            }
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "")
        return result.get("generated_text", "")


class TogetherAIProvider(BaseProvider):
    """Together AI - Free tier with many open models."""
    
    BASE_URL = "https://api.together.xyz/v1"
    ENV_KEY = "TOGETHER_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv(self.ENV_KEY)
        super().__init__(api_key)
    
    @property
    def name(self) -> str:
        return "Together AI"
    
    @property
    def models(self) -> List[ModelInfo]:
        return [
            ModelInfo("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "Llama 3.1 8B Turbo", "together",
                     ProviderTier.FREEMIUM, 128000, "Fast Llama model", "fast", "high"),
            ModelInfo("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral 8x7B", "together",
                     ProviderTier.FREEMIUM, 32768, "MoE model", "medium", "high"),
            ModelInfo("Qwen/Qwen2.5-7B-Instruct-Turbo", "Qwen 2.5 7B Turbo", "together",
                     ProviderTier.FREEMIUM, 32000, "Fast Qwen", "fast", "high"),
            ModelInfo("google/gemma-2-9b-it", "Gemma 2 9B", "together",
                     ProviderTier.FREEMIUM, 8192, "Google's model", "medium", "high"),
        ]
    
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             temperature: float = 0.3) -> str:
        model = model or "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        
        response = self.client.post(
            f"{self.BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class MistralProvider(BaseProvider):
    """Mistral AI - Free tier available."""
    
    BASE_URL = "https://api.mistral.ai/v1"
    ENV_KEY = "MISTRAL_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv(self.ENV_KEY)
        super().__init__(api_key)
    
    @property
    def name(self) -> str:
        return "Mistral"
    
    @property
    def models(self) -> List[ModelInfo]:
        return [
            ModelInfo("mistral-small-latest", "Mistral Small", "mistral",
                     ProviderTier.FREEMIUM, 32000, "Efficient model", "fast", "high"),
            ModelInfo("open-mistral-7b", "Mistral 7B", "mistral",
                     ProviderTier.FREEMIUM, 32000, "Open source base", "fast", "medium"),
            ModelInfo("open-mixtral-8x7b", "Mixtral 8x7B", "mistral",
                     ProviderTier.FREEMIUM, 32000, "MoE architecture", "medium", "high"),
            ModelInfo("codestral-latest", "Codestral", "mistral",
                     ProviderTier.FREEMIUM, 32000, "Code specialized", "fast", "high"),
        ]
    
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             temperature: float = 0.3) -> str:
        model = model or "mistral-small-latest"
        
        response = self.client.post(
            f"{self.BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class OpenRouterProvider(BaseProvider):
    """OpenRouter - Access many models including free ones."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    ENV_KEY = "OPENROUTER_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv(self.ENV_KEY)
        super().__init__(api_key)
    
    @property
    def name(self) -> str:
        return "OpenRouter"
    
    @property
    def models(self) -> List[ModelInfo]:
        return [
            ModelInfo("meta-llama/llama-3.2-3b-instruct:free", "Llama 3.2 3B Free", "openrouter",
                     ProviderTier.FREE, 128000, "Free Llama model", "fast", "medium"),
            ModelInfo("google/gemma-2-9b-it:free", "Gemma 2 9B Free", "openrouter",
                     ProviderTier.FREE, 8192, "Free Google model", "fast", "high"),
            ModelInfo("microsoft/phi-3-mini-128k-instruct:free", "Phi-3 Mini Free", "openrouter",
                     ProviderTier.FREE, 128000, "Free Microsoft model", "fast", "medium"),
            ModelInfo("qwen/qwen-2-7b-instruct:free", "Qwen 2 7B Free", "openrouter",
                     ProviderTier.FREE, 32000, "Free Alibaba model", "fast", "high"),
            ModelInfo("mistralai/mistral-7b-instruct:free", "Mistral 7B Free", "openrouter",
                     ProviderTier.FREE, 32000, "Free Mistral model", "fast", "high"),
        ]
    
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             temperature: float = 0.3) -> str:
        model = model or "meta-llama/llama-3.2-3b-instruct:free"
        
        response = self.client.post(
            f"{self.BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/ai-trader",
                "X-Title": "AI Trader"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class GoogleGeminiProvider(BaseProvider):
    """Google Gemini - Free tier available."""
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    ENV_KEY = "GOOGLE_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv(self.ENV_KEY)
        super().__init__(api_key)
    
    @property
    def name(self) -> str:
        return "Google Gemini"
    
    @property
    def models(self) -> List[ModelInfo]:
        return [
            ModelInfo("gemini-1.5-flash", "Gemini 1.5 Flash", "google",
                     ProviderTier.FREE, 1000000, "Fast multimodal", "fast", "high"),
            ModelInfo("gemini-1.5-flash-8b", "Gemini 1.5 Flash 8B", "google",
                     ProviderTier.FREE, 1000000, "Ultra fast", "very_fast", "medium"),
            ModelInfo("gemini-1.5-pro", "Gemini 1.5 Pro", "google",
                     ProviderTier.FREEMIUM, 2000000, "Most capable", "medium", "very_high"),
        ]
    
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             temperature: float = 0.3) -> str:
        model = model or "gemini-1.5-flash"
        
        # Convert to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        response = self.client.post(
            f"{self.BASE_URL}/models/{model}:generateContent",
            params={"key": self.api_key},
            json={
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 2000
                }
            }
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


class CohereProvider(BaseProvider):
    """Cohere - Free tier available."""
    
    BASE_URL = "https://api.cohere.ai/v1"
    ENV_KEY = "COHERE_API_KEY"
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv(self.ENV_KEY)
        super().__init__(api_key)
    
    @property
    def name(self) -> str:
        return "Cohere"
    
    @property
    def models(self) -> List[ModelInfo]:
        return [
            ModelInfo("command-r", "Command R", "cohere",
                     ProviderTier.FREEMIUM, 128000, "RAG optimized", "medium", "high"),
            ModelInfo("command-r-plus", "Command R+", "cohere",
                     ProviderTier.FREEMIUM, 128000, "Most capable", "medium", "very_high"),
            ModelInfo("command-light", "Command Light", "cohere",
                     ProviderTier.FREEMIUM, 4096, "Fast and light", "fast", "medium"),
        ]
    
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             temperature: float = 0.3) -> str:
        model = model or "command-r"
        
        # Convert messages to Cohere format
        chat_history = []
        preamble = ""
        message = ""
        
        for msg in messages:
            if msg["role"] == "system":
                preamble = msg["content"]
            elif msg["role"] == "user":
                message = msg["content"]
            else:
                chat_history.append({
                    "role": "CHATBOT" if msg["role"] == "assistant" else "USER",
                    "message": msg["content"]
                })
        
        response = self.client.post(
            f"{self.BASE_URL}/chat",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "message": message,
                "preamble": preamble,
                "chat_history": chat_history,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        return response.json()["text"]


class CloudflareAIProvider(BaseProvider):
    """Cloudflare Workers AI - Free tier."""
    
    ENV_KEY = "CLOUDFLARE_API_KEY"
    ACCOUNT_KEY = "CLOUDFLARE_ACCOUNT_ID"
    
    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None):
        api_key = api_key or os.getenv(self.ENV_KEY)
        self.account_id = account_id or os.getenv(self.ACCOUNT_KEY)
        super().__init__(api_key)
    
    @property
    def name(self) -> str:
        return "Cloudflare AI"
    
    @property
    def base_url(self) -> str:
        return f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run"
    
    @property
    def models(self) -> List[ModelInfo]:
        return [
            ModelInfo("@cf/meta/llama-3.1-8b-instruct", "Llama 3.1 8B", "cloudflare",
                     ProviderTier.FREE, 8192, "Meta's efficient model", "fast", "high"),
            ModelInfo("@cf/mistral/mistral-7b-instruct-v0.2", "Mistral 7B", "cloudflare",
                     ProviderTier.FREE, 8192, "Mistral's base model", "fast", "high"),
            ModelInfo("@cf/google/gemma-7b-it", "Gemma 7B", "cloudflare",
                     ProviderTier.FREE, 8192, "Google's model", "fast", "medium"),
            ModelInfo("@cf/qwen/qwen1.5-14b-chat-awq", "Qwen 1.5 14B", "cloudflare",
                     ProviderTier.FREE, 8192, "Alibaba's model", "medium", "high"),
        ]
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.account_id)
    
    def chat(self, messages: List[Dict], model: Optional[str] = None,
             temperature: float = 0.3) -> str:
        model = model or "@cf/meta/llama-3.1-8b-instruct"
        
        response = self.client.post(
            f"{self.base_url}/{model}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000
            }
        )
        response.raise_for_status()
        return response.json()["result"]["response"]


# Registry of all providers
PROVIDERS: Dict[str, type] = {
    "groq": GroqProvider,
    "ollama": OllamaProvider,
    "huggingface": HuggingFaceProvider,
    "together": TogetherAIProvider,
    "mistral": MistralProvider,
    "openrouter": OpenRouterProvider,
    "google": GoogleGeminiProvider,
    "cohere": CohereProvider,
    "cloudflare": CloudflareAIProvider,
}


def get_all_available_providers() -> Dict[str, Dict]:
    """Get all providers and their availability status."""
    result = {}
    for name, provider_class in PROVIDERS.items():
        try:
            provider = provider_class()
            result[name] = {
                "name": provider.name,
                "available": provider.is_available(),
                "models": [m.to_dict() for m in provider.models],
                "env_key": getattr(provider_class, "ENV_KEY", None)
            }
            provider.close()
        except Exception as e:
            result[name] = {
                "name": name,
                "available": False,
                "error": str(e)
            }
    return result


def get_all_free_models() -> List[ModelInfo]:
    """Get all free models from all providers."""
    models = []
    for name, provider_class in PROVIDERS.items():
        try:
            provider = provider_class()
            if provider.is_available():
                for model in provider.models:
                    if model.tier in [ProviderTier.FREE, ProviderTier.LOCAL]:
                        models.append(model)
            provider.close()
        except:
            pass
    return models


def create_provider(name: str, **kwargs) -> BaseProvider:
    """Create a provider instance by name."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name](**kwargs)


class MultiModelAnalyzer:
    """
    Analyzer that can use multiple AI models for consensus-based analysis.
    Tests all available models and finds the best one for trading.
    """
    
    def __init__(self, providers: Optional[List[str]] = None):
        """
        Initialize with specified or all available providers.
        
        Args:
            providers: List of provider names or None for auto-detect
        """
        self.active_providers: Dict[str, BaseProvider] = {}
        
        if providers is None:
            # Auto-detect available providers
            for name, provider_class in PROVIDERS.items():
                try:
                    provider = provider_class()
                    if provider.is_available():
                        self.active_providers[name] = provider
                    else:
                        provider.close()
                except:
                    pass
        else:
            for name in providers:
                try:
                    provider = create_provider(name)
                    if provider.is_available():
                        self.active_providers[name] = provider
                except Exception as e:
                    logger.warning(f"Failed to initialize {name}: {e}")
        
        if not self.active_providers:
            raise ValueError("No AI providers available. Please set API keys.")
        
        logger.info(f"Initialized with providers: {list(self.active_providers.keys())}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using all available models."""
        system_prompt = "You are a financial analyst AI. Respond only with valid JSON."
        user_prompt = f"""Analyze the sentiment of this financial text:

Text: "{text}"

Respond in JSON:
{{"sentiment": "VERY_BULLISH|BULLISH|NEUTRAL|BEARISH|VERY_BEARISH", "score": <-1.0 to 1.0>, "confidence": <0.0 to 1.0>, "reasoning": "<brief>"}}

Return ONLY valid JSON."""

        results = {}
        scores = []
        
        for name, provider in self.active_providers.items():
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = provider.chat(messages)
                
                # Clean response
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1])
                
                data = json.loads(response.strip())
                results[name] = {
                    "sentiment": data.get("sentiment"),
                    "score": float(data.get("score", 0)),
                    "confidence": float(data.get("confidence", 0)),
                    "reasoning": data.get("reasoning", "")
                }
                scores.append(results[name]["score"])
            except Exception as e:
                results[name] = {"error": str(e)}
                logger.warning(f"Provider {name} failed: {e}")
        
        # Calculate consensus
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score >= 0.5:
                consensus = "VERY_BULLISH"
            elif avg_score >= 0.2:
                consensus = "BULLISH"
            elif avg_score >= -0.2:
                consensus = "NEUTRAL"
            elif avg_score >= -0.5:
                consensus = "BEARISH"
            else:
                consensus = "VERY_BEARISH"
            
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            agreement = max(0, 1 - variance * 2)
        else:
            consensus = "NEUTRAL"
            avg_score = 0.0
            agreement = 0.0
        
        return {
            "consensus_sentiment": consensus,
            "consensus_score": avg_score,
            "agreement_level": agreement,
            "providers_used": len(scores),
            "individual_results": results
        }
    
    def benchmark_models(self, test_texts: List[str]) -> Dict[str, Dict]:
        """
        Benchmark all models on test texts.
        
        Returns performance metrics for each model.
        """
        results = {name: {"times": [], "successes": 0, "failures": 0} 
                   for name in self.active_providers}
        
        for text in test_texts:
            for name, provider in self.active_providers.items():
                start_time = time.time()
                try:
                    messages = [
                        {"role": "system", "content": "Analyze sentiment. Return JSON only."},
                        {"role": "user", "content": f'Analyze: "{text}"'}
                    ]
                    provider.chat(messages)
                    elapsed = time.time() - start_time
                    results[name]["times"].append(elapsed)
                    results[name]["successes"] += 1
                except Exception as e:
                    results[name]["failures"] += 1
                    logger.warning(f"{name} failed: {e}")
        
        # Calculate averages
        for name, data in results.items():
            if data["times"]:
                data["avg_time"] = sum(data["times"]) / len(data["times"])
                data["min_time"] = min(data["times"])
                data["max_time"] = max(data["times"])
            data["success_rate"] = data["successes"] / (data["successes"] + data["failures"]) if (data["successes"] + data["failures"]) > 0 else 0
            del data["times"]  # Remove raw times
        
        return results
    
    def close(self):
        """Close all provider connections."""
        for provider in self.active_providers.values():
            provider.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Available AI Providers ===\n")
    providers = get_all_available_providers()
    
    for name, info in providers.items():
        status = "✓ Available" if info.get("available") else "✗ Not configured"
        print(f"{info.get('name', name)}: {status}")
        if info.get("env_key"):
            print(f"  Set {info['env_key']} to enable")
        if info.get("available") and info.get("models"):
            print(f"  Models: {len(info['models'])} available")
    
    print("\n=== Free Models ===\n")
    free_models = get_all_free_models()
    for model in free_models:
        print(f"- {model.name} ({model.provider}): {model.description}")
