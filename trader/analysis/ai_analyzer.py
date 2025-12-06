"""
AIAnalyzer - Module for AI-powered market analysis.

Supports:
- Deepseek API (primary, cost-effective)
- Groq API with Llama 3.1 (fast inference)
- Grok API / X.AI (xAI's model)
- Gemini API (Google's model)

Used for:
- Sentiment analysis of news
- Company fundamental evaluation
- Trading signal generation
- Market trend analysis
"""

import logging
import os
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import json
import time

import httpx

logger = logging.getLogger(__name__)


class Sentiment(Enum):
    """Sentiment classification."""
    VERY_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    VERY_BEARISH = -2


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    sentiment: Sentiment
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    raw_response: Optional[str] = None


@dataclass
class CompanyEvaluation:
    """Result of company fundamental evaluation."""
    symbol: str
    overall_score: float  # 0.0 to 10.0
    strengths: List[str]
    weaknesses: List[str]
    recommendation: Literal['BUY', 'HOLD', 'SELL']
    price_target_direction: Literal['UP', 'FLAT', 'DOWN']
    reasoning: str


class AIAnalyzer:
    """
    AI-powered analysis using Deepseek, Groq, Grok (X.AI), or Gemini APIs.
    
    Usage:
        analyzer = AIAnalyzer(provider='deepseek', api_key='your_key')
        sentiment = analyzer.analyze_sentiment("Apple reports record Q4 earnings")
        evaluation = analyzer.evaluate_company(fundamentals_dict)
        
    Available Providers:
        - 'deepseek': Most cost-effective, good for bulk analysis
        - 'groq': Fastest inference, great for real-time signals
        - 'grok': X.AI's model, excellent reasoning (Grok-2)
        - 'gemini': Google's Gemini Pro, versatile
    """
    
    PROVIDERS = {
        'deepseek': {
            'base_url': 'https://api.deepseek.com',
            'model': 'deepseek-chat',
            'env_key': 'DEEPSEEK_API_KEY',
            'description': 'Cost-effective, good for bulk analysis'
        },
        'groq': {
            'base_url': 'https://api.groq.com/openai/v1',
            'model': 'llama-3.3-70b-versatile',
            'env_key': 'GROQ_API_KEY',
            'description': 'Fastest inference, real-time signals'
        },
        'grok': {
            'base_url': 'https://api.x.ai/v1',
            'model': 'grok-2-latest',
            'env_key': 'XAI_API_KEY',
            'description': 'X.AI Grok-2, excellent reasoning'
        },
        'gemini': {
            'base_url': 'https://generativelanguage.googleapis.com/v1beta',
            'model': 'gemini-pro',
            'env_key': 'GEMINI_API_KEY',
            'description': 'Google Gemini Pro, versatile'
        }
    }
    
    def __init__(self, provider: str = 'deepseek', 
                 api_key: Optional[str] = None,
                 temperature: float = 0.3,
                 max_retries: int = 3):
        """
        Initialize AIAnalyzer.
        
        Args:
            provider: 'deepseek' or 'groq'
            api_key: API key (or set via environment variable)
            temperature: Model temperature (0.0-1.0)
            max_retries: Number of retries on failure
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Provider must be one of: {list(self.PROVIDERS.keys())}")
        
        self.provider = provider
        config = self.PROVIDERS[provider]
        
        self.api_key = api_key or os.getenv(config['env_key'])
        if not self.api_key:
            raise ValueError(f"API key required. Set {config['env_key']} or pass api_key parameter")
        
        self.base_url = config['base_url']
        self.model = config['model']
        self.temperature = temperature
        self.max_retries = max_retries
        
        self.client = httpx.Client(timeout=60.0)
    
    def analyze_sentiment(self, text: str, context: str = "financial news") -> SentimentResult:
        """
        Analyze sentiment of financial text.
        
        Args:
            text: Text to analyze (news headline, article, etc.)
            context: Context for analysis
        
        Returns:
            SentimentResult with sentiment, score, and reasoning
        
        Example:
            >>> result = analyzer.analyze_sentiment(
            ...     "Apple beats earnings expectations, stock soars 5%"
            ... )
            >>> print(result.sentiment)  # Sentiment.BULLISH
            >>> print(result.score)  # 0.8
        """
        prompt = f"""Analyze the sentiment of the following {context} for stock trading purposes.

Text: "{text}"

Respond in JSON format:
{{
    "sentiment": "VERY_BULLISH" | "BULLISH" | "NEUTRAL" | "BEARISH" | "VERY_BEARISH",
    "score": <float between -1.0 and 1.0>,
    "confidence": <float between 0.0 and 1.0>,
    "reasoning": "<brief explanation>"
}}

Consider:
- Impact on stock price (short and long term)
- Market implications
- Company fundamentals signals
- Industry trends

Return ONLY valid JSON, no other text."""

        response = self._call_api(prompt)
        
        try:
            data = json.loads(response)
            return SentimentResult(
                sentiment=Sentiment[data['sentiment']],
                score=float(data['score']),
                confidence=float(data['confidence']),
                reasoning=data['reasoning'],
                raw_response=response
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse sentiment response: {e}")
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                score=0.0,
                confidence=0.0,
                reasoning=f"Failed to parse AI response: {str(e)}",
                raw_response=response
            )
    
    def evaluate_company(self, fundamentals: Dict) -> CompanyEvaluation:
        """
        Evaluate a company based on fundamental data.
        
        Args:
            fundamentals: Dictionary with company fundamentals
        
        Returns:
            CompanyEvaluation with scores and recommendation
        
        Example:
            >>> fundamentals = fetcher.get_fundamentals('AAPL')
            >>> evaluation = analyzer.evaluate_company(fundamentals)
            >>> print(evaluation.recommendation)  # 'BUY'
        """
        prompt = f"""Evaluate this company for investment based on fundamentals:

Company Data:
{json.dumps(fundamentals, indent=2, default=str)}

Analyze and respond in JSON format:
{{
    "overall_score": <float 0.0 to 10.0>,
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...],
    "recommendation": "BUY" | "HOLD" | "SELL",
    "price_target_direction": "UP" | "FLAT" | "DOWN",
    "reasoning": "<comprehensive analysis>"
}}

Consider:
- Valuation metrics (P/E, P/B, PEG)
- Profitability (margins, ROE)
- Financial health (debt ratios, current ratio)
- Growth potential
- Market position

Return ONLY valid JSON, no other text."""

        response = self._call_api(prompt)
        
        try:
            data = json.loads(response)
            return CompanyEvaluation(
                symbol=fundamentals.get('symbol', 'UNKNOWN'),
                overall_score=float(data['overall_score']),
                strengths=data['strengths'],
                weaknesses=data['weaknesses'],
                recommendation=data['recommendation'],
                price_target_direction=data['price_target_direction'],
                reasoning=data['reasoning']
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            return CompanyEvaluation(
                symbol=fundamentals.get('symbol', 'UNKNOWN'),
                overall_score=5.0,
                strengths=[],
                weaknesses=[f"Analysis failed: {str(e)}"],
                recommendation='HOLD',
                price_target_direction='FLAT',
                reasoning=f"Failed to parse AI response: {str(e)}"
            )
    
    def generate_trading_insight(self, symbol: str, 
                                  price_data: Dict, 
                                  fundamentals: Dict,
                                  news_sentiment: Optional[float] = None) -> Dict:
        """
        Generate comprehensive trading insight.
        
        Args:
            symbol: Stock symbol
            price_data: Recent price data summary
            fundamentals: Company fundamentals
            news_sentiment: Average news sentiment (-1 to 1)
        
        Returns:
            Dictionary with trading insight and signals
        """
        prompt = f"""Generate a trading insight for {symbol}:

Price Data:
{json.dumps(price_data, indent=2, default=str)}

Fundamentals:
{json.dumps(fundamentals, indent=2, default=str)}

News Sentiment Score: {news_sentiment if news_sentiment else 'N/A'}

Respond in JSON:
{{
    "signal": "BUY" | "SELL" | "HOLD",
    "signal_strength": <float 0.0 to 1.0>,
    "entry_price_suggestion": <float or null>,
    "stop_loss_pct": <suggested stop loss percentage>,
    "take_profit_pct": <suggested take profit percentage>,
    "time_horizon": "SHORT" | "MEDIUM" | "LONG",
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "key_factors": ["factor1", "factor2", ...],
    "reasoning": "<detailed reasoning>"
}}

Return ONLY valid JSON."""

        response = self._call_api(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse insight response: {e}")
            return {
                'signal': 'HOLD',
                'signal_strength': 0.0,
                'error': str(e),
                'raw_response': response
            }
    
    def _call_api(self, prompt: str) -> str:
        """Make API call with retry logic."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a financial analyst AI. Respond only with valid JSON.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': self.temperature,
            'max_tokens': 1000
        }
        
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f'{self.base_url}/chat/completions',
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Clean up response (remove markdown code blocks if present)
                if content.startswith('```'):
                    lines = content.split('\n')
                    content = '\n'.join(lines[1:-1])
                
                return content.strip()
                
            except httpx.HTTPStatusError as e:
                logger.warning(f"API error (attempt {attempt + 1}): {e}")
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        
        # If all retries failed
        raise last_error or RuntimeError("API call failed after all retries")
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, Dict]:
        """Get info about available providers and which have API keys set."""
        available = {}
        for name, config in cls.PROVIDERS.items():
            api_key = os.getenv(config['env_key'])
            available[name] = {
                'model': config['model'],
                'description': config.get('description', ''),
                'has_api_key': bool(api_key),
                'env_var': config['env_key']
            }
        return available
    
    @classmethod
    def auto_select_provider(cls) -> Optional[str]:
        """
        Automatically select the best available provider based on API keys.
        Priority: grok > deepseek > groq > gemini
        """
        priority = ['grok', 'deepseek', 'groq', 'gemini']
        for provider in priority:
            if os.getenv(cls.PROVIDERS[provider]['env_key']):
                return provider
        return None
    
    def analyze_market_conditions(self, symbols: List[str], market_data: Dict) -> Dict:
        """
        Analyze overall market conditions and sector trends.
        
        Args:
            symbols: List of symbols analyzed
            market_data: Dictionary with market overview data
        
        Returns:
            Dictionary with market analysis
        """
        prompt = f"""Analyze current market conditions based on this data:

Symbols analyzed: {', '.join(symbols[:20])}

Market Overview:
{json.dumps(market_data, indent=2, default=str)}

Respond in JSON format:
{{
    "market_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
    "sentiment_strength": <float 0.0 to 1.0>,
    "trend_direction": "UP" | "DOWN" | "SIDEWAYS",
    "volatility_assessment": "LOW" | "MODERATE" | "HIGH" | "EXTREME",
    "sector_leaders": ["sector1", "sector2"],
    "sector_laggards": ["sector1", "sector2"],
    "key_observations": ["observation1", "observation2", ...],
    "trading_recommendation": "AGGRESSIVE" | "MODERATE" | "DEFENSIVE" | "CASH",
    "risk_factors": ["risk1", "risk2", ...],
    "reasoning": "<comprehensive market analysis>"
}}

Return ONLY valid JSON."""

        response = self._call_api(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse market analysis: {e}")
            return {
                'market_sentiment': 'NEUTRAL',
                'sentiment_strength': 0.5,
                'error': str(e)
            }
    
    def generate_trade_plan(self, symbol: str, signal: str, 
                           entry_price: float, current_data: Dict) -> Dict:
        """
        Generate a detailed trade plan with entry, exit, and risk parameters.
        
        Args:
            symbol: Stock symbol
            signal: 'BUY' or 'SELL'
            entry_price: Suggested entry price
            current_data: Current market data for the symbol
        
        Returns:
            Detailed trade plan
        """
        prompt = f"""Create a detailed trade plan for {symbol}:

Signal: {signal}
Entry Price: ${entry_price:.2f}

Current Data:
{json.dumps(current_data, indent=2, default=str)}

Respond in JSON format:
{{
    "symbol": "{symbol}",
    "action": "{signal}",
    "entry_zone": {{
        "ideal_entry": <price>,
        "entry_range_low": <price>,
        "entry_range_high": <price>
    }},
    "position_sizing": {{
        "aggressive_pct": <portfolio percentage>,
        "moderate_pct": <portfolio percentage>,
        "conservative_pct": <portfolio percentage>
    }},
    "stop_loss": {{
        "tight": <price>,
        "standard": <price>,
        "wide": <price>,
        "stop_loss_reason": "<explanation>"
    }},
    "take_profit": {{
        "target_1": <price>,
        "target_2": <price>,
        "target_3": <price>,
        "trailing_stop_suggestion": "<percentage or price>"
    }},
    "time_horizon": "INTRADAY" | "SWING" | "POSITION",
    "risk_reward_ratio": <float>,
    "success_probability": <float 0.0 to 1.0>,
    "key_levels_to_watch": ["level1: $xx.xx", "level2: $xx.xx"],
    "invalidation_point": "<when to exit regardless>",
    "additional_notes": "<any other relevant information>"
}}

Return ONLY valid JSON."""

        response = self._call_api(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse trade plan: {e}")
            return {
                'symbol': symbol,
                'action': signal,
                'error': str(e)
            }
    
    def analyze_news_batch(self, news_items: List[Dict]) -> Dict:
        """
        Analyze multiple news items at once for efficiency.
        
        Args:
            news_items: List of news items with 'title' and optionally 'summary'
        
        Returns:
            Aggregated sentiment analysis
        """
        if not news_items:
            return {'overall_sentiment': 'NEUTRAL', 'score': 0.0}
        
        news_text = "\n".join([
            f"- {item.get('title', '')}: {item.get('summary', '')[:100]}"
            for item in news_items[:10]
        ])
        
        prompt = f"""Analyze these financial news items for trading sentiment:

{news_text}

Respond in JSON format:
{{
    "overall_sentiment": "VERY_BULLISH" | "BULLISH" | "NEUTRAL" | "BEARISH" | "VERY_BEARISH",
    "sentiment_score": <float -1.0 to 1.0>,
    "confidence": <float 0.0 to 1.0>,
    "news_impact": "HIGH" | "MEDIUM" | "LOW",
    "affected_sectors": ["sector1", "sector2"],
    "key_takeaways": ["takeaway1", "takeaway2", "takeaway3"],
    "trading_implication": "<brief trading recommendation based on news>"
}}

Return ONLY valid JSON."""

        response = self._call_api(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse news batch analysis: {e}")
            return {
                'overall_sentiment': 'NEUTRAL',
                'sentiment_score': 0.0,
                'error': str(e)
            }


class MultiProviderAnalyzer:
    """
    Analyzer that uses multiple AI providers for consensus.
    
    Usage:
        analyzer = MultiProviderAnalyzer(providers=['deepseek', 'groq'])
        result = analyzer.analyze_with_consensus("Apple beats earnings")
    """
    
    def __init__(self, providers: Optional[List[str]] = None):
        """
        Initialize with multiple providers.
        
        Args:
            providers: List of providers to use. Auto-detects if not provided.
        """
        if providers is None:
            providers = []
            for name in AIAnalyzer.PROVIDERS:
                if os.getenv(AIAnalyzer.PROVIDERS[name]['env_key']):
                    providers.append(name)
        
        if not providers:
            raise ValueError("No AI provider API keys found in environment")
        
        self.analyzers = {
            provider: AIAnalyzer(provider=provider)
            for provider in providers
        }
        self.providers = providers
    
    def analyze_with_consensus(self, text: str) -> Dict:
        """
        Get sentiment analysis from multiple providers and calculate consensus.
        
        Args:
            text: Text to analyze
        
        Returns:
            Consensus result with individual provider results
        """
        results = {}
        scores = []
        
        for provider, analyzer in self.analyzers.items():
            try:
                result = analyzer.analyze_sentiment(text)
                results[provider] = {
                    'sentiment': result.sentiment.name,
                    'score': result.score,
                    'confidence': result.confidence
                }
                scores.append(result.score)
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}")
                results[provider] = {'error': str(e)}
        
        if not scores:
            return {'error': 'All providers failed', 'results': results}
        
        avg_score = sum(scores) / len(scores)
        
        # Determine consensus sentiment
        if avg_score >= 0.5:
            consensus = 'VERY_BULLISH'
        elif avg_score >= 0.2:
            consensus = 'BULLISH'
        elif avg_score >= -0.2:
            consensus = 'NEUTRAL'
        elif avg_score >= -0.5:
            consensus = 'BEARISH'
        else:
            consensus = 'VERY_BEARISH'
        
        # Calculate agreement level
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        agreement = max(0, 1 - variance * 2)  # Higher variance = lower agreement
        
        return {
            'consensus_sentiment': consensus,
            'consensus_score': avg_score,
            'agreement_level': agreement,
            'providers_used': len(scores),
            'individual_results': results
        }
    
    def close(self):
        """Close all analyzer clients."""
        for analyzer in self.analyzers.values():
            analyzer.close()


if __name__ == "__main__":
    # Test (requires API key)
    logging.basicConfig(level=logging.INFO)
    
    # Example usage (won't work without valid API key)
    print("AIAnalyzer module loaded successfully")
    print("Available providers:", list(AIAnalyzer.PROVIDERS.keys()))
    
    # To test:
    # analyzer = AIAnalyzer(provider='deepseek', api_key='your_key')
    # result = analyzer.analyze_sentiment("Apple stock rises on strong earnings")
    # print(result)
