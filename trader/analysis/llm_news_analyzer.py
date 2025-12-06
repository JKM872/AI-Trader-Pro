"""
LLM News Analyzer - Real-time news analysis using Large Language Models.

Features:
- Multi-provider LLM support (Gemini, Deepseek, Groq)
- News event classification
- Entity extraction (companies, people, products)
- Impact magnitude prediction
- Sentiment analysis with context
"""

import logging
import os
import json
import hashlib
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
from collections import defaultdict

import httpx

logger = logging.getLogger(__name__)


class NewsEventType(Enum):
    """Types of news events."""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    LAWSUIT = "lawsuit"
    EXECUTIVE_CHANGE = "executive_change"
    REGULATORY = "regulatory"
    ANALYST_RATING = "analyst_rating"
    PARTNERSHIP = "partnership"
    STOCK_BUYBACK = "stock_buyback"
    DIVIDEND = "dividend"
    SCANDAL = "scandal"
    MARKET_TREND = "market_trend"
    ECONOMIC = "economic"
    OTHER = "other"


class ImpactMagnitude(Enum):
    """Expected impact magnitude on stock price."""
    VERY_HIGH = 5      # >5% price move expected
    HIGH = 4           # 2-5% price move
    MEDIUM = 3         # 1-2% price move
    LOW = 2            # 0.5-1% price move
    MINIMAL = 1        # <0.5% price move


class ImpactDirection(Enum):
    """Expected impact direction."""
    VERY_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    VERY_BEARISH = -2


@dataclass
class ExtractedEntity:
    """Extracted entity from news."""
    name: str
    entity_type: str  # company, person, product, location
    relevance: float  # 0-1
    ticker: Optional[str] = None
    context: Optional[str] = None


@dataclass
class NewsAnalysis:
    """Complete analysis of a news article."""
    headline: str
    source: str
    
    # Classification
    event_type: NewsEventType
    primary_tickers: List[str]
    secondary_tickers: List[str]
    
    # Sentiment
    sentiment_score: float  # -1 to 1
    sentiment_direction: ImpactDirection
    
    # Impact prediction
    impact_magnitude: ImpactMagnitude
    expected_price_move_pct: float
    confidence: float  # 0-1
    
    # Entities
    entities: List[ExtractedEntity] = field(default_factory=list)
    
    # Key information
    key_facts: List[str] = field(default_factory=list)
    key_numbers: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    is_breaking: bool = False
    urgency_score: float = 0.5  # 0-1
    
    # Raw analysis
    raw_analysis: Optional[str] = None
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'headline': self.headline,
            'source': self.source,
            'event_type': self.event_type.value,
            'primary_tickers': self.primary_tickers,
            'secondary_tickers': self.secondary_tickers,
            'sentiment_score': self.sentiment_score,
            'sentiment_direction': self.sentiment_direction.name,
            'impact_magnitude': self.impact_magnitude.value,
            'expected_price_move_pct': self.expected_price_move_pct,
            'confidence': self.confidence,
            'entities': [{'name': e.name, 'type': e.entity_type, 'ticker': e.ticker} for e in self.entities],
            'key_facts': self.key_facts,
            'key_numbers': self.key_numbers,
            'is_breaking': self.is_breaking,
            'urgency_score': self.urgency_score,
            'analyzed_at': self.analyzed_at.isoformat()
        }


class LLMProvider(Enum):
    """Available LLM providers."""
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    GROQ = "groq"


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: LLMProvider = LLMProvider.GEMINI
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: float = 30.0


class LLMNewsAnalyzer:
    """
    Analyze news using Large Language Models.
    
    Features:
    - Multi-provider support
    - Structured output parsing
    - Caching for repeated queries
    - Batch processing
    """
    
    # Default models per provider
    DEFAULT_MODELS = {
        LLMProvider.GEMINI: "gemini-pro",
        LLMProvider.DEEPSEEK: "deepseek-chat",
        LLMProvider.GROQ: "llama-3.1-70b-versatile"
    }
    
    # API endpoints
    API_ENDPOINTS = {
        LLMProvider.GEMINI: "https://generativelanguage.googleapis.com/v1beta/models",
        LLMProvider.DEEPSEEK: "https://api.deepseek.com/chat/completions",
        LLMProvider.GROQ: "https://api.groq.com/openai/v1/chat/completions"
    }
    
    # Analysis prompt template
    ANALYSIS_PROMPT = """Analyze this financial news article and extract structured information.

HEADLINE: {headline}

CONTENT: {content}

SOURCE: {source}
PUBLISHED: {published}

Provide your analysis in the following JSON format:
{{
    "event_type": "one of: earnings, merger_acquisition, product_launch, lawsuit, executive_change, regulatory, analyst_rating, partnership, stock_buyback, dividend, scandal, market_trend, economic, other",
    "primary_tickers": ["list of directly affected stock tickers"],
    "secondary_tickers": ["list of indirectly affected tickers"],
    "sentiment_score": float between -1 (very bearish) and 1 (very bullish),
    "sentiment_direction": "one of: VERY_BULLISH, BULLISH, NEUTRAL, BEARISH, VERY_BEARISH",
    "impact_magnitude": integer 1-5 (1=minimal <0.5%, 2=low 0.5-1%, 3=medium 1-2%, 4=high 2-5%, 5=very_high >5%),
    "expected_price_move_pct": expected percentage move (positive or negative),
    "confidence": float 0-1 indicating your confidence in this analysis,
    "entities": [
        {{"name": "entity name", "type": "company/person/product/location", "ticker": "if applicable", "relevance": 0-1}}
    ],
    "key_facts": ["list of key facts from the article"],
    "key_numbers": {{"metric_name": value, ...}},
    "is_breaking": boolean if this is breaking/urgent news,
    "urgency_score": float 0-1 indicating time sensitivity for trading
}}

Focus on trading implications. Be specific about expected price moves and timing. Return ONLY valid JSON."""

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM news analyzer."""
        self.config = config or LLMConfig()
        
        # Get API key from environment if not provided
        if not self.config.api_key:
            self.config.api_key = self._get_api_key()
        
        # Set default model
        if not self.config.model:
            self.config.model = self.DEFAULT_MODELS[self.config.provider]
        
        # HTTP client
        self.client = httpx.Client(timeout=self.config.timeout)
        
        # Cache for repeated queries
        self._cache: Dict[str, NewsAnalysis] = {}
        self._cache_ttl = timedelta(hours=1)
        
        logger.info(f"LLMNewsAnalyzer initialized with {self.config.provider.value}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        key_map = {
            LLMProvider.GEMINI: "GEMINI_API_KEY",
            LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
            LLMProvider.GROQ: "GROQ_API_KEY"
        }
        return os.getenv(key_map.get(self.config.provider, ""))
    
    def _get_cache_key(self, headline: str, content: str) -> str:
        """Generate cache key for a news item."""
        text = f"{headline}:{content[:500]}"
        return hashlib.md5(text.encode()).hexdigest()
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Make API call to LLM provider."""
        if not self.config.api_key:
            logger.warning(f"No API key for {self.config.provider.value}")
            return None
        
        try:
            if self.config.provider == LLMProvider.GEMINI:
                # Use Google's Gemini API
                endpoint = f"{self.API_ENDPOINTS[self.config.provider]}/{self.config.model}:generateContent?key={self.config.api_key}"
                response = self.client.post(
                    endpoint,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": self.config.temperature,
                            "maxOutputTokens": self.config.max_tokens
                        }
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            
            else:  # DeepSeek and Groq (OpenAI-compatible APIs)
                endpoint = self.API_ENDPOINTS[self.config.provider]
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = self.client.post(
                    endpoint,
                    headers=headers,
                    json={
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": "You are a financial news analyst. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
            
            logger.warning(f"LLM API error: {response.status_code} - {response.text[:200]}")
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
        
        return None
    
    def _parse_analysis(
        self,
        response: str,
        headline: str,
        source: str
    ) -> Optional[NewsAnalysis]:
        """Parse LLM response into NewsAnalysis."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response)
            
            # Parse entities
            entities = []
            for e in data.get("entities", []):
                entities.append(ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "unknown"),
                    relevance=float(e.get("relevance", 0.5)),
                    ticker=e.get("ticker"),
                    context=e.get("context")
                ))
            
            # Map event type
            event_type_str = data.get("event_type", "other").lower()
            try:
                event_type = NewsEventType(event_type_str)
            except ValueError:
                event_type = NewsEventType.OTHER
            
            # Map sentiment direction
            sentiment_dir_str = data.get("sentiment_direction", "NEUTRAL").upper()
            try:
                sentiment_direction = ImpactDirection[sentiment_dir_str]
            except KeyError:
                sentiment_direction = ImpactDirection.NEUTRAL
            
            # Map impact magnitude
            impact_mag_val = int(data.get("impact_magnitude", 1))
            impact_magnitude = ImpactMagnitude(min(5, max(1, impact_mag_val)))
            
            return NewsAnalysis(
                headline=headline,
                source=source,
                event_type=event_type,
                primary_tickers=data.get("primary_tickers", []),
                secondary_tickers=data.get("secondary_tickers", []),
                sentiment_score=float(data.get("sentiment_score", 0)),
                sentiment_direction=sentiment_direction,
                impact_magnitude=impact_magnitude,
                expected_price_move_pct=float(data.get("expected_price_move_pct", 0)),
                confidence=float(data.get("confidence", 0.5)),
                entities=entities,
                key_facts=data.get("key_facts", []),
                key_numbers=data.get("key_numbers", {}),
                is_breaking=bool(data.get("is_breaking", False)),
                urgency_score=float(data.get("urgency_score", 0.5)),
                raw_analysis=response
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing analysis: {e}")
            return None
    
    def analyze(
        self,
        headline: str,
        content: str = "",
        source: str = "unknown",
        published: str = "",
        use_cache: bool = True
    ) -> Optional[NewsAnalysis]:
        """
        Analyze a single news article.
        
        Args:
            headline: News headline
            content: Article content
            source: News source
            published: Publication date
            use_cache: Whether to use cached results
        
        Returns:
            NewsAnalysis object or None if analysis failed
        """
        # Check cache
        cache_key = self._get_cache_key(headline, content)
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now(timezone.utc) - cached.analyzed_at < self._cache_ttl:
                return cached
        
        # Build prompt
        prompt = self.ANALYSIS_PROMPT.format(
            headline=headline,
            content=content[:2000] if content else "No content available",
            source=source,
            published=published or "Unknown"
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        if not response:
            return None
        
        # Parse response
        analysis = self._parse_analysis(response, headline, source)
        
        # Cache result
        if analysis:
            self._cache[cache_key] = analysis
        
        return analysis
    
    def analyze_batch(
        self,
        articles: List[Dict[str, str]],
        parallel: bool = False
    ) -> List[Optional[NewsAnalysis]]:
        """
        Analyze multiple articles.
        
        Args:
            articles: List of article dicts with 'headline', 'content', 'source', 'published'
            parallel: Whether to process in parallel (not implemented yet)
        
        Returns:
            List of NewsAnalysis objects
        """
        results = []
        
        for article in articles:
            analysis = self.analyze(
                headline=article.get('headline', article.get('title', '')),
                content=article.get('content', article.get('description', '')),
                source=article.get('source', 'unknown'),
                published=article.get('published', '')
            )
            results.append(analysis)
        
        return results
    
    def get_trading_signals(
        self,
        analyses: List[NewsAnalysis],
        min_confidence: float = 0.6,
        min_impact: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Extract trading signals from analyses.
        
        Args:
            analyses: List of NewsAnalysis objects
            min_confidence: Minimum confidence threshold
            min_impact: Minimum impact magnitude
        
        Returns:
            List of trading signals
        """
        signals = []
        
        for analysis in analyses:
            if not analysis:
                continue
            
            if analysis.confidence < min_confidence:
                continue
            
            if analysis.impact_magnitude.value < min_impact:
                continue
            
            for ticker in analysis.primary_tickers:
                signal = {
                    'ticker': ticker,
                    'direction': 'buy' if analysis.sentiment_score > 0 else 'sell' if analysis.sentiment_score < 0 else 'hold',
                    'strength': abs(analysis.sentiment_score),
                    'confidence': analysis.confidence,
                    'expected_move_pct': analysis.expected_price_move_pct,
                    'event_type': analysis.event_type.value,
                    'headline': analysis.headline,
                    'urgency': analysis.urgency_score,
                    'is_breaking': analysis.is_breaking,
                    'key_facts': analysis.key_facts[:3]
                }
                signals.append(signal)
        
        # Sort by urgency and confidence
        signals.sort(key=lambda x: (x['is_breaking'], x['urgency'], x['confidence']), reverse=True)
        
        return signals
    
    def close(self):
        """Close HTTP client."""
        self.client.close()


class NewsEventStream:
    """
    Real-time news event stream with LLM analysis.
    
    Monitors multiple news sources and provides real-time trading signals.
    """
    
    def __init__(
        self,
        analyzer: Optional[LLMNewsAnalyzer] = None,
        watchlist: Optional[List[str]] = None
    ):
        """
        Initialize news event stream.
        
        Args:
            analyzer: LLM analyzer instance
            watchlist: List of tickers to watch
        """
        self.analyzer = analyzer or LLMNewsAnalyzer()
        self.watchlist = set(watchlist or [])
        
        # Event buffers
        self.recent_analyses: List[NewsAnalysis] = []
        self.pending_signals: List[Dict[str, Any]] = []
        
        # Metrics
        self.total_analyzed = 0
        self.total_signals = 0
    
    def add_to_watchlist(self, tickers: List[str]):
        """Add tickers to watchlist."""
        self.watchlist.update(ticker.upper() for ticker in tickers)
    
    def remove_from_watchlist(self, tickers: List[str]):
        """Remove tickers from watchlist."""
        for ticker in tickers:
            self.watchlist.discard(ticker.upper())
    
    def process_news(
        self,
        headline: str,
        content: str = "",
        source: str = "unknown",
        published: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single news item.
        
        Returns trading signal if relevant to watchlist.
        """
        analysis = self.analyzer.analyze(
            headline=headline,
            content=content,
            source=source,
            published=published
        )
        
        if not analysis:
            return None
        
        self.recent_analyses.append(analysis)
        self.total_analyzed += 1
        
        # Check if relevant to watchlist
        relevant_tickers = set(analysis.primary_tickers + analysis.secondary_tickers)
        
        if self.watchlist and not relevant_tickers.intersection(self.watchlist):
            return None
        
        # Generate signals
        signals = self.analyzer.get_trading_signals([analysis])
        
        if signals:
            self.total_signals += len(signals)
            self.pending_signals.extend(signals)
            return signals[0]  # Return first/primary signal
        
        return None
    
    def get_pending_signals(self, clear: bool = True) -> List[Dict[str, Any]]:
        """Get pending trading signals."""
        signals = self.pending_signals.copy()
        if clear:
            self.pending_signals = []
        return signals
    
    def get_recent_analyses(
        self,
        limit: int = 10,
        ticker: Optional[str] = None
    ) -> List[NewsAnalysis]:
        """Get recent analyses."""
        analyses = self.recent_analyses.copy()
        
        if ticker:
            ticker = ticker.upper()
            analyses = [
                a for a in analyses
                if ticker in a.primary_tickers or ticker in a.secondary_tickers
            ]
        
        return analyses[-limit:]
    
    def get_sentiment_summary(
        self,
        ticker: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get sentiment summary for a ticker.
        
        Args:
            ticker: Stock ticker
            hours: Lookback period in hours
        
        Returns:
            Sentiment summary
        """
        ticker = ticker.upper()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        relevant = [
            a for a in self.recent_analyses
            if (ticker in a.primary_tickers or ticker in a.secondary_tickers)
            and a.analyzed_at >= cutoff
        ]
        
        if not relevant:
            return {
                'ticker': ticker,
                'total_news': 0,
                'avg_sentiment': 0.0,
                'sentiment_direction': 'NEUTRAL',
                'breaking_news_count': 0,
                'event_types': {}
            }
        
        sentiments = [a.sentiment_score for a in relevant]
        event_types = defaultdict(int)
        for a in relevant:
            event_types[a.event_type.value] += 1
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        return {
            'ticker': ticker,
            'total_news': len(relevant),
            'avg_sentiment': avg_sentiment,
            'sentiment_direction': 'BULLISH' if avg_sentiment > 0.2 else 'BEARISH' if avg_sentiment < -0.2 else 'NEUTRAL',
            'breaking_news_count': sum(1 for a in relevant if a.is_breaking),
            'event_types': dict(event_types),
            'latest_headline': relevant[-1].headline if relevant else None
        }
    
    def close(self):
        """Clean up resources."""
        self.analyzer.close()


def create_news_analyzer(
    provider: str = "gemini",
    **kwargs
) -> LLMNewsAnalyzer:
    """
    Factory function to create news analyzer.
    
    Args:
        provider: 'gemini', 'deepseek', or 'groq'
        **kwargs: Additional config parameters
    
    Returns:
        LLMNewsAnalyzer instance
    """
    provider_map = {
        'gemini': LLMProvider.GEMINI,
        'deepseek': LLMProvider.DEEPSEEK,
        'groq': LLMProvider.GROQ
    }
    
    config = LLMConfig(
        provider=provider_map.get(provider.lower(), LLMProvider.GEMINI),
        **{k: v for k, v in kwargs.items() if hasattr(LLMConfig, k)}
    )
    
    return LLMNewsAnalyzer(config=config)
