"""
Fed Speech Analyzer Module.

Analyzes Federal Reserve communications for market impact:
- FOMC statements and minutes
- Fed Chair speeches
- Regional Fed President speeches
- Beige Book analysis
- Dot plot interpretation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)


class FedSpeaker(Enum):
    """Federal Reserve speakers."""
    CHAIR = "chair"
    VICE_CHAIR = "vice_chair"
    GOVERNOR = "governor"
    REGIONAL_PRESIDENT = "regional_president"
    FOMC = "fomc"


class MonetaryBias(Enum):
    """Monetary policy bias."""
    VERY_HAWKISH = "very_hawkish"
    HAWKISH = "hawkish"
    NEUTRAL = "neutral"
    DOVISH = "dovish"
    VERY_DOVISH = "very_dovish"


class PolicyAction(Enum):
    """Expected policy actions."""
    RATE_HIKE = "rate_hike"
    RATE_CUT = "rate_cut"
    HOLD = "hold"
    QT_ACCELERATE = "qt_accelerate"
    QT_SLOW = "qt_slow"
    QE_RESUME = "qe_resume"


@dataclass
class FedEvent:
    """Federal Reserve event information."""
    event_type: str  # FOMC, Speech, Minutes, Beige Book
    date: datetime
    speaker: Optional[str] = None
    speaker_role: FedSpeaker = FedSpeaker.FOMC
    title: str = ""
    importance: int = 5  # 1-10 scale


@dataclass
class SentimentKeyword:
    """Keyword for sentiment analysis."""
    keyword: str
    sentiment_score: float  # -1.0 to 1.0
    category: str  # inflation, employment, growth, policy


@dataclass
class FedAnalysisResult:
    """Result of Fed speech/statement analysis."""
    event: FedEvent
    monetary_bias: MonetaryBias
    bias_score: float  # -1.0 (very dovish) to 1.0 (very hawkish)
    confidence: float
    expected_actions: list[PolicyAction]
    key_themes: list[str]
    inflation_stance: str
    employment_stance: str
    growth_outlook: str
    rate_path_signal: str
    market_implications: dict = field(default_factory=dict)
    key_phrases: list[str] = field(default_factory=list)
    sentiment_breakdown: dict = field(default_factory=dict)


@dataclass
class RateProbability:
    """Rate decision probability."""
    meeting_date: datetime
    current_rate: float
    probabilities: dict[float, float]  # rate -> probability
    most_likely_rate: float
    expected_change: float


class FedSpeechAnalyzer:
    """
    Analyzes Federal Reserve communications for trading signals.
    
    Features:
    - Keyword-based sentiment analysis
    - LLM-enhanced interpretation (if available)
    - Historical comparison
    - Rate path probability
    - Market impact prediction
    """
    
    # Hawkish keywords (positive sentiment = hawkish)
    HAWKISH_KEYWORDS: list[SentimentKeyword] = [
        SentimentKeyword("inflation remains elevated", 0.8, "inflation"),
        SentimentKeyword("inflation too high", 0.9, "inflation"),
        SentimentKeyword("price stability", 0.5, "inflation"),
        SentimentKeyword("tighten", 0.7, "policy"),
        SentimentKeyword("restrictive", 0.6, "policy"),
        SentimentKeyword("further increases", 0.8, "policy"),
        SentimentKeyword("additional firming", 0.7, "policy"),
        SentimentKeyword("higher for longer", 0.8, "policy"),
        SentimentKeyword("committed to 2%", 0.6, "inflation"),
        SentimentKeyword("strong labor market", 0.4, "employment"),
        SentimentKeyword("tight labor market", 0.5, "employment"),
        SentimentKeyword("robust employment", 0.4, "employment"),
        SentimentKeyword("overheating", 0.8, "growth"),
        SentimentKeyword("above trend", 0.5, "growth"),
        SentimentKeyword("resilient", 0.4, "growth"),
        SentimentKeyword("vigilant", 0.6, "policy"),
        SentimentKeyword("upside risks", 0.5, "inflation"),
    ]
    
    # Dovish keywords (negative sentiment = dovish)
    DOVISH_KEYWORDS: list[SentimentKeyword] = [
        SentimentKeyword("inflation easing", -0.6, "inflation"),
        SentimentKeyword("inflation moderating", -0.5, "inflation"),
        SentimentKeyword("disinflation", -0.7, "inflation"),
        SentimentKeyword("price pressures easing", -0.6, "inflation"),
        SentimentKeyword("accommodate", -0.7, "policy"),
        SentimentKeyword("supportive", -0.6, "policy"),
        SentimentKeyword("pause", -0.5, "policy"),
        SentimentKeyword("patient", -0.5, "policy"),
        SentimentKeyword("data dependent", -0.3, "policy"),
        SentimentKeyword("slowing", -0.5, "growth"),
        SentimentKeyword("softening", -0.6, "growth"),
        SentimentKeyword("cooling", -0.5, "growth"),
        SentimentKeyword("labor market normalizing", -0.5, "employment"),
        SentimentKeyword("employment softening", -0.6, "employment"),
        SentimentKeyword("unemployment rising", -0.7, "employment"),
        SentimentKeyword("downside risks", -0.5, "growth"),
        SentimentKeyword("balanced risks", -0.3, "policy"),
        SentimentKeyword("financial conditions", -0.4, "policy"),
        SentimentKeyword("gradual", -0.4, "policy"),
    ]
    
    # Key Fed speakers and their influence weights
    SPEAKER_WEIGHTS = {
        FedSpeaker.CHAIR: 1.0,
        FedSpeaker.VICE_CHAIR: 0.8,
        FedSpeaker.GOVERNOR: 0.6,
        FedSpeaker.REGIONAL_PRESIDENT: 0.4,
        FedSpeaker.FOMC: 1.0,
    }
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str = "gemini",
    ):
        """
        Initialize Fed Speech Analyzer.
        
        Args:
            use_llm: Whether to use LLM for enhanced analysis
            llm_provider: LLM provider to use (gemini, deepseek, groq)
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_client = None
        
        if use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """Initialize LLM client for enhanced analysis."""
        try:
            if self.llm_provider == "gemini":
                import google.generativeai as genai
                import os
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.llm_client = genai.GenerativeModel('gemini-pro')
                else:
                    logger.warning("No Gemini API key found")
                    self.use_llm = False
            elif self.llm_provider == "deepseek":
                import httpx
                import os
                self.llm_client = {
                    "api_key": os.getenv("DEEPSEEK_API_KEY"),
                    "endpoint": "https://api.deepseek.com/chat/completions"
                }
            elif self.llm_provider == "groq":
                import httpx
                import os
                self.llm_client = {
                    "api_key": os.getenv("GROQ_API_KEY"),
                    "endpoint": "https://api.groq.com/openai/v1/chat/completions"
                }
            else:
                logger.warning(f"Unsupported LLM provider: {self.llm_provider}")
        except ImportError as e:
            logger.warning(f"LLM provider {self.llm_provider} not available: {e}")
            self.use_llm = False
    
    def analyze_text(
        self,
        text: str,
        event: Optional[FedEvent] = None,
    ) -> FedAnalysisResult:
        """
        Analyze Fed speech or statement text.
        
        Args:
            text: Text content to analyze
            event: Optional event metadata
            
        Returns:
            FedAnalysisResult with analysis
        """
        if event is None:
            event = FedEvent(
                event_type="Speech",
                date=datetime.now(),
            )
        
        # Keyword-based analysis
        keyword_result = self._keyword_analysis(text)
        
        # LLM-enhanced analysis if available
        if self.use_llm and self.llm_client:
            llm_result = self._llm_analysis(text, event)
            # Combine results
            result = self._combine_analyses(keyword_result, llm_result, event)
        else:
            result = keyword_result
            result.event = event
        
        # Add market implications
        result.market_implications = self._predict_market_impact(result)
        
        return result
    
    def _keyword_analysis(self, text: str) -> FedAnalysisResult:
        """Perform keyword-based sentiment analysis."""
        text_lower = text.lower()
        
        hawkish_score = 0.0
        dovish_score = 0.0
        
        matched_hawkish = []
        matched_dovish = []
        
        sentiment_breakdown = {
            "inflation": 0.0,
            "employment": 0.0,
            "growth": 0.0,
            "policy": 0.0,
        }
        category_counts = {
            "inflation": 0,
            "employment": 0,
            "growth": 0,
            "policy": 0,
        }
        
        # Check hawkish keywords
        for kw in self.HAWKISH_KEYWORDS:
            if kw.keyword.lower() in text_lower:
                hawkish_score += abs(kw.sentiment_score)
                matched_hawkish.append(kw.keyword)
                sentiment_breakdown[kw.category] += kw.sentiment_score
                category_counts[kw.category] += 1
        
        # Check dovish keywords
        for kw in self.DOVISH_KEYWORDS:
            if kw.keyword.lower() in text_lower:
                dovish_score += abs(kw.sentiment_score)
                matched_dovish.append(kw.keyword)
                sentiment_breakdown[kw.category] += kw.sentiment_score
                category_counts[kw.category] += 1
        
        # Normalize sentiment breakdown
        for category in sentiment_breakdown:
            if category_counts[category] > 0:
                sentiment_breakdown[category] /= category_counts[category]
        
        # Calculate overall bias
        total_score = hawkish_score + dovish_score
        if total_score > 0:
            bias_score = (hawkish_score - dovish_score) / total_score
        else:
            bias_score = 0.0
        
        # Determine monetary bias
        if bias_score > 0.5:
            monetary_bias = MonetaryBias.VERY_HAWKISH
        elif bias_score > 0.2:
            monetary_bias = MonetaryBias.HAWKISH
        elif bias_score < -0.5:
            monetary_bias = MonetaryBias.VERY_DOVISH
        elif bias_score < -0.2:
            monetary_bias = MonetaryBias.DOVISH
        else:
            monetary_bias = MonetaryBias.NEUTRAL
        
        # Determine expected actions
        expected_actions = self._determine_expected_actions(
            bias_score, sentiment_breakdown
        )
        
        # Extract key themes
        key_themes = self._extract_themes(text)
        
        # Generate stance descriptions
        inflation_stance = self._describe_stance(
            sentiment_breakdown["inflation"], "inflation"
        )
        employment_stance = self._describe_stance(
            sentiment_breakdown["employment"], "employment"
        )
        growth_outlook = self._describe_stance(
            sentiment_breakdown["growth"], "growth"
        )
        rate_path_signal = self._describe_rate_path(bias_score)
        
        # Calculate confidence
        confidence = min(1.0, (len(matched_hawkish) + len(matched_dovish)) / 10)
        
        return FedAnalysisResult(
            event=FedEvent(event_type="", date=datetime.now()),
            monetary_bias=monetary_bias,
            bias_score=bias_score,
            confidence=confidence,
            expected_actions=expected_actions,
            key_themes=key_themes,
            inflation_stance=inflation_stance,
            employment_stance=employment_stance,
            growth_outlook=growth_outlook,
            rate_path_signal=rate_path_signal,
            key_phrases=matched_hawkish + matched_dovish,
            sentiment_breakdown=sentiment_breakdown,
        )
    
    def _llm_analysis(
        self,
        text: str,
        event: FedEvent,
    ) -> Optional[dict]:
        """Use LLM for enhanced analysis."""
        if not self.llm_client:
            return None
        
        prompt = f"""Analyze the following Federal Reserve communication and provide:
1. Overall monetary policy stance (very_hawkish, hawkish, neutral, dovish, very_dovish)
2. Key themes discussed (list of 3-5 main topics)
3. Inflation assessment
4. Employment assessment
5. Growth outlook
6. Rate path implications
7. Confidence level (0-1)

Fed Communication:
{text[:4000]}

Respond in JSON format ONLY:
{{
    "stance": "hawkish|dovish|neutral",
    "bias_score": 0.5,
    "key_themes": ["theme1", "theme2"],
    "inflation_view": "description",
    "employment_view": "description",
    "growth_view": "description",
    "rate_path": "description",
    "confidence": 0.8
}}"""
        
        try:
            import json
            
            if self.llm_provider == "gemini":
                response = self.llm_client.generate_content(prompt)
                # Extract JSON from response
                response_text = response.text
                # Find JSON in response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end > start:
                    return json.loads(response_text[start:end])
                    
            elif self.llm_provider in ["deepseek", "groq"]:
                import httpx
                
                headers = {
                    "Authorization": f"Bearer {self.llm_client['api_key']}",
                    "Content-Type": "application/json"
                }
                
                model = "deepseek-chat" if self.llm_provider == "deepseek" else "llama-3.1-70b-versatile"
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
                
                with httpx.Client(timeout=30) as client:
                    response = client.post(
                        self.llm_client['endpoint'],
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    # Extract JSON
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end > start:
                        return json.loads(content[start:end])
                        
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None
        
        return None
    
    def _combine_analyses(
        self,
        keyword_result: FedAnalysisResult,
        llm_result: Optional[dict],
        event: FedEvent,
    ) -> FedAnalysisResult:
        """Combine keyword and LLM analyses."""
        if llm_result is None:
            keyword_result.event = event
            return keyword_result
        
        # Weight keyword analysis 40%, LLM 60%
        combined_bias = (
            keyword_result.bias_score * 0.4 +
            llm_result.get("bias_score", 0) * 0.6
        )
        
        # Use LLM themes if available
        themes = llm_result.get("key_themes", keyword_result.key_themes)
        
        # Combine confidence
        llm_confidence = llm_result.get("confidence", 0.5)
        combined_confidence = (keyword_result.confidence + llm_confidence) / 2
        
        # Determine final bias
        if combined_bias > 0.5:
            monetary_bias = MonetaryBias.VERY_HAWKISH
        elif combined_bias > 0.2:
            monetary_bias = MonetaryBias.HAWKISH
        elif combined_bias < -0.5:
            monetary_bias = MonetaryBias.VERY_DOVISH
        elif combined_bias < -0.2:
            monetary_bias = MonetaryBias.DOVISH
        else:
            monetary_bias = MonetaryBias.NEUTRAL
        
        return FedAnalysisResult(
            event=event,
            monetary_bias=monetary_bias,
            bias_score=combined_bias,
            confidence=combined_confidence,
            expected_actions=keyword_result.expected_actions,
            key_themes=themes,
            inflation_stance=llm_result.get("inflation_view", keyword_result.inflation_stance),
            employment_stance=llm_result.get("employment_view", keyword_result.employment_stance),
            growth_outlook=llm_result.get("growth_view", keyword_result.growth_outlook),
            rate_path_signal=llm_result.get("rate_path", keyword_result.rate_path_signal),
            key_phrases=keyword_result.key_phrases,
            sentiment_breakdown=keyword_result.sentiment_breakdown,
        )
    
    def _determine_expected_actions(
        self,
        bias_score: float,
        sentiment_breakdown: dict,
    ) -> list[PolicyAction]:
        """Determine expected policy actions from analysis."""
        actions = []
        
        # Rate actions
        if bias_score > 0.5:
            actions.append(PolicyAction.RATE_HIKE)
        elif bias_score > 0.1:
            actions.append(PolicyAction.HOLD)
        elif bias_score < -0.5:
            actions.append(PolicyAction.RATE_CUT)
        else:
            actions.append(PolicyAction.HOLD)
        
        # QT/QE actions (based on growth/employment outlook)
        growth_score = sentiment_breakdown.get("growth", 0)
        employment_score = sentiment_breakdown.get("employment", 0)
        
        if growth_score < -0.5 or employment_score < -0.5:
            actions.append(PolicyAction.QT_SLOW)
        elif bias_score > 0.5 and growth_score > 0.3:
            actions.append(PolicyAction.QT_ACCELERATE)
        
        return actions
    
    def _extract_themes(self, text: str) -> list[str]:
        """Extract key themes from text."""
        themes = []
        text_lower = text.lower()
        
        theme_keywords = {
            "Inflation": ["inflation", "prices", "cpi", "pce"],
            "Employment": ["employment", "labor", "jobs", "unemployment", "wages"],
            "Growth": ["growth", "gdp", "economic activity", "expansion"],
            "Financial Stability": ["financial stability", "banking", "credit"],
            "Housing": ["housing", "real estate", "mortgage"],
            "Consumer": ["consumer", "spending", "retail"],
            "International": ["global", "international", "trade", "geopolitical"],
            "Monetary Policy": ["rate", "monetary", "policy", "fed funds"],
            "Balance Sheet": ["balance sheet", "qt", "qe", "assets"],
        }
        
        for theme, keywords in theme_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if theme not in themes:
                        themes.append(theme)
                    break
        
        return themes[:5]  # Top 5 themes
    
    def _describe_stance(self, score: float, category: str) -> str:
        """Generate description for a stance score."""
        if category == "inflation":
            if score > 0.3:
                return "Elevated inflation remains a concern, requiring continued vigilance"
            elif score > 0:
                return "Inflation gradually moderating but still above target"
            elif score < -0.3:
                return "Inflation clearly declining toward target"
            else:
                return "Inflation showing mixed signals"
        
        elif category == "employment":
            if score > 0.3:
                return "Labor market remains tight with strong job gains"
            elif score > 0:
                return "Employment conditions solid but showing signs of cooling"
            elif score < -0.3:
                return "Labor market clearly softening"
            else:
                return "Employment conditions normalizing"
        
        elif category == "growth":
            if score > 0.3:
                return "Economic growth remains above trend"
            elif score > 0:
                return "Growth moderating but still positive"
            elif score < -0.3:
                return "Growth slowing significantly"
            else:
                return "Growth outlook balanced"
        
        return "Neutral stance"
    
    def _describe_rate_path(self, bias_score: float) -> str:
        """Describe implied rate path."""
        if bias_score > 0.5:
            return "Further rate increases likely; higher for longer signaled"
        elif bias_score > 0.2:
            return "Rates likely to remain elevated; pause possible but cuts unlikely soon"
        elif bias_score < -0.5:
            return "Rate cuts likely in near term; pivot to easing signaled"
        elif bias_score < -0.2:
            return "Rate cuts becoming more likely; peak rates may be reached"
        else:
            return "Data dependent; no clear signal on direction"
    
    def _predict_market_impact(
        self,
        result: FedAnalysisResult,
    ) -> dict:
        """Predict market impact based on Fed analysis."""
        bias = result.bias_score
        
        implications = {
            "equities": {
                "direction": "",
                "magnitude": "",
                "sectors": [],
            },
            "bonds": {
                "direction": "",
                "magnitude": "",
                "duration_impact": "",
            },
            "dollar": {
                "direction": "",
                "magnitude": "",
            },
            "gold": {
                "direction": "",
                "magnitude": "",
            },
        }
        
        # Equities
        if bias > 0.3:
            implications["equities"]["direction"] = "bearish"
            implications["equities"]["magnitude"] = "moderate to significant"
            implications["equities"]["sectors"] = [
                "Financials may benefit",
                "Growth/Tech may underperform",
                "Rate-sensitive sectors vulnerable",
            ]
        elif bias < -0.3:
            implications["equities"]["direction"] = "bullish"
            implications["equities"]["magnitude"] = "moderate to significant"
            implications["equities"]["sectors"] = [
                "Growth/Tech may outperform",
                "Rate-sensitive sectors may rally",
                "Financials may lag",
            ]
        else:
            implications["equities"]["direction"] = "neutral"
            implications["equities"]["magnitude"] = "limited"
            implications["equities"]["sectors"] = ["Mixed sector impact"]
        
        # Bonds
        if bias > 0.3:
            implications["bonds"]["direction"] = "bearish (yields up)"
            implications["bonds"]["magnitude"] = "significant"
            implications["bonds"]["duration_impact"] = "Long duration most impacted"
        elif bias < -0.3:
            implications["bonds"]["direction"] = "bullish (yields down)"
            implications["bonds"]["magnitude"] = "significant"
            implications["bonds"]["duration_impact"] = "Long duration benefits most"
        else:
            implications["bonds"]["direction"] = "neutral"
            implications["bonds"]["magnitude"] = "limited"
            implications["bonds"]["duration_impact"] = "Curve flattening possible"
        
        # Dollar
        if bias > 0.3:
            implications["dollar"]["direction"] = "bullish"
            implications["dollar"]["magnitude"] = "moderate"
        elif bias < -0.3:
            implications["dollar"]["direction"] = "bearish"
            implications["dollar"]["magnitude"] = "moderate"
        else:
            implications["dollar"]["direction"] = "neutral"
            implications["dollar"]["magnitude"] = "limited"
        
        # Gold
        if bias > 0.3:
            implications["gold"]["direction"] = "bearish"
            implications["gold"]["magnitude"] = "moderate (real rates up)"
        elif bias < -0.3:
            implications["gold"]["direction"] = "bullish"
            implications["gold"]["magnitude"] = "moderate (real rates down)"
        else:
            implications["gold"]["direction"] = "neutral"
            implications["gold"]["magnitude"] = "limited"
        
        return implications
    
    def compare_statements(
        self,
        current_text: str,
        previous_text: str,
    ) -> dict:
        """
        Compare two Fed statements to identify changes.
        
        Args:
            current_text: Current statement text
            previous_text: Previous statement text
            
        Returns:
            Dictionary with comparison results
        """
        current_analysis = self.analyze_text(current_text)
        previous_analysis = self.analyze_text(previous_text)
        
        bias_change = current_analysis.bias_score - previous_analysis.bias_score
        
        comparison = {
            "bias_change": bias_change,
            "direction": "hawkish shift" if bias_change > 0 else "dovish shift" if bias_change < 0 else "unchanged",
            "magnitude": abs(bias_change),
            "current_bias": current_analysis.monetary_bias.value,
            "previous_bias": previous_analysis.monetary_bias.value,
            "new_themes": [t for t in current_analysis.key_themes if t not in previous_analysis.key_themes],
            "removed_themes": [t for t in previous_analysis.key_themes if t not in current_analysis.key_themes],
            "sentiment_changes": {},
        }
        
        # Compare sentiment breakdown
        for category in current_analysis.sentiment_breakdown:
            current = current_analysis.sentiment_breakdown.get(category, 0)
            previous = previous_analysis.sentiment_breakdown.get(category, 0)
            if abs(current - previous) > 0.1:
                comparison["sentiment_changes"][category] = {
                    "change": current - previous,
                    "direction": "more hawkish" if current > previous else "more dovish",
                }
        
        return comparison
    
    def get_fomc_calendar(self, year: Optional[int] = None) -> list[FedEvent]:
        """
        Get FOMC meeting calendar.
        
        Args:
            year: Year to get calendar for (current year if None)
            
        Returns:
            List of FedEvent for FOMC meetings
        """
        if year is None:
            year = datetime.now().year
        
        # 2024 FOMC schedule (example - would be updated annually)
        fomc_dates_2024 = [
            datetime(2024, 1, 31),
            datetime(2024, 3, 20),
            datetime(2024, 5, 1),
            datetime(2024, 6, 12),
            datetime(2024, 7, 31),
            datetime(2024, 9, 18),
            datetime(2024, 11, 7),
            datetime(2024, 12, 18),
        ]
        
        # 2025 FOMC schedule
        fomc_dates_2025 = [
            datetime(2025, 1, 29),
            datetime(2025, 3, 19),
            datetime(2025, 5, 7),
            datetime(2025, 6, 18),
            datetime(2025, 7, 30),
            datetime(2025, 9, 17),
            datetime(2025, 11, 5),
            datetime(2025, 12, 17),
        ]
        
        dates_map = {
            2024: fomc_dates_2024,
            2025: fomc_dates_2025,
        }
        
        dates = dates_map.get(year, [])
        
        events = []
        for date in dates:
            # Determine if it's a "dot plot" meeting (March, June, September, December)
            is_sep = date.month in [3, 6, 9, 12]
            importance = 10 if is_sep else 8
            
            events.append(FedEvent(
                event_type="FOMC Meeting",
                date=date,
                speaker_role=FedSpeaker.FOMC,
                title=f"FOMC Meeting {'with SEP' if is_sep else ''}",
                importance=importance,
            ))
        
        return events


class FedWatcher:
    """
    Monitors Fed communications and tracks market expectations.
    
    Aggregates Fed speeches and statements to track policy evolution.
    """
    
    def __init__(self):
        """Initialize Fed Watcher."""
        self.analyzer = FedSpeechAnalyzer()
        self.analysis_history: list[FedAnalysisResult] = []
        self.current_policy_stance: MonetaryBias = MonetaryBias.NEUTRAL
    
    def add_analysis(self, result: FedAnalysisResult) -> None:
        """Add analysis to history."""
        self.analysis_history.append(result)
        self._update_policy_stance()
    
    def _update_policy_stance(self) -> None:
        """Update current policy stance based on recent analyses."""
        if not self.analysis_history:
            return
        
        # Weight recent analyses more heavily
        weights = []
        total_weight = 0
        weighted_bias = 0
        
        for i, analysis in enumerate(reversed(self.analysis_history[-10:])):
            # More recent = higher weight
            weight = 1.0 / (i + 1)
            # Also weight by speaker importance
            speaker_weight = self.analyzer.SPEAKER_WEIGHTS.get(
                analysis.event.speaker_role, 0.5
            )
            combined_weight = weight * speaker_weight * analysis.confidence
            
            weighted_bias += analysis.bias_score * combined_weight
            total_weight += combined_weight
        
        if total_weight > 0:
            avg_bias = weighted_bias / total_weight
            
            if avg_bias > 0.5:
                self.current_policy_stance = MonetaryBias.VERY_HAWKISH
            elif avg_bias > 0.2:
                self.current_policy_stance = MonetaryBias.HAWKISH
            elif avg_bias < -0.5:
                self.current_policy_stance = MonetaryBias.VERY_DOVISH
            elif avg_bias < -0.2:
                self.current_policy_stance = MonetaryBias.DOVISH
            else:
                self.current_policy_stance = MonetaryBias.NEUTRAL
    
    def get_policy_trajectory(self) -> dict:
        """Get policy trajectory over time."""
        if len(self.analysis_history) < 2:
            return {"trajectory": "insufficient_data"}
        
        # Calculate trend
        recent_biases = [a.bias_score for a in self.analysis_history[-5:]]
        older_biases = [a.bias_score for a in self.analysis_history[-10:-5]] if len(self.analysis_history) >= 10 else []
        
        if not older_biases:
            return {"trajectory": "insufficient_data"}
        
        recent_avg = sum(recent_biases) / len(recent_biases)
        older_avg = sum(older_biases) / len(older_biases)
        
        change = recent_avg - older_avg
        
        if change > 0.2:
            trajectory = "turning_hawkish"
        elif change < -0.2:
            trajectory = "turning_dovish"
        elif change > 0.1:
            trajectory = "slightly_hawkish_shift"
        elif change < -0.1:
            trajectory = "slightly_dovish_shift"
        else:
            trajectory = "stable"
        
        return {
            "trajectory": trajectory,
            "recent_bias": recent_avg,
            "older_bias": older_avg,
            "change": change,
            "current_stance": self.current_policy_stance.value,
        }
    
    def generate_trading_signal(self, symbol_type: str = "equity") -> dict:
        """
        Generate trading signal based on Fed analysis.
        
        Args:
            symbol_type: Type of asset (equity, bond, gold, dollar)
            
        Returns:
            Trading signal dictionary
        """
        if not self.analysis_history:
            return {"signal": "neutral", "confidence": 0}
        
        latest = self.analysis_history[-1]
        trajectory = self.get_policy_trajectory()
        
        signal = {
            "timestamp": datetime.now().isoformat(),
            "symbol_type": symbol_type,
            "fed_stance": self.current_policy_stance.value,
            "trajectory": trajectory.get("trajectory", "stable"),
            "signal": "neutral",
            "confidence": latest.confidence,
            "reasoning": [],
        }
        
        bias = latest.bias_score
        
        if symbol_type == "equity":
            if bias < -0.3 and trajectory.get("trajectory") in ["turning_dovish", "slightly_dovish_shift"]:
                signal["signal"] = "bullish"
                signal["reasoning"].append("Dovish Fed supports risk assets")
            elif bias > 0.3 and trajectory.get("trajectory") in ["turning_hawkish", "slightly_hawkish_shift"]:
                signal["signal"] = "bearish"
                signal["reasoning"].append("Hawkish Fed pressures valuations")
        
        elif symbol_type == "bond":
            if bias < -0.3:
                signal["signal"] = "bullish"
                signal["reasoning"].append("Dovish Fed supports bonds (lower yields)")
            elif bias > 0.3:
                signal["signal"] = "bearish"
                signal["reasoning"].append("Hawkish Fed pressures bonds (higher yields)")
        
        elif symbol_type == "gold":
            if bias < -0.3:
                signal["signal"] = "bullish"
                signal["reasoning"].append("Dovish Fed (lower real rates) supports gold")
            elif bias > 0.3:
                signal["signal"] = "bearish"
                signal["reasoning"].append("Hawkish Fed (higher real rates) pressures gold")
        
        elif symbol_type == "dollar":
            if bias > 0.3:
                signal["signal"] = "bullish"
                signal["reasoning"].append("Hawkish Fed supports dollar")
            elif bias < -0.3:
                signal["signal"] = "bearish"
                signal["reasoning"].append("Dovish Fed pressures dollar")
        
        # Add market implications from latest analysis
        signal["market_implications"] = latest.market_implications.get(
            symbol_type.rstrip('s'), {}
        )
        
        return signal
