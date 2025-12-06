"""
News & Sentiment Aggregator - Multi-source news and sentiment analysis.

Sources:
- NewsAPI (global news)
- Finnhub (financial news)
- Yahoo Finance News
- Reddit (r/wallstreetbets, r/stocks, r/investing)
- StockTwits
- Twitter/X (via API)
- Google News RSS
- SEC Filings (8-K announcements)
- Earnings Calendars
"""

import logging
import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import xml.etree.ElementTree as ET

import httpx

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """Sentiment classification levels."""
    VERY_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    VERY_BEARISH = -2


@dataclass
class NewsArticle:
    """Structured news article."""
    title: str
    source: str
    url: str
    published: str
    description: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    sentiment: Optional[SentimentLevel] = None
    relevance_score: float = 0.0
    tickers: List[str] = field(default_factory=list)


@dataclass
class SocialPost:
    """Social media post."""
    platform: str
    content: str
    author: str
    url: str
    timestamp: str
    likes: int = 0
    comments: int = 0
    shares: int = 0
    sentiment: Optional[SentimentLevel] = None
    tickers: List[str] = field(default_factory=list)


class NewsAggregator:
    """
    Aggregate news from multiple sources.
    """
    
    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "AI-Trader/1.0"}
        )
        
        # API keys
        self.newsapi_key = os.getenv("NEWS_API_KEY")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
    
    def get_all_news(self, query: str, days: int = 7, 
                     max_per_source: int = 20) -> List[NewsArticle]:
        """
        Get news from all available sources.
        
        Args:
            query: Search query (ticker or keyword)
            days: Number of days to look back
            max_per_source: Maximum articles per source
        
        Returns:
            List of NewsArticle objects
        """
        all_news = []
        
        # NewsAPI
        news = self._fetch_newsapi(query, days, max_per_source)
        all_news.extend(news)
        
        # Finnhub
        news = self._fetch_finnhub(query, days, max_per_source)
        all_news.extend(news)
        
        # Yahoo Finance
        news = self._fetch_yahoo_news(query, max_per_source)
        all_news.extend(news)
        
        # Google News RSS
        news = self._fetch_google_news(query, max_per_source)
        all_news.extend(news)
        
        # Sort by date (newest first)
        all_news.sort(key=lambda x: x.published, reverse=True)
        
        return all_news
    
    def _fetch_newsapi(self, query: str, days: int, limit: int) -> List[NewsArticle]:
        """Fetch from NewsAPI."""
        if not self.newsapi_key:
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            response = self.client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": limit,
                    "apiKey": self.newsapi_key
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return [
                    NewsArticle(
                        title=article.get("title", ""),
                        source="newsapi",
                        url=article.get("url", ""),
                        published=article.get("publishedAt", ""),
                        description=article.get("description"),
                        author=article.get("author"),
                        category="news"
                    )
                    for article in data.get("articles", [])
                ]
        except Exception as e:
            logger.warning(f"NewsAPI failed: {e}")
        
        return []
    
    def _fetch_finnhub(self, ticker: str, days: int, limit: int) -> List[NewsArticle]:
        """Fetch from Finnhub."""
        if not self.finnhub_key:
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            response = self.client.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": ticker.upper(),
                    "from": from_date,
                    "to": to_date,
                    "token": self.finnhub_key
                }
            )
            
            if response.status_code == 200:
                articles = response.json()[:limit]
                return [
                    NewsArticle(
                        title=article.get("headline", ""),
                        source="finnhub",
                        url=article.get("url", ""),
                        published=datetime.fromtimestamp(
                            article.get("datetime", 0)
                        ).isoformat(),
                        description=article.get("summary"),
                        category=article.get("category"),
                        tickers=[ticker.upper()]
                    )
                    for article in articles
                ]
        except Exception as e:
            logger.warning(f"Finnhub news failed: {e}")
        
        return []
    
    def _fetch_yahoo_news(self, ticker: str, limit: int) -> List[NewsArticle]:
        """Fetch from Yahoo Finance."""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            
            return [
                NewsArticle(
                    title=article.get("title", ""),
                    source="yahoo",
                    url=article.get("link", ""),
                    published=datetime.fromtimestamp(
                        article.get("providerPublishTime", 0)
                    ).isoformat(),
                    category=article.get("type"),
                    tickers=[ticker.upper()]
                )
                for article in stock.news[:limit]
            ]
        except Exception as e:
            logger.warning(f"Yahoo news failed: {e}")
        
        return []
    
    def _fetch_google_news(self, query: str, limit: int) -> List[NewsArticle]:
        """Fetch from Google News RSS."""
        try:
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            
            response = self.client.get(
                f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            )
            
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                articles = []
                
                for item in root.findall(".//item")[:limit]:
                    title = item.find("title")
                    link = item.find("link")
                    pub_date = item.find("pubDate")
                    source = item.find("source")
                    
                    articles.append(NewsArticle(
                        title=title.text if title is not None else "",
                        source="google_news",
                        url=link.text if link is not None else "",
                        published=pub_date.text if pub_date is not None else "",
                        author=source.text if source is not None else None
                    ))
                
                return articles
        except Exception as e:
            logger.warning(f"Google News failed: {e}")
        
        return []
    
    def get_earnings_calendar(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming earnings announcements."""
        if not self.finnhub_key:
            return []
        
        try:
            from_date = datetime.now().strftime("%Y-%m-%d")
            to_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
            response = self.client.get(
                "https://finnhub.io/api/v1/calendar/earnings",
                params={
                    "from": from_date,
                    "to": to_date,
                    "token": self.finnhub_key
                }
            )
            
            if response.status_code == 200:
                return response.json().get("earningsCalendar", [])
        except Exception as e:
            logger.warning(f"Earnings calendar failed: {e}")
        
        return []
    
    def get_sec_filings(self, ticker: str, form_types: Optional[List[str]] = None) -> List[Dict]:
        """Get recent SEC filings."""
        form_types = form_types or ["8-K", "10-K", "10-Q", "4"]
        
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            
            # Get SEC filings
            if hasattr(stock, 'sec_filings'):
                return stock.sec_filings
        except:
            pass
        
        return []
    
    def close(self):
        self.client.close()


class SocialSentimentAnalyzer:
    """
    Analyze social media sentiment.
    """
    
    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "AI-Trader/1.0"}
        )
    
    def get_reddit_sentiment(self, ticker: str, 
                            subreddits: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get Reddit sentiment for a stock.
        
        Args:
            ticker: Stock symbol
            subreddits: List of subreddits to search
        
        Returns:
            Sentiment analysis results
        """
        subreddits = subreddits or [
            "wallstreetbets", "stocks", "investing", 
            "options", "stockmarket", "traders"
        ]
        
        all_posts = []
        
        for subreddit in subreddits:
            try:
                # Use Reddit's JSON API (no auth needed for public posts)
                response = self.client.get(
                    f"https://www.reddit.com/r/{subreddit}/search.json",
                    params={
                        "q": ticker,
                        "sort": "new",
                        "limit": 25,
                        "t": "week",
                        "restrict_sr": "true"
                    },
                    follow_redirects=True
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for post in data.get("data", {}).get("children", []):
                        post_data = post.get("data", {})
                        all_posts.append(SocialPost(
                            platform="reddit",
                            content=post_data.get("title", ""),
                            author=post_data.get("author", ""),
                            url=f"https://reddit.com{post_data.get('permalink', '')}",
                            timestamp=datetime.fromtimestamp(
                                post_data.get("created_utc", 0)
                            ).isoformat(),
                            likes=post_data.get("score", 0),
                            comments=post_data.get("num_comments", 0),
                            tickers=[ticker]
                        ))
                
                # Rate limiting
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Reddit r/{subreddit} failed: {e}")
        
        # Analyze sentiment
        sentiment_scores = self._calculate_simple_sentiment(all_posts)
        
        return {
            "ticker": ticker,
            "platform": "reddit",
            "total_posts": len(all_posts),
            "total_score": sum(p.likes for p in all_posts),
            "total_comments": sum(p.comments for p in all_posts),
            "posts": [self._post_to_dict(p) for p in all_posts[:50]],
            "sentiment": sentiment_scores
        }
    
    def get_stocktwits_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Get StockTwits sentiment.
        
        StockTwits API is public for basic data.
        """
        try:
            response = self.client.get(
                f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
            )
            
            if response.status_code == 200:
                data = response.json()
                messages = data.get("messages", [])
                
                # Count sentiments
                bullish = sum(1 for m in messages 
                             if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish")
                bearish = sum(1 for m in messages 
                             if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bearish")
                
                posts = [
                    SocialPost(
                        platform="stocktwits",
                        content=m.get("body", ""),
                        author=m.get("user", {}).get("username", ""),
                        url=f"https://stocktwits.com/{m.get('user', {}).get('username', '')}/message/{m.get('id', '')}",
                        timestamp=m.get("created_at", ""),
                        likes=m.get("likes", {}).get("total", 0),
                        tickers=[ticker]
                    )
                    for m in messages
                ]
                
                total = bullish + bearish
                sentiment_ratio = bullish / total if total > 0 else 0.5
                
                return {
                    "ticker": ticker,
                    "platform": "stocktwits",
                    "total_posts": len(messages),
                    "bullish_count": bullish,
                    "bearish_count": bearish,
                    "sentiment_ratio": sentiment_ratio,
                    "sentiment_label": self._ratio_to_sentiment(sentiment_ratio),
                    "posts": [self._post_to_dict(p) for p in posts[:20]]
                }
                
        except Exception as e:
            logger.warning(f"StockTwits failed: {e}")
        
        return {"ticker": ticker, "error": "Failed to fetch StockTwits data"}
    
    def get_combined_social_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Get combined social sentiment from all platforms.
        """
        reddit = self.get_reddit_sentiment(ticker)
        stocktwits = self.get_stocktwits_sentiment(ticker)
        
        # Combine scores
        scores = []
        
        if "sentiment" in reddit:
            scores.append(reddit["sentiment"].get("score", 0))
        
        if "sentiment_ratio" in stocktwits:
            # Convert ratio (0-1) to score (-1 to 1)
            scores.append((stocktwits["sentiment_ratio"] - 0.5) * 2)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "ticker": ticker,
            "combined_score": avg_score,
            "combined_sentiment": self._score_to_sentiment(avg_score),
            "platforms": {
                "reddit": reddit,
                "stocktwits": stocktwits
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_simple_sentiment(self, posts: List[SocialPost]) -> Dict:
        """
        Calculate simple keyword-based sentiment.
        """
        bullish_words = [
            "buy", "long", "calls", "moon", "rocket", "bullish", 
            "green", "pump", "yolo", "diamond", "hands", "squeeze",
            "undervalued", "breakout", "upside", "rally"
        ]
        bearish_words = [
            "sell", "short", "puts", "dump", "crash", "bearish",
            "red", "overvalued", "downside", "avoid", "bubble",
            "dead", "rip", "loss", "plunge"
        ]
        
        bullish_count = 0
        bearish_count = 0
        
        for post in posts:
            content_lower = post.content.lower()
            
            for word in bullish_words:
                if word in content_lower:
                    bullish_count += 1
                    break
            
            for word in bearish_words:
                if word in content_lower:
                    bearish_count += 1
                    break
        
        total = bullish_count + bearish_count
        if total == 0:
            return {"score": 0, "label": "NEUTRAL", "bullish": 0, "bearish": 0}
        
        score = (bullish_count - bearish_count) / total
        
        return {
            "score": score,
            "label": self._score_to_sentiment(score),
            "bullish": bullish_count,
            "bearish": bearish_count,
            "total_analyzed": len(posts)
        }
    
    def _ratio_to_sentiment(self, ratio: float) -> str:
        """Convert ratio (0-1) to sentiment label."""
        if ratio >= 0.7:
            return "VERY_BULLISH"
        elif ratio >= 0.55:
            return "BULLISH"
        elif ratio <= 0.3:
            return "VERY_BEARISH"
        elif ratio <= 0.45:
            return "BEARISH"
        return "NEUTRAL"
    
    def _score_to_sentiment(self, score: float) -> str:
        """Convert score (-1 to 1) to sentiment label."""
        if score >= 0.5:
            return "VERY_BULLISH"
        elif score >= 0.2:
            return "BULLISH"
        elif score <= -0.5:
            return "VERY_BEARISH"
        elif score <= -0.2:
            return "BEARISH"
        return "NEUTRAL"
    
    def _post_to_dict(self, post: SocialPost) -> Dict:
        """Convert post to dictionary."""
        return {
            "platform": post.platform,
            "content": post.content[:200],  # Truncate
            "author": post.author,
            "url": post.url,
            "timestamp": post.timestamp,
            "likes": post.likes,
            "comments": post.comments
        }
    
    def close(self):
        self.client.close()


class TrendingAnalyzer:
    """
    Analyze trending stocks and topics.
    """
    
    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "AI-Trader/1.0"}
        )
    
    def get_trending_tickers(self) -> List[Dict]:
        """Get trending tickers from multiple sources."""
        trending = []
        
        # Yahoo Finance trending
        try:
            import yfinance as yf
            # Get trending from Yahoo
            # Note: yfinance doesn't have direct trending API
            # We'll use most active as proxy
        except:
            pass
        
        # StockTwits trending
        try:
            response = self.client.get(
                "https://api.stocktwits.com/api/2/trending/symbols.json"
            )
            if response.status_code == 200:
                data = response.json()
                for symbol in data.get("symbols", []):
                    trending.append({
                        "ticker": symbol.get("symbol"),
                        "title": symbol.get("title"),
                        "source": "stocktwits",
                        "watchlist_count": symbol.get("watchlist_count", 0)
                    })
        except Exception as e:
            logger.warning(f"StockTwits trending failed: {e}")
        
        return trending
    
    def get_wsb_trending(self) -> List[Dict]:
        """Get trending stocks on r/wallstreetbets."""
        try:
            response = self.client.get(
                "https://www.reddit.com/r/wallstreetbets/hot.json",
                params={"limit": 50},
                follow_redirects=True
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract tickers from titles
                ticker_pattern = r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b'
                ticker_counts = Counter()
                
                for post in data.get("data", {}).get("children", []):
                    title = post.get("data", {}).get("title", "")
                    matches = re.findall(ticker_pattern, title)
                    for match in matches:
                        ticker = match[0] or match[1]
                        if len(ticker) >= 2 and ticker not in ["THE", "FOR", "AND", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT"]:
                            ticker_counts[ticker] += 1
                
                return [
                    {"ticker": ticker, "mentions": count, "source": "wsb"}
                    for ticker, count in ticker_counts.most_common(20)
                ]
        except Exception as e:
            logger.warning(f"WSB trending failed: {e}")
        
        return []
    
    def close(self):
        self.client.close()


def get_comprehensive_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Get comprehensive sentiment analysis for a ticker.
    
    Combines news, social media, and other sources.
    """
    news_agg = NewsAggregator()
    social_analyzer = SocialSentimentAnalyzer()
    
    try:
        # Get news
        news = news_agg.get_all_news(ticker, days=7, max_per_source=10)
        
        # Get social sentiment
        social = social_analyzer.get_combined_social_sentiment(ticker)
        
        # Get earnings calendar
        earnings = news_agg.get_earnings_calendar(days_ahead=14)
        ticker_earnings = [e for e in earnings if e.get("symbol") == ticker.upper()]
        
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "news": {
                "total": len(news),
                "sources": list(set(n.source for n in news)),
                "articles": [
                    {
                        "title": n.title,
                        "source": n.source,
                        "url": n.url,
                        "published": n.published
                    }
                    for n in news[:20]
                ]
            },
            "social_sentiment": social,
            "upcoming_earnings": ticker_earnings,
            "overall_sentiment": social.get("combined_sentiment", "NEUTRAL"),
            "overall_score": social.get("combined_score", 0)
        }
    finally:
        news_agg.close()
        social_analyzer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing News & Sentiment Aggregator ===\n")
    
    # Test news
    news_agg = NewsAggregator()
    try:
        news = news_agg.get_all_news("AAPL", days=3, max_per_source=5)
        print(f"Found {len(news)} news articles for AAPL")
        for article in news[:3]:
            print(f"  - {article.title[:60]}... ({article.source})")
    finally:
        news_agg.close()
    
    # Test social sentiment
    print("\n=== Testing Social Sentiment ===\n")
    social = SocialSentimentAnalyzer()
    try:
        sentiment = social.get_combined_social_sentiment("TSLA")
        print(f"TSLA Combined Sentiment: {sentiment.get('combined_sentiment')}")
        print(f"Score: {sentiment.get('combined_score', 0):.2f}")
    finally:
        social.close()
    
    # Test trending
    print("\n=== Trending on WSB ===\n")
    trending = TrendingAnalyzer()
    try:
        wsb = trending.get_wsb_trending()
        for t in wsb[:5]:
            print(f"  ${t['ticker']}: {t['mentions']} mentions")
    finally:
        trending.close()
