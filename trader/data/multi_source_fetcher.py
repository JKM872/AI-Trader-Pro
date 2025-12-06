"""
Multi-Source Data Fetcher - Comprehensive market data from multiple FREE sources.

Data Categories:
1. Market Data: yfinance, Alpha Vantage, Polygon.io, Finnhub, Twelve Data
2. News: NewsAPI, Finnhub News, Yahoo Finance News, Reddit, StockTwits
3. Fundamentals: yfinance, Financial Modeling Prep, SEC Edgar
4. Economic Data: FRED, World Bank, Trading Economics
5. Crypto: CoinGecko, Binance, CoinMarketCap
6. Alternative Data: Reddit, Twitter/X sentiment, GitHub (for tech stocks)
7. Whale/Insider Tracking: OpenInsider, SEC Edgar, WhaleWisdom
"""

import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import pandas as pd
import httpx

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source identifiers."""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    FMP = "fmp"  # Financial Modeling Prep
    FRED = "fred"
    NEWSAPI = "newsapi"
    REDDIT = "reddit"
    COINGECKO = "coingecko"
    SEC_EDGAR = "sec_edgar"
    WHALE_WISDOM = "whale_wisdom"


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    env_key: str
    base_url: str
    rate_limit: int  # requests per minute
    daily_limit: Optional[int] = None
    is_free: bool = True
    description: str = ""


# All available data sources configuration
DATA_SOURCES: Dict[str, DataSourceConfig] = {
    "alpha_vantage": DataSourceConfig(
        name="Alpha Vantage",
        env_key="ALPHA_VANTAGE_API_KEY",
        base_url="https://www.alphavantage.co/query",
        rate_limit=5,
        daily_limit=500,
        description="Stock data, technicals, fundamentals"
    ),
    "finnhub": DataSourceConfig(
        name="Finnhub",
        env_key="FINNHUB_API_KEY",
        base_url="https://finnhub.io/api/v1",
        rate_limit=60,
        description="Real-time data, news, fundamentals"
    ),
    "polygon": DataSourceConfig(
        name="Polygon.io",
        env_key="POLYGON_API_KEY",
        base_url="https://api.polygon.io",
        rate_limit=5,
        description="Historical and real-time market data"
    ),
    "twelve_data": DataSourceConfig(
        name="Twelve Data",
        env_key="TWELVE_DATA_API_KEY",
        base_url="https://api.twelvedata.com",
        rate_limit=8,
        daily_limit=800,
        description="Technical indicators, time series"
    ),
    "fmp": DataSourceConfig(
        name="Financial Modeling Prep",
        env_key="FMP_API_KEY",
        base_url="https://financialmodelingprep.com/api/v3",
        rate_limit=300,
        daily_limit=250,
        description="Financials, ratios, DCF models"
    ),
    "fred": DataSourceConfig(
        name="FRED (Federal Reserve)",
        env_key="FRED_API_KEY",
        base_url="https://api.stlouisfed.org/fred",
        rate_limit=120,
        description="US economic data, rates, indicators"
    ),
    "newsapi": DataSourceConfig(
        name="NewsAPI",
        env_key="NEWS_API_KEY",
        base_url="https://newsapi.org/v2",
        rate_limit=100,
        daily_limit=100,
        description="Global news articles"
    ),
    "coingecko": DataSourceConfig(
        name="CoinGecko",
        env_key="",  # No key needed for free tier
        base_url="https://api.coingecko.com/api/v3",
        rate_limit=10,
        description="Crypto prices, market data"
    ),
}


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_requests: int, period_seconds: int = 60):
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.requests: List[float] = []
    
    def wait_if_needed(self):
        now = time.time()
        self.requests = [r for r in self.requests if now - r < self.period_seconds]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = self.period_seconds - (now - self.requests[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.requests.append(time.time())


class MultiSourceFetcher:
    """
    Comprehensive data fetcher from multiple sources.
    
    Usage:
        fetcher = MultiSourceFetcher()
        data = fetcher.get_stock_data("AAPL")
        news = fetcher.get_news("AAPL")
        fundamentals = fetcher.get_fundamentals("AAPL")
    """
    
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 300  # 5 minutes default
        
        # Initialize rate limiters
        self._limiters: Dict[str, RateLimiter] = {}
        for name, config in DATA_SOURCES.items():
            self._limiters[name] = RateLimiter(config.rate_limit)
        
        # Load API keys
        self._api_keys: Dict[str, Optional[str]] = {}
        for name, config in DATA_SOURCES.items():
            if config.env_key:
                self._api_keys[name] = os.getenv(config.env_key)
    
    def get_available_sources(self) -> Dict[str, bool]:
        """Get status of all data sources."""
        result = {}
        for name, config in DATA_SOURCES.items():
            if not config.env_key:
                result[name] = True  # No key needed
            else:
                result[name] = bool(self._api_keys.get(name))
        return result
    
    # ==================== Market Data ====================
    
    def get_stock_data_multi(self, ticker: str, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data from multiple sources for comparison.
        
        Returns dict with source name -> DataFrame
        """
        import yfinance as yf
        
        results = {}
        
        # yfinance (always available)
        try:
            stock = yf.Ticker(ticker)
            results["yfinance"] = stock.history(period=period)
        except Exception as e:
            logger.warning(f"yfinance failed: {e}")
        
        # Alpha Vantage
        if self._api_keys.get("alpha_vantage"):
            try:
                self._limiters["alpha_vantage"].wait_if_needed()
                response = self.client.get(
                    DATA_SOURCES["alpha_vantage"].base_url,
                    params={
                        "function": "TIME_SERIES_DAILY_ADJUSTED",
                        "symbol": ticker,
                        "outputsize": "full",
                        "apikey": self._api_keys["alpha_vantage"]
                    }
                )
                data = response.json()
                if "Time Series (Daily)" in data:
                    df = pd.DataFrame(data["Time Series (Daily)"]).T
                    df.columns = ["Open", "High", "Low", "Close", "Adj Close", 
                                 "Volume", "Dividend", "Split"]
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float)
                    results["alpha_vantage"] = df.sort_index()
            except Exception as e:
                logger.warning(f"Alpha Vantage failed: {e}")
        
        # Finnhub
        if self._api_keys.get("finnhub"):
            try:
                self._limiters["finnhub"].wait_if_needed()
                end = int(datetime.now().timestamp())
                start = int((datetime.now() - timedelta(days=365)).timestamp())
                response = self.client.get(
                    f"{DATA_SOURCES['finnhub'].base_url}/stock/candle",
                    params={
                        "symbol": ticker,
                        "resolution": "D",
                        "from": start,
                        "to": end,
                        "token": self._api_keys["finnhub"]
                    }
                )
                data = response.json()
                if data.get("s") == "ok":
                    df = pd.DataFrame({
                        "Open": data["o"],
                        "High": data["h"],
                        "Low": data["l"],
                        "Close": data["c"],
                        "Volume": data["v"]
                    }, index=pd.to_datetime(data["t"], unit="s"))
                    results["finnhub"] = df
            except Exception as e:
                logger.warning(f"Finnhub failed: {e}")
        
        # Twelve Data
        if self._api_keys.get("twelve_data"):
            try:
                self._limiters["twelve_data"].wait_if_needed()
                response = self.client.get(
                    f"{DATA_SOURCES['twelve_data'].base_url}/time_series",
                    params={
                        "symbol": ticker,
                        "interval": "1day",
                        "outputsize": 365,
                        "apikey": self._api_keys["twelve_data"]
                    }
                )
                data = response.json()
                if "values" in data:
                    df = pd.DataFrame(data["values"])
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                    df = df.rename(columns={
                        "open": "Open", "high": "High", 
                        "low": "Low", "close": "Close", "volume": "Volume"
                    })
                    df = df.astype(float)
                    results["twelve_data"] = df.sort_index()
            except Exception as e:
                logger.warning(f"Twelve Data failed: {e}")
        
        return results
    
    def get_realtime_quote(self, ticker: str) -> Dict[str, Any]:
        """Get real-time quote from multiple sources."""
        import yfinance as yf
        
        quotes = {}
        
        # yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            quotes["yfinance"] = {
                "price": info.get("regularMarketPrice"),
                "change": info.get("regularMarketChange"),
                "change_pct": info.get("regularMarketChangePercent"),
                "volume": info.get("regularMarketVolume"),
                "bid": info.get("bid"),
                "ask": info.get("ask")
            }
        except:
            pass
        
        # Finnhub
        if self._api_keys.get("finnhub"):
            try:
                self._limiters["finnhub"].wait_if_needed()
                response = self.client.get(
                    f"{DATA_SOURCES['finnhub'].base_url}/quote",
                    params={
                        "symbol": ticker,
                        "token": self._api_keys["finnhub"]
                    }
                )
                data = response.json()
                quotes["finnhub"] = {
                    "price": data.get("c"),
                    "change": data.get("d"),
                    "change_pct": data.get("dp"),
                    "high": data.get("h"),
                    "low": data.get("l"),
                    "open": data.get("o"),
                    "prev_close": data.get("pc")
                }
            except:
                pass
        
        return quotes
    
    # ==================== News Data ====================
    
    def get_news(self, query: str, days: int = 7) -> List[Dict]:
        """
        Fetch news from multiple sources.
        
        Returns list of news articles with source info.
        """
        all_news = []
        
        # NewsAPI
        if self._api_keys.get("newsapi"):
            try:
                self._limiters["newsapi"].wait_if_needed()
                from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                response = self.client.get(
                    f"{DATA_SOURCES['newsapi'].base_url}/everything",
                    params={
                        "q": query,
                        "from": from_date,
                        "language": "en",
                        "sortBy": "relevancy",
                        "pageSize": 50,
                        "apiKey": self._api_keys["newsapi"]
                    }
                )
                data = response.json()
                for article in data.get("articles", []):
                    all_news.append({
                        "source": "newsapi",
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "url": article.get("url"),
                        "published": article.get("publishedAt"),
                        "source_name": article.get("source", {}).get("name")
                    })
            except Exception as e:
                logger.warning(f"NewsAPI failed: {e}")
        
        # Finnhub News
        if self._api_keys.get("finnhub"):
            try:
                self._limiters["finnhub"].wait_if_needed()
                from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                to_date = datetime.now().strftime("%Y-%m-%d")
                response = self.client.get(
                    f"{DATA_SOURCES['finnhub'].base_url}/company-news",
                    params={
                        "symbol": query,
                        "from": from_date,
                        "to": to_date,
                        "token": self._api_keys["finnhub"]
                    }
                )
                for article in response.json()[:50]:
                    all_news.append({
                        "source": "finnhub",
                        "title": article.get("headline"),
                        "description": article.get("summary"),
                        "url": article.get("url"),
                        "published": datetime.fromtimestamp(article.get("datetime", 0)).isoformat(),
                        "source_name": article.get("source")
                    })
            except Exception as e:
                logger.warning(f"Finnhub news failed: {e}")
        
        # Yahoo Finance News (via yfinance)
        try:
            import yfinance as yf
            stock = yf.Ticker(query)
            for article in stock.news[:20]:
                all_news.append({
                    "source": "yahoo",
                    "title": article.get("title"),
                    "description": "",
                    "url": article.get("link"),
                    "published": datetime.fromtimestamp(article.get("providerPublishTime", 0)).isoformat(),
                    "source_name": article.get("publisher")
                })
        except:
            pass
        
        return all_news
    
    def get_reddit_sentiment(self, ticker: str, subreddits: List[str] = None) -> Dict:
        """
        Get Reddit sentiment for a stock.
        Uses Pushshift/Reddit API.
        """
        subreddits = subreddits or ["wallstreetbets", "stocks", "investing", "options"]
        
        mentions = []
        
        # Note: This requires Reddit API access or web scraping
        # For now, we'll use a simple approach via Reddit's JSON API
        for subreddit in subreddits:
            try:
                response = self.client.get(
                    f"https://www.reddit.com/r/{subreddit}/search.json",
                    params={
                        "q": ticker,
                        "sort": "new",
                        "limit": 25,
                        "t": "week"
                    },
                    headers={"User-Agent": "AI-Trader/1.0"}
                )
                if response.status_code == 200:
                    data = response.json()
                    for post in data.get("data", {}).get("children", []):
                        post_data = post.get("data", {})
                        mentions.append({
                            "subreddit": subreddit,
                            "title": post_data.get("title"),
                            "score": post_data.get("score", 0),
                            "comments": post_data.get("num_comments", 0),
                            "url": f"https://reddit.com{post_data.get('permalink', '')}",
                            "created": datetime.fromtimestamp(post_data.get("created_utc", 0)).isoformat()
                        })
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Reddit failed for r/{subreddit}: {e}")
        
        # Calculate sentiment metrics
        total_score = sum(m["score"] for m in mentions)
        total_comments = sum(m["comments"] for m in mentions)
        
        return {
            "ticker": ticker,
            "mentions": mentions,
            "total_mentions": len(mentions),
            "total_score": total_score,
            "total_comments": total_comments,
            "avg_score": total_score / len(mentions) if mentions else 0
        }
    
    # ==================== Fundamentals ====================
    
    def get_fundamentals_multi(self, ticker: str) -> Dict[str, Dict]:
        """Get fundamentals from multiple sources."""
        import yfinance as yf
        
        results = {}
        
        # yfinance
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            results["yfinance"] = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "profit_margin": info.get("profitMargins"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "revenue": info.get("totalRevenue"),
                "earnings": info.get("netIncomeToCommon"),
                "free_cash_flow": info.get("freeCashflow")
            }
        except Exception as e:
            logger.warning(f"yfinance fundamentals failed: {e}")
        
        # Financial Modeling Prep
        if self._api_keys.get("fmp"):
            try:
                self._limiters["fmp"].wait_if_needed()
                
                # Profile
                response = self.client.get(
                    f"{DATA_SOURCES['fmp'].base_url}/profile/{ticker}",
                    params={"apikey": self._api_keys["fmp"]}
                )
                profile = response.json()[0] if response.json() else {}
                
                # Ratios
                response = self.client.get(
                    f"{DATA_SOURCES['fmp'].base_url}/ratios/{ticker}",
                    params={"apikey": self._api_keys["fmp"], "limit": 1}
                )
                ratios = response.json()[0] if response.json() else {}
                
                results["fmp"] = {
                    "market_cap": profile.get("mktCap"),
                    "pe_ratio": ratios.get("peRatioTTM"),
                    "peg_ratio": ratios.get("pegRatioTTM"),
                    "price_to_book": ratios.get("priceToBookRatioTTM"),
                    "roe": ratios.get("returnOnEquityTTM"),
                    "roa": ratios.get("returnOnAssetsTTM"),
                    "debt_to_equity": ratios.get("debtEquityRatioTTM"),
                    "current_ratio": ratios.get("currentRatioTTM"),
                    "quick_ratio": ratios.get("quickRatioTTM"),
                    "dividend_yield": ratios.get("dividendYielTTM"),
                    "dcf": profile.get("dcf"),
                    "dcf_diff": profile.get("dcfDiff")
                }
            except Exception as e:
                logger.warning(f"FMP fundamentals failed: {e}")
        
        # Finnhub
        if self._api_keys.get("finnhub"):
            try:
                self._limiters["finnhub"].wait_if_needed()
                response = self.client.get(
                    f"{DATA_SOURCES['finnhub'].base_url}/stock/metric",
                    params={
                        "symbol": ticker,
                        "metric": "all",
                        "token": self._api_keys["finnhub"]
                    }
                )
                data = response.json().get("metric", {})
                results["finnhub"] = {
                    "pe_ratio": data.get("peBasicExclExtraTTM"),
                    "peg_ratio": data.get("pegRatioTTM"),
                    "price_to_book": data.get("pbQuarterly"),
                    "roe": data.get("roeTTM"),
                    "roa": data.get("roaTTM"),
                    "eps_growth_3y": data.get("epsGrowth3Y"),
                    "revenue_growth_3y": data.get("revenueGrowth3Y"),
                    "dividend_yield": data.get("dividendYieldIndicatedAnnual"),
                    "52w_high": data.get("52WeekHigh"),
                    "52w_low": data.get("52WeekLow"),
                    "beta": data.get("beta")
                }
            except Exception as e:
                logger.warning(f"Finnhub fundamentals failed: {e}")
        
        return results
    
    # ==================== Economic Data (FRED) ====================
    
    def get_economic_indicators(self) -> Dict[str, Any]:
        """Get key economic indicators from FRED."""
        if not self._api_keys.get("fred"):
            logger.warning("FRED API key not set")
            return {}
        
        indicators = {
            "GDP": "GDP",
            "UNRATE": "Unemployment Rate",
            "CPIAUCSL": "Consumer Price Index",
            "FEDFUNDS": "Federal Funds Rate",
            "DGS10": "10-Year Treasury Yield",
            "DGS2": "2-Year Treasury Yield",
            "T10Y2Y": "10Y-2Y Treasury Spread",
            "VIXCLS": "VIX",
            "UMCSENT": "Consumer Sentiment",
            "INDPRO": "Industrial Production"
        }
        
        results = {}
        for series_id, name in indicators.items():
            try:
                self._limiters["fred"].wait_if_needed()
                response = self.client.get(
                    f"{DATA_SOURCES['fred'].base_url}/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": self._api_keys["fred"],
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 1
                    }
                )
                data = response.json()
                if "observations" in data and data["observations"]:
                    obs = data["observations"][0]
                    results[series_id] = {
                        "name": name,
                        "value": float(obs["value"]) if obs["value"] != "." else None,
                        "date": obs["date"]
                    }
            except Exception as e:
                logger.warning(f"FRED {series_id} failed: {e}")
        
        return results
    
    def get_treasury_yields(self) -> Dict[str, float]:
        """Get current Treasury yield curve."""
        if not self._api_keys.get("fred"):
            return {}
        
        series = {
            "DGS1MO": "1 Month",
            "DGS3MO": "3 Month",
            "DGS6MO": "6 Month",
            "DGS1": "1 Year",
            "DGS2": "2 Year",
            "DGS5": "5 Year",
            "DGS10": "10 Year",
            "DGS20": "20 Year",
            "DGS30": "30 Year"
        }
        
        yields = {}
        for series_id, label in series.items():
            try:
                self._limiters["fred"].wait_if_needed()
                response = self.client.get(
                    f"{DATA_SOURCES['fred'].base_url}/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": self._api_keys["fred"],
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 1
                    }
                )
                data = response.json()
                if "observations" in data and data["observations"]:
                    val = data["observations"][0]["value"]
                    if val != ".":
                        yields[label] = float(val)
            except:
                pass
        
        return yields
    
    # ==================== Crypto Data ====================
    
    def get_crypto_data(self, coin_id: str = "bitcoin") -> Dict:
        """Get crypto data from CoinGecko (no API key needed)."""
        try:
            self._limiters["coingecko"].wait_if_needed()
            response = self.client.get(
                f"{DATA_SOURCES['coingecko'].base_url}/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "community_data": "true",
                    "developer_data": "true"
                }
            )
            data = response.json()
            
            return {
                "id": data.get("id"),
                "name": data.get("name"),
                "symbol": data.get("symbol"),
                "price_usd": data.get("market_data", {}).get("current_price", {}).get("usd"),
                "market_cap": data.get("market_data", {}).get("market_cap", {}).get("usd"),
                "volume_24h": data.get("market_data", {}).get("total_volume", {}).get("usd"),
                "change_24h": data.get("market_data", {}).get("price_change_percentage_24h"),
                "change_7d": data.get("market_data", {}).get("price_change_percentage_7d"),
                "change_30d": data.get("market_data", {}).get("price_change_percentage_30d"),
                "ath": data.get("market_data", {}).get("ath", {}).get("usd"),
                "ath_change": data.get("market_data", {}).get("ath_change_percentage", {}).get("usd"),
                "circulating_supply": data.get("market_data", {}).get("circulating_supply"),
                "total_supply": data.get("market_data", {}).get("total_supply")
            }
        except Exception as e:
            logger.error(f"CoinGecko failed: {e}")
            return {}
    
    def get_crypto_market_overview(self) -> Dict:
        """Get crypto market overview."""
        try:
            self._limiters["coingecko"].wait_if_needed()
            response = self.client.get(
                f"{DATA_SOURCES['coingecko'].base_url}/global"
            )
            data = response.json().get("data", {})
            
            return {
                "total_market_cap": data.get("total_market_cap", {}).get("usd"),
                "total_volume": data.get("total_volume", {}).get("usd"),
                "btc_dominance": data.get("market_cap_percentage", {}).get("btc"),
                "eth_dominance": data.get("market_cap_percentage", {}).get("eth"),
                "active_cryptocurrencies": data.get("active_cryptocurrencies"),
                "market_cap_change_24h": data.get("market_cap_change_percentage_24h_usd")
            }
        except Exception as e:
            logger.error(f"CoinGecko global failed: {e}")
            return {}
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class InsiderTracker:
    """
    Track insider trading and whale movements.
    Uses SEC Edgar and other public sources.
    """
    
    SEC_EDGAR_BASE = "https://data.sec.gov"
    OPENINSIDER_BASE = "http://openinsider.com"
    
    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": "AI-Trader/1.0 (contact@example.com)"}
        )
    
    def get_insider_trades(self, ticker: str, days: int = 30) -> List[Dict]:
        """
        Get recent insider trades for a stock.
        """
        trades = []
        
        # SEC Edgar Form 4 filings
        try:
            # Get CIK for ticker
            response = self.client.get(
                f"{self.SEC_EDGAR_BASE}/submissions/CIK{ticker.upper()}.json"
            )
            if response.status_code == 200:
                data = response.json()
                filings = data.get("filings", {}).get("recent", {})
                
                # Find Form 4 filings
                forms = filings.get("form", [])
                dates = filings.get("filingDate", [])
                accessions = filings.get("accessionNumber", [])
                
                for i, form in enumerate(forms[:50]):
                    if form == "4":
                        trades.append({
                            "source": "sec_edgar",
                            "form": form,
                            "filing_date": dates[i] if i < len(dates) else None,
                            "accession": accessions[i] if i < len(accessions) else None
                        })
        except Exception as e:
            logger.warning(f"SEC Edgar failed: {e}")
        
        return trades
    
    def get_institutional_holdings(self, ticker: str) -> Dict:
        """
        Get institutional holdings (13F filings).
        """
        import yfinance as yf
        
        try:
            stock = yf.Ticker(ticker)
            holders = stock.institutional_holders
            
            if holders is not None and not holders.empty:
                return {
                    "ticker": ticker,
                    "top_holders": holders.to_dict("records"),
                    "total_institutional_shares": holders["Shares"].sum() if "Shares" in holders.columns else 0
                }
        except Exception as e:
            logger.warning(f"Institutional holdings failed: {e}")
        
        return {"ticker": ticker, "top_holders": []}
    
    def get_major_holders(self, ticker: str) -> Dict:
        """Get major shareholders."""
        import yfinance as yf
        
        try:
            stock = yf.Ticker(ticker)
            major = stock.major_holders
            
            if major is not None and not major.empty:
                return {
                    "ticker": ticker,
                    "insider_ownership": major.iloc[0, 0] if len(major) > 0 else None,
                    "institutional_ownership": major.iloc[1, 0] if len(major) > 1 else None
                }
        except Exception as e:
            logger.warning(f"Major holders failed: {e}")
        
        return {"ticker": ticker}
    
    def close(self):
        self.client.close()


class WhaleWatcher:
    """
    Track whale wallets and large transactions (crypto).
    """
    
    WHALE_ALERT_BASE = "https://api.whale-alert.io/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("WHALE_ALERT_API_KEY")
        self.client = httpx.Client(timeout=30.0)
    
    def get_large_transactions(self, min_value: int = 1000000) -> List[Dict]:
        """Get recent large crypto transactions (requires API key)."""
        if not self.api_key:
            logger.warning("Whale Alert API key not set")
            return []
        
        try:
            response = self.client.get(
                f"{self.WHALE_ALERT_BASE}/transactions",
                params={
                    "api_key": self.api_key,
                    "min_value": min_value,
                    "limit": 100
                }
            )
            data = response.json()
            
            transactions = []
            for tx in data.get("transactions", []):
                transactions.append({
                    "blockchain": tx.get("blockchain"),
                    "symbol": tx.get("symbol"),
                    "amount": tx.get("amount"),
                    "amount_usd": tx.get("amount_usd"),
                    "from_type": tx.get("from", {}).get("owner_type"),
                    "to_type": tx.get("to", {}).get("owner_type"),
                    "timestamp": tx.get("timestamp"),
                    "hash": tx.get("hash")
                })
            
            return transactions
        except Exception as e:
            logger.error(f"Whale Alert failed: {e}")
            return []
    
    def close(self):
        self.client.close()


def get_all_data_for_analysis(ticker: str) -> Dict[str, Any]:
    """
    Comprehensive data gathering for a single ticker.
    Returns all available data from all sources.
    """
    fetcher = MultiSourceFetcher()
    insider_tracker = InsiderTracker()
    
    try:
        result = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "market_data": fetcher.get_stock_data_multi(ticker),
            "realtime_quote": fetcher.get_realtime_quote(ticker),
            "fundamentals": fetcher.get_fundamentals_multi(ticker),
            "news": fetcher.get_news(ticker, days=7),
            "reddit_sentiment": fetcher.get_reddit_sentiment(ticker),
            "insider_trades": insider_tracker.get_insider_trades(ticker),
            "institutional_holdings": insider_tracker.get_institutional_holdings(ticker),
            "major_holders": insider_tracker.get_major_holders(ticker),
            "economic_indicators": fetcher.get_economic_indicators(),
            "sources_available": fetcher.get_available_sources()
        }
        return result
    finally:
        fetcher.close()
        insider_tracker.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    fetcher = MultiSourceFetcher()
    
    print("=== Available Data Sources ===")
    sources = fetcher.get_available_sources()
    for name, available in sources.items():
        status = "✓" if available else "✗"
        config = DATA_SOURCES.get(name)
        if config:
            print(f"{status} {config.name}: {config.description}")
            if config.env_key and not available:
                print(f"   Set {config.env_key} to enable")
    
    print("\n=== Testing Data Fetch ===")
    try:
        quotes = fetcher.get_realtime_quote("AAPL")
        print(f"AAPL quotes from {len(quotes)} sources")
        for source, data in quotes.items():
            print(f"  {source}: ${data.get('price', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")
    
    fetcher.close()
