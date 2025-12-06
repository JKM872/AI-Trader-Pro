"""
DataFetcher - Module for fetching market data from multiple sources.

Sources:
- yfinance: Price data, fundamentals (free, unlimited)
- Alpha Vantage: Additional fundamentals (500 req/day free)
- NewsAPI: Financial news (100 req/day free)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from functools import lru_cache
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to avoid hitting API limits."""
    
    def __init__(self, max_requests: int, period_seconds: int):
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.requests: List[float] = []
    
    def wait_if_needed(self):
        """Wait if we've hit the rate limit."""
        now = time.time()
        # Remove old requests outside the window
        self.requests = [r for r in self.requests if now - r < self.period_seconds]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = self.period_seconds - (now - self.requests[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.requests.append(time.time())


class DataFetcher:
    """
    Fetch market data from multiple sources.
    
    Usage:
        fetcher = DataFetcher()
        data = fetcher.get_stock_data('AAPL', period='1y')
        fundamentals = fetcher.get_fundamentals('AAPL')
    """
    
    # Rate limiters for different APIs
    _alpha_vantage_limiter = RateLimiter(max_requests=5, period_seconds=60)
    _news_api_limiter = RateLimiter(max_requests=100, period_seconds=86400)
    
    def __init__(self, alpha_vantage_key: Optional[str] = None, 
                 news_api_key: Optional[str] = None):
        """
        Initialize DataFetcher with optional API keys.
        
        Args:
            alpha_vantage_key: Alpha Vantage API key (500 req/day free)
            news_api_key: NewsAPI key (100 req/day free)
        """
        self.alpha_vantage_key = alpha_vantage_key
        self.news_api_key = news_api_key
        self._cache: Dict[str, tuple] = {}  # (data, timestamp)
        self._cache_ttl = 3600  # 1 hour cache
    
    def get_stock_data(self, ticker: str, period: str = '1y', 
                       interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical price data using yfinance.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
        
        Returns:
            DataFrame with OHLCV data
        
        Example:
            >>> fetcher = DataFetcher()
            >>> df = fetcher.get_stock_data('AAPL', period='1y')
            >>> print(df.columns)  # ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        cache_key = f"stock_{ticker}_{period}_{interval}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            self._set_cache(cache_key, data)
            logger.info(f"Fetched {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_fundamentals(self, ticker: str) -> Dict:
        """
        Fetch fundamental data for a stock.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
        
        Returns:
            Dictionary with fundamental metrics
        
        Example:
            >>> fundamentals = fetcher.get_fundamentals('AAPL')
            >>> print(fundamentals['pe_ratio'])
        """
        cache_key = f"fundamentals_{ticker}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals = {
                'symbol': ticker,
                'company_name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'revenue': info.get('totalRevenue'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'roe': info.get('returnOnEquity'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'free_cash_flow': info.get('freeCashflow'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'fetched_at': datetime.now().isoformat()
            }
            
            self._set_cache(cache_key, fundamentals)
            logger.info(f"Fetched fundamentals for {ticker}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {'symbol': ticker, 'error': str(e)}
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get the current/latest price for a stock.
        
        Args:
            ticker: Stock symbol
        
        Returns:
            Current price or None if unavailable
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            return None
    
    def get_multiple_stocks(self, tickers: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks at once.
        
        Args:
            tickers: List of stock symbols
            period: Data period
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        result = {}
        for ticker in tickers:
            result[ticker] = self.get_stock_data(ticker, period=period)
        return result
    
    def _get_cached(self, key: str):
        """Get cached data if still valid."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                logger.debug(f"Cache hit for {key}")
                return data
        return None
    
    def _set_cache(self, key: str, data):
        """Store data in cache."""
        self._cache[key] = (data, datetime.now().timestamp())


# Convenience function for quick data fetching
def fetch_stock(ticker: str, period: str = '1y') -> pd.DataFrame:
    """Quick function to fetch stock data."""
    fetcher = DataFetcher()
    return fetcher.get_stock_data(ticker, period)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    fetcher = DataFetcher()
    
    # Test price data
    df = fetcher.get_stock_data('AAPL', period='1mo')
    print(f"AAPL data shape: {df.shape}")
    print(df.tail())
    
    # Test fundamentals
    fundamentals = fetcher.get_fundamentals('AAPL')
    print(f"\nAAPL PE Ratio: {fundamentals.get('pe_ratio')}")
    print(f"AAPL Market Cap: {fundamentals.get('market_cap')}")
