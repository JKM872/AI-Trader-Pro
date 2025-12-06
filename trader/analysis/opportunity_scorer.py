"""
Opportunity Scorer - Multi-factor stock scoring for higher win probability.

Combines:
- Fundamentals (30%): P/E, Revenue Growth, Profit Margin, Debt/Equity
- Technicals (30%): RSI, MACD, Trend, Volume
- Sentiment (20%): News, Insider activity
- Guru Holdings (10%): Famous investor positions
- Earnings (10%): Beat/miss history, upcoming catalysts

Returns Opportunity Score 0-100:
- 80-100: Strong Buy
- 60-79: Buy
- 40-59: Hold
- 20-39: Avoid
- 0-19: High Risk
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OpportunityScore:
    """Complete opportunity assessment for a stock."""
    symbol: str
    total_score: float
    recommendation: str
    color: str
    
    # Component scores (0-100 each)
    fundamentals_score: float = 0.0
    technicals_score: float = 0.0
    sentiment_score: float = 0.0
    guru_score: float = 0.0
    earnings_score: float = 0.0
    
    # Details
    fundamentals_details: Dict = field(default_factory=dict)
    technicals_details: Dict = field(default_factory=dict)
    sentiment_details: Dict = field(default_factory=dict)
    guru_details: Dict = field(default_factory=dict)
    earnings_details: Dict = field(default_factory=dict)
    
    # Risk metrics
    volatility: float = 0.0
    risk_level: str = "Medium"
    suggested_position_pct: float = 5.0


class OpportunityScorer:
    """
    Multi-factor opportunity scoring system.
    
    Weights:
    - Fundamentals: 30%
    - Technicals: 30%
    - Sentiment: 20%
    - Guru Holdings: 10%
    - Earnings: 10%
    """
    
    WEIGHTS = {
        'fundamentals': 0.30,
        'technicals': 0.30,
        'sentiment': 0.20,
        'guru': 0.10,
        'earnings': 0.10
    }
    
    def __init__(self):
        # Lazy imports to avoid circular dependencies
        pass
    
    def score_stock(self, symbol: str, data: Optional[pd.DataFrame] = None) -> OpportunityScore:
        """
        Calculate comprehensive opportunity score for a stock.
        
        Args:
            symbol: Stock ticker
            data: Optional price data (will fetch if not provided)
            
        Returns:
            OpportunityScore with full breakdown
        """
        from trader.data.fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        # Fetch data if not provided
        if data is None or data.empty:
            data = fetcher.get_stock_data(symbol, period='6mo')
        
        if data.empty:
            return self._empty_score(symbol)
        
        # Get fundamentals
        fundamentals = fetcher.get_fundamentals(symbol)
        
        # Calculate component scores
        fund_score, fund_details = self._score_fundamentals(fundamentals)
        tech_score, tech_details = self._score_technicals(data)
        sent_score, sent_details = self._score_sentiment(symbol)
        guru_score, guru_details = self._score_guru_holdings(symbol)
        earn_score, earn_details = self._score_earnings(symbol, fundamentals)
        
        # Weighted total
        total = (
            fund_score * self.WEIGHTS['fundamentals'] +
            tech_score * self.WEIGHTS['technicals'] +
            sent_score * self.WEIGHTS['sentiment'] +
            guru_score * self.WEIGHTS['guru'] +
            earn_score * self.WEIGHTS['earnings']
        )
        
        # Get recommendation
        recommendation, color = self._get_recommendation(total)
        
        # Risk assessment
        volatility = self._calculate_volatility(data)
        risk_level = self._assess_risk(volatility, fund_details)
        position_pct = self._suggest_position_size(total, volatility)
        
        return OpportunityScore(
            symbol=symbol,
            total_score=round(total, 1),
            recommendation=recommendation,
            color=color,
            fundamentals_score=round(fund_score, 1),
            technicals_score=round(tech_score, 1),
            sentiment_score=round(sent_score, 1),
            guru_score=round(guru_score, 1),
            earnings_score=round(earn_score, 1),
            fundamentals_details=fund_details,
            technicals_details=tech_details,
            sentiment_details=sent_details,
            guru_details=guru_details,
            earnings_details=earn_details,
            volatility=round(volatility, 2),
            risk_level=risk_level,
            suggested_position_pct=round(position_pct, 1)
        )
    
    def _score_fundamentals(self, fundamentals: Dict) -> Tuple[float, Dict]:
        """Score based on fundamental metrics."""
        score = 50.0  # Neutral start
        details = {}
        
        if not fundamentals:
            return score, {"error": "No fundamentals data"}
        
        # P/E Ratio (lower is better, but not negative)
        pe = fundamentals.get('pe_ratio')
        if pe is not None:
            details['pe_ratio'] = pe
            if pe < 0:
                score -= 10  # Negative earnings
            elif pe < 15:
                score += 15  # Cheap
            elif pe < 25:
                score += 5  # Fair
            elif pe > 50:
                score -= 15  # Expensive
        
        # Revenue Growth
        rev_growth = fundamentals.get('revenue_growth')
        if rev_growth is not None:
            details['revenue_growth'] = f"{rev_growth:.1%}" if isinstance(rev_growth, float) else rev_growth
            if isinstance(rev_growth, (int, float)):
                if rev_growth > 0.20:
                    score += 15
                elif rev_growth > 0.10:
                    score += 10
                elif rev_growth > 0:
                    score += 5
                else:
                    score -= 10
        
        # Profit Margin
        margin = fundamentals.get('profit_margin')
        if margin is not None:
            details['profit_margin'] = f"{margin:.1%}" if isinstance(margin, float) else margin
            if isinstance(margin, (int, float)):
                if margin > 0.20:
                    score += 15
                elif margin > 0.10:
                    score += 10
                elif margin > 0:
                    score += 5
                else:
                    score -= 10
        
        # Debt to Equity (lower is better)
        de = fundamentals.get('debt_to_equity')
        if de is not None:
            details['debt_to_equity'] = de
            if de < 0.5:
                score += 10
            elif de < 1.0:
                score += 5
            elif de > 2.0:
                score -= 15
        
        return max(0, min(100, score)), details
    
    def _score_technicals(self, data: pd.DataFrame) -> Tuple[float, Dict]:
        """Score based on technical indicators."""
        score = 50.0
        details = {}
        
        if data.empty or len(data) < 20:
            return score, {"error": "Insufficient data"}
        
        close = data['Close']
        
        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        details['rsi'] = round(current_rsi, 1)
        
        if current_rsi < 30:
            score += 20  # Oversold - bullish
        elif current_rsi < 40:
            score += 10
        elif current_rsi > 70:
            score -= 15  # Overbought - bearish
        elif current_rsi > 60:
            score -= 5
        
        # MACD
        exp12 = close.ewm(span=12).mean()
        exp26 = close.ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        macd_hist = macd - signal
        
        details['macd_signal'] = "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"
        details['macd_momentum'] = "Increasing" if macd_hist.iloc[-1] > macd_hist.iloc[-2] else "Decreasing"
        
        if macd.iloc[-1] > signal.iloc[-1]:
            score += 10
            if macd_hist.iloc[-1] > macd_hist.iloc[-2]:
                score += 5  # Momentum increasing
        else:
            score -= 10
        
        # Trend (50 SMA)
        sma50 = close.rolling(50).mean()
        if len(sma50) >= 50:
            current_price = close.iloc[-1]
            trend = "Uptrend" if current_price > sma50.iloc[-1] else "Downtrend"
            details['trend'] = trend
            details['price_vs_sma50'] = f"{((current_price / sma50.iloc[-1]) - 1) * 100:.1f}%"
            
            if current_price > sma50.iloc[-1]:
                score += 10
            else:
                score -= 10
        
        # Volume spike
        avg_volume = data['Volume'].rolling(20).mean()
        if avg_volume.iloc[-1] > 0:
            volume_ratio = data['Volume'].iloc[-1] / avg_volume.iloc[-1]
            details['volume_spike'] = f"{volume_ratio:.1f}x"
            if volume_ratio > 2:
                score += 5  # High interest
        
        return max(0, min(100, score)), details
    
    def _score_sentiment(self, symbol: str) -> Tuple[float, Dict]:
        """Score based on sentiment and insider activity."""
        score = 50.0
        details = {}
        
        try:
            import yfinance as yf
            stock = yf.Ticker(symbol)
            
            # Insider transactions
            insider_trans = stock.insider_transactions
            if insider_trans is not None and not insider_trans.empty:
                # Count buys vs sells in recent transactions
                buys = 0
                sells = 0
                for _, row in insider_trans.head(10).iterrows():
                    trans_type = str(row.get('Transaction', '')).lower()
                    if 'buy' in trans_type or 'purchase' in trans_type:
                        buys += 1
                    elif 'sell' in trans_type or 'sale' in trans_type:
                        sells += 1
                
                details['insider_buys'] = buys
                details['insider_sells'] = sells
                
                if buys > sells:
                    score += 15
                    details['insider_signal'] = "Bullish"
                elif sells > buys:
                    score -= 10
                    details['insider_signal'] = "Bearish"
                else:
                    details['insider_signal'] = "Neutral"
            
            # Analyst recommendations
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                recent = recommendations.tail(5)
                if 'To Grade' in recent.columns:
                    grades = recent['To Grade'].str.lower()
                    buy_count = grades.str.contains('buy|outperform|overweight', na=False).sum()
                    sell_count = grades.str.contains('sell|underperform|underweight', na=False).sum()
                    
                    details['recent_upgrades'] = int(buy_count)
                    details['recent_downgrades'] = int(sell_count)
                    
                    score += (buy_count - sell_count) * 5
                    
        except Exception as e:
            details['error'] = str(e)
        
        return max(0, min(100, score)), details
    
    def _score_guru_holdings(self, symbol: str) -> Tuple[float, Dict]:
        """Score based on famous investor holdings."""
        score = 50.0
        details = {}
        
        try:
            from trader.data.investor_tracker import PortfolioTracker
            
            tracker = PortfolioTracker()
            try:
                ownership = tracker.get_stock_institutional_owners(symbol)
                
                famous = ownership.get('famous_investors', [])
                details['guru_count'] = len(famous)
                
                if famous:
                    details['gurus'] = [f['investor'] for f in famous[:5]]
                    score += min(30, len(famous) * 15)  # Up to +30 for guru holdings
                
                # Institutional ownership %
                major = ownership.get('major_holders', {})
                inst_pct = major.get('institutions_pct', '')
                if inst_pct:
                    details['institutional_pct'] = inst_pct
                    try:
                        pct_val = float(str(inst_pct).replace('%', ''))
                        if pct_val > 70:
                            score += 10
                        elif pct_val > 50:
                            score += 5
                    except:
                        pass
                        
            finally:
                tracker.close()
                
        except Exception as e:
            details['error'] = str(e)
        
        return max(0, min(100, score)), details
    
    def _score_earnings(self, symbol: str, fundamentals: Dict) -> Tuple[float, Dict]:
        """Score based on earnings history and upcoming catalysts."""
        score = 50.0
        details = {}
        
        try:
            import yfinance as yf
            stock = yf.Ticker(symbol)
            
            # Earnings calendar
            calendar = stock.calendar
            if calendar is not None and isinstance(calendar, dict):
                earnings_date = calendar.get('Earnings Date')
                if earnings_date:
                    details['next_earnings'] = str(earnings_date)
                    # Upcoming earnings can be a catalyst
                    score += 5
            
            # Earnings history (quarterly)
            earnings = stock.quarterly_earnings
            if earnings is not None and not earnings.empty:
                # Check for consistent growth
                if len(earnings) >= 4:
                    recent = earnings.head(4)
                    if 'Earnings' in recent.columns:
                        eps_values = recent['Earnings'].values
                        if all(e > 0 for e in eps_values if pd.notna(e)):
                            score += 15
                            details['eps_trend'] = "Positive"
                        elif eps_values[0] > eps_values[-1]:
                            score += 10
                            details['eps_trend'] = "Improving"
            
            # Earnings estimates
            earnings_est = stock.earnings_estimate
            if earnings_est is not None and not earnings_est.empty:
                details['has_estimates'] = True
                
        except Exception as e:
            details['error'] = str(e)
        
        return max(0, min(100, score)), details
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate annualized volatility."""
        if data.empty or len(data) < 20:
            return 0.25  # Default 25%
        
        returns = data['Close'].pct_change().dropna()
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        return annual_vol
    
    def _assess_risk(self, volatility: float, fund_details: Dict) -> str:
        """Assess overall risk level."""
        risk_score = 0
        
        # Volatility contribution
        if volatility > 0.50:
            risk_score += 3
        elif volatility > 0.30:
            risk_score += 2
        elif volatility > 0.20:
            risk_score += 1
        
        # Debt contribution
        de = fund_details.get('debt_to_equity', 0)
        if de and de > 2:
            risk_score += 2
        elif de and de > 1:
            risk_score += 1
        
        # PE contribution
        pe = fund_details.get('pe_ratio', 0)
        if pe and (pe < 0 or pe > 50):
            risk_score += 1
        
        if risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Medium"
        else:
            return "Low"
    
    def _suggest_position_size(self, score: float, volatility: float) -> float:
        """Suggest position size as % of portfolio."""
        # Base: higher score = larger position
        base = 2.0 + (score / 100) * 8  # 2-10%
        
        # Adjust for volatility
        if volatility > 0.40:
            base *= 0.5
        elif volatility > 0.30:
            base *= 0.75
        
        return min(10, max(1, base))
    
    def _get_recommendation(self, score: float) -> Tuple[str, str]:
        """Get recommendation text and color."""
        if score >= 80:
            return "Strong Buy", "#089981"
        elif score >= 60:
            return "Buy", "#4caf50"
        elif score >= 40:
            return "Hold", "#787b86"
        elif score >= 20:
            return "Avoid", "#ff9800"
        else:
            return "High Risk", "#f23645"
    
    def _empty_score(self, symbol: str) -> OpportunityScore:
        """Return empty score for failed analysis."""
        return OpportunityScore(
            symbol=symbol,
            total_score=0,
            recommendation="No Data",
            color="#787b86"
        )
    
    def scan_opportunities(self, symbols: List[str], min_score: float = 60) -> List[OpportunityScore]:
        """
        Scan multiple symbols and return top opportunities.
        
        Args:
            symbols: List of tickers to scan
            min_score: Minimum score to include
            
        Returns:
            List of OpportunityScore sorted by score descending
        """
        results = []
        
        for symbol in symbols:
            try:
                score = self.score_stock(symbol)
                if score.total_score >= min_score:
                    results.append(score)
            except Exception as e:
                logger.warning(f"Failed to score {symbol}: {e}")
        
        # Sort by score descending
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        return results


def get_top_opportunities(symbols: List[str], top_n: int = 10) -> List[OpportunityScore]:
    """Quick function to get top opportunities."""
    scorer = OpportunityScorer()
    all_scores = []
    
    for symbol in symbols:
        try:
            score = scorer.score_stock(symbol)
            all_scores.append(score)
        except Exception as e:
            logger.warning(f"Failed to score {symbol}: {e}")
    
    all_scores.sort(key=lambda x: x.total_score, reverse=True)
    return all_scores[:top_n]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    scorer = OpportunityScorer()
    
    # Test with some stocks
    test_symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
    
    for symbol in test_symbols:
        print(f"\n=== {symbol} ===")
        score = scorer.score_stock(symbol)
        print(f"Total Score: {score.total_score}/100 - {score.recommendation}")
        print(f"  Fundamentals: {score.fundamentals_score}")
        print(f"  Technicals: {score.technicals_score}")
        print(f"  Sentiment: {score.sentiment_score}")
        print(f"  Guru Holdings: {score.guru_score}")
        print(f"  Earnings: {score.earnings_score}")
        print(f"  Risk: {score.risk_level}, Volatility: {score.volatility:.1%}")
        print(f"  Suggested Position: {score.suggested_position_pct}%")
