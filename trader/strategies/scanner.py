"""
Signal Scanner - Real-time signal scanning and ranking system.

Features:
- Multi-strategy signal generation
- Signal ranking and scoring
- Confluence detection
- Market regime analysis
- Watchlist scanning
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

from .base import Signal, SignalType, TradingStrategy
from .technical import TechnicalStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .smart_money import SmartMoneyStrategy, MultiTimeframeStrategy
from ..analysis.indicators import TradingViewIndicators, TrendDirection
from ..data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class ScoredSignal:
    """Signal with additional scoring information."""
    signal: Signal
    strategy_name: str
    score: float  # 0-100 overall score
    components: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    
    @property
    def is_strong(self) -> bool:
        return self.score >= 70
    
    @property
    def is_actionable(self) -> bool:
        return self.score >= 50 and self.signal.signal_type != SignalType.HOLD


@dataclass
class MarketContext:
    """Market context for signal filtering."""
    regime: MarketRegime
    vix_level: float  # Simulated or actual VIX
    trend_strength: float
    volatility: float
    market_breadth: float  # % of stocks above key MA
    
    @property
    def is_favorable_for_longs(self) -> bool:
        return (self.regime in [MarketRegime.TRENDING_UP, MarketRegime.RANGING] 
                and self.trend_strength > 0.3)
    
    @property
    def is_favorable_for_shorts(self) -> bool:
        return (self.regime == MarketRegime.TRENDING_DOWN 
                and self.trend_strength > 0.3)


class SignalScorer:
    """Scores signals based on multiple factors."""
    
    WEIGHTS = {
        'confidence': 25,
        'risk_reward': 20,
        'trend_alignment': 20,
        'volume_confirmation': 15,
        'indicator_confluence': 20
    }
    
    @staticmethod
    def score_signal(signal: Signal, data: pd.DataFrame, 
                     market_context: Optional[MarketContext] = None) -> Tuple[float, Dict[str, float]]:
        """
        Score a signal based on multiple factors.
        
        Returns:
            Tuple of (total_score, component_scores)
        """
        components = {}
        
        # 1. Confidence Score (0-25)
        components['confidence'] = signal.confidence * SignalScorer.WEIGHTS['confidence']
        
        # 2. Risk/Reward Ratio (0-20)
        rr = signal.risk_reward_ratio or 1.0
        rr_score = min(rr / 3, 1.0)  # Max score at 3:1 RR
        components['risk_reward'] = rr_score * SignalScorer.WEIGHTS['risk_reward']
        
        # 3. Trend Alignment (0-20)
        trend_score = SignalScorer._calculate_trend_alignment(signal, data)
        components['trend_alignment'] = trend_score * SignalScorer.WEIGHTS['trend_alignment']
        
        # 4. Volume Confirmation (0-15)
        volume_score = SignalScorer._calculate_volume_confirmation(signal, data)
        components['volume_confirmation'] = volume_score * SignalScorer.WEIGHTS['volume_confirmation']
        
        # 5. Indicator Confluence (0-20)
        confluence_score = SignalScorer._calculate_indicator_confluence(signal, data)
        components['indicator_confluence'] = confluence_score * SignalScorer.WEIGHTS['indicator_confluence']
        
        # Market context adjustment
        if market_context:
            context_mult = SignalScorer._get_context_multiplier(signal, market_context)
            total = sum(components.values()) * context_mult
        else:
            total = sum(components.values())
        
        return min(total, 100), components
    
    @staticmethod
    def _calculate_trend_alignment(signal: Signal, data: pd.DataFrame) -> float:
        """Check if signal aligns with overall trend."""
        if len(data) < 50:
            return 0.5
        
        # Calculate trend
        sma20 = data['Close'].rolling(20).mean().iloc[-1]
        sma50 = data['Close'].rolling(50).mean().iloc[-1]
        price = data['Close'].iloc[-1]
        
        if signal.signal_type == SignalType.BUY:
            if price > sma20 > sma50:
                return 1.0
            elif price > sma20:
                return 0.7
            elif price > sma50:
                return 0.4
            else:
                return 0.2
        elif signal.signal_type == SignalType.SELL:
            if price < sma20 < sma50:
                return 1.0
            elif price < sma20:
                return 0.7
            elif price < sma50:
                return 0.4
            else:
                return 0.2
        
        return 0.5
    
    @staticmethod
    def _calculate_volume_confirmation(signal: Signal, data: pd.DataFrame) -> float:
        """Check if volume confirms the signal."""
        if 'Volume' not in data.columns or len(data) < 20:
            return 0.5
        
        # Compare recent volume to average
        recent_vol = data['Volume'].tail(5).mean()
        avg_vol = data['Volume'].tail(20).mean()
        
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
        
        if signal.signal_type != SignalType.HOLD:
            # Higher volume on signal is better
            if vol_ratio > 1.5:
                return 1.0
            elif vol_ratio > 1.2:
                return 0.8
            elif vol_ratio > 0.8:
                return 0.5
            else:
                return 0.3
        
        return 0.5
    
    @staticmethod
    def _calculate_indicator_confluence(signal: Signal, data: pd.DataFrame) -> float:
        """Calculate confluence of multiple indicators."""
        if len(data) < 50:
            return 0.5
        
        indicators = TradingViewIndicators()
        confluence_count = 0
        total_indicators = 5
        
        # 1. Supertrend
        st, direction = indicators.supertrend(data)
        st_bullish = direction.iloc[-1] == 1
        if (signal.signal_type == SignalType.BUY and st_bullish) or \
           (signal.signal_type == SignalType.SELL and not st_bullish):
            confluence_count += 1
        
        # 2. ADX trend strength
        adx, plus_di, minus_di = indicators.adx_dmi(data)
        if adx.iloc[-1] > 25:  # Strong trend
            if (signal.signal_type == SignalType.BUY and plus_di.iloc[-1] > minus_di.iloc[-1]) or \
               (signal.signal_type == SignalType.SELL and minus_di.iloc[-1] > plus_di.iloc[-1]):
                confluence_count += 1
        
        # 3. RSI
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        if (signal.signal_type == SignalType.BUY and current_rsi < 50) or \
           (signal.signal_type == SignalType.SELL and current_rsi > 50):
            confluence_count += 1
        
        # 4. MACD
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        if (signal.signal_type == SignalType.BUY and macd.iloc[-1] > signal_line.iloc[-1]) or \
           (signal.signal_type == SignalType.SELL and macd.iloc[-1] < signal_line.iloc[-1]):
            confluence_count += 1
        
        # 5. Price action
        last_candle = data.iloc[-1]
        bullish_candle = last_candle['Close'] > last_candle['Open']
        
        if (signal.signal_type == SignalType.BUY and bullish_candle) or \
           (signal.signal_type == SignalType.SELL and not bullish_candle):
            confluence_count += 1
        
        return confluence_count / total_indicators
    
    @staticmethod
    def _get_context_multiplier(signal: Signal, context: MarketContext) -> float:
        """Adjust score based on market context."""
        if signal.signal_type == SignalType.BUY:
            if context.is_favorable_for_longs:
                return 1.1
            elif context.regime == MarketRegime.TRENDING_DOWN:
                return 0.7
        elif signal.signal_type == SignalType.SELL:
            if context.is_favorable_for_shorts:
                return 1.1
            elif context.regime == MarketRegime.TRENDING_UP:
                return 0.7
        
        return 1.0


class MarketAnalyzer:
    """Analyzes overall market conditions."""
    
    def __init__(self, market_symbol: str = 'SPY'):
        self.market_symbol = market_symbol
        self.fetcher = DataFetcher()
    
    def get_market_context(self) -> MarketContext:
        """Get current market context."""
        try:
            data = self.fetcher.get_stock_data(self.market_symbol, period='3mo')
            
            if data.empty:
                return self._default_context()
            
            indicators = TradingViewIndicators()
            
            # Calculate regime
            regime = self._determine_regime(data)
            
            # ADX for trend strength
            adx, _, _ = indicators.adx_dmi(data)
            trend_strength = min(adx.iloc[-1] / 50, 1.0) if not pd.isna(adx.iloc[-1]) else 0.5
            
            # Volatility (ATR-based)
            atr = indicators.calculate_atr(data)
            volatility = atr.iloc[-1] / data['Close'].iloc[-1] if data['Close'].iloc[-1] > 0 else 0
            
            # Simulated VIX (based on volatility)
            vix_level = volatility * 100 * 10  # Rough approximation
            
            return MarketContext(
                regime=regime,
                vix_level=vix_level,
                trend_strength=trend_strength,
                volatility=volatility,
                market_breadth=0.5  # Would need actual breadth data
            )
        except Exception as e:
            logger.warning(f"Failed to get market context: {e}")
            return self._default_context()
    
    def _determine_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Determine current market regime."""
        sma50 = data['Close'].rolling(50).mean()
        sma200 = data['Close'].rolling(200).mean() if len(data) >= 200 else sma50
        
        price = data['Close'].iloc[-1]
        
        # Trend determination
        if len(sma200) >= 200:
            if price > sma50.iloc[-1] > sma200.iloc[-1]:
                return MarketRegime.TRENDING_UP
            elif price < sma50.iloc[-1] < sma200.iloc[-1]:
                return MarketRegime.TRENDING_DOWN
        else:
            if price > sma50.iloc[-1]:
                return MarketRegime.TRENDING_UP
            elif price < sma50.iloc[-1]:
                return MarketRegime.TRENDING_DOWN
        
        # Check volatility
        returns = data['Close'].pct_change().tail(20)
        volatility = returns.std() * np.sqrt(252)
        
        if volatility > 0.3:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.1:
            return MarketRegime.LOW_VOLATILITY
        
        return MarketRegime.RANGING
    
    def _default_context(self) -> MarketContext:
        """Return default market context."""
        return MarketContext(
            regime=MarketRegime.RANGING,
            vix_level=15.0,
            trend_strength=0.5,
            volatility=0.02,
            market_breadth=0.5
        )


class SignalScanner:
    """
    Scans watchlist for trading signals across multiple strategies.
    """
    
    STRATEGIES = {
        'Technical': TechnicalStrategy,
        'Momentum': MomentumStrategy,
        'Mean Reversion': MeanReversionStrategy,
        'Breakout': BreakoutStrategy,
        'Smart Money': SmartMoneyStrategy,
        'Multi-Timeframe': MultiTimeframeStrategy,
    }
    
    def __init__(self,
                 strategies: Optional[List[str]] = None,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10,
                 min_score: float = 50.0):
        """
        Initialize Signal Scanner.
        
        Args:
            strategies: List of strategy names to use (default: all)
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            min_score: Minimum score threshold for signals
        """
        self.fetcher = DataFetcher()
        self.market_analyzer = MarketAnalyzer()
        self.scorer = SignalScorer()
        self.min_score = min_score
        
        # Initialize strategies
        strategy_names = strategies or list(self.STRATEGIES.keys())
        self.strategies = {}
        
        for name in strategy_names:
            if name in self.STRATEGIES:
                self.strategies[name] = self.STRATEGIES[name](
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct
                )
    
    def scan_symbol(self, symbol: str, 
                    period: str = '6mo') -> List[ScoredSignal]:
        """
        Scan a single symbol for signals.
        
        Returns:
            List of scored signals from all strategies
        """
        try:
            data = self.fetcher.get_stock_data(symbol, period=period)
            
            if data.empty:
                logger.warning(f"No data for {symbol}")
                return []
            
            signals = []
            market_context = self.market_analyzer.get_market_context()
            
            for name, strategy in self.strategies.items():
                try:
                    signal = strategy.generate_signal(symbol, data)
                    score, components = self.scorer.score_signal(
                        signal, data, market_context
                    )
                    
                    scored = ScoredSignal(
                        signal=signal,
                        strategy_name=name,
                        score=score,
                        components=components
                    )
                    
                    signals.append(scored)
                    
                except Exception as e:
                    logger.error(f"Error generating signal for {symbol} with {name}: {e}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return []
    
    def scan_watchlist(self, symbols: List[str],
                       period: str = '6mo',
                       max_workers: int = 5) -> Dict[str, List[ScoredSignal]]:
        """
        Scan multiple symbols in parallel.
        
        Returns:
            Dict of symbol -> list of scored signals
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.scan_symbol, symbol, period): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signals = future.result()
                    results[symbol] = signals
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    results[symbol] = []
        
        return results
    
    def get_top_signals(self, symbols: List[str],
                        top_n: int = 10,
                        signal_type: Optional[SignalType] = None) -> List[ScoredSignal]:
        """
        Get top N signals across all symbols.
        
        Args:
            symbols: List of symbols to scan
            top_n: Number of top signals to return
            signal_type: Filter by signal type (BUY/SELL)
            
        Returns:
            List of top scored signals
        """
        all_signals = []
        results = self.scan_watchlist(symbols)
        
        for symbol, signals in results.items():
            for scored in signals:
                if scored.score >= self.min_score:
                    if signal_type is None or scored.signal.signal_type == signal_type:
                        all_signals.append(scored)
        
        # Sort by score
        all_signals.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, sig in enumerate(all_signals):
            sig.rank = i + 1
        
        return all_signals[:top_n]
    
    def get_consensus_signals(self, symbol: str,
                              min_agreement: float = 0.6) -> Optional[ScoredSignal]:
        """
        Get signal when majority of strategies agree.
        
        Args:
            symbol: Symbol to analyze
            min_agreement: Minimum percentage of strategies that must agree
            
        Returns:
            Consensus signal if agreement threshold met
        """
        signals = self.scan_symbol(symbol)
        
        if not signals:
            return None
        
        buy_count = sum(1 for s in signals if s.signal.signal_type == SignalType.BUY)
        sell_count = sum(1 for s in signals if s.signal.signal_type == SignalType.SELL)
        total = len(signals)
        
        if buy_count / total >= min_agreement:
            # Find best BUY signal
            buy_signals = [s for s in signals if s.signal.signal_type == SignalType.BUY]
            best = max(buy_signals, key=lambda x: x.score)
            best.signal.reasons.append(f"Consensus: {buy_count}/{total} strategies agree on BUY")
            return best
        
        elif sell_count / total >= min_agreement:
            # Find best SELL signal
            sell_signals = [s for s in signals if s.signal.signal_type == SignalType.SELL]
            best = max(sell_signals, key=lambda x: x.score)
            best.signal.reasons.append(f"Consensus: {sell_count}/{total} strategies agree on SELL")
            return best
        
        return None
    
    def generate_report(self, symbols: List[str]) -> pd.DataFrame:
        """
        Generate a signal report for watchlist.
        
        Returns:
            DataFrame with signal summary
        """
        results = self.scan_watchlist(symbols)
        
        report_data = []
        
        for symbol, signals in results.items():
            for scored in signals:
                report_data.append({
                    'Symbol': symbol,
                    'Strategy': scored.strategy_name,
                    'Signal': scored.signal.signal_type.value,
                    'Score': round(scored.score, 1),
                    'Confidence': f"{scored.signal.confidence:.1%}",
                    'Price': f"${scored.signal.price:.2f}",
                    'Stop Loss': f"${scored.signal.stop_loss:.2f}" if scored.signal.stop_loss else 'N/A',
                    'Take Profit': f"${scored.signal.take_profit:.2f}" if scored.signal.take_profit else 'N/A',
                    'R:R': f"{scored.signal.risk_reward_ratio:.2f}" if scored.signal.risk_reward_ratio else 'N/A',
                    'Top Reason': scored.signal.reasons[0] if scored.signal.reasons else 'N/A'
                })
        
        df = pd.DataFrame(report_data)
        
        if not df.empty:
            df = df.sort_values('Score', ascending=False)
        
        return df


def get_signal_summary(symbol: str, period: str = '6mo') -> Dict:
    """
    Get a comprehensive signal summary for a symbol.
    
    Useful for dashboard display.
    """
    scanner = SignalScanner()
    signals = scanner.scan_symbol(symbol, period)
    
    if not signals:
        return {
            'symbol': symbol,
            'overall_signal': 'HOLD',
            'consensus': 0,
            'best_score': 0,
            'signals': []
        }
    
    # Calculate consensus
    buy_count = sum(1 for s in signals if s.signal.signal_type == SignalType.BUY)
    sell_count = sum(1 for s in signals if s.signal.signal_type == SignalType.SELL)
    total = len(signals)
    
    if buy_count > sell_count:
        overall = 'BUY'
        consensus = buy_count / total
    elif sell_count > buy_count:
        overall = 'SELL'
        consensus = sell_count / total
    else:
        overall = 'HOLD'
        consensus = 0.5
    
    best_signal = max(signals, key=lambda x: x.score)
    
    return {
        'symbol': symbol,
        'overall_signal': overall,
        'consensus': consensus,
        'best_score': best_signal.score,
        'best_strategy': best_signal.strategy_name,
        'price': best_signal.signal.price,
        'stop_loss': best_signal.signal.stop_loss,
        'take_profit': best_signal.signal.take_profit,
        'signals': [
            {
                'strategy': s.strategy_name,
                'signal': s.signal.signal_type.value,
                'score': s.score,
                'confidence': s.signal.confidence
            }
            for s in sorted(signals, key=lambda x: x.score, reverse=True)
        ]
    }


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Test Signal Scanner
    print("Signal Scanner Test")
    print("=" * 60)
    
    scanner = SignalScanner(min_score=30)
    
    # Scan single symbol
    print("\nScanning AAPL...")
    signals = scanner.scan_symbol('AAPL')
    
    print(f"\nFound {len(signals)} signals:")
    for sig in sorted(signals, key=lambda x: x.score, reverse=True):
        print(f"  {sig.strategy_name}: {sig.signal.signal_type.value} "
              f"(Score: {sig.score:.1f}, Confidence: {sig.signal.confidence:.1%})")
    
    # Get summary
    print("\n" + "=" * 60)
    summary = get_signal_summary('AAPL')
    print(f"\nSignal Summary for {summary['symbol']}:")
    print(f"  Overall: {summary['overall_signal']}")
    print(f"  Consensus: {summary['consensus']:.0%}")
    print(f"  Best Score: {summary['best_score']:.1f} ({summary['best_strategy']})")
    
    # Test top signals across watchlist
    print("\n" + "=" * 60)
    watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    print(f"\nTop BUY signals across {watchlist}:")
    
    top_buys = scanner.get_top_signals(watchlist, top_n=5, signal_type=SignalType.BUY)
    for sig in top_buys:
        print(f"  #{sig.rank} {sig.signal.symbol} ({sig.strategy_name}): "
              f"Score {sig.score:.1f}")
