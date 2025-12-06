"""
Unified Trading Engine - Full Integration Module.

Integrates all trading components:
- Multi-AI Ensemble Analysis
- Market Regime Detection  
- ML Price Prediction
- Dynamic Risk Management
- Trade Journal
- Signal Generation
- Order Execution

This is the main entry point for the AI trading system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from trader.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading system modes."""
    PAPER = "paper"           # Paper trading (simulation)
    BACKTEST = "backtest"     # Historical backtesting
    LIVE = "live"             # Live trading (not implemented)
    ANALYSIS = "analysis"     # Analysis only (no trading)


class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class TradingEngineConfig:
    """Configuration for the trading engine."""
    
    # Mode
    mode: TradingMode = TradingMode.PAPER
    
    # Capital and risk
    initial_capital: float = 100000.0
    max_position_pct: float = 0.15
    risk_per_trade_pct: float = 0.02
    max_open_positions: int = 10
    
    # Signal thresholds
    min_confidence: float = 0.6
    strong_signal_threshold: float = 0.8
    consensus_threshold: float = 0.5
    
    # AI Ensemble settings
    use_ai_ensemble: bool = True
    ensemble_min_providers: int = 2
    
    # Market regime settings
    use_regime_detection: bool = True
    avoid_trading_in_crisis: bool = True
    
    # ML settings
    use_ml_prediction: bool = True
    ml_min_confidence: float = 0.55
    
    # Risk settings
    use_dynamic_risk: bool = True
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    trailing_stop_pct: Optional[float] = 0.03
    
    # Journal settings
    auto_journal_trades: bool = True
    
    # Execution
    allow_same_day_trades: bool = True
    max_daily_trades: int = 20


@dataclass
class AnalysisResult:
    """Comprehensive analysis result for a symbol."""
    
    symbol: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Final signal
    signal_type: SignalType = SignalType.HOLD
    signal_strength: SignalStrength = SignalStrength.NEUTRAL
    confidence: float = 0.0
    
    # Component signals
    technical_signal: Optional[Signal] = None
    ai_consensus: Optional[Dict] = None
    ml_prediction: Optional[Dict] = None
    regime_context: Optional[Dict] = None
    
    # Risk assessment
    risk_adjusted_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Analysis details
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Actionable
    is_actionable: bool = False
    suggested_action: str = "hold"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type.value.lower(),
            'signal_strength': self.signal_strength.value,
            'confidence': self.confidence,
            'risk_adjusted_size': self.risk_adjusted_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': self.risk_reward_ratio,
            'is_actionable': self.is_actionable,
            'suggested_action': self.suggested_action,
            'reasons': self.reasons,
            'warnings': self.warnings
        }


@dataclass
class TradeExecution:
    """Trade execution record."""
    
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Order details
    order_type: str = "market"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Execution result
    executed: bool = False
    execution_price: Optional[float] = None
    commission: float = 0.0
    
    # Tracking
    trade_id: Optional[str] = None
    analysis_id: Optional[str] = None
    
    @property
    def total_value(self) -> float:
        """Total trade value."""
        price = self.execution_price or self.price
        return self.quantity * price + self.commission


class TradingEngine:
    """
    Unified Trading Engine integrating all AI trading components.
    
    This is the main orchestrator that:
    1. Fetches and processes market data
    2. Runs AI ensemble analysis
    3. Detects market regime
    4. Generates ML predictions
    5. Combines signals with risk management
    6. Executes trades (paper/live)
    7. Logs to trade journal
    """
    
    def __init__(self, config: Optional[TradingEngineConfig] = None):
        """
        Initialize the trading engine.
        
        Args:
            config: Engine configuration
        """
        self.config = config or TradingEngineConfig()
        self.is_initialized = False
        
        # Core components (lazy loaded)
        self._data_fetcher = None
        self._technical_strategy = None
        self._ai_ensemble = None
        self._regime_detector = None
        self._ml_predictor = None
        self._dynamic_risk = None
        self._portfolio = None
        self._trade_journal = None
        self._executor = None
        
        # State
        self.current_capital = self.config.initial_capital
        self.daily_trades = 0
        self.last_trade_date: Optional[datetime] = None
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        
        # Performance tracking
        self.trades_executed: List[TradeExecution] = []
        self.total_pnl = 0.0
        
        logger.info(f"TradingEngine initialized in {self.config.mode.value} mode")
    
    def initialize(self) -> bool:
        """
        Initialize all trading components.
        
        Returns:
            True if successful
        """
        try:
            logger.info("Initializing trading engine components...")
            
            # Data Fetcher
            from trader.data.fetcher import DataFetcher
            self._data_fetcher = DataFetcher()
            
            # Technical Strategy
            from trader.strategies.technical import TechnicalStrategy
            self._technical_strategy = TechnicalStrategy(
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct
            )
            
            # AI Ensemble (optional)
            if self.config.use_ai_ensemble:
                try:
                    from trader.ensemble import AIEnsemble
                    self._ai_ensemble = AIEnsemble()
                except ImportError:
                    logger.warning("AI Ensemble not available")
            
            # Regime Detector (optional)
            if self.config.use_regime_detection:
                try:
                    from trader.market_analysis import MarketRegimeDetector
                    self._regime_detector = MarketRegimeDetector()
                except ImportError:
                    logger.warning("Market Regime Detector not available")
            
            # ML Predictor (optional)
            if self.config.use_ml_prediction:
                try:
                    from trader.ml import PricePredictor
                    self._ml_predictor = PricePredictor()
                except ImportError:
                    logger.warning("ML Predictor not available")
            
            # Dynamic Risk Manager
            if self.config.use_dynamic_risk:
                try:
                    from trader.risk import DynamicRiskManager, DynamicRiskConfig
                    risk_config = DynamicRiskConfig(
                        base_risk_per_trade=self.config.risk_per_trade_pct,
                        base_max_position=self.config.max_position_pct
                    )
                    self._dynamic_risk = DynamicRiskManager(
                        config=risk_config,
                        initial_capital=self.config.initial_capital
                    )
                except ImportError:
                    logger.warning("Dynamic Risk Manager not available")
            
            # Portfolio
            from trader.portfolio import Portfolio
            self._portfolio = Portfolio(initial_capital=self.config.initial_capital)
            
            # Trade Journal
            if self.config.auto_journal_trades:
                try:
                    from trader.journal import TradeJournal
                    self._trade_journal = TradeJournal()
                except ImportError:
                    logger.warning("Trade Journal not available")
            
            # Paper Trading Executor
            if self.config.mode == TradingMode.PAPER:
                from trader.execution import PaperTradingExecutor
                self._executor = PaperTradingExecutor()
            
            self.is_initialized = True
            logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            return False
    
    def analyze_symbol(
        self,
        symbol: str,
        data: Optional[pd.DataFrame] = None,
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis on a symbol.
        
        Args:
            symbol: Stock symbol
            data: Optional price data (fetched if not provided)
            news_context: Optional news for AI analysis
            
        Returns:
            AnalysisResult with all analysis components
        """
        if not self.is_initialized:
            self.initialize()
        
        result = AnalysisResult(symbol=symbol)
        
        try:
            # 1. Fetch data if needed
            if data is None:
                data = self._data_fetcher.get_stock_data(symbol, period='6mo')
            
            if data is None or len(data) < 50:
                result.warnings.append("Insufficient data for analysis")
                return result
            
            current_price = float(data['Close'].iloc[-1])
            
            # 2. Technical Analysis
            tech_signal = self._technical_strategy.generate_signal(symbol, data)
            result.technical_signal = tech_signal
            
            if tech_signal:
                result.reasons.extend(tech_signal.reasons or [])
            
            # 3. Market Regime Detection
            if self._regime_detector:
                try:
                    regime = self._regime_detector.detect_regime(data)
                    result.regime_context = {
                        'regime': regime.value if hasattr(regime, 'value') else str(regime),
                        'description': self._get_regime_description(regime)
                    }
                    
                    # Check if we should avoid trading
                    if self.config.avoid_trading_in_crisis:
                        if 'crisis' in str(regime).lower():
                            result.warnings.append("Crisis regime detected - trading restricted")
                except Exception as e:
                    logger.debug(f"Regime detection failed: {e}")
            
            # 4. AI Ensemble Analysis
            if self._ai_ensemble and news_context:
                try:
                    consensus = self._ai_ensemble.analyze_with_consensus(
                        query=f"Analyze {symbol}: {news_context}"
                    )
                    result.ai_consensus = consensus
                    
                    if consensus.get('consensus_sentiment'):
                        result.reasons.append(
                            f"AI Consensus: {consensus['consensus_sentiment']}"
                        )
                except Exception as e:
                    logger.debug(f"AI ensemble analysis failed: {e}")
            
            # 5. ML Prediction
            if self._ml_predictor:
                try:
                    from trader.ml import FeatureEngineer
                    engineer = FeatureEngineer()
                    features = engineer.create_features(data)
                    
                    if features is not None and len(features) > 0:
                        prediction = self._ml_predictor.predict(features.iloc[-1:])
                        if prediction is not None:
                            result.ml_prediction = {
                                'direction': 'bullish' if prediction[0] > 0 else 'bearish',
                                'confidence': min(abs(prediction[0]) * 10, 1.0)
                            }
                except Exception as e:
                    logger.debug(f"ML prediction failed: {e}")
            
            # 6. Combine Signals
            combined_signal, confidence = self._combine_signals(result)
            result.signal_type = combined_signal
            result.confidence = confidence
            result.signal_strength = self._get_signal_strength(combined_signal, confidence)
            
            # 7. Risk-Adjusted Sizing
            if self._dynamic_risk and combined_signal != SignalType.HOLD:
                volatility = float(data['Close'].pct_change().std() * np.sqrt(252))
                
                adj = self._dynamic_risk.calculate_risk_adjustment(
                    current_capital=self.current_capital,
                    current_volatility=volatility
                )
                
                if adj.is_restricted:
                    result.warnings.append(f"Trading restricted: {adj.restriction_reason}")
                    result.is_actionable = False
                else:
                    base_size = self.current_capital * self.config.max_position_pct
                    result.risk_adjusted_size = base_size * adj.overall_scale
            
            # 8. Calculate Stop Loss / Take Profit
            if combined_signal == SignalType.BUY:
                result.stop_loss = current_price * (1 - self.config.stop_loss_pct)
                result.take_profit = current_price * (1 + self.config.take_profit_pct)
            elif combined_signal == SignalType.SELL:
                result.stop_loss = current_price * (1 + self.config.stop_loss_pct)
                result.take_profit = current_price * (1 - self.config.take_profit_pct)
            
            # Risk/Reward ratio
            if result.stop_loss > 0 and combined_signal != SignalType.HOLD:
                risk = abs(current_price - result.stop_loss)
                reward = abs(result.take_profit - current_price)
                result.risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # 9. Determine if actionable
            result.is_actionable = (
                confidence >= self.config.min_confidence and
                combined_signal != SignalType.HOLD and
                len(result.warnings) == 0
            )
            
            if result.is_actionable:
                result.suggested_action = combined_signal.value.lower()
            
            # Cache result
            self.analysis_cache[symbol] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            result.warnings.append(f"Analysis error: {str(e)}")
            return result
    
    def _combine_signals(
        self,
        result: AnalysisResult
    ) -> Tuple[SignalType, float]:
        """
        Combine signals from multiple sources into final signal.
        
        Args:
            result: Analysis result with component signals
            
        Returns:
            Tuple of (signal_type, confidence)
        """
        votes = {'buy': 0, 'sell': 0, 'hold': 0}
        weights = {'buy': 0, 'sell': 0, 'hold': 0}
        
        # Technical signal (weight: 0.3)
        if result.technical_signal:
            tech = result.technical_signal
            signal_key = tech.signal_type.value.lower()
            votes[signal_key] += 1
            weights[signal_key] += 0.3 * tech.confidence
        
        # AI Consensus (weight: 0.3)
        if result.ai_consensus:
            sentiment = result.ai_consensus.get('consensus_sentiment', '').lower()
            if 'bullish' in sentiment or 'positive' in sentiment:
                votes['buy'] += 1
                weights['buy'] += 0.3 * result.ai_consensus.get('confidence', 0.5)
            elif 'bearish' in sentiment or 'negative' in sentiment:
                votes['sell'] += 1
                weights['sell'] += 0.3 * result.ai_consensus.get('confidence', 0.5)
            else:
                votes['hold'] += 1
                weights['hold'] += 0.3 * 0.5
        
        # ML Prediction (weight: 0.2)
        if result.ml_prediction:
            direction = result.ml_prediction.get('direction', '').lower()
            conf = result.ml_prediction.get('confidence', 0.5)
            if direction == 'bullish':
                votes['buy'] += 1
                weights['buy'] += 0.2 * conf
            elif direction == 'bearish':
                votes['sell'] += 1
                weights['sell'] += 0.2 * conf
        
        # Regime context (weight: 0.2)
        if result.regime_context:
            regime = result.regime_context.get('regime', '').lower()
            if 'bull' in regime or 'trend_up' in regime:
                weights['buy'] += 0.2 * 0.6
            elif 'bear' in regime or 'trend_down' in regime:
                weights['sell'] += 0.2 * 0.6
            elif 'crisis' in regime:
                weights['hold'] += 0.2 * 0.8
        
        # Determine final signal
        if weights['buy'] > weights['sell'] and weights['buy'] > weights['hold']:
            signal = SignalType.BUY
            confidence = min(weights['buy'] / 0.8, 1.0)  # Normalize
        elif weights['sell'] > weights['buy'] and weights['sell'] > weights['hold']:
            signal = SignalType.SELL
            confidence = min(weights['sell'] / 0.8, 1.0)
        else:
            signal = SignalType.HOLD
            confidence = 0.5
        
        return signal, confidence
    
    def _get_signal_strength(
        self,
        signal_type: SignalType,
        confidence: float
    ) -> SignalStrength:
        """Get signal strength classification."""
        if signal_type == SignalType.HOLD:
            return SignalStrength.NEUTRAL
        
        threshold = self.config.strong_signal_threshold
        
        if signal_type == SignalType.BUY:
            if confidence >= threshold:
                return SignalStrength.STRONG_BUY
            elif confidence >= 0.6:
                return SignalStrength.BUY
            else:
                return SignalStrength.WEAK_BUY
        else:  # SELL
            if confidence >= threshold:
                return SignalStrength.STRONG_SELL
            elif confidence >= 0.6:
                return SignalStrength.SELL
            else:
                return SignalStrength.WEAK_SELL
    
    def _get_regime_description(self, regime) -> str:
        """Get human-readable regime description."""
        regime_str = str(regime).lower()
        
        descriptions = {
            'bull': 'Bullish trending market',
            'bear': 'Bearish trending market',
            'sideways': 'Range-bound market',
            'volatile': 'High volatility conditions',
            'crisis': 'Crisis/extreme conditions',
            'recovery': 'Market recovering'
        }
        
        for key, desc in descriptions.items():
            if key in regime_str:
                return desc
        
        return 'Unknown market regime'
    
    def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> TradeExecution:
        """
        Execute a trade (paper or live based on mode).
        
        Args:
            symbol: Stock symbol
            action: BUY or SELL
            quantity: Number of shares
            price: Target price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            TradeExecution record
        """
        execution = TradeExecution(
            symbol=symbol,
            action=action.upper(),
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Check daily trade limit
        today = datetime.now(timezone.utc).date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        
        if self.daily_trades >= self.config.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return execution
        
        try:
            if self.config.mode == TradingMode.PAPER:
                # Paper trading execution
                if self._executor:
                    order = self._executor.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side=action.lower(),
                        order_type='market'
                    )
                    execution.executed = order.status in ['filled', 'FILLED']
                    execution.execution_price = order.filled_price or price
                else:
                    # Simulate execution
                    execution.executed = True
                    execution.execution_price = price
                
                # Update portfolio
                if execution.executed and self._portfolio:
                    if action.upper() == 'BUY':
                        self._portfolio.add_position(
                            symbol=symbol,
                            quantity=quantity,
                            price=execution.execution_price or price,
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )
                    else:
                        self._portfolio.close_position(
                            symbol=symbol,
                            price=execution.execution_price or price
                        )
            
            elif self.config.mode == TradingMode.ANALYSIS:
                # Analysis only - don't execute
                logger.info(f"Analysis mode - trade not executed: {action} {quantity} {symbol}")
                return execution
            
            # Log to journal
            if execution.executed and self._trade_journal:
                try:
                    self._trade_journal.log_trade(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        price=execution.execution_price or price,
                        reason=f"Trading Engine {self.config.mode.value}"
                    )
                except Exception as e:
                    logger.debug(f"Journal logging failed: {e}")
            
            # Track
            self.daily_trades += 1
            self.trades_executed.append(execution)
            
            logger.info(
                f"Trade executed: {action} {quantity} {symbol} @ "
                f"${execution.execution_price or price:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            execution.executed = False
        
        return execution
    
    def scan_watchlist(
        self,
        symbols: List[str],
        min_confidence: Optional[float] = None
    ) -> List[AnalysisResult]:
        """
        Scan a watchlist for trading opportunities.
        
        Args:
            symbols: List of symbols to scan
            min_confidence: Minimum confidence for inclusion
            
        Returns:
            List of analysis results sorted by confidence
        """
        min_conf = min_confidence or self.config.min_confidence
        results = []
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol(symbol)
                if result.confidence >= min_conf or result.is_actionable:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Scan failed for {symbol}: {e}")
        
        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status."""
        if not self._portfolio:
            return {}
        
        try:
            metrics = self._portfolio.get_metrics()
            return {
                'total_value': getattr(metrics, 'total_value', 0),
                'cash': getattr(metrics, 'cash', 0),
                'positions_count': len(self._portfolio.positions) if hasattr(self._portfolio, 'positions') else 0,
                'total_pnl': getattr(metrics, 'total_pnl', 0),
                'total_pnl_pct': getattr(metrics, 'total_pnl_pct', 0)
            }
        except Exception as e:
            logger.debug(f"Portfolio status failed: {e}")
            return {}
    
    def get_status_report(self) -> str:
        """Generate comprehensive status report."""
        report = []
        report.append("=" * 60)
        report.append("         TRADING ENGINE STATUS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Mode and State
        report.append(f"Mode: {self.config.mode.value.upper()}")
        report.append(f"Initialized: {'Yes' if self.is_initialized else 'No'}")
        report.append("")
        
        # Components
        report.append("📦 COMPONENTS:")
        report.append(f"   Data Fetcher: {'✓' if self._data_fetcher else '✗'}")
        report.append(f"   Technical Strategy: {'✓' if self._technical_strategy else '✗'}")
        report.append(f"   AI Ensemble: {'✓' if self._ai_ensemble else '✗'}")
        report.append(f"   Regime Detector: {'✓' if self._regime_detector else '✗'}")
        report.append(f"   ML Predictor: {'✓' if self._ml_predictor else '✗'}")
        report.append(f"   Dynamic Risk: {'✓' if self._dynamic_risk else '✗'}")
        report.append(f"   Portfolio: {'✓' if self._portfolio else '✗'}")
        report.append(f"   Trade Journal: {'✓' if self._trade_journal else '✗'}")
        report.append("")
        
        # Trading Stats
        report.append("📊 TRADING STATS:")
        report.append(f"   Daily Trades: {self.daily_trades}/{self.config.max_daily_trades}")
        report.append(f"   Total Trades: {len(self.trades_executed)}")
        report.append(f"   Cached Analyses: {len(self.analysis_cache)}")
        report.append("")
        
        # Portfolio
        status = self.get_portfolio_status()
        if status:
            report.append("💰 PORTFOLIO:")
            report.append(f"   Total Value: ${status.get('total_value', 0):,.2f}")
            report.append(f"   Cash: ${status.get('cash', 0):,.2f}")
            report.append(f"   Positions: {status.get('positions_count', 0)}")
            report.append(f"   P/L: ${status.get('total_pnl', 0):,.2f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def create_trading_engine(
    mode: str = "paper",
    initial_capital: float = 100000.0,
    **kwargs
) -> TradingEngine:
    """
    Factory function to create a configured trading engine.
    
    Args:
        mode: Trading mode (paper, backtest, analysis)
        initial_capital: Starting capital
        **kwargs: Additional config options
        
    Returns:
        Configured TradingEngine instance
    """
    mode_map = {
        'paper': TradingMode.PAPER,
        'backtest': TradingMode.BACKTEST,
        'live': TradingMode.LIVE,
        'analysis': TradingMode.ANALYSIS
    }
    
    trading_mode = mode_map.get(mode.lower(), TradingMode.PAPER)
    
    config = TradingEngineConfig(
        mode=trading_mode,
        initial_capital=initial_capital,
        **kwargs
    )
    
    engine = TradingEngine(config=config)
    engine.initialize()
    
    return engine
