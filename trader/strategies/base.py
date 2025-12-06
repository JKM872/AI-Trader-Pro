"""
Base Strategy - Abstract base class for trading strategies.

All strategies should:
1. Return standardized signals: 'BUY', 'SELL', 'HOLD'
2. Include confidence scores (0.0 to 1.0)
3. Implement risk management (stop-loss, position sizing)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from datetime import datetime
from enum import Enum
import pandas as pd


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """
    Trading signal with metadata.
    
    Attributes:
        symbol: Stock symbol
        signal_type: BUY, SELL, or HOLD
        confidence: Confidence score (0.0 to 1.0)
        price: Current price at signal generation
        stop_loss: Suggested stop loss price
        take_profit: Suggested take profit price
        reasons: List of reasons for the signal
        timestamp: When the signal was generated
        metadata: Additional strategy-specific data
    """
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.price <= 0:
            raise ValueError("Price must be positive")
    
    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (not HOLD with high confidence)."""
        return self.signal_type != SignalType.HOLD and self.confidence >= 0.5
    
    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio if stop_loss and take_profit are set."""
        if self.stop_loss and self.take_profit and self.price:
            risk = abs(self.price - self.stop_loss)
            reward = abs(self.take_profit - self.price)
            if risk > 0:
                return reward / risk
        return None
    
    def to_dict(self) -> Dict:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'signal': self.signal_type.value,
            'confidence': self.confidence,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward': self.risk_reward_ratio,
            'reasons': self.reasons,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must implement:
    - generate_signal(): Generate trading signal for a symbol
    - calculate_position_size(): Determine position size based on risk
    """
    
    def __init__(self, name: str, 
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10,
                 min_confidence: float = 0.5):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            stop_loss_pct: Default stop loss percentage (0.05 = 5%)
            take_profit_pct: Default take profit percentage (0.10 = 10%)
            min_confidence: Minimum confidence for actionable signals
        """
        self.name = name
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_confidence = min_confidence
    
    @abstractmethod
    def generate_signal(self, symbol: str, data: pd.DataFrame, 
                        **kwargs) -> Signal:
        """
        Generate trading signal for a symbol.
        
        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV data
            **kwargs: Additional data (fundamentals, sentiment, etc.)
        
        Returns:
            Signal object with trading recommendation
        """
        pass
    
    def calculate_position_size(self, capital: float, price: float,
                                 risk_per_trade: float = 0.02,
                                 signal: Optional[Signal] = None) -> int:
        """
        Calculate position size based on risk management.
        
        Args:
            capital: Available capital
            price: Current stock price
            risk_per_trade: Max risk per trade as fraction of capital (0.02 = 2%)
            signal: Optional signal with stop_loss for precise calculation
        
        Returns:
            Number of shares to trade
        """
        max_risk_amount = capital * risk_per_trade
        
        if signal and signal.stop_loss:
            # Calculate based on actual stop loss distance
            risk_per_share = abs(price - signal.stop_loss)
            if risk_per_share > 0:
                shares = int(max_risk_amount / risk_per_share)
            else:
                shares = int(max_risk_amount / (price * self.stop_loss_pct))
        else:
            # Use default stop loss percentage
            risk_per_share = price * self.stop_loss_pct
            shares = int(max_risk_amount / risk_per_share)
        
        # Ensure we don't exceed available capital
        max_shares = int(capital / price)
        return min(shares, max_shares)
    
    def calculate_stop_loss(self, price: float, 
                            signal_type: SignalType) -> float:
        """Calculate stop loss price based on signal type."""
        if signal_type == SignalType.BUY:
            return price * (1 - self.stop_loss_pct)
        elif signal_type == SignalType.SELL:
            return price * (1 + self.stop_loss_pct)
        return price
    
    def calculate_take_profit(self, price: float, 
                              signal_type: SignalType) -> float:
        """Calculate take profit price based on signal type."""
        if signal_type == SignalType.BUY:
            return price * (1 + self.take_profit_pct)
        elif signal_type == SignalType.SELL:
            return price * (1 - self.take_profit_pct)
        return price
    
    def validate_data(self, data: pd.DataFrame, 
                      min_rows: int = 20) -> bool:
        """
        Validate input data.
        
        Args:
            data: DataFrame to validate
            min_rows: Minimum required rows
        
        Returns:
            True if data is valid
        """
        if data is None or data.empty:
            return False
        if len(data) < min_rows:
            return False
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)
