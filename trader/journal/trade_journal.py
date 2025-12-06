"""
Trade Journal - Comprehensive trade logging and analysis.

Records all trades with rich metadata for performance analysis.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd


class TradeStatus(Enum):
    """Trade status."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class TradeType(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeEntry:
    """Complete trade record."""
    # Identification
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    
    # Trade details
    trade_type: TradeType = TradeType.LONG
    status: TradeStatus = TradeStatus.PENDING
    
    # Entry
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_quantity: int = 0
    
    # Exit
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_quantity: Optional[int] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Signal info
    strategy_name: str = ""
    signal_confidence: float = 0.0
    signal_reasons: List[str] = field(default_factory=list)
    
    # Market context
    market_regime: str = ""
    volatility_level: str = ""
    trend_direction: str = ""
    
    # AI/ML predictions
    ai_predictions: Dict[str, Any] = field(default_factory=dict)
    ensemble_consensus: Optional[str] = None
    ml_confidence: Optional[float] = None
    
    # Performance
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    max_favorable_excursion: Optional[float] = None  # MFE
    max_adverse_excursion: Optional[float] = None    # MAE
    
    # Notes and tags
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def calculate_pnl(self) -> Optional[float]:
        """Calculate P/L if trade is closed."""
        if self.exit_price is None or self.entry_price == 0:
            return None
            
        quantity = self.exit_quantity or self.entry_quantity
        
        if self.trade_type == TradeType.LONG:
            return (self.exit_price - self.entry_price) * quantity
        else:  # SHORT
            return (self.entry_price - self.exit_price) * quantity
    
    def calculate_pnl_pct(self) -> Optional[float]:
        """Calculate P/L percentage."""
        if self.exit_price is None or self.entry_price == 0:
            return None
            
        if self.trade_type == TradeType.LONG:
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * 100
    
    def duration(self) -> Optional[timedelta]:
        """Get trade duration."""
        if self.entry_time and self.exit_time:
            return self.exit_time - self.entry_time
        return None
    
    def risk_reward_actual(self) -> Optional[float]:
        """Calculate actual risk/reward ratio."""
        if self.max_adverse_excursion and self.max_adverse_excursion != 0:
            if self.realized_pnl:
                return abs(self.realized_pnl / self.max_adverse_excursion)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert enums and datetime
        data['trade_type'] = self.trade_type.value
        data['status'] = self.status.value
        data['entry_time'] = self.entry_time.isoformat() if self.entry_time else None
        data['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeEntry':
        """Create from dictionary."""
        # Convert strings back to enums
        if isinstance(data.get('trade_type'), str):
            data['trade_type'] = TradeType(data['trade_type'])
        if isinstance(data.get('status'), str):
            data['status'] = TradeStatus(data['status'])
        
        # Convert datetime strings
        for field_name in ['entry_time', 'exit_time', 'created_at', 'updated_at']:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)


@dataclass
class JournalStats:
    """Aggregated journal statistics."""
    total_trades: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    average_pnl: float = 0.0
    average_winner: float = 0.0
    average_loser: float = 0.0
    
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    average_duration: Optional[timedelta] = None
    average_mfe: float = 0.0
    average_mae: float = 0.0
    
    trades_by_strategy: Dict[str, int] = field(default_factory=dict)
    pnl_by_strategy: Dict[str, float] = field(default_factory=dict)
    
    trades_by_symbol: Dict[str, int] = field(default_factory=dict)
    pnl_by_symbol: Dict[str, float] = field(default_factory=dict)


class TradeJournal:
    """
    Comprehensive trade journal for logging and analyzing trades.
    
    Features:
    - Detailed trade records with rich metadata
    - Automatic P/L calculation
    - MFE/MAE tracking
    - Strategy and symbol analysis
    - JSON persistence
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize trade journal."""
        self.storage_path = Path(storage_path) if storage_path else Path("trades.json")
        self.trades: Dict[str, TradeEntry] = {}
        self._load_trades()
    
    def _load_trades(self) -> None:
        """Load trades from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for trade_data in data.get('trades', []):
                        trade = TradeEntry.from_dict(trade_data)
                        self.trades[trade.trade_id] = trade
            except (json.JSONDecodeError, KeyError):
                self.trades = {}
    
    def _save_trades(self) -> None:
        """Save trades to storage."""
        data = {
            'trades': [trade.to_dict() for trade in self.trades.values()],
            'updated_at': datetime.now().isoformat()
        }
        
        # Create directory if needed
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        entry_price: float,
        quantity: int,
        strategy_name: str = "",
        signal_confidence: float = 0.0,
        signal_reasons: Optional[List[str]] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        market_regime: str = "",
        notes: str = "",
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> TradeEntry:
        """Create and log a new trade."""
        trade = TradeEntry(
            symbol=symbol,
            trade_type=trade_type,
            status=TradeStatus.OPEN,
            entry_price=entry_price,
            entry_time=datetime.now(),
            entry_quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
            signal_confidence=signal_confidence,
            signal_reasons=signal_reasons or [],
            market_regime=market_regime,
            notes=notes,
            tags=tags or [],
            **kwargs
        )
        
        self.trades[trade.trade_id] = trade
        self._save_trades()
        
        return trade
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_quantity: Optional[int] = None,
        notes: str = ""
    ) -> Optional[TradeEntry]:
        """Close an existing trade."""
        trade = self.trades.get(trade_id)
        
        if not trade:
            return None
        
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.exit_quantity = exit_quantity or trade.entry_quantity
        trade.status = TradeStatus.CLOSED
        trade.updated_at = datetime.now()
        
        # Calculate P/L
        trade.realized_pnl = trade.calculate_pnl()
        trade.realized_pnl_pct = trade.calculate_pnl_pct()
        
        if notes:
            trade.notes = f"{trade.notes}\n{notes}" if trade.notes else notes
        
        self._save_trades()
        
        return trade
    
    def update_trade(
        self,
        trade_id: str,
        **updates
    ) -> Optional[TradeEntry]:
        """Update trade fields."""
        trade = self.trades.get(trade_id)
        
        if not trade:
            return None
        
        for key, value in updates.items():
            if hasattr(trade, key):
                setattr(trade, key, value)
        
        trade.updated_at = datetime.now()
        self._save_trades()
        
        return trade
    
    def update_excursions(
        self,
        trade_id: str,
        current_price: float
    ) -> Optional[TradeEntry]:
        """Update MFE/MAE based on current price."""
        trade = self.trades.get(trade_id)
        
        if not trade or trade.status != TradeStatus.OPEN:
            return None
        
        if trade.trade_type == TradeType.LONG:
            favorable = (current_price - trade.entry_price) * trade.entry_quantity
            adverse = (trade.entry_price - current_price) * trade.entry_quantity
        else:
            favorable = (trade.entry_price - current_price) * trade.entry_quantity
            adverse = (current_price - trade.entry_price) * trade.entry_quantity
        
        # Update MFE if new high
        if favorable > 0:
            if trade.max_favorable_excursion is None or favorable > trade.max_favorable_excursion:
                trade.max_favorable_excursion = favorable
        
        # Update MAE if new low
        if adverse > 0:
            if trade.max_adverse_excursion is None or adverse > trade.max_adverse_excursion:
                trade.max_adverse_excursion = adverse
        
        trade.updated_at = datetime.now()
        self._save_trades()
        
        return trade
    
    def cancel_trade(self, trade_id: str) -> Optional[TradeEntry]:
        """Cancel a pending trade."""
        trade = self.trades.get(trade_id)
        
        if trade and trade.status == TradeStatus.PENDING:
            trade.status = TradeStatus.CANCELLED
            trade.updated_at = datetime.now()
            self._save_trades()
        
        return trade
    
    def get_trade(self, trade_id: str) -> Optional[TradeEntry]:
        """Get a specific trade."""
        return self.trades.get(trade_id)
    
    def get_all_trades(self) -> List[TradeEntry]:
        """Get all trades."""
        return list(self.trades.values())
    
    def get_open_trades(self) -> List[TradeEntry]:
        """Get all open trades."""
        return [t for t in self.trades.values() if t.status == TradeStatus.OPEN]
    
    def get_closed_trades(self) -> List[TradeEntry]:
        """Get all closed trades."""
        return [t for t in self.trades.values() if t.status == TradeStatus.CLOSED]
    
    def get_trades_by_symbol(self, symbol: str) -> List[TradeEntry]:
        """Get trades for a specific symbol."""
        return [t for t in self.trades.values() if t.symbol == symbol]
    
    def get_trades_by_strategy(self, strategy: str) -> List[TradeEntry]:
        """Get trades for a specific strategy."""
        return [t for t in self.trades.values() if t.strategy_name == strategy]
    
    def get_trades_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[TradeEntry]:
        """Get trades within a date range."""
        return [
            t for t in self.trades.values()
            if t.entry_time and start_date <= t.entry_time <= end_date
        ]
    
    def get_trades_by_tag(self, tag: str) -> List[TradeEntry]:
        """Get trades with a specific tag."""
        return [t for t in self.trades.values() if tag in t.tags]
    
    def search_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        trade_type: Optional[TradeType] = None,
        status: Optional[TradeStatus] = None,
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
        min_confidence: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> List[TradeEntry]:
        """Search trades with multiple filters."""
        results = list(self.trades.values())
        
        if symbol:
            results = [t for t in results if t.symbol == symbol]
        if strategy:
            results = [t for t in results if t.strategy_name == strategy]
        if trade_type:
            results = [t for t in results if t.trade_type == trade_type]
        if status:
            results = [t for t in results if t.status == status]
        if min_pnl is not None:
            results = [t for t in results if t.realized_pnl and t.realized_pnl >= min_pnl]
        if max_pnl is not None:
            results = [t for t in results if t.realized_pnl and t.realized_pnl <= max_pnl]
        if min_confidence is not None:
            results = [t for t in results if t.signal_confidence >= min_confidence]
        if tags:
            results = [t for t in results if any(tag in t.tags for tag in tags)]
        
        return results
    
    def get_statistics(self) -> JournalStats:
        """Calculate comprehensive statistics."""
        stats = JournalStats()
        
        closed_trades = self.get_closed_trades()
        open_trades = self.get_open_trades()
        
        stats.total_trades = len(self.trades)
        stats.open_trades = len(open_trades)
        stats.closed_trades = len(closed_trades)
        
        if not closed_trades:
            return stats
        
        # Win/Loss analysis
        winners = [t for t in closed_trades if t.realized_pnl and t.realized_pnl > 0]
        losers = [t for t in closed_trades if t.realized_pnl and t.realized_pnl < 0]
        breakeven = [t for t in closed_trades if t.realized_pnl == 0]
        
        stats.winning_trades = len(winners)
        stats.losing_trades = len(losers)
        stats.breakeven_trades = len(breakeven)
        
        if closed_trades:
            stats.win_rate = len(winners) / len(closed_trades) * 100
        
        # P/L calculations
        pnls = [t.realized_pnl for t in closed_trades if t.realized_pnl is not None]
        
        if pnls:
            stats.total_pnl = sum(pnls)
            stats.average_pnl = stats.total_pnl / len(pnls)
            stats.largest_win = max(pnls) if pnls else 0
            stats.largest_loss = min(pnls) if pnls else 0
        
        if winners:
            stats.average_winner = sum(t.realized_pnl for t in winners if t.realized_pnl) / len(winners)
        if losers:
            stats.average_loser = sum(t.realized_pnl for t in losers if t.realized_pnl) / len(losers)
        
        # Profit factor
        gross_profit = sum(t.realized_pnl for t in winners if t.realized_pnl) if winners else 0
        gross_loss = abs(sum(t.realized_pnl for t in losers if t.realized_pnl)) if losers else 0
        
        if gross_loss > 0:
            stats.profit_factor = gross_profit / gross_loss
        
        # Expectancy
        if stats.win_rate > 0 and stats.average_winner and stats.average_loser:
            win_prob = stats.win_rate / 100
            loss_prob = 1 - win_prob
            stats.expectancy = (win_prob * stats.average_winner) + (loss_prob * stats.average_loser)
        
        # Duration analysis
        durations = [t.duration() for t in closed_trades if t.duration()]
        if durations:
            total_seconds = sum(d.total_seconds() for d in durations)
            stats.average_duration = timedelta(seconds=total_seconds / len(durations))
        
        # MFE/MAE analysis
        mfes = [t.max_favorable_excursion for t in closed_trades if t.max_favorable_excursion]
        maes = [t.max_adverse_excursion for t in closed_trades if t.max_adverse_excursion]
        
        if mfes:
            stats.average_mfe = sum(mfes) / len(mfes)
        if maes:
            stats.average_mae = sum(maes) / len(maes)
        
        # By strategy
        for trade in closed_trades:
            strategy = trade.strategy_name or "Unknown"
            stats.trades_by_strategy[strategy] = stats.trades_by_strategy.get(strategy, 0) + 1
            if trade.realized_pnl:
                stats.pnl_by_strategy[strategy] = stats.pnl_by_strategy.get(strategy, 0) + trade.realized_pnl
        
        # By symbol
        for trade in closed_trades:
            symbol = trade.symbol
            stats.trades_by_symbol[symbol] = stats.trades_by_symbol.get(symbol, 0) + 1
            if trade.realized_pnl:
                stats.pnl_by_symbol[symbol] = stats.pnl_by_symbol.get(symbol, 0) + trade.realized_pnl
        
        return stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades.values():
            row = {
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'trade_type': trade.trade_type.value,
                'status': trade.status.value,
                'entry_price': trade.entry_price,
                'entry_time': trade.entry_time,
                'entry_quantity': trade.entry_quantity,
                'exit_price': trade.exit_price,
                'exit_time': trade.exit_time,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'strategy_name': trade.strategy_name,
                'signal_confidence': trade.signal_confidence,
                'market_regime': trade.market_regime,
                'realized_pnl': trade.realized_pnl,
                'realized_pnl_pct': trade.realized_pnl_pct,
                'mfe': trade.max_favorable_excursion,
                'mae': trade.max_adverse_excursion,
                'tags': ','.join(trade.tags),
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_to_csv(self, filepath: str) -> None:
        """Export trades to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
    
    def clear_all_trades(self) -> None:
        """Clear all trades (use with caution)."""
        self.trades = {}
        self._save_trades()
