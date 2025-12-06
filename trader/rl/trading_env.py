"""
Trading Environment - Gymnasium-compatible environment for RL trading.

Features:
- Configurable action space (discrete/continuous)
- Realistic trading simulation with fees
- Multiple reward functions (PnL, Sharpe, risk-adjusted)
- Position management
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Check for gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("Gymnasium not available. RL environment will use fallback.")


class RewardType(Enum):
    """Types of reward functions."""
    PNL = "pnl"                    # Simple profit/loss
    SHARPE = "sharpe"              # Sharpe ratio based
    SORTINO = "sortino"            # Sortino ratio based
    RISK_ADJUSTED = "risk_adjusted"  # Risk-adjusted returns


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment."""
    
    # Initial conditions
    initial_cash: float = 100000.0
    max_position_size: float = 0.25  # Max 25% of portfolio per position
    
    # Trading costs
    commission_rate: float = 0.001  # 0.1%
    slippage: float = 0.0005       # 0.05%
    
    # Risk management
    max_drawdown: float = 0.20     # 20% max drawdown
    stop_loss_pct: float = 0.05    # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    
    # Environment settings
    window_size: int = 60          # Observation window
    reward_type: RewardType = RewardType.RISK_ADJUSTED
    normalize_obs: bool = True
    
    # Episode settings
    max_steps: int = 252           # ~1 year of trading days


@dataclass
class Position:
    """Track a single position."""
    symbol: str
    shares: float
    entry_price: float
    entry_step: int
    
    @property
    def value(self) -> float:
        return self.shares * self.entry_price


if GYM_AVAILABLE:
    
    class TradingEnvironment(gym.Env):
        """
        Gymnasium-compatible trading environment.
        
        Observation Space:
        - Price features (OHLCV normalized)
        - Technical indicators
        - Portfolio state (cash, positions, PnL)
        - Market regime
        
        Action Space:
        - 0: Hold
        - 1: Buy
        - 2: Sell
        - Continuous: Position sizing (-1 to 1)
        """
        
        metadata = {'render_modes': ['human', 'ansi']}
        
        def __init__(
            self,
            prices: np.ndarray,
            features: Optional[np.ndarray] = None,
            config: Optional[TradingEnvConfig] = None,
            render_mode: Optional[str] = None
        ):
            """
            Initialize trading environment.
            
            Args:
                prices: Price array [n_steps]
                features: Optional feature array [n_steps, n_features]
                config: Environment configuration
                render_mode: Rendering mode
            """
            super().__init__()
            
            self.config = config or TradingEnvConfig()
            self.render_mode = render_mode
            
            # Store data
            self.prices = prices
            self.n_steps = len(prices)
            
            # Features
            if features is not None:
                self.features = features
                self.n_features = features.shape[1]
            else:
                self.features = self._compute_default_features(prices)
                self.n_features = self.features.shape[1]
            
            # Normalize features if needed
            if self.config.normalize_obs:
                self.features = self._normalize_features(self.features)
            
            # Define spaces
            # Observation: features + portfolio state
            obs_dim = self.n_features + 5  # +5 for portfolio state
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.config.window_size, obs_dim),
                dtype=np.float32
            )
            
            # Action: discrete (0=hold, 1=buy, 2=sell) or continuous
            self.action_space = spaces.Discrete(3)
            
            # State variables
            self.current_step = 0
            self.cash = self.config.initial_cash
            self.shares = 0.0
            self.entry_price = 0.0
            self.portfolio_value = self.config.initial_cash
            self.peak_value = self.config.initial_cash
            
            # History
            self.portfolio_history: List[float] = []
            self.action_history: List[int] = []
            self.return_history: List[float] = []
            
            logger.info(
                f"TradingEnvironment initialized: {self.n_steps} steps, "
                f"{self.n_features} features"
            )
        
        def _compute_default_features(self, prices: np.ndarray) -> np.ndarray:
            """Compute default features from prices."""
            n = len(prices)
            features = []
            
            # Returns
            returns = np.zeros(n)
            returns[1:] = np.diff(prices) / prices[:-1]
            features.append(returns)
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                ma = np.convolve(prices, np.ones(window)/window, mode='same')
                features.append((prices - ma) / (ma + 1e-8))
            
            # Volatility
            vol = np.zeros(n)
            for i in range(20, n):
                vol[i] = np.std(returns[i-20:i])
            features.append(vol)
            
            # RSI approximation
            rsi = np.zeros(n)
            for i in range(14, n):
                gains = np.maximum(0, np.diff(prices[i-14:i]))
                losses = np.maximum(0, -np.diff(prices[i-14:i]))
                avg_gain = np.mean(gains) + 1e-8
                avg_loss = np.mean(losses) + 1e-8
                rsi[i] = 100 - (100 / (1 + avg_gain/avg_loss))
            features.append((rsi - 50) / 50)  # Normalize to [-1, 1]
            
            return np.column_stack(features)
        
        def _normalize_features(self, features: np.ndarray) -> np.ndarray:
            """Z-score normalize features."""
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0) + 1e-8
            return (features - mean) / std
        
        def _get_observation(self) -> np.ndarray:
            """Get current observation."""
            # Get feature window
            start_idx = max(0, self.current_step - self.config.window_size + 1)
            end_idx = self.current_step + 1
            
            feature_window = self.features[start_idx:end_idx]
            
            # Pad if needed
            if len(feature_window) < self.config.window_size:
                padding = np.zeros((self.config.window_size - len(feature_window), self.n_features))
                feature_window = np.vstack([padding, feature_window])
            
            # Portfolio state
            current_price = self.prices[self.current_step]
            position_value = self.shares * current_price
            total_value = self.cash + position_value
            
            portfolio_state = np.array([
                self.cash / self.config.initial_cash,           # Normalized cash
                position_value / self.config.initial_cash,       # Normalized position
                total_value / self.config.initial_cash,          # Normalized total
                (total_value - self.peak_value) / self.peak_value if self.peak_value > 0 else 0,  # Drawdown
                1.0 if self.shares > 0 else 0.0                 # Has position
            ])
            
            # Repeat portfolio state for each time step
            portfolio_expanded = np.tile(portfolio_state, (self.config.window_size, 1))
            
            # Combine
            obs = np.hstack([feature_window, portfolio_expanded])
            
            return obs.astype(np.float32)
        
        def _calculate_reward(self, action: int, prev_value: float) -> float:
            """Calculate reward based on configuration."""
            current_price = self.prices[self.current_step]
            position_value = self.shares * current_price
            total_value = self.cash + position_value
            
            # Track history
            self.portfolio_history.append(total_value)
            
            # Calculate step return
            step_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0
            self.return_history.append(step_return)
            
            if self.config.reward_type == RewardType.PNL:
                return step_return * 100  # Scale up
            
            elif self.config.reward_type == RewardType.SHARPE:
                if len(self.return_history) < 20:
                    return step_return * 100
                
                returns = np.array(self.return_history[-20:])
                mean_ret = np.mean(returns)
                std_ret = np.std(returns) + 1e-8
                sharpe = mean_ret / std_ret * np.sqrt(252)
                return sharpe
            
            elif self.config.reward_type == RewardType.SORTINO:
                if len(self.return_history) < 20:
                    return step_return * 100
                
                returns = np.array(self.return_history[-20:])
                mean_ret = np.mean(returns)
                downside = returns[returns < 0]
                downside_std = np.std(downside) + 1e-8 if len(downside) > 0 else 1e-8
                sortino = mean_ret / downside_std * np.sqrt(252)
                return sortino
            
            else:  # RISK_ADJUSTED
                # Reward PnL but penalize drawdown
                drawdown = (self.peak_value - total_value) / self.peak_value if self.peak_value > 0 else 0
                reward = step_return * 100 - drawdown * 50
                
                # Penalize trading costs
                if action != 0:  # Buy or sell
                    reward -= self.config.commission_rate * 10
                
                return reward
        
        def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            """
            Execute one step in the environment.
            
            Args:
                action: 0=hold, 1=buy, 2=sell
            
            Returns:
                observation, reward, terminated, truncated, info
            """
            self.action_history.append(action)
            
            current_price = self.prices[self.current_step]
            prev_value = self.cash + self.shares * current_price
            
            # Execute action
            if action == 1:  # Buy
                if self.shares == 0 and self.cash > 0:
                    # Calculate position size
                    max_invest = self.cash * self.config.max_position_size
                    price_with_slippage = current_price * (1 + self.config.slippage)
                    shares_to_buy = max_invest / price_with_slippage
                    
                    # Apply commission
                    cost = shares_to_buy * price_with_slippage * (1 + self.config.commission_rate)
                    
                    if cost <= self.cash:
                        self.shares = shares_to_buy
                        self.cash -= cost
                        self.entry_price = price_with_slippage
            
            elif action == 2:  # Sell
                if self.shares > 0:
                    price_with_slippage = current_price * (1 - self.config.slippage)
                    proceeds = self.shares * price_with_slippage * (1 - self.config.commission_rate)
                    self.cash += proceeds
                    self.shares = 0
                    self.entry_price = 0
            
            # Move to next step
            self.current_step += 1
            
            # Check stop-loss and take-profit
            if self.shares > 0:
                current_price = self.prices[min(self.current_step, self.n_steps - 1)]
                position_return = (current_price - self.entry_price) / self.entry_price
                
                if position_return <= -self.config.stop_loss_pct:
                    # Stop loss triggered
                    proceeds = self.shares * current_price * (1 - self.config.commission_rate)
                    self.cash += proceeds
                    self.shares = 0
                    self.entry_price = 0
                
                elif position_return >= self.config.take_profit_pct:
                    # Take profit triggered
                    proceeds = self.shares * current_price * (1 - self.config.commission_rate)
                    self.cash += proceeds
                    self.shares = 0
                    self.entry_price = 0
            
            # Calculate reward
            reward = self._calculate_reward(action, prev_value)
            
            # Update peak value
            current_value = self.cash + self.shares * self.prices[min(self.current_step, self.n_steps - 1)]
            self.peak_value = max(self.peak_value, current_value)
            
            # Check termination
            terminated = False
            truncated = False
            
            # Max drawdown exceeded
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > self.config.max_drawdown:
                terminated = True
            
            # Bankrupt
            if current_value < self.config.initial_cash * 0.1:
                terminated = True
            
            # End of data
            if self.current_step >= self.n_steps - 1:
                truncated = True
            
            # Max steps
            if self.current_step >= self.config.max_steps:
                truncated = True
            
            # Get observation
            obs = self._get_observation()
            
            # Info
            info = {
                'portfolio_value': current_value,
                'cash': self.cash,
                'shares': self.shares,
                'total_return': (current_value - self.config.initial_cash) / self.config.initial_cash,
                'drawdown': drawdown,
                'step': self.current_step
            }
            
            return obs, reward, terminated, truncated, info
        
        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None
        ) -> Tuple[np.ndarray, Dict]:
            """Reset the environment."""
            super().reset(seed=seed)
            
            # Reset state
            self.current_step = self.config.window_size
            self.cash = self.config.initial_cash
            self.shares = 0.0
            self.entry_price = 0.0
            self.portfolio_value = self.config.initial_cash
            self.peak_value = self.config.initial_cash
            
            # Clear history
            self.portfolio_history = [self.config.initial_cash]
            self.action_history = []
            self.return_history = []
            
            obs = self._get_observation()
            info = {'portfolio_value': self.config.initial_cash}
            
            return obs, info
        
        def render(self) -> Optional[str]:
            """Render environment state."""
            if self.render_mode == 'ansi':
                current_price = self.prices[min(self.current_step, self.n_steps - 1)]
                position_value = self.shares * current_price
                total_value = self.cash + position_value
                
                output = (
                    f"\n{'='*50}\n"
                    f"Step: {self.current_step}/{self.n_steps}\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Cash: ${self.cash:,.2f}\n"
                    f"Shares: {self.shares:.4f}\n"
                    f"Position Value: ${position_value:,.2f}\n"
                    f"Total Value: ${total_value:,.2f}\n"
                    f"Total Return: {(total_value/self.config.initial_cash - 1)*100:.2f}%\n"
                    f"{'='*50}\n"
                )
                return output
            return None
        
        def get_performance_metrics(self) -> Dict[str, float]:
            """Get performance metrics."""
            returns = np.array(self.return_history)
            
            if len(returns) < 2:
                return {}
            
            total_return = (self.portfolio_history[-1] / self.config.initial_cash - 1) * 100
            
            # Sharpe ratio
            mean_ret = np.mean(returns)
            std_ret = np.std(returns) + 1e-8
            sharpe = mean_ret / std_ret * np.sqrt(252)
            
            # Sortino ratio
            downside = returns[returns < 0]
            downside_std = np.std(downside) + 1e-8 if len(downside) > 0 else 1e-8
            sortino = mean_ret / downside_std * np.sqrt(252)
            
            # Max drawdown
            peak = np.maximum.accumulate(self.portfolio_history)
            drawdowns = (peak - self.portfolio_history) / peak
            max_drawdown = np.max(drawdowns) * 100
            
            # Win rate
            winning_trades = sum(1 for r in returns if r > 0)
            total_trades = sum(1 for a in self.action_history if a != 0)
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            return {
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown_pct': max_drawdown,
                'win_rate_pct': win_rate,
                'total_trades': total_trades
            }

else:
    # Fallback implementation without gymnasium
    class TradingEnvironment:
        """Fallback trading environment when gymnasium is not available."""
        
        def __init__(self, prices: np.ndarray, **kwargs):
            self.prices = prices
            self.config = TradingEnvConfig()
            logger.warning("Gymnasium not available. Using fallback TradingEnvironment.")
        
        def reset(self):
            return np.zeros((60, 10)), {}
        
        def step(self, action):
            return np.zeros((60, 10)), 0.0, True, False, {}
        
        def render(self):
            pass
        
        def get_performance_metrics(self):
            return {}
