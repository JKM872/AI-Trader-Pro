"""
RL Agent - Deep Reinforcement Learning agent for trading.

Supports multiple algorithms:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)

Uses stable-baselines3 for implementation.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Union, Tuple
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for stable-baselines3
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not available. RL agent will use fallback.")

from .trading_env import TradingEnvironment, TradingEnvConfig


class RLAlgorithm(Enum):
    """Available RL algorithms."""
    DQN = "dqn"
    PPO = "ppo"
    A2C = "a2c"


class TradingAction(Enum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class RLAgentConfig:
    """Configuration for RL agent."""
    
    # Algorithm
    algorithm: RLAlgorithm = RLAlgorithm.PPO
    
    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99          # Discount factor
    batch_size: int = 64
    buffer_size: int = 100000    # Replay buffer size (DQN)
    n_steps: int = 2048          # Steps per update (PPO/A2C)
    n_epochs: int = 10           # Epochs per update (PPO)
    
    # Network
    policy: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    
    # Exploration (DQN)
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    
    # PPO specific
    clip_range: float = 0.2
    ent_coef: float = 0.01       # Entropy coefficient
    
    # Training settings
    total_timesteps: int = 100000
    eval_freq: int = 5000
    verbose: int = 1


@dataclass
class AgentState:
    """Agent's current state."""
    portfolio_value: float
    cash: float
    position: float
    unrealized_pnl: float
    current_action: TradingAction
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RLAgent:
    """
    Deep Reinforcement Learning trading agent.
    
    Features:
    - Multiple algorithm support (DQN, PPO, A2C)
    - Training with early stopping
    - Action with confidence estimation
    - Model persistence
    """
    
    def __init__(
        self,
        config: Optional[RLAgentConfig] = None,
        env_config: Optional[TradingEnvConfig] = None
    ):
        """
        Initialize RL agent.
        
        Args:
            config: Agent configuration
            env_config: Environment configuration
        """
        self.config = config or RLAgentConfig()
        self.env_config = env_config or TradingEnvConfig()
        
        self.model: Optional[Any] = None
        self.env: Optional[TradingEnvironment] = None
        self.is_trained = False
        
        self.training_history: List[Dict] = []
        self.action_history: List[TradingAction] = []
        
        logger.info(f"RLAgent initialized with {self.config.algorithm.value}")
    
    def _create_model(self, env: Any) -> Any:
        """Create RL model based on configuration."""
        if not SB3_AVAILABLE:
            return None
        
        policy_kwargs = {
            "net_arch": self.config.net_arch
        }
        
        if self.config.algorithm == RLAlgorithm.DQN:
            model = DQN(
                self.config.policy,
                env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                exploration_fraction=self.config.exploration_fraction,
                exploration_final_eps=self.config.exploration_final_eps,
                policy_kwargs=policy_kwargs,
                verbose=self.config.verbose
            )
        
        elif self.config.algorithm == RLAlgorithm.PPO:
            model = PPO(
                self.config.policy,
                env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                batch_size=self.config.batch_size,
                n_steps=self.config.n_steps,
                n_epochs=self.config.n_epochs,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                policy_kwargs=policy_kwargs,
                verbose=self.config.verbose
            )
        
        else:  # A2C
            model = A2C(
                self.config.policy,
                env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                n_steps=self.config.n_steps,
                ent_coef=self.config.ent_coef,
                policy_kwargs=policy_kwargs,
                verbose=self.config.verbose
            )
        
        return model
    
    def train(
        self,
        prices: np.ndarray,
        features: Optional[np.ndarray] = None,
        eval_prices: Optional[np.ndarray] = None,
        eval_features: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            prices: Training price data
            features: Optional training features
            eval_prices: Evaluation price data
            eval_features: Optional evaluation features
            save_path: Path to save best model
        
        Returns:
            Training results
        """
        if not SB3_AVAILABLE:
            logger.warning("stable-baselines3 not available. Cannot train.")
            return {'error': 'stable-baselines3 not available'}
        
        # Create training environment
        train_env = TradingEnvironment(
            prices=prices,
            features=features,
            config=self.env_config
        )
        train_env = DummyVecEnv([lambda: train_env])
        
        # Create model
        self.model = self._create_model(train_env)
        
        # Setup callbacks
        callbacks = []
        
        if eval_prices is not None:
            eval_env = TradingEnvironment(
                prices=eval_prices,
                features=eval_features,
                config=self.env_config
            )
            eval_env = DummyVecEnv([lambda: eval_env])
            
            # Early stopping callback
            stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=5,
                min_evals=10
            )
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path if save_path else "./rl_models/",
                log_path="./rl_logs/",
                eval_freq=self.config.eval_freq,
                deterministic=True,
                render=False,
                callback_after_eval=stop_callback
            )
            callbacks.append(eval_callback)
        
        # Train
        logger.info(f"Training {self.config.algorithm.value} for {self.config.total_timesteps} timesteps")
        
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=True
        )
        
        self.is_trained = True
        self.env = train_env
        
        # Get final metrics
        metrics = self._evaluate_model(train_env)
        
        logger.info(f"Training complete. Final return: {metrics.get('total_return_pct', 0):.2f}%")
        
        return metrics
    
    def _evaluate_model(self, env: Any, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            return {}
        
        returns = []
        sharpes = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            
            if isinstance(info, list) and len(info) > 0:
                returns.append(info[0].get('total_return', 0))
        
        return {
            'mean_return_pct': np.mean(returns) * 100 if returns else 0,
            'std_return_pct': np.std(returns) * 100 if returns else 0,
            'total_return_pct': returns[-1] * 100 if returns else 0
        }
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[TradingAction, float]:
        """
        Predict action for given observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
        
        Returns:
            Tuple of (action, confidence)
        """
        if not self.is_trained or self.model is None:
            return TradingAction.HOLD, 0.0
        
        # Get action
        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        # Estimate confidence
        if hasattr(self.model, 'policy'):
            try:
                import torch
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                
                with torch.no_grad():
                    if hasattr(self.model.policy, 'get_distribution'):
                        dist = self.model.policy.get_distribution(obs_tensor)
                        probs = dist.distribution.probs.numpy()[0]
                        confidence = float(probs[action])
                    else:
                        # For DQN, use Q-value difference
                        q_values = self.model.q_net(obs_tensor).numpy()[0]
                        q_exp = np.exp(q_values - np.max(q_values))
                        probs = q_exp / q_exp.sum()
                        confidence = float(probs[action])
            except Exception:
                confidence = 0.5
        else:
            confidence = 0.5
        
        trading_action = TradingAction(int(action) if isinstance(action, (int, np.integer)) else int(action[0]))
        self.action_history.append(trading_action)
        
        return trading_action, confidence
    
    def get_action_probabilities(self, observation: np.ndarray) -> Dict[TradingAction, float]:
        """
        Get action probabilities.
        
        Args:
            observation: Current observation
        
        Returns:
            Dict mapping actions to probabilities
        """
        if not self.is_trained or self.model is None:
            return {a: 1/3 for a in TradingAction}
        
        try:
            import torch
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            
            with torch.no_grad():
                if hasattr(self.model.policy, 'get_distribution'):
                    dist = self.model.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.numpy()[0]
                else:
                    q_values = self.model.q_net(obs_tensor).numpy()[0]
                    q_exp = np.exp(q_values - np.max(q_values))
                    probs = q_exp / q_exp.sum()
            
            return {
                TradingAction.HOLD: float(probs[0]),
                TradingAction.BUY: float(probs[1]),
                TradingAction.SELL: float(probs[2])
            }
        except Exception:
            return {a: 1/3 for a in TradingAction}
    
    def get_state(self, portfolio_value: float, cash: float, position: float) -> AgentState:
        """
        Get current agent state.
        
        Args:
            portfolio_value: Current portfolio value
            cash: Available cash
            position: Current position value
        
        Returns:
            AgentState object
        """
        unrealized_pnl = position  # Simplified
        
        # Get current recommendation
        if self.action_history:
            current_action = self.action_history[-1]
        else:
            current_action = TradingAction.HOLD
        
        return AgentState(
            portfolio_value=portfolio_value,
            cash=cash,
            position=position,
            unrealized_pnl=unrealized_pnl,
            current_action=current_action,
            confidence=0.5
        )
    
    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        if not SB3_AVAILABLE:
            logger.warning("stable-baselines3 not available. Cannot load.")
            return
        
        algorithm_map = {
            RLAlgorithm.DQN: DQN,
            RLAlgorithm.PPO: PPO,
            RLAlgorithm.A2C: A2C
        }
        
        model_class = algorithm_map[self.config.algorithm]
        self.model = model_class.load(path)
        self.is_trained = True
        
        logger.info(f"Model loaded from {path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'algorithm': self.config.algorithm.value,
            'is_trained': self.is_trained,
            'total_actions': len(self.action_history),
            'action_distribution': {
                action.name: sum(1 for a in self.action_history if a == action)
                for action in TradingAction
            } if self.action_history else {}
        }


class EnsembleRLAgent:
    """
    Ensemble of multiple RL agents for robust trading decisions.
    
    Combines predictions from multiple agents using voting or averaging.
    """
    
    def __init__(
        self,
        algorithms: Optional[List[RLAlgorithm]] = None,
        env_config: Optional[TradingEnvConfig] = None
    ):
        """
        Initialize ensemble of RL agents.
        
        Args:
            algorithms: List of algorithms to use
            env_config: Environment configuration
        """
        self.algorithms = algorithms or [RLAlgorithm.PPO, RLAlgorithm.A2C, RLAlgorithm.DQN]
        self.env_config = env_config or TradingEnvConfig()
        
        self.agents: List[RLAgent] = []
        self.agent_weights: List[float] = []
        
        # Create agents
        for algo in self.algorithms:
            config = RLAgentConfig(algorithm=algo)
            agent = RLAgent(config=config, env_config=self.env_config)
            self.agents.append(agent)
            self.agent_weights.append(1.0 / len(self.algorithms))
    
    def train_all(
        self,
        prices: np.ndarray,
        features: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Train all agents."""
        results = []
        
        for i, agent in enumerate(self.agents):
            logger.info(f"Training agent {i+1}/{len(self.agents)}: {agent.config.algorithm.value}")
            result = agent.train(prices, features, **kwargs)
            results.append(result)
            
            # Update weights based on performance
            if 'total_return_pct' in result:
                self.agent_weights[i] = max(0.1, result['total_return_pct'] / 100 + 1)
        
        # Normalize weights
        total_weight = sum(self.agent_weights)
        self.agent_weights = [w / total_weight for w in self.agent_weights]
        
        return results
    
    def predict(
        self,
        observation: np.ndarray,
        voting: str = 'weighted'
    ) -> Tuple[TradingAction, float]:
        """
        Ensemble prediction.
        
        Args:
            observation: Current observation
            voting: 'weighted', 'majority', or 'confidence'
        
        Returns:
            Tuple of (action, confidence)
        """
        if not any(a.is_trained for a in self.agents):
            return TradingAction.HOLD, 0.0
        
        votes = {action: 0.0 for action in TradingAction}
        
        for agent, weight in zip(self.agents, self.agent_weights):
            if not agent.is_trained:
                continue
            
            action, confidence = agent.predict(observation)
            
            if voting == 'weighted':
                votes[action] += weight
            elif voting == 'majority':
                votes[action] += 1
            else:  # confidence
                votes[action] += weight * confidence
        
        # Get winning action
        best_action = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        confidence = votes[best_action] / total_votes if total_votes > 0 else 0
        
        return best_action, confidence


def create_rl_agent(
    algorithm: str = 'ppo',
    learning_rate: float = 3e-4,
    **kwargs
) -> RLAgent:
    """
    Factory function to create RL agent.
    
    Args:
        algorithm: 'dqn', 'ppo', or 'a2c'
        learning_rate: Learning rate
        **kwargs: Additional config parameters
    
    Returns:
        RLAgent instance
    """
    algo_map = {
        'dqn': RLAlgorithm.DQN,
        'ppo': RLAlgorithm.PPO,
        'a2c': RLAlgorithm.A2C
    }
    
    config = RLAgentConfig(
        algorithm=algo_map.get(algorithm.lower(), RLAlgorithm.PPO),
        learning_rate=learning_rate,
        **{k: v for k, v in kwargs.items() if hasattr(RLAgentConfig, k)}
    )
    
    return RLAgent(config=config)


# Type alias for tuple return
from typing import Tuple
