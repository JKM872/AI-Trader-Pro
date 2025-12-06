"""
Configuration loader for AI Trader.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """
    Configuration manager for AI Trader.
    
    Loads settings from:
    1. config/config.yaml (default settings)
    2. .env file (API keys and secrets)
    3. Environment variables (override)
    
    Usage:
        config = Config()
        api_key = config.get('ALPACA_API_KEY')
        stop_loss = config.get('risk.stop_loss_pct', default=0.05)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml (auto-detected if not provided)
        """
        self._config: Dict[str, Any] = {}
        self._env_loaded = False
        
        # Load .env file
        self._load_env()
        
        # Load YAML config
        if config_path is None:
            config_path = self._find_config_file()
        
        if config_path and Path(config_path).exists():
            self._load_yaml(config_path)
    
    def _find_config_file(self) -> Optional[str]:
        """Find config.yaml in common locations."""
        locations = [
            Path(__file__).parent.parent / 'config' / 'config.yaml',
            Path.cwd() / 'config' / 'config.yaml',
            Path.cwd() / 'config.yaml',
        ]
        
        for loc in locations:
            if loc.exists():
                return str(loc)
        return None
    
    def _load_env(self):
        """Load environment variables from .env file."""
        env_locations = [
            Path.cwd() / '.env',
            Path(__file__).parent.parent.parent / '.env',
        ]
        
        for loc in env_locations:
            if loc.exists():
                load_dotenv(loc)
                self._env_loaded = True
                break
    
    def _load_yaml(self, path: str):
        """Load YAML configuration file."""
        with open(path, 'r') as f:
            self._config = yaml.safe_load(f) or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Supports dot notation for nested keys: 'risk.stop_loss_pct'
        
        Priority:
        1. Environment variables (for secrets)
        2. YAML config
        3. Default value
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if not found
        
        Returns:
            Configuration value
        """
        # Check environment first (for secrets/API keys)
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value
        
        # Navigate nested config with dot notation
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self._config.get(section, {})
    
    @property
    def trading_mode(self) -> str:
        """Get trading mode (paper, backtest, live)."""
        return self.get('trading.mode', 'paper')
    
    @property
    def symbols(self) -> list:
        """Get list of symbols to trade."""
        return self.get('trading.symbols', ['AAPL', 'MSFT', 'GOOGL'])
    
    @property
    def ai_provider(self) -> str:
        """Get AI provider (deepseek, groq)."""
        return self.get('ai.provider', 'deepseek')
    
    @property
    def stop_loss_pct(self) -> float:
        """Get stop loss percentage."""
        return self.get('risk.stop_loss_pct', 0.05)
    
    @property
    def take_profit_pct(self) -> float:
        """Get take profit percentage."""
        return self.get('risk.take_profit_pct', 0.10)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config as dictionary (excluding secrets)."""
        return self._config.copy()


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


if __name__ == "__main__":
    # Test configuration
    config = Config()
    
    print("=== Configuration ===")
    print(f"Trading Mode: {config.trading_mode}")
    print(f"Symbols: {config.symbols}")
    print(f"AI Provider: {config.ai_provider}")
    print(f"Stop Loss: {config.stop_loss_pct:.1%}")
    print(f"Take Profit: {config.take_profit_pct:.1%}")
    
    print("\n=== Environment Variables ===")
    print(f"ALPACA_API_KEY: {'*' * 10 if config.get('ALPACA_API_KEY') else 'Not set'}")
    print(f"DEEPSEEK_API_KEY: {'*' * 10 if config.get('DEEPSEEK_API_KEY') else 'Not set'}")
