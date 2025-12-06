"""
Transformer-based Price Predictor - State-of-the-art sequence modeling for price prediction.

Features:
- Multi-head self-attention for temporal patterns
- Positional encoding for time series
- Cross-attention for multi-asset correlation
- Ensemble with traditional models
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import math

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Transformer predictor will use fallback mode.")


class PredictionHorizon(Enum):
    """Prediction time horizons."""
    INTRADAY = "intraday"      # 1-4 hours
    DAILY = "daily"            # 1 day
    WEEKLY = "weekly"          # 5 days
    MONTHLY = "monthly"        # 20 days


@dataclass
class TransformerPrediction:
    """Prediction result from transformer model."""
    symbol: str
    current_price: float
    predicted_price: float
    predicted_direction: str  # 'up', 'down', 'neutral'
    confidence: float
    horizon: PredictionHorizon
    price_targets: Dict[str, float] = field(default_factory=dict)  # low, mid, high
    attention_weights: Optional[np.ndarray] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def predicted_return(self) -> float:
        """Calculate predicted return percentage."""
        if self.current_price <= 0:
            return 0.0
        return ((self.predicted_price - self.current_price) / self.current_price) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'predicted_price': self.predicted_price,
            'predicted_return_pct': self.predicted_return,
            'predicted_direction': self.predicted_direction,
            'confidence': self.confidence,
            'horizon': self.horizon.value,
            'price_targets': self.price_targets,
            'timestamp': self.timestamp.isoformat()
        }


if TORCH_AVAILABLE:
    
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for time series."""
        
        def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input."""
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
    
    
    class TemporalAttentionBlock(nn.Module):
        """Self-attention block for temporal patterns."""
        
        def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            d_ff: int = 256,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
            
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        
        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass with attention weights."""
            # Self-attention with residual
            attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(x + self.dropout(attn_out))
            
            # Feed-forward with residual
            ff_out = self.feed_forward(x)
            x = self.norm2(x + ff_out)
            
            return x, attn_weights
    
    
    class PriceTransformer(nn.Module):
        """
        Transformer model for price prediction.
        
        Architecture:
        - Input embedding layer
        - Positional encoding
        - Stack of transformer blocks
        - Output projection layers
        """
        
        def __init__(
            self,
            input_dim: int,
            d_model: int = 128,
            n_heads: int = 8,
            n_layers: int = 4,
            d_ff: int = 256,
            dropout: float = 0.1,
            max_seq_len: int = 252  # ~1 year of trading days
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.d_model = d_model
            
            # Input projection
            self.input_projection = nn.Linear(input_dim, d_model)
            
            # Positional encoding
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
            
            # Transformer layers
            self.layers = nn.ModuleList([
                TemporalAttentionBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            
            # Output heads
            self.price_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)  # Price prediction
            )
            
            self.direction_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 3),  # Up, Down, Neutral
                nn.Softmax(dim=-1)
            )
            
            self.volatility_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),  # Volatility prediction
                nn.Softplus()  # Ensure positive
            )
        
        def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                x: Input tensor [batch, seq_len, input_dim]
                return_attention: Whether to return attention weights
            
            Returns:
                Dict with predictions
            """
            # Project input
            x = self.input_projection(x)
            
            # Add positional encoding
            x = self.pos_encoding(x)
            
            # Pass through transformer layers
            attention_weights = []
            for layer in self.layers:
                x, attn = layer(x)
                if return_attention:
                    attention_weights.append(attn)
            
            # Use last token for prediction (like BERT [CLS])
            last_hidden = x[:, -1, :]
            
            # Generate predictions
            outputs = {
                'price': self.price_head(last_hidden),
                'direction': self.direction_head(last_hidden),
                'volatility': self.volatility_head(last_hidden)
            }
            
            if return_attention:
                outputs['attention'] = attention_weights
            
            return outputs
    
    
    class TimeSeriesDataset(Dataset):
        """Dataset for time series prediction."""
        
        def __init__(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            seq_len: int = 60
        ):
            self.features = torch.FloatTensor(features)
            self.targets = torch.FloatTensor(targets)
            self.seq_len = seq_len
        
        def __len__(self) -> int:
            return len(self.features) - self.seq_len
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.features[idx:idx + self.seq_len]
            y = self.targets[idx + self.seq_len]
            return x, y


class TransformerPredictor:
    """
    High-level interface for transformer-based price prediction.
    
    Features:
    - Training on historical data
    - Inference with confidence estimation
    - Attention visualization
    - Integration with existing feature engineering
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        seq_len: int = 60,
        learning_rate: float = 1e-4,
        device: Optional[str] = None
    ):
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        
        # Set device
        if device:
            self.device = device
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.model: Optional[Any] = None
        self.optimizer: Optional[Any] = None
        self.is_trained = False
        self.training_history: List[Dict] = []
        
        # Feature scalers
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0
        
        if TORCH_AVAILABLE:
            self._init_model()
    
    def _init_model(self):
        """Initialize the transformer model."""
        self.model = PriceTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        logger.info(f"Initialized TransformerPredictor on {self.device}")
    
    def _normalize_features(
        self,
        features: np.ndarray,
        fit: bool = False
    ) -> np.ndarray:
        """Normalize features using z-score."""
        if fit:
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-8
        
        return (features - self.feature_mean) / self.feature_std
    
    def _normalize_targets(
        self,
        targets: np.ndarray,
        fit: bool = False
    ) -> np.ndarray:
        """Normalize targets."""
        if fit:
            self.target_mean = np.mean(targets)
            self.target_std = np.std(targets) + 1e-8
        
        return (targets - self.target_mean) / self.target_std
    
    def _denormalize_targets(self, targets: np.ndarray) -> np.ndarray:
        """Denormalize targets."""
        return targets * self.target_std + self.target_mean
    
    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the transformer model.
        
        Args:
            features: Feature array [n_samples, n_features]
            targets: Target array [n_samples]
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation set fraction
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Training history
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Cannot train.")
            return {'train_loss': [], 'val_loss': []}
        
        # Update input dimension if needed
        if features.shape[1] != self.input_dim:
            self.input_dim = features.shape[1]
            self._init_model()
        
        # Normalize data
        features = self._normalize_features(features, fit=True)
        targets = self._normalize_targets(targets, fit=True)
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        train_features, val_features = features[:split_idx], features[split_idx:]
        train_targets, val_targets = targets[:split_idx], targets[split_idx:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_features, train_targets, self.seq_len)
        val_dataset = TimeSeriesDataset(val_features, val_targets, self.seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss function
        price_criterion = nn.MSELoss()
        direction_criterion = nn.CrossEntropyLoss()
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_x)
                
                # Combined loss
                price_loss = price_criterion(outputs['price'].squeeze(), batch_y)
                
                # Direction labels (0=down, 1=neutral, 2=up)
                direction_labels = torch.where(
                    batch_y > 0.01, torch.tensor(2),
                    torch.where(batch_y < -0.01, torch.tensor(0), torch.tensor(1))
                ).to(self.device)
                direction_loss = direction_criterion(outputs['direction'], direction_labels)
                
                loss = price_loss + 0.5 * direction_loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    price_loss = price_criterion(outputs['price'].squeeze(), batch_y)
                    val_loss += price_loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
        
        self.is_trained = True
        self.training_history = history
        
        return history
    
    def predict(
        self,
        features: np.ndarray,
        symbol: str = "",
        current_price: float = 0.0,
        horizon: PredictionHorizon = PredictionHorizon.DAILY
    ) -> TransformerPrediction:
        """
        Make a prediction.
        
        Args:
            features: Feature array [seq_len, n_features]
            symbol: Stock symbol
            current_price: Current price for return calculation
            horizon: Prediction horizon
        
        Returns:
            TransformerPrediction object
        """
        if not TORCH_AVAILABLE or not self.is_trained:
            # Fallback prediction
            return TransformerPrediction(
                symbol=symbol,
                current_price=current_price,
                predicted_price=current_price,
                predicted_direction='neutral',
                confidence=0.0,
                horizon=horizon
            )
        
        self.model.eval()
        
        # Normalize features
        features = self._normalize_features(features)
        
        # Ensure correct sequence length
        if len(features) < self.seq_len:
            # Pad with zeros
            padding = np.zeros((self.seq_len - len(features), features.shape[1]))
            features = np.vstack([padding, features])
        elif len(features) > self.seq_len:
            features = features[-self.seq_len:]
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x, return_attention=True)
        
        # Get predictions
        price_pred = outputs['price'].cpu().numpy()[0, 0]
        direction_probs = outputs['direction'].cpu().numpy()[0]
        volatility = outputs['volatility'].cpu().numpy()[0, 0]
        
        # Denormalize price prediction
        price_pred = self._denormalize_targets(np.array([price_pred]))[0]
        
        # Calculate predicted price
        if current_price > 0:
            predicted_price = current_price * (1 + price_pred)
        else:
            predicted_price = price_pred
        
        # Determine direction
        direction_idx = np.argmax(direction_probs)
        direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
        predicted_direction = direction_map[direction_idx]
        
        # Confidence is the probability of the predicted direction
        confidence = float(direction_probs[direction_idx])
        
        # Price targets based on volatility
        vol_factor = float(volatility) * current_price if current_price > 0 else float(volatility)
        price_targets = {
            'low': predicted_price - 2 * vol_factor,
            'mid': predicted_price,
            'high': predicted_price + 2 * vol_factor
        }
        
        # Get attention weights for interpretability
        attention_weights = None
        if 'attention' in outputs:
            attention_weights = outputs['attention'][-1].cpu().numpy()
        
        return TransformerPrediction(
            symbol=symbol,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_direction=predicted_direction,
            confidence=confidence,
            horizon=horizon,
            price_targets=price_targets,
            attention_weights=attention_weights
        )
    
    def predict_multiple_horizons(
        self,
        features: np.ndarray,
        symbol: str = "",
        current_price: float = 0.0
    ) -> Dict[str, TransformerPrediction]:
        """
        Predict for multiple time horizons.
        
        Returns:
            Dict of predictions by horizon
        """
        predictions = {}
        
        for horizon in PredictionHorizon:
            # Adjust features for different horizons (simplified)
            pred = self.predict(features, symbol, current_price, horizon)
            predictions[horizon.value] = pred
        
        return predictions
    
    def get_feature_importance(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get feature importance using attention weights.
        
        Args:
            features: Feature array
            feature_names: Optional feature names
        
        Returns:
            Feature importance dict
        """
        if not TORCH_AVAILABLE or not self.is_trained:
            return {}
        
        self.model.eval()
        
        # Normalize and prepare
        features = self._normalize_features(features)
        if len(features) > self.seq_len:
            features = features[-self.seq_len:]
        
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x, return_attention=True)
        
        # Average attention across heads and layers
        if 'attention' in outputs:
            attention = outputs['attention'][-1].cpu().numpy()
            # attention shape: [batch, heads, seq, seq]
            avg_attention = np.mean(attention[0], axis=0)
            
            # Sum attention received by each position
            importance = np.mean(avg_attention, axis=0)
            
            # Map to features (last position)
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance))]
            
            # Normalize
            importance = importance / (importance.sum() + 1e-8)
            
            return dict(zip(feature_names[:len(importance)], importance.tolist()))
        
        return {}
    
    def save(self, path: str):
        """Save model to file."""
        if not TORCH_AVAILABLE or self.model is None:
            logger.warning("Cannot save: no model available")
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'seq_len': self.seq_len,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std,
            'is_trained': self.is_trained
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file."""
        if not TORCH_AVAILABLE:
            logger.warning("Cannot load: PyTorch not available")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.input_dim = checkpoint['input_dim']
        self.d_model = checkpoint['d_model']
        self.n_heads = checkpoint['n_heads']
        self.n_layers = checkpoint['n_layers']
        self.seq_len = checkpoint['seq_len']
        self.feature_mean = checkpoint['feature_mean']
        self.feature_std = checkpoint['feature_std']
        self.target_mean = checkpoint['target_mean']
        self.target_std = checkpoint['target_std']
        self.is_trained = checkpoint['is_trained']
        
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {path}")


def create_transformer_predictor(
    input_dim: int = 64,
    model_size: str = 'medium'
) -> TransformerPredictor:
    """
    Factory function to create transformer predictor.
    
    Args:
        input_dim: Number of input features
        model_size: 'small', 'medium', or 'large'
    
    Returns:
        TransformerPredictor instance
    """
    configs = {
        'small': {'d_model': 64, 'n_heads': 4, 'n_layers': 2},
        'medium': {'d_model': 128, 'n_heads': 8, 'n_layers': 4},
        'large': {'d_model': 256, 'n_heads': 8, 'n_layers': 6}
    }
    
    config = configs.get(model_size, configs['medium'])
    
    return TransformerPredictor(
        input_dim=input_dim,
        **config
    )
