import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional, Tuple
from ..core.base import BeaconModel

class TransformerModel(BeaconModel):
    """Transformer model for sequence data processing"""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        config = super()._get_default_config()
        config.update({
            'input_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'max_seq_length': 1000,
            'positional_encoding': 'sinusoidal',
            'output_dim': None,
            'task': 'classification'  # or 'regression'
        })
        return config
    
    def _build_model(self) -> nn.Module:
        """Build model architecture"""
        class TransformerArchitecture(nn.Module):
            def __init__(self, config: Dict[str, Any]):
                super().__init__()
                
                self.input_dim = config['input_dim']
                self.num_heads = config['num_heads']
                self.num_layers = config['num_layers']
                self.dim_feedforward = config['dim_feedforward']
                self.dropout = config['dropout']
                self.max_seq_length = config['max_seq_length']
                self.output_dim = config['output_dim']
                
                # Input embedding
                self.embedding = nn.Linear(self.input_dim, self.input_dim)
                
                # Positional encoding
                if config['positional_encoding'] == 'sinusoidal':
                    self.pos_encoding = self._create_sinusoidal_encoding()
                else:
                    self.pos_encoding = nn.Parameter(
                        torch.randn(1, self.max_seq_length, self.input_dim)
                    )
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.input_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=self.num_layers
                )
                
                # Output head
                self.output_head = nn.Sequential(
                    nn.Linear(self.input_dim, self.dim_feedforward),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.dim_feedforward, self.output_dim)
                )
            
            def _create_sinusoidal_encoding(self) -> torch.Tensor:
                """Create sinusoidal positional encoding"""
                position = torch.arange(self.max_seq_length).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, self.input_dim, 2) * 
                    (-math.log(10000.0) / self.input_dim)
                )
                pos_encoding = torch.zeros(1, self.max_seq_length, self.input_dim)
                pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
                pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
                return pos_encoding
            
            def forward(self, x: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                """
                Forward pass
                Args:
                    x: Input tensor of shape (batch_size, seq_length, input_dim)
                    mask: Optional attention mask
                Returns:
                    Output tensor
                """
                # Apply input embedding
                x = self.embedding(x)
                
                # Add positional encoding
                x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
                
                # Apply transformer
                if mask is not None:
                    x = self.transformer(x, src_key_padding_mask=mask)
                else:
                    x = self.transformer(x)
                
                # Global average pooling
                x = torch.mean(x, dim=1)
                
                # Apply output head
                x = self.output_head(x)
                
                return x
        
        # Ensure output dimension is set
        if self.config['output_dim'] is None:
            if self.config['task'] == 'classification':
                self.config['output_dim'] = 2  # Binary classification by default
            else:
                self.config['output_dim'] = 1  # Regression
        
        model = TransformerArchitecture(self.config)
        self.logger.info(f"Created transformer model with architecture:\n{model}")
        return model
    
    def create_padding_mask(self, lengths: torch.Tensor, 
                           max_len: Optional[int] = None) -> torch.Tensor:
        """
        Create padding mask for variable length sequences
        Args:
            lengths: Sequence lengths
            max_len: Maximum sequence length
        Returns:
            Boolean mask tensor
        """
        if max_len is None:
            max_len = max(lengths)
        
        mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        return mask
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step
        Args:
            batch: Batch dictionary containing sequences and targets
        Returns:
            Tuple of (loss, predictions)
        """
        sequences = batch['sequences'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        # Create padding mask if lengths are provided
        mask = None
        if 'lengths' in batch:
            mask = self.create_padding_mask(batch['lengths']).to(self.device)
        
        # Forward pass
        predictions = self.model(sequences, mask)
        
        # Calculate loss
        if self.config['task'] == 'classification':
            loss = nn.CrossEntropyLoss()(predictions, targets)
        else:
            loss = nn.MSELoss()(predictions, targets)
        
        return loss, predictions
    
    def predict_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Prediction step
        Args:
            batch: Batch dictionary containing sequences
        Returns:
            Model predictions
        """
        sequences = batch['sequences'].to(self.device)
        
        # Create padding mask if lengths are provided
        mask = None
        if 'lengths' in batch:
            mask = self.create_padding_mask(batch['lengths']).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.model(sequences, mask)
            
            if self.config['task'] == 'classification':
                predictions = torch.softmax(predictions, dim=1)
        
        return predictions
    
    def get_attention_weights(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Get attention weights from all layers
        Args:
            batch: Batch dictionary containing sequences
        Returns:
            List of attention weight tensors
        """
        sequences = batch['sequences'].to(self.device)
        
        # Create padding mask if lengths are provided
        mask = None
        if 'lengths' in batch:
            mask = self.create_padding_mask(batch['lengths']).to(self.device)
        
        attention_weights = []
        
        def hook_fn(module, input, output):
            attention_weights.append(output[1])  # attention weights
        
        # Register hooks for all attention layers
        hooks = []
        for layer in self.model.transformer.layers:
            hooks.append(layer.self_attn.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            self.model(sequences, mask)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights 