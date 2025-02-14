import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BeaconModel
from typing import Dict, Any, List

class GenomicModel(BeaconModel):
    """Model for genomic data analysis"""
    
    def _build_model(self) -> nn.Module:
        """
        Build model architecture for genomic data analysis
        Returns:
            PyTorch model
        """
        class GenomicArchitecture(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
                super().__init__()
                
                # Build layers dynamically
                layers = []
                prev_dim = input_dim
                
                # Add hidden layers
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_dim = hidden_dim
                
                # Add output layer
                layers.append(nn.Linear(prev_dim, output_dim))
                
                self.network = nn.Sequential(*layers)
                
                # Attention mechanism for feature importance
                self.attention = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.Tanh(),
                    nn.Linear(input_dim, 1),
                    nn.Softmax(dim=1)
                )
            
            def forward(self, x):
                # Apply attention
                attention_weights = self.attention(x)
                x = x * attention_weights
                
                # Forward pass through main network
                return self.network(x)
        
        model = GenomicArchitecture(
            input_dim=self.config['input_dim'],
            hidden_dims=self.config.get('hidden_dims', [256, 128, 64]),
            output_dim=self.config['output_dim'],
            dropout_rate=self.config.get('dropout_rate', 0.3)
        ).to(self.device)
        
        self.logger.info(f"Created genomic model with architecture:\n{model}")
        return model
    
    def train(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Train the model
        Args:
            data: Dictionary containing features and labels
            **kwargs: Additional training parameters
        Returns:
            Training history
        """
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        if self.config.get('task_type') == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:  # regression
            criterion = nn.MSELoss()
        
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_metrics = {}
            batches = self._create_batches(data, batch_size)
            
            for batch in batches:
                optimizer.zero_grad()
                outputs = self.model(batch['features'])
                loss = criterion(outputs, batch['labels'])
                
                # Add L1 regularization for feature selection
                if self.config.get('l1_lambda'):
                    l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                    loss += self.config['l1_lambda'] * l1_norm
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(batches)
            history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                **epoch_metrics
            })
            
            self.logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return history
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions
        Args:
            data: Input features
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data.to(self.device))
            if self.config.get('task_type') == 'classification':
                return F.softmax(predictions, dim=1)
            return predictions
    
    def get_feature_importance(self, data: torch.Tensor) -> torch.Tensor:
        """
        Get feature importance scores using attention weights
        Args:
            data: Input features
        Returns:
            Feature importance scores
        """
        self.model.eval()
        with torch.no_grad():
            attention_weights = self.model.attention(data.to(self.device))
            return attention_weights.mean(dim=0)  # Average attention across samples
    
    def _create_batches(self, data: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Create batches from data
        Args:
            data: Input data dictionary
            batch_size: Size of each batch
        Returns:
            List of batch dictionaries
        """
        n_samples = len(data['features'])
        indices = torch.randperm(n_samples)
        batches = []
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = {
                'features': data['features'][batch_indices].to(self.device),
                'labels': data['labels'][batch_indices].to(self.device)
            }
            batches.append(batch)
        
        return batches 