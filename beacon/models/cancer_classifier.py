import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BeaconModel
from typing import Dict, Any, List

class CancerClassifier(BeaconModel):
    """Basic cancer classification model"""
    
    def _build_model(self) -> nn.Module:
        """
        Build model architecture
        Returns:
            PyTorch model
        """
        model = nn.Sequential(
            nn.Linear(self.config['input_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout', 0.3)),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout', 0.3)),
            nn.Linear(self.config['hidden_dim'] // 2, self.config['output_dim'])
        ).to(self.device)
        
        self.logger.info(f"Model architecture:\n{model}")
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
            lr=self.config.get('learning_rate', 0.001)
        )
        criterion = nn.CrossEntropyLoss()
        
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = self._create_batches(data, batch_size)
            
            for batch in batches:
                optimizer.zero_grad()
                outputs = self.model(batch['features'])
                loss = criterion(outputs, batch['labels'])
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(batches)
            history.append({
                'epoch': epoch + 1,
                'loss': avg_loss
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
            return F.softmax(predictions, dim=1)
    
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