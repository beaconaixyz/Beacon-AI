import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BeaconModel
from typing import Dict, Any, List, Tuple
import numpy as np

class SurvivalModel(BeaconModel):
    """Neural network based survival analysis model"""
    
    def _build_model(self) -> nn.Module:
        """
        Build model architecture for survival analysis
        Returns:
            PyTorch model
        """
        class SurvivalNet(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3):
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
                
                # Output layer (hazard ratio)
                layers.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        model = SurvivalNet(
            input_dim=self.config['input_dim'],
            hidden_dims=self.config.get('hidden_dims', [256, 128, 64]),
            dropout_rate=self.config.get('dropout_rate', 0.3)
        ).to(self.device)
        
        self.logger.info(f"Created survival analysis model with architecture:\n{model}")
        return model
    
    def negative_log_likelihood(self, risk_scores: torch.Tensor, 
                              survival_time: torch.Tensor,
                              event_indicator: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative log likelihood for Cox model
        Args:
            risk_scores: Predicted risk scores
            survival_time: Time to event
            event_indicator: Binary indicator of event occurrence
        Returns:
            Negative log likelihood loss
        """
        # Sort by survival time in descending order
        sorted_indices = torch.argsort(survival_time, descending=True)
        risk_scores = risk_scores[sorted_indices]
        event_indicator = event_indicator[sorted_indices]
        
        # Calculate cumulative sum of exp(risk_scores)
        cumsum_exp_risk = torch.cumsum(torch.exp(risk_scores), dim=0)
        
        # Calculate log of cumulative sum
        log_cumsum = torch.log(cumsum_exp_risk)
        
        # Calculate loss for events only
        event_loss = event_indicator * (risk_scores - log_cumsum)
        
        return -torch.mean(event_loss)
    
    def train(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Train the model
        Args:
            data: Dictionary containing features, survival times, and event indicators
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
        
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = self._create_batches(data, batch_size)
            
            for batch in batches:
                optimizer.zero_grad()
                risk_scores = self.model(batch['features']).squeeze()
                loss = self.negative_log_likelihood(
                    risk_scores,
                    batch['survival_time'],
                    batch['event_indicator']
                )
                
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
    
    def predict_risk(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predict risk scores
        Args:
            data: Input features
        Returns:
            Predicted risk scores
        """
        self.model.eval()
        with torch.no_grad():
            risk_scores = self.model(data.to(self.device))
            return torch.exp(risk_scores)  # Convert to hazard ratio
    
    def predict_survival_function(self, data: torch.Tensor, 
                                time_points: torch.Tensor) -> torch.Tensor:
        """
        Predict survival function
        Args:
            data: Input features
            time_points: Time points for survival function
        Returns:
            Survival probabilities at each time point
        """
        risk_scores = self.predict_risk(data)
        baseline_hazard = self._estimate_baseline_hazard()
        
        # Calculate cumulative hazard
        cumulative_hazard = baseline_hazard.unsqueeze(0) * risk_scores
        
        # Convert to survival probabilities
        survival_probs = torch.exp(-cumulative_hazard)
        return survival_probs
    
    def _estimate_baseline_hazard(self) -> torch.Tensor:
        """Estimate baseline hazard function"""
        # This is a simplified version
        # In practice, you would estimate this from training data
        return torch.ones(1).to(self.device)
    
    def _create_batches(self, data: Dict[str, torch.Tensor], 
                       batch_size: int) -> List[Dict[str, torch.Tensor]]:
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
                'survival_time': data['survival_time'][batch_indices].to(self.device),
                'event_indicator': data['event_indicator'][batch_indices].to(self.device)
            }
            batches.append(batch)
        
        return batches 