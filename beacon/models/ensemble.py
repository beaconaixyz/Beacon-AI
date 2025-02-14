import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..core.base import BeaconBase
from .multimodal import MultimodalFusion

class MultimodalEnsemble(BeaconBase):
    """Ensemble of multimodal models with uncertainty estimation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ensemble
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.models = []
        self._initialize_models()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'n_models': 5,  # Number of models in ensemble
            'model_config': {  # Base model configuration
                'hidden_dim': 256,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            },
            'bootstrap_ratio': 0.8,  # Ratio of data to sample for each model
            'aggregation_method': 'mean',  # mean, weighted_mean, or voting
            'uncertainty_method': 'entropy',  # entropy or variance
            'dropout_samples': 20,  # Number of MC dropout samples
            'temperature': 1.0,  # Temperature for calibration
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def _initialize_models(self):
        """Initialize ensemble models"""
        for _ in range(self.config['n_models']):
            model = MultimodalFusion(self.config['model_config'])
            model.to(self.config['device'])
            self.models.append(model)
    
    def train(self, data: Dict[str, torch.Tensor], **kwargs) -> List[Dict[str, Any]]:
        """
        Train ensemble models
        Args:
            data: Training data dictionary
            **kwargs: Additional training parameters
        Returns:
            List of training histories
        """
        histories = []
        n_samples = len(next(iter(data.values())))
        bootstrap_size = int(n_samples * self.config['bootstrap_ratio'])
        
        for i, model in enumerate(self.models):
            # Bootstrap sampling
            indices = torch.randint(0, n_samples, (bootstrap_size,))
            bootstrap_data = {
                key: value[indices] for key, value in data.items()
            }
            
            self.logger.info(f"Training model {i+1}/{len(self.models)}")
            history = model.train(bootstrap_data, **kwargs)
            histories.append(history)
        
        return histories
    
    def predict(self, batch: Dict[str, torch.Tensor], 
               return_uncertainty: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Make predictions with uncertainty estimation
        Args:
            batch: Input batch
            return_uncertainty: Whether to return uncertainty estimates
        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = []
        
        # Get predictions from each model
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model.predict(batch)
                if self.config['uncertainty_method'] == 'entropy':
                    # Apply temperature scaling
                    pred = pred / self.config['temperature']
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Aggregate predictions
        if self.config['aggregation_method'] == 'mean':
            ensemble_pred = predictions.mean(0)
        elif self.config['aggregation_method'] == 'voting':
            ensemble_pred = torch.mode(predictions.argmax(dim=-1), dim=0)[0]
        else:  # weighted_mean
            weights = torch.ones(len(self.models)) / len(self.models)
            ensemble_pred = (predictions * weights.view(-1, 1, 1)).sum(0)
        
        if not return_uncertainty:
            return ensemble_pred, None
        
        # Calculate uncertainty
        if self.config['uncertainty_method'] == 'entropy':
            # Entropy of the average prediction
            uncertainty = -torch.sum(ensemble_pred * torch.log(ensemble_pred + 1e-10), dim=-1)
        else:  # variance
            uncertainty = predictions.var(0).mean(-1)
        
        return ensemble_pred, uncertainty
    
    def monte_carlo_dropout(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Monte Carlo dropout for uncertainty estimation
        Args:
            batch: Input batch
        Returns:
            Tuple of (mean predictions, uncertainties)
        """
        predictions = []
        
        # Enable dropout at inference time
        for model in self.models:
            model.train()  # Enable dropout
            for _ in range(self.config['dropout_samples']):
                with torch.no_grad():
                    pred = model.predict(batch)
                    predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(0)
        uncertainty = predictions.var(0).mean(-1)
        
        return mean_pred, uncertainty
    
    def calibrate(self, val_data: Dict[str, torch.Tensor], 
                 temperature_range: Tuple[float, float] = (0.1, 5.0)) -> float:
        """
        Calibrate ensemble using temperature scaling
        Args:
            val_data: Validation data
            temperature_range: Range of temperatures to search
        Returns:
            Optimal temperature
        """
        temperatures = torch.linspace(temperature_range[0], temperature_range[1], 100)
        best_temp = 1.0
        best_nll = float('inf')
        
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model.predict(val_data)
                predictions.append(pred)
        
        predictions = torch.stack(predictions).mean(0)
        labels = val_data['labels']
        
        for temp in temperatures:
            scaled_preds = predictions / temp
            nll = -torch.mean(torch.sum(
                torch.log(scaled_preds + 1e-10) * labels, dim=-1
            ))
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.config['temperature'] = best_temp.item()
        return best_temp.item()
    
    def get_feature_importance(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get ensemble feature importance
        Args:
            batch: Input batch
        Returns:
            Dictionary of feature importance scores
        """
        importance_scores = []
        
        for model in self.models:
            scores = model.get_modality_embeddings(batch)
            importance_scores.append(scores)
        
        # Average importance scores across models
        ensemble_scores = {}
        for key in importance_scores[0].keys():
            scores = torch.stack([score[key] for score in importance_scores])
            ensemble_scores[key] = scores.mean(0)
        
        return ensemble_scores
    
    def save_ensemble(self, path: str):
        """Save ensemble models"""
        for i, model in enumerate(self.models):
            model_path = f"{path}_model_{i}.pt"
            model.save(model_path)
        
        # Save ensemble configuration
        config_path = f"{path}_config.pt"
        torch.save(self.config, config_path)
    
    def load_ensemble(self, path: str):
        """Load ensemble models"""
        # Load configuration
        config_path = f"{path}_config.pt"
        self.config = torch.load(config_path)
        
        # Load models
        self.models = []
        for i in range(self.config['n_models']):
            model_path = f"{path}_model_{i}.pt"
            model = MultimodalFusion(self.config['model_config'])
            model.load(model_path)
            model.to(self.config['device'])
            self.models.append(model) 