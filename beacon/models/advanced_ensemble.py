import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from ..core.base import BeaconBase
from .multimodal import MultimodalFusion
from .ensemble import MultimodalEnsemble

class AdvancedEnsemble(BeaconBase):
    """Advanced ensemble methods for multimodal models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize advanced ensemble
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.base_models = []
        self.meta_model = None
        self._initialize_models()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'ensemble_method': 'stacking',  # stacking, boosting, or deep_ensemble
            'n_base_models': 5,
            'base_model_config': {
                'hidden_dim': 256,
                'dropout_rate': 0.3,
                'learning_rate': 0.001
            },
            'meta_model_config': {
                'hidden_dim': 128,
                'dropout_rate': 0.2,
                'learning_rate': 0.0005
            },
            'boosting_config': {
                'n_rounds': 10,
                'learning_rate': 0.1,
                'subsample_ratio': 0.8
            },
            'uncertainty_config': {
                'method': 'evidential',  # evidential, bayesian, or ensemble
                'prior_scale': 1.0,
                'n_samples': 50
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def _initialize_models(self):
        """Initialize models based on ensemble method"""
        if self.config['ensemble_method'] == 'stacking':
            self._initialize_stacking_models()
        elif self.config['ensemble_method'] == 'boosting':
            self._initialize_boosting_models()
        else:  # deep_ensemble
            self._initialize_deep_ensemble()
    
    def _initialize_stacking_models(self):
        """Initialize models for stacking"""
        # Initialize base models
        for _ in range(self.config['n_base_models']):
            model = MultimodalFusion(self.config['base_model_config'])
            model.to(self.config['device'])
            self.base_models.append(model)
        
        # Initialize meta model
        self.meta_model = nn.Sequential(
            nn.Linear(self.config['n_base_models'] * 2, 
                     self.config['meta_model_config']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.config['meta_model_config']['dropout_rate']),
            nn.Linear(self.config['meta_model_config']['hidden_dim'], 2)
        ).to(self.config['device'])
    
    def _initialize_boosting_models(self):
        """Initialize models for boosting"""
        self.base_models = []  # Will be filled during training
        self.model_weights = []  # Weights for each model
    
    def _initialize_deep_ensemble(self):
        """Initialize deep ensemble"""
        for _ in range(self.config['n_base_models']):
            model = MultimodalFusion(self.config['base_model_config'])
            model.to(self.config['device'])
            self.base_models.append(model)
    
    def train(self, data: Dict[str, torch.Tensor], 
              val_data: Optional[Dict[str, torch.Tensor]] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train ensemble
        Args:
            data: Training data
            val_data: Validation data
            **kwargs: Additional training parameters
        Returns:
            Training history
        """
        if self.config['ensemble_method'] == 'stacking':
            return self._train_stacking(data, val_data, **kwargs)
        elif self.config['ensemble_method'] == 'boosting':
            return self._train_boosting(data, **kwargs)
        else:  # deep_ensemble
            return self._train_deep_ensemble(data, **kwargs)
    
    def _train_stacking(self, data: Dict[str, torch.Tensor],
                       val_data: Dict[str, torch.Tensor],
                       **kwargs) -> Dict[str, Any]:
        """Train stacking ensemble"""
        # Train base models
        base_histories = []
        base_predictions = []
        
        for i, model in enumerate(self.base_models):
            print(f"Training base model {i+1}/{len(self.base_models)}")
            history = model.train(data, **kwargs)
            base_histories.append(history)
            
            # Get predictions for meta-model training
            with torch.no_grad():
                pred = model.predict(val_data)
                base_predictions.append(pred)
        
        # Prepare meta-features
        meta_features = torch.stack(base_predictions, dim=1)
        meta_features = meta_features.view(meta_features.size(0), -1)
        
        # Train meta-model
        print("Training meta-model...")
        meta_optimizer = torch.optim.Adam(
            self.meta_model.parameters(),
            lr=self.config['meta_model_config']['learning_rate']
        )
        criterion = nn.CrossEntropyLoss()
        
        meta_history = []
        for epoch in range(kwargs.get('meta_epochs', 50)):
            meta_optimizer.zero_grad()
            meta_output = self.meta_model(meta_features)
            loss = criterion(meta_output, val_data['labels'])
            loss.backward()
            meta_optimizer.step()
            meta_history.append({'epoch': epoch, 'loss': loss.item()})
        
        return {
            'base_histories': base_histories,
            'meta_history': meta_history
        }
    
    def _train_boosting(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Train boosting ensemble"""
        n_samples = len(next(iter(data.values())))
        weights = torch.ones(n_samples) / n_samples
        histories = []
        
        for round in range(self.config['boosting_config']['n_rounds']):
            print(f"Boosting round {round+1}/{self.config['boosting_config']['n_rounds']}")
            
            # Sample data based on weights
            indices = torch.multinomial(
                weights,
                int(n_samples * self.config['boosting_config']['subsample_ratio']),
                replacement=True
            )
            round_data = {k: v[indices] for k, v in data.items()}
            
            # Train new model
            model = MultimodalFusion(self.config['base_model_config'])
            model.to(self.config['device'])
            history = model.train(round_data, **kwargs)
            
            # Get predictions and update weights
            with torch.no_grad():
                predictions = model.predict(data)
                pred_labels = predictions.argmax(dim=1)
                errors = (pred_labels != data['labels']).float()
                
                # Calculate model weight
                error_rate = (errors * weights).sum() / weights.sum()
                model_weight = 0.5 * torch.log((1 - error_rate) / error_rate)
                
                # Update sample weights
                weights *= torch.exp(model_weight * errors)
                weights /= weights.sum()
            
            self.base_models.append(model)
            self.model_weights.append(model_weight)
            histories.append(history)
        
        return {'histories': histories, 'model_weights': self.model_weights}
    
    def _train_deep_ensemble(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """Train deep ensemble"""
        histories = []
        
        for i, model in enumerate(self.base_models):
            print(f"Training model {i+1}/{len(self.base_models)}")
            # Initialize with different random weights
            model.apply(self._init_weights)
            history = model.train(data, **kwargs)
            histories.append(history)
        
        return {'histories': histories}
    
    def _init_weights(self, module):
        """Initialize weights with random values"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
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
        if self.config['ensemble_method'] == 'stacking':
            return self._predict_stacking(batch, return_uncertainty)
        elif self.config['ensemble_method'] == 'boosting':
            return self._predict_boosting(batch, return_uncertainty)
        else:  # deep_ensemble
            return self._predict_deep_ensemble(batch, return_uncertainty)
    
    def _predict_stacking(self, batch: Dict[str, torch.Tensor],
                         return_uncertainty: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Make predictions using stacking"""
        base_predictions = []
        
        # Get base model predictions
        for model in self.base_models:
            with torch.no_grad():
                pred = model.predict(batch)
                base_predictions.append(pred)
        
        # Prepare meta-features
        meta_features = torch.stack(base_predictions, dim=1)
        meta_features = meta_features.view(meta_features.size(0), -1)
        
        # Get meta-model predictions
        with torch.no_grad():
            predictions = self.meta_model(meta_features)
        
        if not return_uncertainty:
            return predictions, None
        
        # Calculate uncertainty using prediction variance
        uncertainties = torch.stack(base_predictions).var(0).mean(-1)
        return predictions, uncertainties
    
    def _predict_boosting(self, batch: Dict[str, torch.Tensor],
                         return_uncertainty: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Make predictions using boosting"""
        predictions = []
        weights = torch.tensor(self.model_weights).to(self.config['device'])
        
        # Get weighted predictions from all models
        for model in self.base_models:
            with torch.no_grad():
                pred = model.predict(batch)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        weighted_sum = (predictions * weights.view(-1, 1, 1)).sum(0)
        ensemble_pred = torch.softmax(weighted_sum, dim=-1)
        
        if not return_uncertainty:
            return ensemble_pred, None
        
        # Calculate uncertainty using weighted variance
        uncertainties = (predictions.var(0) * weights.view(-1, 1, 1)).sum(0).mean(-1)
        return ensemble_pred, uncertainties
    
    def _predict_deep_ensemble(self, batch: Dict[str, torch.Tensor],
                             return_uncertainty: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Make predictions using deep ensemble"""
        predictions = []
        
        # Get predictions from all models
        for model in self.base_models:
            with torch.no_grad():
                pred = model.predict(batch)
                predictions.append(pred)
        
        # Calculate ensemble prediction
        predictions = torch.stack(predictions)
        ensemble_pred = predictions.mean(0)
        
        if not return_uncertainty:
            return ensemble_pred, None
        
        if self.config['uncertainty_config']['method'] == 'evidential':
            # Implement evidential uncertainty
            evidence = torch.exp(predictions)
            uncertainty = self.config['uncertainty_config']['prior_scale'] / evidence.sum(dim=-1).mean(0)
        elif self.config['uncertainty_config']['method'] == 'bayesian':
            # Implement Bayesian uncertainty
            uncertainty = self._estimate_bayesian_uncertainty(predictions)
        else:  # ensemble
            # Use ensemble variance
            uncertainty = predictions.var(0).mean(-1)
        
        return ensemble_pred, uncertainty
    
    def _estimate_bayesian_uncertainty(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty using Bayesian approximation
        Args:
            predictions: Model predictions
        Returns:
            Uncertainty estimates
        """
        # Calculate aleatoric uncertainty (data uncertainty)
        probs = torch.softmax(predictions, dim=-1)
        aleatoric = -(probs * torch.log(probs + 1e-10)).sum(-1).mean(0)
        
        # Calculate epistemic uncertainty (model uncertainty)
        epistemic = predictions.var(0).mean(-1)
        
        # Combine uncertainties
        return aleatoric + epistemic
    
    def save_ensemble(self, path: str):
        """Save ensemble models"""
        # Save base models
        for i, model in enumerate(self.base_models):
            model_path = f"{path}_base_{i}.pt"
            model.save(model_path)
        
        # Save meta model if using stacking
        if self.config['ensemble_method'] == 'stacking':
            meta_path = f"{path}_meta.pt"
            torch.save(self.meta_model.state_dict(), meta_path)
        
        # Save model weights if using boosting
        if self.config['ensemble_method'] == 'boosting':
            weights_path = f"{path}_weights.pt"
            torch.save(self.model_weights, weights_path)
        
        # Save configuration
        config_path = f"{path}_config.pt"
        torch.save(self.config, config_path)
    
    def load_ensemble(self, path: str):
        """Load ensemble models"""
        # Load configuration
        config_path = f"{path}_config.pt"
        self.config = torch.load(config_path)
        
        # Initialize models based on loaded configuration
        self._initialize_models()
        
        # Load base models
        for i in range(len(self.base_models)):
            model_path = f"{path}_base_{i}.pt"
            self.base_models[i].load(model_path)
        
        # Load meta model if using stacking
        if self.config['ensemble_method'] == 'stacking':
            meta_path = f"{path}_meta.pt"
            self.meta_model.load_state_dict(torch.load(meta_path))
        
        # Load model weights if using boosting
        if self.config['ensemble_method'] == 'boosting':
            weights_path = f"{path}_weights.pt"
            self.model_weights = torch.load(weights_path) 