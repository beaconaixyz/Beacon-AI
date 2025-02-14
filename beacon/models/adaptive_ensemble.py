import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .advanced_ensemble import AdvancedEnsemble

class AdaptiveEnsemble(AdvancedEnsemble):
    """Adaptive ensemble learning with dynamic modality weighting"""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        config = super()._get_default_config()
        config.update({
            'adaptation_method': 'performance',  # performance, uncertainty, or gradient
            'adaptation_frequency': 10,  # Update weights every N steps
            'adaptation_window': 100,  # Window size for performance tracking
            'min_weight': 0.1,  # Minimum weight for any modality
            'temperature': 1.0,  # Temperature for softmax weight normalization
            'modality_specific_learning': True,  # Whether to use modality-specific learning rates
            'weight_decay_factor': 0.99,  # Decay factor for historical weights
            'uncertainty_threshold': 0.8,  # Threshold for uncertainty-based adaptation
            'gradient_threshold': 0.1,  # Threshold for gradient-based adaptation
        })
        return config
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive ensemble"""
        super().__init__(config)
        self.modality_weights = None
        self.performance_history = {
            'image': [],
            'sequence': [],
            'clinical': []
        }
        self.uncertainty_history = {
            'image': [],
            'sequence': [],
            'clinical': []
        }
        self.gradient_history = {
            'image': [],
            'sequence': [],
            'clinical': []
        }
        self.step_counter = 0
        
    def _initialize_weights(self):
        """Initialize modality weights"""
        self.modality_weights = {
            'image': torch.tensor(1.0),
            'sequence': torch.tensor(1.0),
            'clinical': torch.tensor(1.0)
        }
    
    def update_weights(self, batch: Dict[str, torch.Tensor], 
                      labels: torch.Tensor) -> Dict[str, float]:
        """Update modality weights based on performance"""
        self.step_counter += 1
        
        if self.step_counter % self.config['adaptation_frequency'] != 0:
            return self.modality_weights
        
        if self.config['adaptation_method'] == 'performance':
            weights = self._update_weights_by_performance(batch, labels)
        elif self.config['adaptation_method'] == 'uncertainty':
            weights = self._update_weights_by_uncertainty(batch)
        else:  # gradient
            weights = self._update_weights_by_gradient(batch, labels)
            
        # Apply weight decay
        for modality in weights:
            historical_weight = self.modality_weights[modality]
            current_weight = weights[modality]
            weights[modality] = (self.config['weight_decay_factor'] * historical_weight + 
                               (1 - self.config['weight_decay_factor']) * current_weight)
        
        # Ensure minimum weight
        min_weight = self.config['min_weight']
        for modality in weights:
            weights[modality] = max(weights[modality], min_weight)
        
        # Normalize weights
        total_weight = sum(weights.values())
        for modality in weights:
            weights[modality] /= total_weight
            
        self.modality_weights = weights
        return weights
    
    def _update_weights_by_performance(self, batch: Dict[str, torch.Tensor],
                                     labels: torch.Tensor) -> Dict[str, float]:
        """Update weights based on individual modality performance"""
        weights = {}
        
        for modality in self.modality_weights:
            # Get predictions using only this modality
            single_modal_batch = {modality: batch[modality]}
            predictions = self._predict_single_modality(single_modal_batch)
            
            # Calculate performance (accuracy for classification)
            accuracy = (predictions.argmax(dim=1) == labels).float().mean().item()
            self.performance_history[modality].append(accuracy)
            
            # Keep only recent history
            if len(self.performance_history[modality]) > self.config['adaptation_window']:
                self.performance_history[modality].pop(0)
            
            # Calculate average performance
            avg_performance = np.mean(self.performance_history[modality])
            weights[modality] = avg_performance
        
        return weights
    
    def _update_weights_by_uncertainty(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update weights based on prediction uncertainty"""
        weights = {}
        
        for modality in self.modality_weights:
            # Get predictions and uncertainty for this modality
            single_modal_batch = {modality: batch[modality]}
            _, uncertainty = self._predict_single_modality(single_modal_batch, return_uncertainty=True)
            
            # Calculate average uncertainty
            avg_uncertainty = uncertainty.mean().item()
            self.uncertainty_history[modality].append(avg_uncertainty)
            
            # Keep only recent history
            if len(self.uncertainty_history[modality]) > self.config['adaptation_window']:
                self.uncertainty_history[modality].pop(0)
            
            # Convert uncertainty to weight (lower uncertainty -> higher weight)
            avg_uncertainty = np.mean(self.uncertainty_history[modality])
            weights[modality] = 1.0 / (avg_uncertainty + 1e-6)
        
        return weights
    
    def _update_weights_by_gradient(self, batch: Dict[str, torch.Tensor],
                                  labels: torch.Tensor) -> Dict[str, float]:
        """Update weights based on gradient magnitudes"""
        weights = {}
        
        for modality in self.modality_weights:
            # Calculate gradients for this modality
            single_modal_batch = {modality: batch[modality]}
            grad_magnitude = self._compute_gradient_magnitude(single_modal_batch, labels)
            
            self.gradient_history[modality].append(grad_magnitude)
            
            # Keep only recent history
            if len(self.gradient_history[modality]) > self.config['adaptation_window']:
                self.gradient_history[modality].pop(0)
            
            # Calculate average gradient magnitude
            avg_gradient = np.mean(self.gradient_history[modality])
            weights[modality] = avg_gradient
        
        return weights
    
    def _compute_gradient_magnitude(self, batch: Dict[str, torch.Tensor],
                                  labels: torch.Tensor) -> float:
        """Compute gradient magnitude for a single modality"""
        # Enable gradient computation
        for param in self.model.parameters():
            param.requires_grad = True
            
        # Forward pass
        predictions = self._predict_single_modality(batch)
        loss = nn.CrossEntropyLoss()(predictions, labels)
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient magnitude
        total_grad = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad += param.grad.norm().item()
                
        # Reset gradients
        for param in self.model.parameters():
            param.grad = None
            param.requires_grad = False
            
        return total_grad
    
    def _predict_single_modality(self, batch: Dict[str, torch.Tensor],
                               return_uncertainty: bool = False) -> torch.Tensor:
        """Make prediction using a single modality"""
        if self.config['ensemble_method'] == 'stacking':
            return self._predict_stacking(batch, return_uncertainty)
        elif self.config['ensemble_method'] == 'boosting':
            return self._predict_boosting(batch, return_uncertainty)
        else:  # deep ensemble
            return self._predict_deep_ensemble(batch, return_uncertainty)
    
    def train(self, data: Dict[str, torch.Tensor],
              val_data: Optional[Dict[str, torch.Tensor]] = None,
              **kwargs) -> Dict[str, Any]:
        """Train the adaptive ensemble"""
        if self.modality_weights is None:
            self._initialize_weights()
            
        history = super().train(data, val_data, **kwargs)
        
        # Add modality weights to history
        history['modality_weights'] = {
            modality: self.modality_weights[modality].item()
            for modality in self.modality_weights
        }
        
        return history
    
    def predict(self, batch: Dict[str, torch.Tensor],
                return_uncertainty: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Make predictions with weighted modalities"""
        predictions = []
        uncertainties = []
        
        # Get predictions from each modality
        for modality in self.modality_weights:
            single_modal_batch = {modality: batch[modality]}
            pred, unc = super().predict(single_modal_batch, return_uncertainty=True)
            
            # Weight the predictions
            pred = pred * self.modality_weights[modality]
            predictions.append(pred)
            
            if return_uncertainty:
                unc = unc * self.modality_weights[modality]
                uncertainties.append(unc)
        
        # Combine predictions
        final_pred = torch.stack(predictions).sum(dim=0)
        
        if return_uncertainty:
            final_unc = torch.stack(uncertainties).sum(dim=0)
            return final_pred, final_unc
        
        return final_pred, None 