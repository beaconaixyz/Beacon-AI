import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class AdaptiveLearning:
    """Handles adaptive learning rates and dynamic weight adjustment for ensemble models"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize adaptive learning mechanism
        
        Args:
            config: Configuration dictionary containing:
                - initial_lr: Initial learning rate
                - min_lr: Minimum learning rate
                - max_lr: Maximum learning rate
                - adaptation_window: Window size for performance tracking
                - momentum: Momentum factor for weight updates
                - beta1: Exponential decay rate for first moment
                - beta2: Exponential decay rate for second moment
                - epsilon: Small constant for numerical stability
        """
        self.config = self._get_default_config()
        self.config.update(config)
        
        self.learning_rates = {}
        self.momentum_buffer = {}
        self.first_moment = {}
        self.second_moment = {}
        self.step_count = 0
        
        self.performance_history = []
        self.lr_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'initial_lr': 0.001,
            'min_lr': 1e-6,
            'max_lr': 0.1,
            'adaptation_window': 50,
            'momentum': 0.9,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        }
    
    def initialize_learning_rates(self, modalities: List[str]):
        """Initialize learning rates for each modality
        
        Args:
            modalities: List of modality names
        """
        for modality in modalities:
            self.learning_rates[modality] = self.config['initial_lr']
            self.momentum_buffer[modality] = 0.0
            self.first_moment[modality] = 0.0
            self.second_moment[modality] = 0.0
    
    def update_learning_rates(self, 
                            performance_metrics: Dict[str, float],
                            gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update learning rates based on performance and gradients
        
        Args:
            performance_metrics: Dictionary of performance metrics for each modality
            gradients: Dictionary of gradients for each modality
            
        Returns:
            Dictionary of updated learning rates
        """
        self.step_count += 1
        
        # Store performance history
        self.performance_history.append(performance_metrics)
        if len(self.performance_history) > self.config['adaptation_window']:
            self.performance_history.pop(0)
        
        updated_lrs = {}
        for modality in self.learning_rates:
            # Calculate gradient statistics
            grad = gradients[modality]
            grad_norm = grad.norm().item()
            
            # Update moments
            self.first_moment[modality] = (
                self.config['beta1'] * self.first_moment[modality] +
                (1 - self.config['beta1']) * grad_norm
            )
            self.second_moment[modality] = (
                self.config['beta2'] * self.second_moment[modality] +
                (1 - self.config['beta2']) * (grad_norm ** 2)
            )
            
            # Bias correction
            first_moment_corrected = self.first_moment[modality] / (1 - self.config['beta1'] ** self.step_count)
            second_moment_corrected = self.second_moment[modality] / (1 - self.config['beta2'] ** self.step_count)
            
            # Calculate adaptive learning rate
            adaptive_factor = first_moment_corrected / (np.sqrt(second_moment_corrected) + self.config['epsilon'])
            
            # Get performance trend
            if len(self.performance_history) >= 2:
                current_perf = performance_metrics[modality]
                prev_perf = self.performance_history[-2][modality]
                perf_improvement = (current_perf - prev_perf) / abs(prev_perf + self.config['epsilon'])
            else:
                perf_improvement = 0.0
            
            # Update learning rate
            momentum = self.config['momentum']
            current_lr = self.learning_rates[modality]
            
            # Adjust learning rate based on performance and gradient statistics
            if perf_improvement > 0:
                # Increase learning rate if performance is improving
                lr_update = current_lr * (1 + 0.1 * perf_improvement)
            else:
                # Decrease learning rate if performance is degrading
                lr_update = current_lr * (1 + perf_improvement)
            
            # Apply momentum
            self.momentum_buffer[modality] = (
                momentum * self.momentum_buffer[modality] +
                (1 - momentum) * (lr_update - current_lr)
            )
            
            # Update learning rate with bounds
            new_lr = current_lr + self.momentum_buffer[modality]
            new_lr = np.clip(new_lr, self.config['min_lr'], self.config['max_lr'])
            
            updated_lrs[modality] = new_lr
            self.learning_rates[modality] = new_lr
        
        # Store learning rate history
        self.lr_history.append(updated_lrs)
        if len(self.lr_history) > self.config['adaptation_window']:
            self.lr_history.pop(0)
        
        return updated_lrs
    
    def get_learning_rate_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about learning rates
        
        Returns:
            Dictionary containing mean, std, min, max learning rates for each modality
        """
        stats = {}
        for modality in self.learning_rates:
            modality_lrs = [h[modality] for h in self.lr_history]
            stats[modality] = {
                'mean': np.mean(modality_lrs),
                'std': np.std(modality_lrs),
                'min': np.min(modality_lrs),
                'max': np.max(modality_lrs)
            }
        return stats
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about performance history
        
        Returns:
            Dictionary containing mean, std, min, max performance for each modality
        """
        stats = {}
        for modality in self.learning_rates:
            modality_perf = [h[modality] for h in self.performance_history]
            stats[modality] = {
                'mean': np.mean(modality_perf),
                'std': np.std(modality_perf),
                'min': np.min(modality_perf),
                'max': np.max(modality_perf)
            }
        return stats
    
    def reset(self):
        """Reset the adaptive learning mechanism"""
        self.learning_rates = {}
        self.momentum_buffer = {}
        self.first_moment = {}
        self.second_moment = {}
        self.step_count = 0
        self.performance_history = []
        self.lr_history = [] 