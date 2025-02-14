import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ..core.base import BeaconBase
from ..models.multimodal import MultimodalFusion

class MultimodalRobustnessAnalyzer(BeaconBase):
    """Analyzer for multimodal model robustness"""
    
    def __init__(self, model: MultimodalFusion, config: Dict[str, Any]):
        """
        Initialize analyzer
        Args:
            model: Trained multimodal fusion model
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model = model
        self.device = next(model.parameters()).device
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'adversarial': {
                'epsilon': 0.1,
                'alpha': 0.01,
                'num_steps': 10,
                'random_start': True
            },
            'sensitivity': {
                'noise_types': ['gaussian', 'uniform', 'salt_and_pepper'],
                'noise_levels': [0.01, 0.05, 0.1],
                'n_samples': 100
            },
            'feature_ablation': {
                'n_features': 10,
                'strategy': 'importance'  # or 'random'
            },
            'cross_modality': {
                'enabled': True,
                'n_permutations': 100
            }
        }
    
    def generate_adversarial_examples(self, batch: Dict[str, torch.Tensor],
                                    target: Optional[torch.Tensor] = None,
                                    targeted: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate adversarial examples using PGD attack
        Args:
            batch: Input batch
            target: Target labels (optional)
            targeted: Whether to perform targeted attack
        Returns:
            Dictionary of adversarial examples for each modality
        """
        config = self.config['adversarial']
        adversarial_batch = {}
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get original predictions if target not provided
        if target is None:
            with torch.no_grad():
                predictions = self.model(batch)
                target = predictions.argmax(dim=1)
        
        # Initialize adversarial examples
        for modality, data in batch.items():
            if isinstance(data, torch.Tensor) and modality != 'target':
                if config['random_start']:
                    noise = torch.rand_like(data) * 2 * config['epsilon'] - config['epsilon']
                    adversarial_batch[modality] = data + noise
                else:
                    adversarial_batch[modality] = data.clone()
        
        # PGD attack
        for _ in range(config['num_steps']):
            # Forward pass
            for modality in adversarial_batch:
                adversarial_batch[modality].requires_grad_(True)
            
            outputs = self.model(adversarial_batch)
            loss = nn.CrossEntropyLoss()(outputs, target)
            
            if targeted:
                loss = -loss
            
            # Backward pass
            loss.backward()
            
            # Update adversarial examples
            with torch.no_grad():
                for modality, data in adversarial_batch.items():
                    grad = data.grad.sign()
                    data.add_(config['alpha'] * grad)
                    
                    # Project back to epsilon ball
                    delta = data - batch[modality]
                    delta.clamp_(-config['epsilon'], config['epsilon'])
                    data.copy_(batch[modality] + delta)
                    data.grad.zero_()
        
        return adversarial_batch
    
    def analyze_sensitivity(self, batch: Dict[str, torch.Tensor],
                          noise_type: str = 'gaussian') -> Dict[str, np.ndarray]:
        """
        Analyze model sensitivity to input perturbations
        Args:
            batch: Input batch
            noise_type: Type of noise to add
        Returns:
            Dictionary of sensitivity scores for each modality
        """
        config = self.config['sensitivity']
        sensitivity = {}
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get original predictions
        with torch.no_grad():
            original_pred = self.model(batch)
        
        for noise_level in config['noise_levels']:
            for modality, data in batch.items():
                if isinstance(data, torch.Tensor) and modality != 'target':
                    predictions = []
                    
                    for _ in range(config['n_samples']):
                        # Add noise
                        if noise_type == 'gaussian':
                            noise = torch.randn_like(data) * noise_level
                        elif noise_type == 'uniform':
                            noise = (torch.rand_like(data) * 2 - 1) * noise_level
                        elif noise_type == 'salt_and_pepper':
                            mask = torch.rand_like(data) < noise_level
                            noise = torch.randn_like(data) * mask
                        else:
                            raise ValueError(f"Unknown noise type: {noise_type}")
                        
                        noisy_batch = {k: v.clone() for k, v in batch.items()}
                        noisy_batch[modality] = data + noise
                        
                        # Get predictions
                        with torch.no_grad():
                            pred = self.model(noisy_batch)
                            predictions.append(pred)
                    
                    # Calculate sensitivity
                    predictions = torch.stack(predictions)
                    sensitivity[f"{modality}_{noise_level}"] = (
                        (predictions != original_pred).float().mean().cpu().numpy()
                    )
        
        return sensitivity
    
    def analyze_feature_importance(self, batch: Dict[str, torch.Tensor]
                                 ) -> Dict[str, torch.Tensor]:
        """
        Analyze feature importance through ablation
        Args:
            batch: Input batch
        Returns:
            Dictionary of feature importance scores
        """
        config = self.config['feature_ablation']
        importance = {}
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get original predictions
        with torch.no_grad():
            original_pred = self.model(batch)
        
        for modality, data in batch.items():
            if isinstance(data, torch.Tensor) and modality != 'target':
                n_features = data.shape[1]  # Assume feature dimension is 1
                scores = torch.zeros(n_features, device=self.device)
                
                # Ablate each feature
                for i in range(n_features):
                    ablated_batch = {k: v.clone() for k, v in batch.items()}
                    ablated_batch[modality][:, i] = 0
                    
                    with torch.no_grad():
                        pred = self.model(ablated_batch)
                        scores[i] = nn.functional.kl_div(
                            original_pred.log_softmax(dim=1),
                            pred.softmax(dim=1),
                            reduction='batchmean'
                        )
                
                importance[modality] = scores
        
        return importance
    
    def analyze_cross_modality_robustness(self, batch: Dict[str, torch.Tensor]
                                        ) -> Dict[str, float]:
        """
        Analyze robustness across modalities
        Args:
            batch: Input batch
        Returns:
            Dictionary of cross-modality robustness scores
        """
        if not self.config['cross_modality']['enabled']:
            return {}
        
        robustness = {}
        n_permutations = self.config['cross_modality']['n_permutations']
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get original predictions
        with torch.no_grad():
            original_pred = self.model(batch)
        
        modalities = [k for k, v in batch.items() 
                     if isinstance(v, torch.Tensor) and k != 'target']
        
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                consistency = 0
                
                for _ in range(n_permutations):
                    # Permute features between modalities
                    permuted_batch = {k: v.clone() for k, v in batch.items()}
                    idx = torch.randperm(batch[mod1].size(0))
                    permuted_batch[mod1] = batch[mod1][idx]
                    permuted_batch[mod2] = batch[mod2][idx]
                    
                    with torch.no_grad():
                        pred = self.model(permuted_batch)
                        consistency += (pred.argmax(dim=1) == 
                                     original_pred.argmax(dim=1)).float().mean()
                
                robustness[f"{mod1}_{mod2}"] = (consistency / n_permutations).cpu().item()
        
        return robustness
    
    def get_robustness_metrics(self, batch: Dict[str, torch.Tensor]
                              ) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive robustness metrics
        Args:
            batch: Input batch
        Returns:
            Dictionary of robustness metrics
        """
        metrics = {}
        
        # Adversarial robustness
        adv_batch = self.generate_adversarial_examples(batch)
        with torch.no_grad():
            original_pred = self.model(batch)
            adv_pred = self.model(adv_batch)
        
        metrics['adversarial'] = {
            'accuracy_drop': (
                (original_pred.argmax(dim=1) != adv_pred.argmax(dim=1))
                .float().mean().cpu().item()
            ),
            'confidence_drop': (
                original_pred.softmax(dim=1).max(dim=1)[0].mean() -
                adv_pred.softmax(dim=1).max(dim=1)[0].mean()
            ).cpu().item()
        }
        
        # Sensitivity analysis
        for noise_type in self.config['sensitivity']['noise_types']:
            metrics[f'sensitivity_{noise_type}'] = self.analyze_sensitivity(
                batch, noise_type
            )
        
        # Feature importance
        importance = self.analyze_feature_importance(batch)
        metrics['feature_importance'] = {
            modality: scores.mean().cpu().item()
            for modality, scores in importance.items()
        }
        
        # Cross-modality robustness
        metrics['cross_modality'] = self.analyze_cross_modality_robustness(batch)
        
        return metrics 