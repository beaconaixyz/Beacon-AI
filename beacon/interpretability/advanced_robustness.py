import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from ..core.base import BeaconBase
from ..models.multimodal import MultimodalFusion

class AdvancedRobustnessAnalyzer(BeaconBase):
    """Advanced analyzer for multimodal model robustness"""
    
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
            'fgsm': {
                'epsilon': 0.1
            },
            'pgd': {
                'epsilon': 0.1,
                'alpha': 0.01,
                'num_steps': 10,
                'random_start': True
            },
            'carlini_wagner': {
                'confidence': 0,
                'learning_rate': 0.01,
                'num_steps': 100,
                'binary_search_steps': 9
            },
            'deepfool': {
                'num_steps': 50,
                'overshoot': 0.02
            },
            'boundary': {
                'num_steps': 100,
                'spherical_step': 0.01,
                'source_step': 0.01,
                'step_adaptation': 1.5
            },
            'universal': {
                'delta': 0.2,
                'max_iter': 100,
                'num_samples': 1000
            },
            'analysis': {
                'lipschitz_estimation': {
                    'num_samples': 1000,
                    'radius': 0.1
                },
                'decision_boundary': {
                    'num_points': 100,
                    'radius': 1.0
                },
                'gradient_analysis': {
                    'num_samples': 100,
                    'step_size': 0.01
                }
            }
        }
    
    def fgsm_attack(self, batch: Dict[str, torch.Tensor],
                   target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Fast Gradient Sign Method attack
        Args:
            batch: Input batch
            target: Target labels (optional)
        Returns:
            Dictionary of adversarial examples
        """
        config = self.config['fgsm']
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get original predictions if target not provided
        if target is None:
            with torch.no_grad():
                predictions = self.model(batch)
                target = predictions.argmax(dim=1)
        
        adversarial_batch = {}
        for modality, data in batch.items():
            if isinstance(data, torch.Tensor) and modality != 'target':
                data.requires_grad_(True)
                adversarial_batch[modality] = data
        
        # Forward pass
        outputs = self.model(adversarial_batch)
        loss = nn.CrossEntropyLoss()(outputs, target)
        
        # Backward pass
        loss.backward()
        
        # Generate adversarial examples
        with torch.no_grad():
            for modality, data in adversarial_batch.items():
                perturbation = config['epsilon'] * data.grad.sign()
                adversarial_batch[modality] = data + perturbation
                adversarial_batch[modality].clamp_(0, 1)  # Ensure valid range
        
        return adversarial_batch
    
    def carlini_wagner_attack(self, batch: Dict[str, torch.Tensor],
                            target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Carlini & Wagner L2 attack
        Args:
            batch: Input batch
            target: Target labels (optional)
        Returns:
            Dictionary of adversarial examples
        """
        config = self.config['carlini_wagner']
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        if target is None:
            with torch.no_grad():
                predictions = self.model(batch)
                target = predictions.argmax(dim=1)
        
        def tanh_space(x: torch.Tensor) -> torch.Tensor:
            return 1/2 * (torch.tanh(x) + 1)
        
        def inverse_tanh_space(x: torch.Tensor) -> torch.Tensor:
            return torch.atanh(2 * x - 1)
        
        # Initialize attack
        adversarial_batch = {}
        w = {modality: inverse_tanh_space(data).detach().requires_grad_(True)
             for modality, data in batch.items()
             if isinstance(data, torch.Tensor) and modality != 'target'}
        
        optimizer = torch.optim.Adam(w.values(), lr=config['learning_rate'])
        
        best_dist = {modality: 1e10 * torch.ones_like(data)
                    for modality, data in batch.items()
                    if isinstance(data, torch.Tensor) and modality != 'target'}
        best_attack = {modality: data.clone()
                      for modality, data in batch.items()
                      if isinstance(data, torch.Tensor) and modality != 'target'}
        
        # Binary search for c
        c = torch.ones(len(target), device=self.device)
        c_high = 1e10 * torch.ones_like(c)
        c_low = torch.zeros_like(c)
        
        for binary_step in range(config['binary_search_steps']):
            for iteration in range(config['num_steps']):
                optimizer.zero_grad()
                
                # Forward pass
                attack = {modality: tanh_space(w[modality])
                         for modality in w}
                outputs = self.model(attack)
                
                # Calculate loss
                l2_dist = sum(((attack[modality] - batch[modality])**2).view(len(target), -1).sum(1)
                             for modality in attack)
                
                real = torch.gather(outputs, 1, target.unsqueeze(1)).squeeze(1)
                other = torch.max(
                    torch.cat([
                        outputs[:, :target[0]],
                        outputs[:, target[0] + 1:]
                    ], dim=1),
                    dim=1
                )[0]
                
                loss = torch.mean(
                    c * torch.clamp(real - other + config['confidence'], min=0) +
                    l2_dist
                )
                
                loss.backward()
                optimizer.step()
                
                # Update best results
                for modality in attack:
                    mask = l2_dist < best_dist[modality]
                    best_dist[modality][mask] = l2_dist[mask]
                    best_attack[modality][mask] = attack[modality][mask]
            
            # Binary search step
            success = (outputs.argmax(dim=1) == target)
            c_high[success] = torch.min(c_high[success], c[success])
            c_low[~success] = torch.max(c_low[~success], c[~success])
            c = (c_high + c_low) / 2
        
        return best_attack
    
    def deepfool_attack(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        DeepFool attack
        Args:
            batch: Input batch
        Returns:
            Dictionary of adversarial examples
        """
        config = self.config['deepfool']
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        adversarial_batch = {
            modality: data.clone()
            for modality, data in batch.items()
            if isinstance(data, torch.Tensor) and modality != 'target'
        }
        
        with torch.no_grad():
            outputs = self.model(batch)
            num_classes = outputs.shape[1]
        
        for _ in range(config['num_steps']):
            for modality in adversarial_batch:
                adversarial_batch[modality].requires_grad_(True)
            
            outputs = self.model(adversarial_batch)
            current_class = outputs.argmax(dim=1)
            
            # Calculate gradients for all classes
            gradients = []
            for i in range(num_classes):
                if i == current_class:
                    continue
                
                loss = outputs[:, i] - outputs[:, current_class]
                grad_i = torch.autograd.grad(loss.sum(), 
                                          [adversarial_batch[m] for m in adversarial_batch],
                                          retain_graph=True)
                gradients.append(grad_i)
            
            # Find closest hyperplane
            with torch.no_grad():
                for modality in adversarial_batch:
                    min_perturbation = None
                    min_dist = float('inf')
                    
                    for i, grad_i in enumerate(gradients):
                        w_i = grad_i[list(adversarial_batch.keys()).index(modality)]
                        f_i = outputs[:, i] - outputs[:, current_class]
                        dist_i = torch.abs(f_i) / (w_i.view(len(w_i), -1).norm(dim=1) + 1e-8)
                        
                        if dist_i < min_dist:
                            min_dist = dist_i
                            min_perturbation = w_i
                    
                    # Update adversarial example
                    if min_perturbation is not None:
                        perturbation = (min_perturbation + 1e-8) / \
                                     (min_perturbation.view(len(min_perturbation), -1).norm(dim=1) + 1e-8).view(-1, 1, 1, 1)
                        adversarial_batch[modality] += config['overshoot'] * perturbation
                        adversarial_batch[modality].clamp_(0, 1)
        
        return adversarial_batch
    
    def estimate_lipschitz_constant(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Estimate Lipschitz constant for each modality
        Args:
            batch: Input batch
        Returns:
            Dictionary of Lipschitz constants
        """
        config = self.config['analysis']['lipschitz_estimation']
        batch = {k: v.to(self.device) for k, v in batch.items()}
        lipschitz = {}
        
        for modality, data in batch.items():
            if isinstance(data, torch.Tensor) and modality != 'target':
                max_ratio = 0
                
                for _ in range(config['num_samples']):
                    # Generate random perturbation
                    perturbation = torch.randn_like(data) * config['radius']
                    perturbed_batch = {k: v.clone() for k, v in batch.items()}
                    perturbed_batch[modality] = data + perturbation
                    
                    with torch.no_grad():
                        original_output = self.model(batch)
                        perturbed_output = self.model(perturbed_batch)
                        
                        output_diff = (perturbed_output - original_output).norm()
                        input_diff = perturbation.norm()
                        ratio = output_diff / (input_diff + 1e-8)
                        max_ratio = max(max_ratio, ratio.item())
                
                lipschitz[modality] = max_ratio
        
        return lipschitz
    
    def analyze_decision_boundary(self, batch: Dict[str, torch.Tensor]
                                ) -> Dict[str, np.ndarray]:
        """
        Analyze decision boundary around samples
        Args:
            batch: Input batch
        Returns:
            Dictionary of decision boundary analysis results
        """
        config = self.config['analysis']['decision_boundary']
        batch = {k: v.to(self.device) for k, v in batch.items()}
        results = {}
        
        # Get original predictions
        with torch.no_grad():
            original_pred = self.model(batch)
            original_class = original_pred.argmax(dim=1)
        
        for modality, data in batch.items():
            if isinstance(data, torch.Tensor) and modality != 'target':
                distances = []
                directions = []
                
                # Sample points in different directions
                for _ in range(config['num_points']):
                    direction = torch.randn_like(data)
                    direction /= direction.view(len(direction), -1).norm(dim=1).view(-1, 1, 1, 1)
                    
                    # Binary search for decision boundary
                    low = torch.zeros(len(data), device=self.device)
                    high = config['radius'] * torch.ones_like(low)
                    
                    for _ in range(10):  # Binary search steps
                        mid = (low + high) / 2
                        perturbed_batch = {k: v.clone() for k, v in batch.items()}
                        perturbed_batch[modality] = data + mid.view(-1, 1, 1, 1) * direction
                        
                        with torch.no_grad():
                            pred = self.model(perturbed_batch)
                            crossed = (pred.argmax(dim=1) != original_class)
                            
                            high[crossed] = mid[crossed]
                            low[~crossed] = mid[~crossed]
                    
                    distances.append(((low + high) / 2).cpu().numpy())
                    directions.append(direction.cpu().numpy())
                
                results[modality] = {
                    'distances': np.array(distances),
                    'directions': np.array(directions)
                }
        
        return results
    
    def analyze_gradient_landscape(self, batch: Dict[str, torch.Tensor]
                                 ) -> Dict[str, np.ndarray]:
        """
        Analyze gradient landscape around samples
        Args:
            batch: Input batch
        Returns:
            Dictionary of gradient landscape analysis results
        """
        config = self.config['analysis']['gradient_analysis']
        batch = {k: v.to(self.device) for k, v in batch.items()}
        results = {}
        
        for modality, data in batch.items():
            if isinstance(data, torch.Tensor) and modality != 'target':
                gradients = []
                points = []
                
                for _ in range(config['num_samples']):
                    # Sample random point around original data
                    noise = torch.randn_like(data) * config['step_size']
                    perturbed_batch = {k: v.clone() for k, v in batch.items()}
                    perturbed_batch[modality] = data + noise
                    perturbed_batch[modality].requires_grad_(True)
                    
                    # Calculate gradient
                    outputs = self.model(perturbed_batch)
                    loss = outputs.sum()  # Any loss function would work
                    grad = torch.autograd.grad(loss, perturbed_batch[modality])[0]
                    
                    gradients.append(grad.cpu().numpy())
                    points.append(perturbed_batch[modality].detach().cpu().numpy())
                
                results[modality] = {
                    'gradients': np.array(gradients),
                    'points': np.array(points)
                }
        
        return results 