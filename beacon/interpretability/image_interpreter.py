import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from captum.attr import (
    IntegratedGradients,
    GuidedGradCam,
    Occlusion,
    GradientShap,
    DeepLift,
    NoiseTunnel,
    FeatureAblation
)
from ..core.base import BeaconBase
from ..models.medical_cnn import MedicalCNN

class ImageInterpreter(BeaconBase):
    """Interpreter for medical image analysis models"""
    
    def __init__(self, model: MedicalCNN, config: Dict[str, Any]):
        """
        Initialize interpreter
        Args:
            model: Model to interpret
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model = model
        self._initialize_interpreters()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'method': 'integrated_gradients',
            'n_steps': 50,
            'internal_batch_size': 32,
            'noise_tunnel_samples': 10,
            'feature_mask_shape': (7, 7),
            'occlusion_sliding_window_shapes': ((3, 3),),
            'baseline_type': 'zero',
            'baseline_value': 0.0
        }
    
    def _initialize_interpreters(self):
        """Initialize interpretation methods"""
        self.interpreters = {
            'integrated_gradients': IntegratedGradients(self.model),
            'guided_gradcam': GuidedGradCam(self.model, self.model.model.layer4),
            'occlusion': Occlusion(self.model),
            'gradient_shap': GradientShap(self.model),
            'deep_lift': DeepLift(self.model),
            'feature_ablation': FeatureAblation(self.model)
        }
    
    def _get_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get baseline tensor
        Args:
            input_tensor: Input tensor
        Returns:
            Baseline tensor
        """
        if self.config['baseline_type'] == 'zero':
            return torch.zeros_like(input_tensor)
        elif self.config['baseline_type'] == 'random':
            return torch.rand_like(input_tensor)
        elif self.config['baseline_type'] == 'mean':
            return torch.ones_like(input_tensor) * self.config['baseline_value']
        else:
            raise ValueError(f"Unsupported baseline type: {self.config['baseline_type']}")
    
    def interpret(self, input_tensor: torch.Tensor, target: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate interpretation
        Args:
            input_tensor: Input tensor
            target: Target class (optional)
        Returns:
            Dictionary of attribution maps
        """
        # Move input to device
        input_tensor = input_tensor.to(self.model.device)
        
        # Get interpreter
        method = self.config['method'].lower()
        if method not in self.interpreters:
            raise ValueError(f"Unsupported interpretation method: {method}")
        
        interpreter = self.interpreters[method]
        
        # Get baseline
        baseline = self._get_baseline(input_tensor)
        
        # Generate attribution
        if method == 'integrated_gradients':
            attribution = interpreter.attribute(
                input_tensor,
                target=target,
                n_steps=self.config['n_steps'],
                internal_batch_size=self.config['internal_batch_size'],
                baselines=baseline
            )
        
        elif method == 'guided_gradcam':
            attribution = interpreter.attribute(
                input_tensor,
                target=target
            )
        
        elif method == 'occlusion':
            attribution = interpreter.attribute(
                input_tensor,
                target=target,
                sliding_window_shapes=self.config['occlusion_sliding_window_shapes']
            )
        
        elif method == 'gradient_shap':
            attribution = interpreter.attribute(
                input_tensor,
                target=target,
                n_samples=self.config['noise_tunnel_samples'],
                baselines=baseline
            )
        
        elif method == 'deep_lift':
            attribution = interpreter.attribute(
                input_tensor,
                target=target,
                baselines=baseline
            )
        
        elif method == 'feature_ablation':
            attribution = interpreter.attribute(
                input_tensor,
                target=target,
                feature_mask=self._create_feature_mask(input_tensor)
            )
        
        # Apply noise tunnel if specified
        if self.config.get('use_noise_tunnel', False):
            nt = NoiseTunnel(interpreter)
            attribution = nt.attribute(
                input_tensor,
                target=target,
                nt_type='smoothgrad',
                n_samples=self.config['noise_tunnel_samples']
            )
        
        return {
            'attribution': attribution,
            'method': method
        }
    
    def _create_feature_mask(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Create feature mask for ablation
        Args:
            input_tensor: Input tensor
        Returns:
            Feature mask tensor
        """
        mask_shape = self.config['feature_mask_shape']
        if not isinstance(mask_shape, tuple) or len(mask_shape) != 2:
            raise ValueError("feature_mask_shape must be a tuple of length 2")
        
        h, w = input_tensor.shape[-2:]
        mask_h, mask_w = mask_shape
        
        if h % mask_h != 0 or w % mask_w != 0:
            raise ValueError("Input dimensions must be divisible by mask dimensions")
        
        mask = torch.zeros((h, w))
        block_h = h // mask_h
        block_w = w // mask_w
        
        for i in range(mask_h):
            for j in range(mask_w):
                mask[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = i * mask_w + j
        
        return mask.expand(input_tensor.shape[:-2] + mask.shape).to(input_tensor.device)
    
    def aggregate_attributions(self, attributions: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple attribution maps
        Args:
            attributions: List of attribution tensors
        Returns:
            Aggregated attribution tensor
        """
        if not attributions:
            raise ValueError("Empty attribution list")
        
        # Stack attributions
        stacked = torch.stack(attributions)
        
        # Normalize each attribution
        normalized = []
        for attr in attributions:
            attr_abs = torch.abs(attr)
            if attr_abs.sum() > 0:
                normalized.append(attr_abs / attr_abs.sum())
            else:
                normalized.append(attr_abs)
        
        # Average normalized attributions
        return torch.mean(torch.stack(normalized), dim=0)
    
    def get_interpretation_stats(self, attribution: torch.Tensor) -> Dict[str, float]:
        """
        Calculate interpretation statistics
        Args:
            attribution: Attribution tensor
        Returns:
            Dictionary of statistics
        """
        # Convert to numpy for calculations
        attr = attribution.cpu().numpy()
        
        # Calculate statistics
        stats = {
            'mean': float(np.mean(attr)),
            'std': float(np.std(attr)),
            'min': float(np.min(attr)),
            'max': float(np.max(attr)),
            'sparsity': float((attr == 0).mean()),
            'positive_ratio': float((attr > 0).mean()),
            'negative_ratio': float((attr < 0).mean())
        }
        
        # Calculate percentiles
        for p in [1, 5, 25, 50, 75, 95, 99]:
            stats[f'percentile_{p}'] = float(np.percentile(attr, p))
        
        return stats
    
    def compare_interpretations(self, input_tensor: torch.Tensor, 
                              methods: List[str],
                              target: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare different interpretation methods
        Args:
            input_tensor: Input tensor
            methods: List of methods to compare
            target: Target class (optional)
        Returns:
            Dictionary of results for each method
        """
        results = {}
        
        for method in methods:
            # Save current method
            original_method = self.config['method']
            self.config['method'] = method
            
            # Generate interpretation
            interpretation = self.interpret(input_tensor, target)
            
            # Calculate statistics
            stats = self.get_interpretation_stats(interpretation['attribution'])
            
            # Store results
            results[method] = {
                'attribution': interpretation['attribution'],
                'stats': stats
            }
            
            # Restore original method
            self.config['method'] = original_method
        
        return results 