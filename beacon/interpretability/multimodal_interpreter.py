import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GuidedGradCam,
    Occlusion,
    ShapleyValueSampling,
    LayerGradCam,
    NoiseTunnel
)
from ..core.base import BeaconBase
from ..models.multimodal import MultimodalFusion

class MultimodalInterpreter(BeaconBase):
    """Interpreter for multimodal fusion model"""
    
    def __init__(self, model: MultimodalFusion, config: Dict[str, Any]):
        """
        Initialize interpreter
        Args:
            model: Trained multimodal fusion model
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model = model
        self.device = next(model.parameters()).device
        self._setup_interpreters()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'methods': ['integrated_gradients', 'deep_lift', 'guided_gradcam', 
                       'occlusion', 'shap', 'layer_gradcam'],
            'integrated_gradients': {
                'n_steps': 50,
                'internal_batch_size': 32
            },
            'deep_lift': {
                'multiply_by_inputs': True
            },
            'guided_gradcam': {
                'abs': True
            },
            'occlusion': {
                'sliding_window_shapes': {
                    'image': (1, 8, 8),
                    'clinical': (4,),
                    'genomic': (10,)
                },
                'strides': {
                    'image': (1, 4, 4),
                    'clinical': (2,),
                    'genomic': (5,)
                }
            },
            'shap': {
                'n_samples': 100,
                'batch_size': 32
            },
            'layer_gradcam': {
                'layer_names': ['image_model.conv1', 'genomic_model.conv1']
            },
            'noise_tunnel': {
                'n_samples': 10,
                'stdevs': 0.1,
                'nt_type': 'smoothgrad'
            }
        }
    
    def _setup_interpreters(self):
        """Initialize interpretation methods"""
        self.interpreters = {}
        
        if 'integrated_gradients' in self.config['methods']:
            self.interpreters['integrated_gradients'] = IntegratedGradients(
                self.model.forward_wrapper
            )
        
        if 'deep_lift' in self.config['methods']:
            self.interpreters['deep_lift'] = DeepLift(
                self.model.forward_wrapper
            )
        
        if 'guided_gradcam' in self.config['methods']:
            self.interpreters['guided_gradcam'] = GuidedGradCam(
                self.model.forward_wrapper,
                self.model.model.image_model.conv1
            )
        
        if 'occlusion' in self.config['methods']:
            self.interpreters['occlusion'] = Occlusion(
                self.model.forward_wrapper
            )
        
        if 'shap' in self.config['methods']:
            self.interpreters['shap'] = ShapleyValueSampling(
                self.model.forward_wrapper
            )
        
        if 'layer_gradcam' in self.config['methods']:
            self.interpreters['layer_gradcam'] = {}
            for layer_name in self.config['layer_gradcam']['layer_names']:
                layer = self._get_layer(layer_name)
                if layer is not None:
                    self.interpreters['layer_gradcam'][layer_name] = LayerGradCam(
                        self.model.forward_wrapper,
                        layer
                    )
    
    def _get_layer(self, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name"""
        names = layer_name.split('.')
        current = self.model.model
        for name in names:
            if hasattr(current, name):
                current = getattr(current, name)
            else:
                return None
        return current
    
    def interpret(self, batch: Dict[str, torch.Tensor], 
                 target: Optional[torch.Tensor] = None,
                 method: str = 'integrated_gradients') -> Dict[str, torch.Tensor]:
        """
        Interpret model predictions
        Args:
            batch: Input batch
            target: Target class (optional)
            method: Interpretation method
        Returns:
            Dictionary of attributions for each modality
        """
        if method not in self.interpreters:
            raise ValueError(f"Unknown interpretation method: {method}")
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get predictions if target not provided
        if target is None:
            with torch.no_grad():
                predictions = self.model(batch)
                target = predictions.argmax(dim=1)
        
        # Get attributions based on method
        if method == 'integrated_gradients':
            return self._interpret_integrated_gradients(batch, target)
        elif method == 'deep_lift':
            return self._interpret_deep_lift(batch, target)
        elif method == 'guided_gradcam':
            return self._interpret_guided_gradcam(batch, target)
        elif method == 'occlusion':
            return self._interpret_occlusion(batch, target)
        elif method == 'shap':
            return self._interpret_shap(batch, target)
        elif method == 'layer_gradcam':
            return self._interpret_layer_gradcam(batch, target)
    
    def _interpret_integrated_gradients(self, batch: Dict[str, torch.Tensor],
                                      target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply Integrated Gradients"""
        attributions = {}
        config = self.config['integrated_gradients']
        
        for modality in ['image', 'clinical', 'genomic']:
            if modality in batch:
                attr = self.interpreters['integrated_gradients'].attribute(
                    batch[modality],
                    target=target,
                    n_steps=config['n_steps'],
                    internal_batch_size=config['internal_batch_size']
                )
                attributions[modality] = attr
        
        return attributions
    
    def _interpret_deep_lift(self, batch: Dict[str, torch.Tensor],
                           target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply DeepLIFT"""
        attributions = {}
        config = self.config['deep_lift']
        
        for modality in ['image', 'clinical', 'genomic']:
            if modality in batch:
                attr = self.interpreters['deep_lift'].attribute(
                    batch[modality],
                    target=target,
                    multiply_by_inputs=config['multiply_by_inputs']
                )
                attributions[modality] = attr
        
        return attributions
    
    def _interpret_guided_gradcam(self, batch: Dict[str, torch.Tensor],
                                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply Guided GradCAM"""
        if 'image' not in batch:
            return {}
        
        attr = self.interpreters['guided_gradcam'].attribute(
            batch['image'],
            target=target,
            abs=self.config['guided_gradcam']['abs']
        )
        
        return {'image': attr}
    
    def _interpret_occlusion(self, batch: Dict[str, torch.Tensor],
                           target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply Occlusion"""
        attributions = {}
        config = self.config['occlusion']
        
        for modality in ['image', 'clinical', 'genomic']:
            if modality in batch:
                attr = self.interpreters['occlusion'].attribute(
                    batch[modality],
                    target=target,
                    sliding_window_shapes=config['sliding_window_shapes'][modality],
                    strides=config['strides'][modality]
                )
                attributions[modality] = attr
        
        return attributions
    
    def _interpret_shap(self, batch: Dict[str, torch.Tensor],
                       target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply SHAP"""
        attributions = {}
        config = self.config['shap']
        
        for modality in ['image', 'clinical', 'genomic']:
            if modality in batch:
                attr = self.interpreters['shap'].attribute(
                    batch[modality],
                    target=target,
                    n_samples=config['n_samples'],
                    batch_size=config['batch_size']
                )
                attributions[modality] = attr
        
        return attributions
    
    def _interpret_layer_gradcam(self, batch: Dict[str, torch.Tensor],
                               target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply Layer GradCAM"""
        attributions = {}
        
        for layer_name, interpreter in self.interpreters['layer_gradcam'].items():
            modality = layer_name.split('.')[0].replace('_model', '')
            if modality in batch:
                attr = interpreter.attribute(
                    batch[modality],
                    target=target
                )
                attributions[f"{modality}_{layer_name}"] = attr
        
        return attributions
    
    def add_noise_tunnel(self, attributions: Dict[str, torch.Tensor],
                        batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add noise tunnel to attributions
        Args:
            attributions: Original attributions
            batch: Input batch
        Returns:
            Smoothed attributions
        """
        config = self.config['noise_tunnel']
        smoothed = {}
        
        for modality, attr in attributions.items():
            nt = NoiseTunnel(lambda x: attr)
            smoothed[modality] = nt.attribute(
                batch[modality],
                nt_type=config['nt_type'],
                n_samples=config['n_samples'],
                stdevs=config['stdevs']
            )
        
        return smoothed
    
    def analyze_feature_interactions(self, batch: Dict[str, torch.Tensor],
                                   target: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Analyze feature interactions between modalities
        Args:
            batch: Input batch
            target: Target class
        Returns:
            Dictionary of interaction matrices
        """
        interactions = {}
        
        # Get base attributions
        base_attr = self.interpret(batch, target, method='integrated_gradients')
        
        # Analyze interactions between modalities
        modalities = [m for m in ['image', 'clinical', 'genomic'] if m in batch]
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                # Get joint attribution
                joint_batch = {
                    mod1: batch[mod1],
                    mod2: batch[mod2]
                }
                joint_attr = self.interpret(joint_batch, target, 
                                          method='integrated_gradients')
                
                # Calculate interaction strength
                interaction = (
                    joint_attr[mod1].sum() + joint_attr[mod2].sum() -
                    base_attr[mod1].sum() - base_attr[mod2].sum()
                ).abs().cpu().numpy()
                
                interactions[f"{mod1}_{mod2}"] = interaction
        
        return interactions
    
    def get_interpretation_stats(self, attributions: Dict[str, torch.Tensor]
                               ) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for attributions
        Args:
            attributions: Attribution dictionary
        Returns:
            Dictionary of statistics for each modality
        """
        stats = {}
        
        for modality, attr in attributions.items():
            attr_np = attr.abs().cpu().numpy()
            stats[modality] = {
                'mean': float(np.mean(attr_np)),
                'std': float(np.std(attr_np)),
                'max': float(np.max(attr_np)),
                'min': float(np.min(attr_np)),
                'sparsity': float((attr_np == 0).mean())
            }
        
        return stats
    
    def compare_methods(self, batch: Dict[str, torch.Tensor],
                       target: torch.Tensor,
                       methods: Optional[List[str]] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compare different interpretation methods
        Args:
            batch: Input batch
            target: Target class
            methods: List of methods to compare
        Returns:
            Dictionary of attributions for each method
        """
        if methods is None:
            methods = self.config['methods']
        
        results = {}
        for method in methods:
            results[method] = self.interpret(batch, target, method)
        
        return results 