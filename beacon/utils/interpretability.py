import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from ..core.base import BeaconBase
import shap
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GuidedGradCam,
    Occlusion,
    NoiseTunnel,
    visualization
)

class ModelInterpreter(BeaconBase):
    """Model interpretation and explanation tools"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model interpreter
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.method = self.config.get('method', 'integrated_gradients')
        self.baseline = self.config.get('baseline', None)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'method': 'integrated_gradients',
            'n_steps': 50,
            'internal_batch_size': 32,
            'feature_names': None,
            'output_dir': 'interpretability_results'
        }
    
    def explain_prediction(self, 
                          model: torch.nn.Module,
                          input_data: torch.Tensor,
                          target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate feature attributions for model prediction
        Args:
            model: Trained PyTorch model
            input_data: Input data tensor
            target: Optional target labels
        Returns:
            Tuple of (attributions, metadata)
        """
        if self.method == 'integrated_gradients':
            return self._integrated_gradients(model, input_data, target)
        elif self.method == 'deep_lift':
            return self._deep_lift(model, input_data, target)
        elif self.method == 'guided_gradcam':
            return self._guided_gradcam(model, input_data, target)
        elif self.method == 'shap':
            return self._shap_values(model, input_data)
        else:
            raise ValueError(f"Unknown interpretation method: {self.method}")
    
    def _integrated_gradients(self, 
                            model: torch.nn.Module,
                            input_data: torch.Tensor,
                            target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Calculate Integrated Gradients attributions
        """
        ig = IntegratedGradients(model)
        attributions, delta = ig.attribute(
            input_data,
            target=target,
            n_steps=self.config['n_steps'],
            internal_batch_size=self.config['internal_batch_size'],
            return_convergence_delta=True
        )
        
        metadata = {
            'method': 'integrated_gradients',
            'convergence_delta': delta.detach().numpy()
        }
        
        return attributions, metadata
    
    def _deep_lift(self, 
                   model: torch.nn.Module,
                   input_data: torch.Tensor,
                   target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Calculate DeepLIFT attributions
        """
        dl = DeepLift(model)
        attributions = dl.attribute(
            input_data,
            target=target,
            baselines=self.baseline
        )
        
        metadata = {
            'method': 'deep_lift'
        }
        
        return attributions, metadata
    
    def _guided_gradcam(self, 
                        model: torch.nn.Module,
                        input_data: torch.Tensor,
                        target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Calculate Guided GradCAM attributions
        """
        guided_gc = GuidedGradCam(model, model.conv_layers[-1])
        attributions = guided_gc.attribute(
            input_data,
            target=target
        )
        
        metadata = {
            'method': 'guided_gradcam'
        }
        
        return attributions, metadata
    
    def _shap_values(self, 
                     model: torch.nn.Module,
                     input_data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Calculate SHAP values
        """
        # Convert model to CPU for SHAP
        model = model.cpu()
        background = torch.zeros_like(input_data[:100])  # Use first 100 samples as background
        
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(input_data)
        
        metadata = {
            'method': 'shap'
        }
        
        return torch.tensor(shap_values), metadata
    
    def visualize_attributions(self,
                             attributions: torch.Tensor,
                             input_data: torch.Tensor,
                             feature_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None):
        """
        Visualize feature attributions
        Args:
            attributions: Feature attributions
            input_data: Original input data
            feature_names: Optional list of feature names
            save_path: Optional path to save visualization
        """
        if len(input_data.shape) == 4:  # Image data
            self._visualize_image_attributions(attributions, input_data, save_path)
        else:  # Tabular data
            self._visualize_tabular_attributions(
                attributions, input_data, feature_names, save_path
            )
    
    def _visualize_image_attributions(self,
                                    attributions: torch.Tensor,
                                    images: torch.Tensor,
                                    save_path: Optional[str] = None):
        """
        Visualize attributions for image data
        """
        # Convert to numpy
        attr_np = attributions.detach().numpy()
        img_np = images.detach().numpy()
        
        for i in range(min(len(images), 5)):  # Show up to 5 images
            plt.figure(figsize=(12, 4))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(img_np[i].squeeze(), cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            # Attribution map
            plt.subplot(1, 3, 2)
            plt.imshow(attr_np[i].squeeze(), cmap='RdBu')
            plt.title('Attribution Map')
            plt.axis('off')
            
            # Overlay
            plt.subplot(1, 3, 3)
            overlay = 0.5 * img_np[i].squeeze() + 0.5 * attr_np[i].squeeze()
            plt.imshow(overlay, cmap='RdBu')
            plt.title('Overlay')
            plt.axis('off')
            
            if save_path:
                plt.savefig(f"{save_path}_image_{i}.png")
            plt.close()
    
    def _visualize_tabular_attributions(self,
                                      attributions: torch.Tensor,
                                      data: torch.Tensor,
                                      feature_names: Optional[List[str]] = None,
                                      save_path: Optional[str] = None):
        """
        Visualize attributions for tabular data
        """
        # Convert to numpy
        attr_np = attributions.detach().numpy()
        
        # Use feature indices if names not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(attr_np.shape[1])]
        
        # Calculate mean absolute attribution for each feature
        mean_attr = np.abs(attr_np).mean(axis=0)
        
        # Sort features by importance
        sorted_idx = np.argsort(mean_attr)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_attr = mean_attr[sorted_idx]
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_features)), sorted_attr)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Mean Absolute Attribution')
        plt.title('Feature Importance')
        
        if save_path:
            plt.savefig(f"{save_path}_feature_importance.png")
        plt.close()
    
    def analyze_feature_interactions(self,
                                   model: torch.nn.Module,
                                   input_data: torch.Tensor,
                                   feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Analyze feature interactions using SHAP interaction values
        Args:
            model: Trained PyTorch model
            input_data: Input data tensor
            feature_names: Optional list of feature names
        Returns:
            Dictionary of interaction scores
        """
        # Convert to numpy for SHAP
        model = model.cpu()
        background = torch.zeros_like(input_data[:100])
        
        # Calculate SHAP interaction values
        explainer = shap.DeepExplainer(model, background)
        interaction_values = explainer.shap_interaction_values(input_data)
        
        # Calculate interaction strength
        interaction_strength = np.abs(interaction_values).mean(axis=0)
        
        # Create interaction dictionary
        interactions = {}
        n_features = interaction_strength.shape[0]
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                feature1 = feature_names[i] if feature_names else f"Feature {i}"
                feature2 = feature_names[j] if feature_names else f"Feature {j}"
                key = f"{feature1} × {feature2}"
                interactions[key] = float(interaction_strength[i, j])
        
        return interactions
    
    def generate_counterfactuals(self,
                                model: torch.nn.Module,
                                input_data: torch.Tensor,
                                target: torch.Tensor,
                                n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate counterfactual examples
        Args:
            model: Trained PyTorch model
            input_data: Input data tensor
            target: Target predictions
            n_samples: Number of counterfactual samples to generate
        Returns:
            Tuple of (counterfactual examples, distances)
        """
        # Initialize counterfactual examples with noise
        counterfactuals = input_data.clone() + torch.randn_like(input_data) * 0.1
        counterfactuals.requires_grad = True
        
        # Optimize counterfactuals
        optimizer = torch.optim.Adam([counterfactuals], lr=0.01)
        
        for _ in range(100):  # Maximum iterations
            optimizer.zero_grad()
            
            # Prediction loss
            pred = model(counterfactuals)
            pred_loss = torch.nn.functional.cross_entropy(pred, target)
            
            # Distance loss
            dist_loss = torch.norm(counterfactuals - input_data, p=2)
            
            # Total loss
            loss = pred_loss + 0.1 * dist_loss
            loss.backward()
            optimizer.step()
        
        # Calculate distances
        distances = torch.norm(counterfactuals.detach() - input_data, p=2, dim=1)
        
        return counterfactuals.detach(), distances 