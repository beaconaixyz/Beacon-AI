import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Union
from ..core.base import BeaconBase
import cv2
from tqdm import tqdm
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import joblib
from datetime import datetime
import hashlib

class ImageInterpreter(BeaconBase):
    """Interpreter for medical image model explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image interpreter
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.method = self.config.get('method', 'guided_gradcam')
        self._setup_cache()
    
    def _setup_cache(self):
        """Setup caching system"""
        if self.config['caching']['enabled']:
            cache_dir = self.config['caching']['directory']
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up expired cache files"""
        if not self.config['caching']['enabled']:
            return
        
        max_size = self.config['caching']['max_size_gb'] * 1024 * 1024 * 1024
        expiration_days = self.config['caching']['expiration_days']
        current_time = datetime.now()
        
        # Get all cache files
        cache_files = []
        total_size = 0
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(self.cache_dir, file)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                file_size = os.path.getsize(file_path)
                
                # Check if file is expired
                if (current_time - file_time).days > expiration_days:
                    os.remove(file_path)
                    continue
                
                cache_files.append((file_path, file_time, file_size))
                total_size += file_size
        
        # Remove oldest files if cache is too large
        if total_size > max_size:
            cache_files.sort(key=lambda x: x[1])  # Sort by time
            for file_path, _, file_size in cache_files:
                os.remove(file_path)
                total_size -= file_size
                if total_size <= max_size:
                    break
    
    def _get_cache_key(self, model: nn.Module, image: torch.Tensor, method: str) -> str:
        """Generate cache key for given inputs"""
        # Create unique identifier for model architecture
        model_str = str(model)
        # Hash image tensor
        image_hash = hashlib.md5(image.cpu().numpy().tobytes()).hexdigest()
        # Combine with method name
        key = f"{model_str}_{image_hash}_{method}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Get results from cache"""
        if not self.config['caching']['enabled']:
            return None
        
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            try:
                return joblib.load(cache_path)
            except:
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, result: Tuple[torch.Tensor, Dict[str, Any]]):
        """Save results to cache"""
        if not self.config['caching']['enabled']:
            return
        
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        joblib.dump(result, cache_path)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        config = super()._get_default_config()
        config.update({
            'method': 'guided_gradcam',
            'occlusion_window': 8,
            'occlusion_stride': 4,
            'n_samples': 50,
            'integrated_gradients': {
                'steps': 50,
                'baseline': 'black'
            },
            'deep_lift': {
                'baseline': 'black',
                'multiply_by_inputs': True
            },
            'lime': {
                'num_samples': 1000,
                'num_segments': 50,
                'kernel_size': 4,
                'random_seed': 42
            },
            'stability_analysis': {
                'n_perturbations': 10,
                'noise_level': 0.1,
                'methods': ['gradient', 'integrated_gradients', 'deep_lift', 'lime']
            },
            'visualization': {
                'cmap': 'jet',
                'alpha': 0.5,
                'dpi': 300,
                'saliency_threshold': 0.2,
                'top_k_features': 5,
                'figure_size': {
                    'width': 12,
                    'height': 8
                },
                'save_format': 'png'
            },
            'analysis': {
                'compute_rank_correlation': True,
                'compute_sparseness': True,
                'compute_smoothness': True
            },
            'clustering': {
                'n_clusters': 5,
                'tsne_perplexity': 30,
                'random_state': 42
            },
            'caching': {
                'enabled': True,
                'directory': 'cache',
                'max_size_gb': 10,
                'expiration_days': 7
            }
        })
        return config
    
    def explain_prediction(self,
                         model: nn.Module,
                         image: torch.Tensor,
                         target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Explain model prediction for an image with caching
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
        Returns:
            Attribution map and metadata
        """
        # Try to get from cache
        cache_key = self._get_cache_key(model, image, self.method)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Compute attribution
        if self.method == 'guided_gradcam':
            result = self._guided_gradcam(model, image, target)
        elif self.method == 'occlusion':
            result = self._occlusion(model, image, target)
        elif self.method == 'gradient':
            result = self._gradient(model, image, target)
        elif self.method == 'gradshap':
            result = self._gradient_shap(model, image, target)
        elif self.method == 'integrated_gradients':
            result = self._integrated_gradients(model, image, target)
        elif self.method == 'lime':
            result = self._lime(model, image, target)
        elif self.method == 'deep_lift':
            result = self._deep_lift(model, image, target)
        elif self.method == 'gradcam_plus_plus':
            result = self._gradcam_plus_plus(model, image, target)
        elif self.method == 'smooth_grad':
            result = self._smooth_grad(model, image, target)
        else:
            raise ValueError(f"Unknown interpretation method: {self.method}")
        
        # Save to cache
        self._save_to_cache(cache_key, result)
        
        return result
    
    def _guided_gradcam(self,
                       model: nn.Module,
                       image: torch.Tensor,
                       target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute Guided Grad-CAM attribution
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
        Returns:
            Attribution map and metadata
        """
        model.eval()
        image.requires_grad_(True)
        
        # Get the last convolutional layer
        conv_layer = None
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                break
        
        if conv_layer is None:
            raise ValueError("No convolutional layer found in the model")
        
        # Forward pass
        output = model(image)
        if target is None:
            target = output.argmax(dim=1)
        
        # Compute gradients
        output[:, target].backward()
        gradients = image.grad.detach()
        
        # Get feature maps from the last conv layer
        feature_maps = conv_layer.output
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Compute weighted combination of feature maps
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Combine with gradients for guided grad-cam
        guided_gradcam = gradients * cam
        
        return guided_gradcam, {'method': 'guided_gradcam'}
    
    def _occlusion(self,
                   model: nn.Module,
                   image: torch.Tensor,
                   target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute occlusion sensitivity map
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
        Returns:
            Attribution map and metadata
        """
        model.eval()
        window_size = self.config['occlusion_window']
        stride = self.config['occlusion_stride']
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(image)
            if target is None:
                target = original_output.argmax(dim=1)
        
        # Initialize attribution map
        attribution = torch.zeros_like(image)
        count = torch.zeros_like(image)
        
        # Slide window over image
        for h in range(0, image.shape[2] - window_size + 1, stride):
            for w in range(0, image.shape[3] - window_size + 1, stride):
                # Create occluded image
                occluded = image.clone()
                occluded[..., h:h+window_size, w:w+window_size] = 0
                
                # Get prediction for occluded image
                with torch.no_grad():
                    output = model(occluded)
                
                # Compute importance as difference in target class probability
                diff = original_output[:, target] - output[:, target]
                attribution[..., h:h+window_size, w:w+window_size] += diff.view(-1, 1, 1, 1)
                count[..., h:h+window_size, w:w+window_size] += 1
        
        # Average attribution
        attribution = attribution / (count + 1e-8)
        
        return attribution, {
            'method': 'occlusion',
            'window_size': window_size,
            'stride': stride
        }
    
    def _gradient(self,
                  model: nn.Module,
                  image: torch.Tensor,
                  target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute vanilla gradient attribution
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
        Returns:
            Attribution map and metadata
        """
        model.eval()
        image.requires_grad_(True)
        
        # Forward pass
        output = model(image)
        if target is None:
            target = output.argmax(dim=1)
        
        # Compute gradients
        output[:, target].backward()
        attribution = image.grad.detach()
        
        return attribution, {'method': 'gradient'}
    
    def _gradient_shap(self,
                      model: nn.Module,
                      image: torch.Tensor,
                      target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute GradientSHAP attribution
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
        Returns:
            Attribution map and metadata
        """
        model.eval()
        n_samples = self.config['n_samples']
        
        # Generate random reference images
        references = torch.randn(n_samples, *image.shape[1:]) * 0.1
        references = references.to(image.device)
        
        # Compute integrated gradients along the path
        attribution = torch.zeros_like(image)
        for alpha in torch.linspace(0, 1, n_samples):
            interpolated = image * alpha + references * (1 - alpha)
            interpolated.requires_grad_(True)
            
            output = model(interpolated)
            if target is None:
                target = output.argmax(dim=1)
            
            output[:, target].backward()
            attribution += interpolated.grad.detach()
        
        attribution = attribution / n_samples
        
        return attribution, {'method': 'gradshap'}
    
    def _integrated_gradients(self,
                            model: nn.Module,
                            image: torch.Tensor,
                            target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute Integrated Gradients attribution
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
        Returns:
            Attribution map and metadata
        """
        model.eval()
        steps = self.config['integrated_gradients']['steps']
        
        # Create baseline
        if self.config['integrated_gradients']['baseline'] == 'black':
            baseline = torch.zeros_like(image)
        elif self.config['integrated_gradients']['baseline'] == 'white':
            baseline = torch.ones_like(image)
        else:  # random
            baseline = torch.randn_like(image) * 0.1
        
        # Generate path
        alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
        path = baseline + alphas * (image - baseline)
        path.requires_grad_(True)
        
        # Compute gradients along path
        grads = []
        for interpolated in path:
            interpolated = interpolated.unsqueeze(0)
            output = model(interpolated)
            
            if target is None:
                target = output.argmax(dim=1)
            
            output[:, target].backward()
            grads.append(interpolated.grad.detach())
        
        # Compute attribution
        grads = torch.stack(grads)
        attribution = torch.mean(grads, dim=0) * (image - baseline)
        
        # Compute convergence delta
        convergence_delta = torch.sum(attribution) - (output[:, target] - model(baseline)[:, target])
        
        return attribution, {
            'method': 'integrated_gradients',
            'convergence_delta': convergence_delta.item(),
            'steps': steps
        }
    
    def _deep_lift(self,
                  model: nn.Module,
                  image: torch.Tensor,
                  target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute DeepLIFT attribution
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
        Returns:
            Tuple of (attribution map, metadata)
        """
        model.eval()
        image = image.requires_grad_(True)
        
        # Create baseline (black image)
        baseline = torch.zeros_like(image)
        
        # Forward pass with input and baseline
        input_output = model(image)
        baseline_output = model(baseline)
        
        if target is None:
            target = input_output.argmax(dim=1)
        
        # Compute gradients
        score = input_output[:, target].sum()
        baseline_score = baseline_output[:, target].sum()
        
        # Compute multipliers
        delta_out = score - baseline_score
        delta_in = image - baseline
        
        # Compute attribution
        delta_out.backward()
        attribution = (delta_in * image.grad).detach()
        
        metadata = {
            'method': 'deep_lift',
            'target_class': target.item(),
            'score_diff': delta_out.item()
        }
        
        return attribution, metadata
    
    def _lime(self,
             model: nn.Module,
             image: torch.Tensor,
             target: Optional[torch.Tensor] = None,
             n_samples: int = 1000,
             n_segments: int = 50) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute LIME attribution
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
            n_samples: Number of perturbed samples
            n_segments: Number of superpixels
        Returns:
            Tuple of (attribution map, metadata)
        """
        model.eval()
        
        # Convert to numpy for segmentation
        image_np = image.squeeze().cpu().numpy()
        
        # Create superpixels
        segments = self._create_superpixels(image_np, n_segments)
        
        # Generate perturbed samples
        perturbed_data = []
        binary_labels = []
        
        for _ in range(n_samples):
            perturbed = self._perturb_image(image_np, segments)
            perturbed_tensor = torch.FloatTensor(perturbed).unsqueeze(0).unsqueeze(0)
            perturbed_data.append(perturbed_tensor)
            binary_labels.append(np.ones(n_segments))
        
        # Stack all perturbed samples
        perturbed_data = torch.cat(perturbed_data, dim=0)
        
        # Get model predictions
        with torch.no_grad():
            predictions = model(perturbed_data)
            if target is None:
                target = predictions[0].argmax().item()
            scores = predictions[:, target]
        
        # Fit linear model
        from sklearn.linear_model import Ridge
        classifier = Ridge(alpha=1.0)
        classifier.fit(np.array(binary_labels), scores.cpu().numpy())
        
        # Create attribution map
        attribution = np.zeros_like(image_np)
        for segment_id in range(n_segments):
            attribution[segments == segment_id] = classifier.coef_[segment_id]
        
        attribution_tensor = torch.FloatTensor(attribution).unsqueeze(0).unsqueeze(0)
        
        metadata = {
            'method': 'lime',
            'n_segments': n_segments,
            'n_samples': n_samples,
            'target_class': target,
            'model_score': classifier.score(np.array(binary_labels), scores.cpu().numpy())
        }
        
        return attribution_tensor, metadata
    
    def _create_superpixels(self, image: np.ndarray, num_segments: int) -> np.ndarray:
        """Create superpixels from image using SLIC"""
        from skimage.segmentation import slic
        segments = slic(image, n_segments=num_segments, compactness=10)
        return segments
    
    def _perturb_image(self, image: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """Perturb image by randomly turning off segments"""
        perturbed = image.copy()
        active_segments = np.random.binomial(1, 0.5, np.unique(segments).size)
        for i, active in enumerate(active_segments):
            if not active:
                perturbed[segments == i] = 0
        return perturbed
    
    def analyze_saliency_stability(self,
                                 model: nn.Module,
                                 image: torch.Tensor,
                                 noise_std: float = 0.1,
                                 n_samples: int = 10) -> Dict[str, float]:
        """
        Analyze stability of saliency maps under input perturbations
        Args:
            model: Neural network model
            image: Input image tensor
            noise_std: Standard deviation of Gaussian noise
            n_samples: Number of noisy samples
        Returns:
            Dictionary of stability metrics
        """
        original_attribution, _ = self.explain_prediction(model, image)
        
        # Generate noisy samples and their attributions
        noisy_attributions = []
        for _ in range(n_samples):
            noisy_image = image + torch.randn_like(image) * noise_std
            noisy_attribution, _ = self.explain_prediction(model, noisy_image)
            noisy_attributions.append(noisy_attribution)
        
        # Compute stability metrics
        noisy_attributions = torch.stack(noisy_attributions)
        mean_attribution = torch.mean(noisy_attributions, dim=0)
        std_attribution = torch.std(noisy_attributions, dim=0)
        
        # Compute metrics
        stability_score = torch.mean(torch.corrcoef(
            noisy_attributions.view(n_samples, -1)
        )).item()
        
        max_deviation = torch.max(torch.abs(
            original_attribution - mean_attribution
        )).item()
        
        return {
            'stability_score': stability_score,
            'max_deviation': max_deviation,
            'mean_std': torch.mean(std_attribution).item()
        }
    
    def visualize_saliency_comparison(self,
                                    image: torch.Tensor,
                                    attributions: Dict[str, torch.Tensor],
                                    save_path: Optional[str] = None):
        """
        Compare saliency maps from different methods
        Args:
            image: Original image tensor
            attributions: Dictionary of attribution maps from different methods
            save_path: Path to save visualization (optional)
        """
        n_methods = len(attributions)
        fig_width = 4 * (n_methods + 1)
        plt.figure(figsize=(fig_width, 4))
        
        # Plot original image
        plt.subplot(1, n_methods + 1, 1)
        plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot attribution maps
        for i, (method, attribution) in enumerate(attributions.items(), 2):
            plt.subplot(1, n_methods + 1, i)
            attr = attribution.squeeze().cpu().numpy()
            attr = np.abs(attr)
            attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
            
            plt.imshow(attr, cmap=self.config['visualization']['cmap'])
            plt.title(f'{method}')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def compute_attribution_metrics(self,
                                  attribution: torch.Tensor,
                                  image: torch.Tensor) -> Dict[str, float]:
        """
        Compute quantitative metrics for attribution map
        Args:
            attribution: Attribution map tensor
            image: Original image tensor
        Returns:
            Dictionary of attribution metrics
        """
        # Normalize attribution
        attribution = attribution.squeeze()
        attribution = torch.abs(attribution)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        # Compute sparseness
        threshold = self.config['visualization']['saliency_threshold']
        sparseness = torch.mean((attribution > threshold).float()).item()
        
        # Compute smoothness
        dx = attribution[1:, :] - attribution[:-1, :]
        dy = attribution[:, 1:] - attribution[:, :-1]
        smoothness = -(torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))).item()
        
        # Compute correlation with image intensity
        image = image.squeeze()
        correlation = torch.corrcoef(torch.stack([
            attribution.view(-1),
            image.view(-1)
        ]))[0, 1].item()
        
        return {
            'sparseness': sparseness,
            'smoothness': smoothness,
            'intensity_correlation': correlation
        }
    
    def visualize_attribution(self,
                            image: torch.Tensor,
                            attribution: torch.Tensor,
                            save_path: Optional[str] = None):
        """
        Visualize attribution map overlaid on image
        Args:
            image: Original image tensor
            attribution: Attribution map tensor
            save_path: Path to save visualization (optional)
        """
        # Convert tensors to numpy arrays
        image = image.squeeze().cpu().numpy()
        attribution = attribution.squeeze().cpu().numpy()
        
        # Normalize attribution map
        attribution = np.abs(attribution)
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        # Create visualization
        plt.figure(figsize=(12, 4))
        
        # Plot original image
        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot attribution map
        plt.subplot(132)
        plt.imshow(attribution, cmap=self.config['visualization']['cmap'])
        plt.title('Attribution Map')
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(133)
        plt.imshow(image, cmap='gray')
        plt.imshow(attribution, cmap=self.config['visualization']['cmap'],
                  alpha=self.config['visualization']['alpha'])
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_class_activation_map(self,
                                    model: nn.Module,
                                    image: torch.Tensor,
                                    target_layer: nn.Module) -> torch.Tensor:
        """
        Generate Class Activation Map (CAM)
        Args:
            model: Neural network model
            image: Input image tensor
            target_layer: Target layer for CAM
        Returns:
            Class activation map
        """
        model.eval()
        
        # Register hooks
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        # Register hooks
        handle1 = target_layer.register_forward_hook(forward_hook)
        handle2 = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        output = model(image)
        target = output.argmax(dim=1)
        
        # Backward pass
        output[:, target].backward()
        
        # Remove hooks
        handle1.remove()
        handle2.remove()
        
        # Generate CAM
        weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def analyze_activation_patterns(self,
                                  model: nn.Module,
                                  image: torch.Tensor,
                                  layer_name: str) -> Dict[str, torch.Tensor]:
        """
        Analyze activation patterns in a specific layer
        Args:
            model: Neural network model
            image: Input image tensor
            layer_name: Name of the target layer
        Returns:
            Dictionary of activation statistics
        """
        model.eval()
        activations = {}
        
        def hook_fn(module, input, output):
            activations['output'] = output.detach()
        
        # Find target layer
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            model(image)
        
        # Remove hook
        handle.remove()
        
        # Compute statistics
        layer_output = activations['output']
        mean_activation = torch.mean(layer_output, dim=(2, 3))
        max_activation = torch.max(layer_output.view(layer_output.size(0), layer_output.size(1), -1), dim=2)[0]
        
        # Compute channel correlations
        flat_activations = layer_output.view(layer_output.size(0), layer_output.size(1), -1)
        channel_correlation = torch.zeros((layer_output.size(1), layer_output.size(1)))
        for i in range(layer_output.size(1)):
            for j in range(layer_output.size(1)):
                correlation = torch.corrcoef(flat_activations[0, [i, j], :])
                channel_correlation[i, j] = correlation[0, 1]
        
        return {
            'mean_activation': mean_activation,
            'max_activation': max_activation,
            'channel_correlation': channel_correlation
        }
    
    def visualize_layer_activations(self,
                                  activations: torch.Tensor,
                                  n_channels: int = 16,
                                  save_path: Optional[str] = None):
        """
        Visualize layer activations
        Args:
            activations: Layer activation tensor
            n_channels: Number of channels to visualize
            save_path: Path to save visualization (optional)
        """
        # Select subset of channels
        n_channels = min(n_channels, activations.size(1))
        activations = activations[0, :n_channels].cpu().numpy()
        
        # Create grid plot
        n_rows = int(np.ceil(np.sqrt(n_channels)))
        n_cols = int(np.ceil(n_channels / n_rows))
        
        plt.figure(figsize=(2*n_cols, 2*n_rows))
        
        for i in range(n_channels):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(activations[i], cmap='viridis')
            plt.title(f'Channel {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_feature_importance(self,
                                   attribution_maps: Dict[str, torch.Tensor],
                                   top_k: int = 5,
                                   save_path: Optional[str] = None):
        """
        Visualize feature importance across different interpretation methods
        Args:
            attribution_maps: Dictionary of attribution maps from different methods
            top_k: Number of top features to display
            save_path: Path to save visualization (optional)
        """
        plt.figure(figsize=(12, 4 * len(attribution_maps)))
        
        for i, (method, attribution) in enumerate(attribution_maps.items()):
            # Convert to numpy and flatten
            attr = attribution.squeeze().cpu().numpy()
            attr_flat = attr.flatten()
            
            # Get top k features
            top_indices = np.argsort(np.abs(attr_flat))[-top_k:]
            top_values = attr_flat[top_indices]
            
            # Plot feature importance
            plt.subplot(len(attribution_maps), 1, i+1)
            plt.barh(range(top_k), np.abs(top_values))
            plt.title(f'Top {top_k} Features - {method}')
            plt.xlabel('Attribution Magnitude')
            plt.ylabel('Feature Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def analyze_attribution_stability(self,
                                    model: nn.Module,
                                    image: torch.Tensor,
                                    n_perturbations: int = 10,
                                    noise_level: float = 0.1) -> Dict[str, Dict[str, float]]:
        """
        Analyze stability of different attribution methods under input perturbations
        Args:
            model: Neural network model
            image: Input image tensor
            n_perturbations: Number of perturbations to generate
            noise_level: Level of Gaussian noise to add
        Returns:
            Dictionary of stability metrics for each method
        """
        methods = ['gradient', 'integrated_gradients', 'deep_lift', 'lime']
        stability_metrics = {}
        
        for method in methods:
            original_method = self.method
            self.method = method
            
            # Get original attribution
            original_attr, _ = self.explain_prediction(model, image)
            
            # Generate perturbed attributions
            perturbed_attrs = []
            for _ in range(n_perturbations):
                noisy_image = image + torch.randn_like(image) * noise_level
                attr, _ = self.explain_prediction(model, noisy_image)
                perturbed_attrs.append(attr)
            
            # Compute stability metrics
            stability = self._compute_stability_metrics(original_attr, perturbed_attrs)
            stability_metrics[method] = stability
            
            # Restore original method
            self.method = original_method
        
        return stability_metrics
    
    def _compute_stability_metrics(self,
                                 original: torch.Tensor,
                                 perturbed: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute stability metrics between original and perturbed attributions
        Args:
            original: Original attribution map
            perturbed: List of perturbed attribution maps
        Returns:
            Dictionary of stability metrics
        """
        # Convert to numpy
        original = original.squeeze().cpu().numpy()
        perturbed = [p.squeeze().cpu().numpy() for p in perturbed]
        
        # Compute metrics
        correlations = [np.corrcoef(original.flatten(), p.flatten())[0, 1] for p in perturbed]
        mean_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
        
        # Compute rank stability
        rank_correlations = []
        for p in perturbed:
            orig_ranks = np.argsort(np.abs(original.flatten()))
            pert_ranks = np.argsort(np.abs(p.flatten()))
            rank_corr = np.corrcoef(orig_ranks, pert_ranks)[0, 1]
            rank_correlations.append(rank_corr)
        
        mean_rank_correlation = np.mean(rank_correlations)
        
        return {
            'mean_correlation': mean_correlation,
            'std_correlation': std_correlation,
            'mean_rank_correlation': mean_rank_correlation
        }
    
    def visualize_attribution_comparison(self,
                                       image: torch.Tensor,
                                       attribution_maps: Dict[str, torch.Tensor],
                                       save_path: Optional[str] = None):
        """
        Create a comprehensive comparison of different attribution methods
        Args:
            image: Original image tensor
            attribution_maps: Dictionary of attribution maps from different methods
            save_path: Path to save visualization (optional)
        """
        n_methods = len(attribution_maps)
        fig = plt.figure(figsize=(4 * (n_methods + 1), 8))
        
        # Plot original image
        plt.subplot(2, n_methods + 1, 1)
        plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot attribution maps
        for i, (method, attribution) in enumerate(attribution_maps.items(), 1):
            # Plot attribution map
            plt.subplot(2, n_methods + 1, i + 1)
            attr = attribution.squeeze().cpu().numpy()
            attr = np.abs(attr)
            attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
            plt.imshow(attr, cmap=self.config['visualization']['cmap'])
            plt.title(f'{method}')
            plt.axis('off')
            
            # Plot overlay
            plt.subplot(2, n_methods + 1, i + n_methods + 2)
            plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
            plt.imshow(attr, cmap=self.config['visualization']['cmap'],
                      alpha=self.config['visualization']['alpha'])
            plt.title(f'{method} Overlay')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _gradcam_plus_plus(self,
                          model: nn.Module,
                          image: torch.Tensor,
                          target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute Grad-CAM++ attribution
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
        Returns:
            Tuple of (attribution map, metadata)
        """
        model.eval()
        image = image.requires_grad_(True)
        
        # Get the last convolutional layer
        conv_layer = None
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                break
        
        if conv_layer is None:
            raise ValueError("No convolutional layer found in the model")
        
        # Forward pass
        output = model(image)
        if target is None:
            target = output.argmax(dim=1)
        
        # First-order gradient
        first_grad = torch.autograd.grad(output[:, target], image, create_graph=True)[0]
        
        # Second-order gradient
        second_grad = torch.autograd.grad(torch.sum(first_grad), image, create_graph=True)[0]
        
        # Third-order gradient
        third_grad = torch.autograd.grad(torch.sum(second_grad), image)[0]
        
        # Get feature maps and gradients from the last conv layer
        feature_maps = conv_layer.output
        alpha_num = second_grad.relu()
        alpha_denom = second_grad.relu() + third_grad.relu() * first_grad.relu()
        alpha = alpha_num / (alpha_denom + 1e-7)
        
        weights = torch.sum(alpha * first_grad.relu(), dim=(2, 3), keepdim=True)
        
        # Compute weighted combination of feature maps
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam, {'method': 'gradcam_plus_plus'}
    
    def _smooth_grad(self,
                    model: nn.Module,
                    image: torch.Tensor,
                    target: Optional[torch.Tensor] = None,
                    n_samples: int = 50,
                    noise_level: float = 0.1) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute SmoothGrad attribution
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
            n_samples: Number of noisy samples
            noise_level: Standard deviation of Gaussian noise
        Returns:
            Tuple of (attribution map, metadata)
        """
        model.eval()
        
        # Initialize accumulated gradients
        accumulated_grads = torch.zeros_like(image)
        
        # Generate noisy samples and compute gradients
        for _ in range(n_samples):
            # Add noise to input
            noisy_image = image + torch.randn_like(image) * noise_level
            noisy_image.requires_grad_(True)
            
            # Forward pass
            output = model(noisy_image)
            if target is None:
                target = output.argmax(dim=1)
            
            # Compute gradients
            output[:, target].backward()
            accumulated_grads += noisy_image.grad.detach()
        
        # Average gradients
        smoothed_grad = accumulated_grads / n_samples
        
        return smoothed_grad, {
            'method': 'smooth_grad',
            'n_samples': n_samples,
            'noise_level': noise_level
        }
    
    def visualize_3d_attribution(self,
                               image: torch.Tensor,
                               attribution: torch.Tensor,
                               save_path: Optional[str] = None):
        """
        Create 3D visualization of attribution map
        Args:
            image: Original image tensor
            attribution: Attribution map tensor
            save_path: Path to save visualization (optional)
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Convert tensors to numpy arrays
        image_np = image.squeeze().cpu().numpy()
        attr_np = attribution.squeeze().cpu().numpy()
        
        # Normalize attribution
        attr_np = np.abs(attr_np)
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(attr_np.shape[1]), np.arange(attr_np.shape[0]))
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(x, y, attr_np,
                              cmap=self.config['visualization']['cmap'],
                              linewidth=0,
                              antialiased=True)
        
        # Customize plot
        ax.set_title('3D Attribution Map')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Attribution Value')
        
        # Add colorbar
        fig.colorbar(surf)
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_interactive_visualization(self,
                                      image: torch.Tensor,
                                      attribution_maps: Dict[str, torch.Tensor],
                                      save_path: Optional[str] = None):
        """
        Create interactive HTML visualization of attribution maps
        Args:
            image: Original image tensor
            attribution_maps: Dictionary of attribution maps from different methods
            save_path: Path to save HTML file (optional)
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Convert tensors to numpy arrays
        image_np = image.squeeze().cpu().numpy()
        
        # Create subplot grid
        n_methods = len(attribution_maps)
        fig = make_subplots(rows=2, cols=n_methods + 1,
                           subplot_titles=['Original Image'] + 
                                        [f'{method} Attribution' for method in attribution_maps.keys()] +
                                        [''] +  # Placeholder for original image in second row
                                        [f'{method} Overlay' for method in attribution_maps.keys()])
        
        # Add original image
        fig.add_trace(
            go.Heatmap(z=image_np, colorscale='gray', showscale=False),
            row=1, col=1
        )
        
        # Add attribution maps and overlays
        for i, (method, attr) in enumerate(attribution_maps.items(), 1):
            attr_np = attr.squeeze().cpu().numpy()
            attr_np = np.abs(attr_np)
            attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
            
            # Attribution map
            fig.add_trace(
                go.Heatmap(z=attr_np,
                          colorscale=self.config['visualization']['cmap'],
                          showscale=True),
                row=1, col=i + 1
            )
            
            # Overlay
            fig.add_trace(
                go.Heatmap(z=image_np, colorscale='gray',
                          opacity=1 - self.config['visualization']['alpha'],
                          showscale=False),
                row=2, col=i + 1
            )
            fig.add_trace(
                go.Heatmap(z=attr_np,
                          colorscale=self.config['visualization']['cmap'],
                          opacity=self.config['visualization']['alpha'],
                          showscale=False),
                row=2, col=i + 1
            )
        
        # Update layout
        fig.update_layout(
            title='Interactive Attribution Map Comparison',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def explain_batch(self,
                     model: nn.Module,
                     images: torch.Tensor,
                     targets: Optional[torch.Tensor] = None,
                     batch_size: int = 8) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Generate explanations for a batch of images
        Args:
            model: Neural network model
            images: Batch of input images
            targets: Optional target classes
            batch_size: Size of mini-batches for processing
        Returns:
            Tuple of (batch attribution maps, list of metadata)
        """
        n_samples = images.size(0)
        attribution_maps = []
        metadata_list = []
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Generating explanations"):
            batch_images = images[i:i+batch_size]
            batch_targets = targets[i:i+batch_size] if targets is not None else None
            
            # Process mini-batch
            attribution, metadata = self.explain_prediction(model, batch_images, batch_targets)
            attribution_maps.append(attribution)
            metadata_list.extend([metadata] * len(batch_images))
        
        # Concatenate attribution maps
        attribution_maps = torch.cat(attribution_maps, dim=0)
        
        return attribution_maps, metadata_list
    
    def analyze_feature_interactions(self,
                                   model: nn.Module,
                                   image: torch.Tensor,
                                   target: Optional[torch.Tensor] = None,
                                   n_perturbations: int = 10) -> Dict[str, np.ndarray]:
        """
        Analyze feature interactions using perturbation analysis
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
            n_perturbations: Number of perturbations per feature
        Returns:
            Dictionary containing interaction matrices
        """
        # Get base attribution
        base_attr, _ = self.explain_prediction(model, image, target)
        base_attr = base_attr.squeeze().cpu().numpy()
        
        # Initialize interaction matrix
        n_features = np.prod(base_attr.shape)
        interactions = np.zeros((n_features, n_features))
        
        # Analyze pairwise interactions
        for i in tqdm(range(n_features), desc="Analyzing feature interactions"):
            for j in range(i+1, n_features):
                # Create perturbed image with both features modified
                perturbed = image.clone()
                idx_i = np.unravel_index(i, base_attr.shape)
                idx_j = np.unravel_index(j, base_attr.shape)
                
                interaction_score = 0
                for _ in range(n_perturbations):
                    # Perturb features
                    noise = torch.randn(2) * 0.1
                    perturbed[..., idx_i[0], idx_i[1]] += noise[0]
                    perturbed[..., idx_j[0], idx_j[1]] += noise[1]
                    
                    # Get attribution for perturbed image
                    pert_attr, _ = self.explain_prediction(model, perturbed, target)
                    pert_attr = pert_attr.squeeze().cpu().numpy()
                    
                    # Compute interaction score
                    interaction_score += np.abs(
                        pert_attr[idx_i] - base_attr[idx_i] +
                        pert_attr[idx_j] - base_attr[idx_j]
                    )
                
                interaction_score /= n_perturbations
                interactions[i, j] = interaction_score
                interactions[j, i] = interaction_score
        
        return {
            'interaction_matrix': interactions,
            'feature_importance': np.sum(interactions, axis=0)
        }
    
    def analyze_model_sensitivity(self,
                                model: nn.Module,
                                image: torch.Tensor,
                                target: Optional[torch.Tensor] = None,
                                n_scales: int = 5) -> Dict[str, Any]:
        """
        Analyze model sensitivity to input transformations
        Args:
            model: Neural network model
            image: Input image tensor
            target: Target class (optional)
            n_scales: Number of scales to analyze
        Returns:
            Dictionary of sensitivity analysis results
        """
        results = {
            'scale_sensitivity': [],
            'rotation_sensitivity': [],
            'noise_sensitivity': []
        }
        
        # Scale sensitivity
        scales = np.linspace(0.5, 1.5, n_scales)
        for scale in scales:
            scaled_image = torch.nn.functional.interpolate(
                image,
                scale_factor=scale,
                mode='bilinear',
                align_corners=False
            )
            scaled_image = torch.nn.functional.interpolate(
                scaled_image,
                size=image.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            attr, _ = self.explain_prediction(model, scaled_image, target)
            results['scale_sensitivity'].append(torch.mean(torch.abs(attr)).item())
        
        # Rotation sensitivity
        angles = np.linspace(-30, 30, n_scales)
        for angle in angles:
            rotated_image = torch.from_numpy(
                cv2.warpAffine(
                    image.squeeze().cpu().numpy(),
                    cv2.getRotationMatrix2D(
                        (image.shape[2]//2, image.shape[3]//2),
                        angle,
                        1.0
                    ),
                    (image.shape[2], image.shape[3])
            ).unsqueeze(0).unsqueeze(0).to(image.device)
            
            attr, _ = self.explain_prediction(model, rotated_image, target)
            results['rotation_sensitivity'].append(torch.mean(torch.abs(attr)).item())
        
        # Noise sensitivity
        noise_levels = np.linspace(0, 0.2, n_scales)
        for noise_level in noise_levels:
            noisy_image = image + torch.randn_like(image) * noise_level
            attr, _ = self.explain_prediction(model, noisy_image, target)
            results['noise_sensitivity'].append(torch.mean(torch.abs(attr)).item())
        
        return {
            'scale_sensitivity': np.array(results['scale_sensitivity']),
            'rotation_sensitivity': np.array(results['rotation_sensitivity']),
            'noise_sensitivity': np.array(results['noise_sensitivity']),
            'scales': scales,
            'angles': angles,
            'noise_levels': noise_levels
        }
    
    def visualize_sensitivity_analysis(self,
                                     sensitivity_results: Dict[str, Any],
                                     save_path: Optional[str] = None):
        """
        Visualize sensitivity analysis results
        Args:
            sensitivity_results: Results from analyze_model_sensitivity
            save_path: Path to save visualization (optional)
        """
        plt.figure(figsize=(15, 5))
        
        # Plot scale sensitivity
        plt.subplot(131)
        plt.plot(sensitivity_results['scales'],
                 sensitivity_results['scale_sensitivity'])
        plt.title('Scale Sensitivity')
        plt.xlabel('Scale Factor')
        plt.ylabel('Mean Attribution Magnitude')
        
        # Plot rotation sensitivity
        plt.subplot(132)
        plt.plot(sensitivity_results['angles'],
                 sensitivity_results['rotation_sensitivity'])
        plt.title('Rotation Sensitivity')
        plt.xlabel('Rotation Angle (degrees)')
        plt.ylabel('Mean Attribution Magnitude')
        
        # Plot noise sensitivity
        plt.subplot(133)
        plt.plot(sensitivity_results['noise_levels'],
                 sensitivity_results['noise_sensitivity'])
        plt.title('Noise Sensitivity')
        plt.xlabel('Noise Level (std)')
        plt.ylabel('Mean Attribution Magnitude')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_feature_interactions(self,
                                     interaction_results: Dict[str, np.ndarray],
                                     top_k: int = 10,
                                     save_path: Optional[str] = None):
        """
        Visualize feature interactions
        Args:
            interaction_results: Results from analyze_feature_interactions
            top_k: Number of top interactions to highlight
            save_path: Path to save visualization (optional)
        """
        plt.figure(figsize=(12, 5))
        
        # Plot interaction matrix
        plt.subplot(121)
        plt.imshow(interaction_results['interaction_matrix'],
                   cmap='viridis')
        plt.colorbar(label='Interaction Strength')
        plt.title('Feature Interaction Matrix')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Index')
        
        # Plot top feature importance
        plt.subplot(122)
        importance = interaction_results['feature_importance']
        top_indices = np.argsort(importance)[-top_k:]
        plt.barh(range(top_k), importance[top_indices])
        plt.title(f'Top {top_k} Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def analyze_feature_clusters(self,
                               model: nn.Module,
                               images: torch.Tensor,
                               layer_name: str) -> Dict[str, Any]:
        """
        Analyze feature clusters in network activations
        Args:
            model: Neural network model
            images: Batch of input images
            layer_name: Name of the target layer
        Returns:
            Dictionary containing clustering results
        """
        # Get layer activations
        activations = self.analyze_activation_patterns(model, images, layer_name)
        features = activations['mean_activation'].cpu().numpy()
        
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(
            n_components=2,
            perplexity=self.config['clustering']['tsne_perplexity'],
            random_state=self.config['clustering']['random_state']
        )
        features_2d = tsne.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=self.config['clustering']['n_clusters'],
            random_state=self.config['clustering']['random_state']
        )
        clusters = kmeans.fit_predict(features)
        
        # Compute cluster statistics
        cluster_stats = {}
        for i in range(self.config['clustering']['n_clusters']):
            cluster_mask = clusters == i
            cluster_stats[f'cluster_{i}'] = {
                'size': np.sum(cluster_mask),
                'mean_activation': np.mean(features[cluster_mask], axis=0),
                'std_activation': np.std(features[cluster_mask], axis=0)
            }
        
        return {
            'features_2d': features_2d,
            'clusters': clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_stats': cluster_stats
        }
    
    def visualize_feature_clusters(self,
                                 clustering_results: Dict[str, Any],
                                 save_path: Optional[str] = None):
        """
        Visualize feature clusters
        Args:
            clustering_results: Results from analyze_feature_clusters
            save_path: Path to save visualization (optional)
        """
        features_2d = clustering_results['features_2d']
        clusters = clustering_results['clusters']
        centers = clustering_results['cluster_centers']
        
        plt.figure(figsize=(15, 5))
        
        # Plot t-SNE visualization with clusters
        plt.subplot(121)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                             c=clusters, cmap='viridis')
        plt.colorbar(scatter, label='Cluster')
        plt.title('Feature Clusters (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # Plot cluster statistics
        plt.subplot(122)
        cluster_sizes = [
            clustering_results['cluster_stats'][f'cluster_{i}']['size']
            for i in range(len(clustering_results['cluster_stats']))
        ]
        plt.bar(range(len(cluster_sizes)), cluster_sizes)
        plt.title('Cluster Sizes')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['visualization']['dpi'],
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show() 