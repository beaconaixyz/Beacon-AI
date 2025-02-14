import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple, Union
import seaborn as sns
from PIL import Image
import cv2
from ..core.base import BeaconBase

class Visualizer(BeaconBase):
    """Base class for visualization tools"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.style = config.get('style', 'seaborn')
        plt.style.use(self.style)
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'style': 'seaborn',
            'dpi': 300,
            'figsize': (10, 6),
            'save_format': 'png'
        }
    
    def plot_feature_importance(self, 
                              importance_scores: torch.Tensor,
                              feature_names: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> None:
        """Plot feature importance scores"""
        plt.figure(figsize=self.config['figsize'])
        scores = importance_scores.cpu().numpy()
        n_features = len(scores)
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
            
        plt.barh(range(n_features), scores)
        plt.yticks(range(n_features), feature_names)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'], 
                       format=self.config['save_format'])
        plt.close()
    
    def plot_training_history(self,
                            history: Dict[str, List[float]],
                            save_path: Optional[str] = None) -> None:
        """Plot training history"""
        plt.figure(figsize=self.config['figsize'])
        
        for metric, values in history.items():
            plt.plot(values, label=metric)
            
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training History')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'],
                       format=self.config['save_format'])
        plt.close()
    
    def plot_confusion_matrix(self,
                            matrix: torch.Tensor,
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> None:
        """Plot confusion matrix"""
        plt.figure(figsize=self.config['figsize'])
        plt.imshow(matrix.cpu().numpy(), cmap='Blues')
        
        if class_names is None:
            class_names = [str(i) for i in range(matrix.shape[0])]
            
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=self.config['dpi'],
                       format=self.config['save_format'])
        plt.close()
    
    def visualize_attribution(self, image: torch.Tensor, 
                            attribution: torch.Tensor,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize attribution map overlaid on image
        Args:
            image: Input image tensor
            attribution: Attribution tensor
            save_path: Optional path to save visualization
        Returns:
            Matplotlib figure
        """
        # Convert tensors to numpy arrays
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.cpu().numpy()
        
        # Ensure correct shape
        if image.shape[0] == 3:  # CHW to HWC
            image = np.transpose(image, (1, 2, 0))
        if attribution.shape[0] == 1:  # CHW to HWC
            attribution = np.transpose(attribution, (1, 2, 0))
        
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        
        # Normalize attribution
        attribution = np.abs(attribution)
        if attribution.sum() > 0:
            attribution = attribution / attribution.max()
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.config['figsize'])
        
        # Plot original image
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=self.config['title_fontsize'])
        ax1.axis('off')
        
        # Plot attribution map
        attribution_map = ax2.imshow(
            attribution,
            cmap=self.config['colormap'],
            interpolation='nearest'
        )
        ax2.set_title('Attribution Map', fontsize=self.config['title_fontsize'])
        ax2.axis('off')
        plt.colorbar(attribution_map, ax=ax2)
        
        # Plot overlay
        ax3.imshow(image)
        overlay = ax3.imshow(
            attribution,
            cmap=self.config['colormap'],
            alpha=self.config['alpha']
        )
        ax3.set_title('Overlay', fontsize=self.config['title_fontsize'])
        ax3.axis('off')
        plt.colorbar(overlay, ax=ax3)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(
                save_path,
                format=self.config['save_format'],
                dpi=self.config['dpi'],
                bbox_inches='tight'
            )
        
        return fig
    
    def visualize_feature_maps(self, feature_maps: List[torch.Tensor],
                             max_features: int = 16,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize feature maps
        Args:
            feature_maps: List of feature map tensors
            max_features: Maximum number of features to display
            save_path: Optional path to save visualization
        Returns:
            Matplotlib figure
        """
        n_maps = len(feature_maps)
        
        # Create figure
        fig = plt.figure(figsize=(15, 5 * n_maps))
        
        for i, feat_map in enumerate(feature_maps):
            # Convert to numpy if needed
            if isinstance(feat_map, torch.Tensor):
                feat_map = feat_map.cpu().numpy()
            
            # Get number of features
            n_features = min(feat_map.shape[1], max_features)
            
            # Create subplot grid
            grid_size = int(np.ceil(np.sqrt(n_features)))
            
            for j in range(n_features):
                plt.subplot(n_maps, grid_size, i * grid_size + j + 1)
                plt.imshow(feat_map[0, j], cmap='viridis')
                plt.axis('off')
                plt.title(f'Layer {i+1}, Feature {j+1}')
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(
                save_path,
                format=self.config['save_format'],
                dpi=self.config['dpi'],
                bbox_inches='tight'
            )
        
        return fig
    
    def visualize_attention(self, image: torch.Tensor,
                          attention_weights: torch.Tensor,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize attention weights
        Args:
            image: Input image tensor
            attention_weights: Attention weights tensor
            save_path: Optional path to save visualization
        Returns:
            Matplotlib figure
        """
        # Convert tensors to numpy arrays
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # Ensure correct shape
        if image.shape[0] == 3:  # CHW to HWC
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        
        # Resize attention weights if needed
        if attention_weights.shape[-2:] != image.shape[-2:]:
            attention_weights = cv2.resize(
                attention_weights,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config['figsize'])
        
        # Plot original image
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=self.config['title_fontsize'])
        ax1.axis('off')
        
        # Plot attention overlay
        ax2.imshow(image)
        attention_map = ax2.imshow(
            attention_weights,
            cmap=self.config['colormap'],
            alpha=self.config['alpha']
        )
        ax2.set_title('Attention Map', fontsize=self.config['title_fontsize'])
        ax2.axis('off')
        plt.colorbar(attention_map, ax=ax2)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(
                save_path,
                format=self.config['save_format'],
                dpi=self.config['dpi'],
                bbox_inches='tight'
            )
        
        return fig
    
    def visualize_comparison(self, results: Dict[str, Dict[str, Any]],
                           image: torch.Tensor,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize comparison of different interpretation methods
        Args:
            results: Dictionary of interpretation results
            image: Input image tensor
            save_path: Optional path to save visualization
        Returns:
            Matplotlib figure
        """
        n_methods = len(results)
        
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.shape[0] == 3:  # CHW to HWC
            image = np.transpose(image, (1, 2, 0))
        
        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())
        
        # Create figure
        fig = plt.figure(figsize=(15, 5 * (n_methods + 1)))
        
        # Plot original image
        plt.subplot(n_methods + 1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image', fontsize=self.config['title_fontsize'])
        plt.axis('off')
        
        # Plot results for each method
        for i, (method, result) in enumerate(results.items()):
            attribution = result['attribution']
            if isinstance(attribution, torch.Tensor):
                attribution = attribution.cpu().numpy()
            if attribution.shape[0] == 1:
                attribution = np.transpose(attribution, (1, 2, 0))
            
            # Normalize attribution
            attribution = np.abs(attribution)
            if attribution.sum() > 0:
                attribution = attribution / attribution.max()
            
            # Plot attribution map
            plt.subplot(n_methods + 1, 3, (i + 1) * 3 - 1)
            plt.imshow(attribution, cmap=self.config['colormap'])
            plt.title(f'{method} Attribution', fontsize=self.config['title_fontsize'])
            plt.axis('off')
            plt.colorbar()
            
            # Plot overlay
            plt.subplot(n_methods + 1, 3, (i + 1) * 3)
            plt.imshow(image)
            plt.imshow(attribution, cmap=self.config['colormap'], alpha=self.config['alpha'])
            plt.title(f'{method} Overlay', fontsize=self.config['title_fontsize'])
            plt.axis('off')
            plt.colorbar()
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(
                save_path,
                format=self.config['save_format'],
                dpi=self.config['dpi'],
                bbox_inches='tight'
            )
        
        return fig
    
    def plot_interpretation_stats(self, stats: Dict[str, float],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot interpretation statistics
        Args:
            stats: Dictionary of statistics
            save_path: Optional path to save visualization
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config['figsize'])
        
        # Plot basic statistics
        basic_stats = {
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'sparsity': stats['sparsity'],
            'positive_ratio': stats['positive_ratio'],
            'negative_ratio': stats['negative_ratio']
        }
        
        sns.barplot(
            x=list(basic_stats.keys()),
            y=list(basic_stats.values()),
            ax=ax1
        )
        ax1.set_title('Basic Statistics', fontsize=self.config['title_fontsize'])
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot percentiles
        percentiles = {k: v for k, v in stats.items() if 'percentile' in k}
        sns.lineplot(
            x=[int(k.split('_')[1]) for k in percentiles.keys()],
            y=list(percentiles.values()),
            ax=ax2
        )
        ax2.set_title('Percentile Distribution', fontsize=self.config['title_fontsize'])
        ax2.set_xlabel('Percentile')
        ax2.set_ylabel('Value')
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(
                save_path,
                format=self.config['save_format'],
                dpi=self.config['dpi'],
                bbox_inches='tight'
            )
        
        return fig 