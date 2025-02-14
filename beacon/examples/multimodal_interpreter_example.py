import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from beacon.models.multimodal import MultimodalFusion
from beacon.interpretability.multimodal_interpreter import MultimodalInterpreter
from beacon.examples.multimodal_example import (
    generate_synthetic_data,
    MultimodalDataset,
    collate_fn
)

def plot_modality_attributions(attributions: Dict[str, torch.Tensor],
                             batch: Dict[str, torch.Tensor],
                             sample_idx: int = 0):
    """
    Plot attributions for each modality
    Args:
        attributions: Dictionary of attributions
        batch: Input batch
        sample_idx: Index of sample to plot
    """
    n_modalities = len(attributions)
    fig, axes = plt.subplots(1, n_modalities, figsize=(5*n_modalities, 4))
    
    if n_modalities == 1:
        axes = [axes]
    
    for ax, (modality, attr) in zip(axes, attributions.items()):
        attr_np = attr[sample_idx].abs().sum(dim=0).cpu().numpy()
        
        if modality == 'image':
            # Plot image attributions as heatmap
            im = ax.imshow(attr_np, cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{modality.capitalize()} Attributions')
            ax.axis('off')
        else:
            # Plot feature attributions as bar plot
            ax.bar(range(len(attr_np)), attr_np)
            ax.set_title(f'{modality.capitalize()} Feature Attributions')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Attribution Magnitude')
    
    plt.tight_layout()
    plt.show()

def plot_method_comparison(comparison_results: Dict[str, Dict[str, torch.Tensor]],
                         sample_idx: int = 0):
    """
    Plot comparison of different interpretation methods
    Args:
        comparison_results: Results from different methods
        sample_idx: Index of sample to plot
    """
    n_methods = len(comparison_results)
    n_modalities = len(next(iter(comparison_results.values())))
    
    fig, axes = plt.subplots(n_methods, n_modalities, 
                            figsize=(5*n_modalities, 4*n_methods))
    
    for i, (method, results) in enumerate(comparison_results.items()):
        for j, (modality, attr) in enumerate(results.items()):
            ax = axes[i, j] if n_methods > 1 else axes[j]
            
            attr_np = attr[sample_idx].abs().sum(dim=0).cpu().numpy()
            
            if modality == 'image':
                im = ax.imshow(attr_np, cmap='viridis')
                plt.colorbar(im, ax=ax)
                ax.axis('off')
            else:
                ax.bar(range(len(attr_np)), attr_np)
            
            if i == 0:
                ax.set_title(modality.capitalize())
            if j == 0:
                ax.set_ylabel(method)
    
    plt.tight_layout()
    plt.show()

def plot_feature_interactions(interactions: Dict[str, np.ndarray]):
    """
    Plot feature interactions between modalities
    Args:
        interactions: Dictionary of interaction matrices
    """
    n_pairs = len(interactions)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 4))
    
    if n_pairs == 1:
        axes = [axes]
    
    for ax, (pair, interaction) in zip(axes, interactions.items()):
        im = ax.imshow(interaction, cmap='coolwarm', center=0)
        plt.colorbar(im, ax=ax)
        ax.set_title(f'{pair} Interactions')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_interpretation_stats(stats: Dict[str, Dict[str, float]]):
    """
    Plot interpretation statistics
    Args:
        stats: Dictionary of statistics for each modality
    """
    metrics = ['mean', 'std', 'max', 'min', 'sparsity']
    modalities = list(stats.keys())
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3*len(metrics)))
    
    for i, metric in enumerate(metrics):
        values = [stats[mod][metric] for mod in modalities]
        axes[i].bar(modalities, values)
        axes[i].set_title(f'{metric.capitalize()} Attribution Values')
        axes[i].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()

def main():
    """Example of using the multimodal interpreter"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    n_samples = 100
    train_data, val_data = generate_synthetic_data(n_samples)
    
    # Create datasets
    train_dataset = MultimodalDataset(train_data)
    val_dataset = MultimodalDataset(val_data)
    
    # Model configuration
    model_config = {
        'image': {
            'enabled': True,
            'in_channels': 1,
            'base_filters': 32,
            'n_blocks': 3
        },
        'genomic': {
            'enabled': True,
            'conv_type': 'gat',
            'input_dim': 119,
            'hidden_dims': [64, 128],
            'num_heads': 4
        },
        'clinical': {
            'enabled': True,
            'input_dim': 32,
            'hidden_dims': [64, 32]
        },
        'fusion': {
            'method': 'attention',
            'hidden_dim': 256,
            'num_heads': 4,
            'dropout_rate': 0.3
        },
        'output_dim': 2,
        'task': 'classification',
        'learning_rate': 0.001
    }
    
    # Initialize and train model
    print("\nInitializing and training model...")
    model = MultimodalFusion(model_config)
    
    # Interpreter configuration
    interpreter_config = {
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
        }
    }
    
    # Initialize interpreter
    print("\nInitializing interpreter...")
    interpreter = MultimodalInterpreter(model, interpreter_config)
    
    # Get a sample batch
    sample_batch = next(iter(val_dataset))
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            sample_batch[k] = v.unsqueeze(0)  # Add batch dimension
    
    # Get attributions using different methods
    print("\nComputing attributions using different methods...")
    methods = ['integrated_gradients', 'deep_lift', 'occlusion']
    comparison_results = interpreter.compare_methods(
        sample_batch,
        sample_batch['target'],
        methods=methods
    )
    
    # Plot comparison of methods
    print("\nPlotting method comparison...")
    plot_method_comparison(comparison_results)
    
    # Analyze feature interactions
    print("\nAnalyzing feature interactions...")
    interactions = interpreter.analyze_feature_interactions(
        sample_batch,
        sample_batch['target']
    )
    plot_feature_interactions(interactions)
    
    # Get interpretation statistics
    print("\nComputing interpretation statistics...")
    base_attr = interpreter.interpret(sample_batch, sample_batch['target'])
    stats = interpreter.get_interpretation_stats(base_attr)
    plot_interpretation_stats(stats)
    
    # Apply noise tunnel
    print("\nApplying noise tunnel...")
    smoothed_attr = interpreter.add_noise_tunnel(base_attr, sample_batch)
    plot_modality_attributions(smoothed_attr, sample_batch)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 