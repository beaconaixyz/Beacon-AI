import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from beacon.models.multimodal import MultimodalFusion
from beacon.interpretability.robustness_analyzer import MultimodalRobustnessAnalyzer
from beacon.examples.multimodal_example import (
    generate_synthetic_data,
    MultimodalDataset,
    collate_fn
)

def plot_adversarial_examples(original_batch: Dict[str, torch.Tensor],
                            adversarial_batch: Dict[str, torch.Tensor],
                            sample_idx: int = 0):
    """
    Plot original and adversarial examples
    Args:
        original_batch: Original input batch
        adversarial_batch: Adversarial examples
        sample_idx: Index of sample to plot
    """
    modalities = ['image', 'clinical', 'genomic']
    fig, axes = plt.subplots(2, len(modalities), figsize=(15, 8))
    
    for i, modality in enumerate(modalities):
        if modality in original_batch and modality in adversarial_batch:
            # Plot original
            data = original_batch[modality][sample_idx]
            if modality == 'image':
                axes[0, i].imshow(data.permute(1, 2, 0).cpu().numpy())
            else:
                axes[0, i].bar(range(len(data)), data.cpu().numpy())
            axes[0, i].set_title(f'Original {modality}')
            
            # Plot adversarial
            data = adversarial_batch[modality][sample_idx]
            if modality == 'image':
                axes[1, i].imshow(data.permute(1, 2, 0).cpu().numpy())
            else:
                axes[1, i].bar(range(len(data)), data.cpu().numpy())
            axes[1, i].set_title(f'Adversarial {modality}')
    
    plt.tight_layout()
    plt.show()

def plot_sensitivity_heatmap(sensitivity: Dict[str, np.ndarray],
                           noise_types: List[str],
                           noise_levels: List[float]):
    """
    Plot sensitivity analysis results as heatmap
    Args:
        sensitivity: Sensitivity scores
        noise_types: List of noise types
        noise_levels: List of noise levels
    """
    modalities = ['image', 'clinical', 'genomic']
    n_types = len(noise_types)
    n_levels = len(noise_levels)
    
    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 4))
    
    for i, modality in enumerate(modalities):
        data = np.zeros((n_types, n_levels))
        
        for j, noise_type in enumerate(noise_types):
            for k, level in enumerate(noise_levels):
                key = f"sensitivity_{noise_type}"
                if key in sensitivity:
                    data[j, k] = sensitivity[key][f"{modality}_{level}"]
        
        sns.heatmap(data, ax=axes[i], xticklabels=noise_levels,
                   yticklabels=noise_types, cmap='YlOrRd')
        axes[i].set_title(f'{modality} Sensitivity')
        axes[i].set_xlabel('Noise Level')
        axes[i].set_ylabel('Noise Type')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance: Dict[str, torch.Tensor],
                          top_k: int = 10):
    """
    Plot feature importance scores
    Args:
        importance: Feature importance scores
        top_k: Number of top features to plot
    """
    modalities = ['image', 'clinical', 'genomic']
    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 4))
    
    for i, modality in enumerate(modalities):
        if modality in importance:
            scores = importance[modality].cpu().numpy()
            top_indices = np.argsort(scores)[-top_k:]
            top_scores = scores[top_indices]
            
            axes[i].barh(range(top_k), top_scores)
            axes[i].set_yticks(range(top_k))
            axes[i].set_yticklabels([f'Feature {idx}' for idx in top_indices])
            axes[i].set_title(f'{modality} Feature Importance')
            axes[i].set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.show()

def plot_cross_modality_robustness(robustness: Dict[str, float]):
    """
    Plot cross-modality robustness scores
    Args:
        robustness: Cross-modality robustness scores
    """
    pairs = list(robustness.keys())
    scores = list(robustness.values())
    
    plt.figure(figsize=(10, 4))
    plt.bar(pairs, scores)
    plt.title('Cross-Modality Robustness')
    plt.xlabel('Modality Pair')
    plt.ylabel('Robustness Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """Example of using the multimodal robustness analyzer"""
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
    
    # Analyzer configuration
    analyzer_config = {
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
            'strategy': 'importance'
        },
        'cross_modality': {
            'enabled': True,
            'n_permutations': 100
        }
    }
    
    # Initialize analyzer
    print("\nInitializing robustness analyzer...")
    analyzer = MultimodalRobustnessAnalyzer(model, analyzer_config)
    
    # Get a sample batch
    sample_batch = next(iter(val_dataset))
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            sample_batch[k] = v.unsqueeze(0)  # Add batch dimension
    
    # Generate and visualize adversarial examples
    print("\nGenerating adversarial examples...")
    adv_batch = analyzer.generate_adversarial_examples(sample_batch)
    plot_adversarial_examples(sample_batch, adv_batch)
    
    # Perform sensitivity analysis
    print("\nPerforming sensitivity analysis...")
    metrics = analyzer.get_robustness_metrics(sample_batch)
    plot_sensitivity_heatmap(
        metrics,
        analyzer_config['sensitivity']['noise_types'],
        analyzer_config['sensitivity']['noise_levels']
    )
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance = analyzer.analyze_feature_importance(sample_batch)
    plot_feature_importance(importance)
    
    # Analyze cross-modality robustness
    print("\nAnalyzing cross-modality robustness...")
    robustness = analyzer.analyze_cross_modality_robustness(sample_batch)
    plot_cross_modality_robustness(robustness)
    
    # Print comprehensive metrics
    print("\nComprehensive robustness metrics:")
    for category, values in metrics.items():
        print(f"\n{category.upper()}:")
        if isinstance(values, dict):
            for metric, score in values.items():
                print(f"  {metric}: {score:.4f}")
        else:
            print(f"  Score: {values:.4f}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 