import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from beacon.models.multimodal import MultimodalFusion
from beacon.interpretability.advanced_robustness import AdvancedRobustnessAnalyzer
from beacon.examples.multimodal_example import (
    generate_synthetic_data,
    MultimodalDataset,
    collate_fn
)

def plot_adversarial_examples(original_batch: Dict[str, torch.Tensor],
                            adversarial_batches: Dict[str, Dict[str, torch.Tensor]],
                            sample_idx: int = 0):
    """
    Plot original and adversarial examples from different attacks
    Args:
        original_batch: Original input batch
        adversarial_batches: Dictionary of adversarial examples from different attacks
        sample_idx: Index of sample to plot
    """
    modalities = ['image', 'sequence', 'clinical']
    attacks = list(adversarial_batches.keys())
    
    fig, axes = plt.subplots(len(attacks) + 1, len(modalities),
                            figsize=(15, 5 * (len(attacks) + 1)))
    
    # Plot original data
    for i, modality in enumerate(modalities):
        if modality in original_batch:
            data = original_batch[modality][sample_idx]
            if modality == 'image':
                axes[0, i].imshow(data.permute(1, 2, 0).cpu().numpy())
            else:
                axes[0, i].bar(range(len(data)), data.cpu().numpy())
            axes[0, i].set_title(f'Original {modality}')
    
    # Plot adversarial examples
    for j, attack in enumerate(attacks, 1):
        for i, modality in enumerate(modalities):
            if modality in adversarial_batches[attack]:
                data = adversarial_batches[attack][modality][sample_idx]
                if modality == 'image':
                    axes[j, i].imshow(data.permute(1, 2, 0).cpu().numpy())
                else:
                    axes[j, i].bar(range(len(data)), data.cpu().numpy())
                axes[j, i].set_title(f'{attack} {modality}')
    
    plt.tight_layout()
    plt.show()

def plot_decision_boundary_analysis(results: Dict[str, Dict[str, np.ndarray]]):
    """
    Plot decision boundary analysis results
    Args:
        results: Dictionary of decision boundary analysis results
    """
    modalities = list(results.keys())
    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 5))
    
    for i, modality in enumerate(modalities):
        distances = results[modality]['distances']
        
        # Plot histogram of distances to decision boundary
        axes[i].hist(distances.flatten(), bins=30)
        axes[i].set_title(f'{modality} Decision Boundary Distances')
        axes[i].set_xlabel('Distance')
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def plot_gradient_landscape(results: Dict[str, Dict[str, np.ndarray]],
                          sample_idx: int = 0):
    """
    Plot gradient landscape analysis results
    Args:
        results: Dictionary of gradient landscape analysis results
        sample_idx: Index of sample to plot
    """
    modalities = list(results.keys())
    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 5))
    
    for i, modality in enumerate(modalities):
        gradients = results[modality]['gradients'][sample_idx]
        points = results[modality]['points'][sample_idx]
        
        # Plot gradient magnitudes
        grad_mag = np.linalg.norm(gradients, axis=1)
        axes[i].scatter(points[:, 0], points[:, 1], c=grad_mag, cmap='viridis')
        axes[i].set_title(f'{modality} Gradient Landscape')
        axes[i].set_xlabel('Direction 1')
        axes[i].set_ylabel('Direction 2')
        plt.colorbar(axes[i].collections[0], ax=axes[i], label='Gradient Magnitude')
    
    plt.tight_layout()
    plt.show()

def plot_lipschitz_constants(lipschitz: Dict[str, float]):
    """
    Plot Lipschitz constants for each modality
    Args:
        lipschitz: Dictionary of Lipschitz constants
    """
    modalities = list(lipschitz.keys())
    constants = [lipschitz[m] for m in modalities]
    
    plt.figure(figsize=(10, 5))
    plt.bar(modalities, constants)
    plt.title('Lipschitz Constants by Modality')
    plt.xlabel('Modality')
    plt.ylabel('Lipschitz Constant')
    plt.show()

def main():
    """Example of using the advanced robustness analyzer"""
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
        'sequence': {
            'enabled': True,
            'input_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2
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
        'output_dim': 2
    }
    
    # Initialize and train model
    print("\nInitializing and training model...")
    model = MultimodalFusion(model_config)
    
    # Analyzer configuration
    analyzer_config = {
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
    
    # Initialize analyzer
    print("\nInitializing advanced robustness analyzer...")
    analyzer = AdvancedRobustnessAnalyzer(model, analyzer_config)
    
    # Get a sample batch
    sample_batch = next(iter(val_dataset))
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor):
            sample_batch[k] = v.unsqueeze(0)  # Add batch dimension
    
    # Generate adversarial examples using different attacks
    print("\nGenerating adversarial examples...")
    adversarial_batches = {
        'FGSM': analyzer.fgsm_attack(sample_batch),
        'C&W': analyzer.carlini_wagner_attack(sample_batch),
        'DeepFool': analyzer.deepfool_attack(sample_batch)
    }
    
    # Plot adversarial examples
    plot_adversarial_examples(sample_batch, adversarial_batches)
    
    # Analyze decision boundary
    print("\nAnalyzing decision boundary...")
    boundary_results = analyzer.analyze_decision_boundary(sample_batch)
    plot_decision_boundary_analysis(boundary_results)
    
    # Analyze gradient landscape
    print("\nAnalyzing gradient landscape...")
    landscape_results = analyzer.analyze_gradient_landscape(sample_batch)
    plot_gradient_landscape(landscape_results)
    
    # Estimate Lipschitz constants
    print("\nEstimating Lipschitz constants...")
    lipschitz = analyzer.estimate_lipschitz_constant(sample_batch)
    plot_lipschitz_constants(lipschitz)
    
    # Print comprehensive analysis results
    print("\nComprehensive robustness analysis results:")
    
    print("\nLipschitz Constants:")
    for modality, constant in lipschitz.items():
        print(f"  {modality}: {constant:.4f}")
    
    print("\nDecision Boundary Statistics:")
    for modality, results in boundary_results.items():
        distances = results['distances']
        print(f"  {modality}:")
        print(f"    Mean distance: {np.mean(distances):.4f}")
        print(f"    Min distance: {np.min(distances):.4f}")
        print(f"    Max distance: {np.max(distances):.4f}")
    
    print("\nGradient Landscape Statistics:")
    for modality, results in landscape_results.items():
        gradients = results['gradients']
        grad_norms = np.linalg.norm(gradients, axis=1)
        print(f"  {modality}:")
        print(f"    Mean gradient norm: {np.mean(grad_norms):.4f}")
        print(f"    Max gradient norm: {np.max(grad_norms):.4f}")
        print(f"    Gradient norm std: {np.std(grad_norms):.4f}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 