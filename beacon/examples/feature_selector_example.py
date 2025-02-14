import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from beacon.models.feature_selector import AdaptiveFeatureSelector
from beacon.models.adaptive_ensemble import AdaptiveEnsemble
from beacon.models.multimodal import MultimodalFusion
from typing import Dict, List

def generate_synthetic_data(n_samples: int = 1000):
    """Generate synthetic multimodal data with known important features"""
    # Generate base features
    images = torch.randn(n_samples, 64)
    sequences = torch.randn(n_samples, 128)
    clinical = torch.randn(n_samples, 20)
    
    # Add important patterns to specific features
    important_features = {
        'images': [0, 10, 20, 30],
        'sequences': [0, 25, 50, 75, 100],
        'clinical': [0, 5, 10, 15]
    }
    
    for modality in important_features:
        data = locals()[modality]
        for idx in important_features[modality]:
            if modality == 'images':
                data[:, idx] += torch.sin(torch.linspace(0, 4*np.pi, n_samples))
            elif modality == 'sequences':
                data[:, idx] += torch.cos(torch.linspace(0, 2*np.pi, n_samples))
            else:
                data[:, idx] *= 2.0
    
    # Generate labels based on important features
    combined_signal = (
        images[:, important_features['images']].mean(1) +
        sequences[:, important_features['sequences']].mean(1) +
        clinical[:, important_features['clinical']].mean(1)
    )
    labels = (combined_signal > combined_signal.mean()).long()
    
    return {
        'images': images,
        'sequences': sequences,
        'clinical': clinical,
        'labels': labels,
        'true_important_features': important_features
    }

def plot_feature_importance(importance_history: dict, 
                          true_important_features: dict,
                          save_path: str = None):
    """
    Plot feature importance evolution
    Args:
        importance_history: History of importance scores
        true_important_features: True important features for validation
        save_path: Optional path to save plot
    """
    n_modalities = len(importance_history)
    fig, axes = plt.subplots(n_modalities, 1, figsize=(12, 4*n_modalities))
    if n_modalities == 1:
        axes = [axes]
    
    for ax, (modality, history) in zip(axes, importance_history.items()):
        scores = torch.stack(history)
        
        # Plot heatmap of importance scores over time
        sns.heatmap(scores.cpu().numpy(), 
                   ax=ax,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Importance Score'})
        
        # Highlight true important features
        if modality in true_important_features:
            true_features = true_important_features[modality]
            ax.vlines(true_features, 
                     0, 
                     len(history), 
                     colors='blue', 
                     linestyles='dashed',
                     alpha=0.5)
        
        ax.set_title(f'{modality} Feature Importance Evolution')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Update Step')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_selected_features(feature_masks: dict,
                         true_important_features: dict,
                         save_path: str = None):
    """Plot currently selected features"""
    n_modalities = len(feature_masks)
    fig, axes = plt.subplots(n_modalities, 1, figsize=(12, 2*n_modalities))
    
    for idx, (modality, mask) in enumerate(feature_masks.items()):
        ax = axes[idx]
        
        # Plot feature mask
        ax.bar(range(len(mask)), mask.numpy())
        
        # Highlight true important features
        true_features = true_important_features[modality]
        for feature in true_features:
            ax.axvline(x=feature, color='red', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{modality} Selected Features')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Selection Status')
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_performance_comparison(original_perf: list,
                              selected_perf: list,
                              save_path: str = None):
    """Plot performance comparison"""
    plt.figure(figsize=(10, 6))
    
    steps = range(len(original_perf))
    plt.plot(steps, original_perf, label='Original Features')
    plt.plot(steps, selected_perf, label='Selected Features')
    
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_groups(feature_groups: List[List[int]],
                       importance_scores: torch.Tensor,
                       modality: str,
                       save_path: str = None):
    """
    Visualize feature groups and their importance
    Args:
        feature_groups: List of feature groups
        importance_scores: Feature importance scores
        modality: Modality name
        save_path: Optional path to save plot
    """
    n_groups = len(feature_groups)
    n_features = len(importance_scores)
    
    # Create group assignment matrix
    group_matrix = np.zeros((n_groups, n_features))
    for i, group in enumerate(feature_groups):
        group_matrix[i, group] = 1
    
    plt.figure(figsize=(12, 6))
    
    # Plot group assignments
    plt.subplot(2, 1, 1)
    sns.heatmap(group_matrix, 
                cmap='Set3',
                cbar_kws={'label': 'Group Assignment'})
    plt.title(f'{modality} Feature Groups')
    plt.xlabel('Feature Index')
    plt.ylabel('Group Index')
    
    # Plot mean importance per group
    plt.subplot(2, 1, 2)
    group_importance = [importance_scores[group].mean().item() for group in feature_groups]
    sns.barplot(x=range(n_groups), y=group_importance)
    plt.title('Group Importance Scores')
    plt.xlabel('Group Index')
    plt.ylabel('Mean Importance Score')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_cross_modal_interactions(interactions: Dict[str, torch.Tensor],
                                save_path: str = None):
    """
    Visualize cross-modal feature interactions
    Args:
        interactions: Dictionary of interaction matrices
        save_path: Optional path to save plot
    """
    n_pairs = len(interactions)
    fig = plt.figure(figsize=(15, 5 * ((n_pairs + 1) // 2)))
    
    for i, (pair, matrix) in enumerate(interactions.items(), 1):
        plt.subplot(((n_pairs + 1) // 2), 2, i)
        sns.heatmap(matrix.cpu().numpy(),
                   cmap='coolwarm',
                   center=0,
                   cbar_kws={'label': 'Interaction Strength'})
        plt.title(f'Interaction: {pair}')
        plt.xlabel('Features Modality 2')
        plt.ylabel('Features Modality 1')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_selection_metrics(selector: AdaptiveFeatureSelector,
                         save_path: str = None):
    """
    Plot feature selection metrics over time
    Args:
        selector: Feature selector instance
        save_path: Optional path to save plot
    """
    metrics = ['stability_scores', 'reduction_ratios']
    n_modalities = len(selector.feature_masks)
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    
    for ax, metric in zip(axes, metrics):
        for modality in selector.feature_masks.keys():
            values = getattr(selector, metric)[modality]
            ax.plot(values, label=modality)
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_method_comparison(importance_scores: Dict[str, Dict[str, torch.Tensor]],
                         save_path: str = None):
    """
    Compare feature importance scores from different methods
    Args:
        importance_scores: Dictionary of importance scores per method and modality
        save_path: Optional path to save plot
    """
    methods = list(importance_scores.keys())
    modalities = list(importance_scores[methods[0]].keys())
    
    fig = plt.figure(figsize=(15, 5 * len(modalities)))
    
    for i, modality in enumerate(modalities, 1):
        plt.subplot(len(modalities), 1, i)
        
        for method in methods:
            scores = importance_scores[method][modality].cpu().numpy()
            plt.plot(scores, label=method, alpha=0.7)
        
        plt.title(f'{modality} Feature Importance by Method')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    """Example of using adaptive feature selection"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path('outputs/feature_selection')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("Generating synthetic data...")
    data = generate_synthetic_data(1000)
    feature_dims = {
        'images': data['images'].shape[1],
        'sequences': data['sequences'].shape[1],
        'clinical': data['clinical'].shape[1]
    }
    
    # Configure feature selector
    selector_config = {
        'selection_method': 'ensemble',
        'ensemble_weights': {
            'attention': 0.2,
            'gradient': 0.2,
            'mutual_info': 0.2,
            'lasso': 0.2,
            'elastic_net': 0.2
        },
        'group_selection': {
            'enabled': True,
            'n_groups': 5
        },
        'cross_modal': {
            'enabled': True
        }
    }
    
    # Initialize feature selector
    print("Initializing feature selector...")
    selector = AdaptiveFeatureSelector(selector_config)
    selector.initialize_masks(feature_dims)
    
    # Configure ensemble model
    model_config = {
        'ensemble_method': 'stacking',
        'n_base_models': 3,
        'base_model_config': {
            'hidden_dims': [64, 32],
            'dropout_rate': 0.3
        },
        'meta_model_config': {
            'hidden_dims': [32, 16],
            'dropout_rate': 0.2
        }
    }
    
    # Initialize model
    print("Initializing ensemble model...")
    model = AdaptiveEnsemble(model_config)
    
    # Training loop
    print("Starting training...")
    n_epochs = 5
    batch_size = 32
    n_batches = len(data['labels']) // batch_size
    
    original_performance = []
    selected_performance = []
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Shuffle data
        indices = torch.randperm(len(data['labels']))
        
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # Create batch
            batch = {
                'images': data['images'][batch_indices],
                'sequences': data['sequences'][batch_indices],
                'clinical': data['clinical'][batch_indices],
                'labels': data['labels'][batch_indices]
            }
            
            # Forward pass with original features
            original_pred = model(batch)
            original_acc = (original_pred.argmax(dim=1) == 
                          batch['labels']).float().mean().item()
            original_performance.append(original_acc)
            
            # Update feature selection
            step = epoch * n_batches + batch_idx
            selector.update_feature_selection(batch, model, step)
            
            # Forward pass with selected features
            masked_batch = selector.apply_masks(batch)
            selected_pred = model(masked_batch)
            selected_acc = (selected_pred.argmax(dim=1) == 
                          batch['labels']).float().mean().item()
            selected_performance.append(selected_acc)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{n_batches}")
                print(f"Original accuracy: {original_acc:.4f}")
                print(f"Selected accuracy: {selected_acc:.4f}")
        
        # Plot current state
        plot_feature_importance(
            selector.get_importance_history(),
            data['true_important_features'],
            output_dir / f'importance_epoch_{epoch+1}.png'
        )
        
        plot_selected_features(
            selector.get_selected_features(),
            data['true_important_features'],
            output_dir / f'selected_features_epoch_{epoch+1}.png'
        )
    
    # Plot final performance comparison
    plot_performance_comparison(
        original_performance,
        selected_performance,
        output_dir / 'performance_comparison.png'
    )
    
    # Print final statistics
    print("\nFeature selection statistics:")
    selected_features = selector.get_selected_features()
    for modality, mask in selected_features.items():
        n_selected = mask.sum().item()
        n_total = len(mask)
        print(f"\n{modality}:")
        print(f"Selected features: {n_selected}/{n_total} "
              f"({100*n_selected/n_total:.1f}%)")
        
        true_features = data['true_important_features'][modality]
        n_correct = sum(mask[i] == 1 for i in true_features)
        print(f"True important features identified: "
              f"{n_correct}/{len(true_features)} "
              f"({100*n_correct/len(true_features):.1f}%)")

    # Plot feature groups for each modality
    for modality in feature_dims.keys():
        features = data['images'] if modality == 'images' else (data['sequences'] if modality == 'sequences' else data['clinical'])
        scores = selector.importance_history[modality][-1]
        groups, _ = selector._identify_feature_groups(features, scores)
        plot_feature_groups(groups, scores, modality)
    
    # Plot cross-modal interactions
    interactions = selector._compute_feature_interactions(batch)
    plot_cross_modal_interactions(interactions)
    
    # Plot selection metrics
    plot_selection_metrics(selector)
    
    # Compare different methods
    methods = ['attention', 'gradient', 'lasso', 'elastic_net']
    scores = {}
    for method in methods:
        config = {
            'selection_method': method,
            'ensemble_weights': {
                'attention': 0.2,
                'gradient': 0.2,
                'mutual_info': 0.2,
                'lasso': 0.2,
                'elastic_net': 0.2
            },
            'group_selection': {
                'enabled': True,
                'n_groups': 5
            },
            'cross_modal': {
                'enabled': True
            }
        }
        selector = AdaptiveFeatureSelector(config)
        scores[method] = selector._compute_importance_scores(batch, model, data['labels'])
    
    plot_method_comparison(scores)

if __name__ == '__main__':
    main() 