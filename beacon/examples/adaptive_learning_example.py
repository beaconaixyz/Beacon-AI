import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from beacon.models.adaptive_learning import AdaptiveLearning
from beacon.models.adaptive_ensemble import AdaptiveEnsemble

def generate_synthetic_data(n_samples: int = 1000):
    """Generate synthetic multimodal data with varying difficulty levels"""
    # Generate data with different noise levels for each modality
    images = torch.randn(n_samples, 3, 64, 64)
    sequences = torch.randn(n_samples, 50, 128)
    clinical = torch.randn(n_samples, 20)
    
    # Add structured patterns
    images[:, 0] += torch.sin(torch.linspace(0, 10, n_samples)).unsqueeze(1).unsqueeze(1)
    sequences[:, 0] += torch.cos(torch.linspace(0, 10, n_samples)).unsqueeze(1)
    clinical += torch.randn(n_samples, 20) * 0.1
    
    # Generate labels with modality-specific contributions
    image_signal = images.mean(dim=(1, 2, 3))
    sequence_signal = sequences.mean(dim=(1, 2))
    clinical_signal = clinical.mean(dim=1)
    
    combined_signal = (0.4 * image_signal + 
                      0.3 * sequence_signal + 
                      0.3 * clinical_signal)
    
    labels = (combined_signal > combined_signal.mean()).long()
    
    return {
        'images': images,
        'sequences': sequences,
        'clinical': clinical,
        'labels': labels
    }

def plot_learning_rates(lr_history: list, save_path: str = None):
    """Plot learning rate evolution for each modality"""
    plt.figure(figsize=(10, 6))
    
    modalities = list(lr_history[0].keys())
    steps = range(len(lr_history))
    
    for modality in modalities:
        lrs = [h[modality] for h in lr_history]
        plt.plot(steps, lrs, label=f'{modality} Learning Rate')
    
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Evolution')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_performance_history(perf_history: list, save_path: str = None):
    """Plot performance evolution for each modality"""
    plt.figure(figsize=(10, 6))
    
    modalities = list(perf_history[0].keys())
    steps = range(len(perf_history))
    
    for modality in modalities:
        perfs = [h[modality] for h in perf_history]
        plt.plot(steps, perfs, label=f'{modality} Performance')
    
    plt.xlabel('Training Step')
    plt.ylabel('Performance')
    plt.title('Performance Evolution')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_modality_contributions(model: AdaptiveEnsemble, save_path: str = None):
    """Plot final modality contributions"""
    plt.figure(figsize=(8, 6))
    
    weights = model.modality_weights
    modalities = list(weights.keys())
    values = list(weights.values())
    
    plt.bar(modalities, values)
    plt.ylabel('Contribution Weight')
    plt.title('Final Modality Contributions')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    """Example of using adaptive learning"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path('outputs/adaptive_learning')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("Generating synthetic data...")
    train_data = generate_synthetic_data(1000)
    val_data = generate_synthetic_data(200)
    test_data = generate_synthetic_data(200)
    
    # Configure adaptive learning
    adaptive_config = {
        'initial_lr': 0.001,
        'min_lr': 1e-6,
        'max_lr': 0.1,
        'adaptation_window': 50,
        'momentum': 0.9,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8
    }
    
    # Initialize adaptive learning
    print("Initializing adaptive learning...")
    adaptive = AdaptiveLearning(adaptive_config)
    adaptive.initialize_learning_rates(['images', 'sequences', 'clinical'])
    
    # Configure ensemble model
    model_config = {
        'ensemble_method': 'stacking',
        'n_base_models': 5,
        'base_model_config': {
            'hidden_dims': [128, 64],
            'dropout_rate': 0.3
        },
        'meta_model_config': {
            'hidden_dims': [64, 32],
            'dropout_rate': 0.2
        },
        'adaptation_method': 'performance',
        'adaptation_frequency': 10
    }
    
    # Initialize ensemble model
    print("Initializing ensemble model...")
    model = AdaptiveEnsemble(model_config)
    
    # Training loop
    print("Starting training...")
    n_epochs = 10
    batch_size = 32
    n_batches = len(train_data['labels']) // batch_size
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Shuffle data
        indices = torch.randperm(len(train_data['labels']))
        
        for batch_idx in range(n_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # Create batch
            batch = {
                'images': train_data['images'][batch_indices],
                'sequences': train_data['sequences'][batch_indices],
                'clinical': train_data['clinical'][batch_indices],
                'labels': train_data['labels'][batch_indices]
            }
            
            # Forward pass and compute gradients
            predictions, gradients = model.forward_with_gradients(batch)
            
            # Compute performance metrics
            performance = {}
            for modality in ['images', 'sequences', 'clinical']:
                accuracy = (predictions[modality].argmax(dim=1) == 
                          batch['labels']).float().mean().item()
                performance[modality] = accuracy
            
            # Update learning rates
            learning_rates = adaptive.update_learning_rates(performance, gradients)
            
            # Update model with new learning rates
            model.update_learning_rates(learning_rates)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{n_batches}")
                print("Learning rates:", learning_rates)
                print("Performance:", performance)
        
        # Evaluate on validation set
        val_predictions = model.predict(val_data)
        val_accuracy = (val_predictions.argmax(dim=1) == 
                       val_data['labels']).float().mean().item()
        print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Plot training history
    print("\nPlotting training history...")
    plot_learning_rates(adaptive.lr_history, 
                       output_dir / 'learning_rates.png')
    plot_performance_history(adaptive.performance_history,
                           output_dir / 'performance.png')
    plot_modality_contributions(model,
                              output_dir / 'modality_contributions.png')
    
    # Final evaluation
    print("\nFinal evaluation...")
    test_predictions = model.predict(test_data)
    test_accuracy = (test_predictions.argmax(dim=1) == 
                    test_data['labels']).float().mean().item()
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Print learning rate statistics
    print("\nLearning rate statistics:")
    lr_stats = adaptive.get_learning_rate_stats()
    for modality, stats in lr_stats.items():
        print(f"\n{modality}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.6f}")
    
    # Print performance statistics
    print("\nPerformance statistics:")
    perf_stats = adaptive.get_performance_stats()
    for modality, stats in perf_stats.items():
        print(f"\n{modality}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save model and adaptive learning state
    print("\nSaving model and adaptive learning state...")
    torch.save({
        'model_state': model.state_dict(),
        'adaptive_state': {
            'learning_rates': adaptive.learning_rates,
            'momentum_buffer': adaptive.momentum_buffer,
            'first_moment': adaptive.first_moment,
            'second_moment': adaptive.second_moment,
            'step_count': adaptive.step_count,
            'performance_history': adaptive.performance_history,
            'lr_history': adaptive.lr_history
        }
    }, output_dir / 'adaptive_learning_state.pt')

if __name__ == '__main__':
    main() 