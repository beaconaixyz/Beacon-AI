import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from beacon.models.adaptive_ensemble import AdaptiveEnsemble
from beacon.data.processor import DataProcessor

def generate_synthetic_data(n_samples: int = 100):
    """Generate synthetic multimodal data"""
    # Generate image data (batch_size, channels, height, width)
    images = torch.randn(n_samples, 3, 32, 32)
    
    # Generate sequence data (batch_size, sequence_length, features)
    sequences = torch.randn(n_samples, 10, 20)
    
    # Generate clinical data (batch_size, features)
    clinical = torch.randn(n_samples, 15)
    
    # Generate labels with modality-specific patterns
    image_contrib = torch.sum(images.view(n_samples, -1), dim=1)
    seq_contrib = torch.sum(sequences.view(n_samples, -1), dim=1)
    clinical_contrib = torch.sum(clinical, dim=1)
    
    # Create labels with different modality contributions
    combined_score = (0.4 * image_contrib + 0.3 * seq_contrib + 0.3 * clinical_contrib)
    labels = (combined_score > combined_score.mean()).long()
    
    return {
        'image': images,
        'sequence': sequences,
        'clinical': clinical,
        'labels': labels
    }

def plot_weight_evolution(weight_history: list, save_path: str = None):
    """Plot evolution of modality weights over time"""
    plt.figure(figsize=(10, 6))
    
    # Extract weights for each modality
    steps = range(len(weight_history))
    image_weights = [w['image'] for w in weight_history]
    sequence_weights = [w['sequence'] for w in weight_history]
    clinical_weights = [w['clinical'] for w in weight_history]
    
    # Plot weights
    plt.plot(steps, image_weights, label='Image', marker='o')
    plt.plot(steps, sequence_weights, label='Sequence', marker='s')
    plt.plot(steps, clinical_weights, label='Clinical', marker='^')
    
    plt.xlabel('Training Step')
    plt.ylabel('Modality Weight')
    plt.title('Evolution of Modality Weights')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_performance_comparison(performances: dict, save_path: str = None):
    """Plot performance comparison between modalities"""
    plt.figure(figsize=(8, 6))
    
    modalities = list(performances.keys())
    accuracies = [performances[m] for m in modalities]
    
    plt.bar(modalities, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Performance Comparison Across Modalities')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_uncertainty_distribution(uncertainties: dict, save_path: str = None):
    """Plot uncertainty distribution for each modality"""
    plt.figure(figsize=(12, 4))
    
    for i, (modality, values) in enumerate(uncertainties.items(), 1):
        plt.subplot(1, 3, i)
        sns.histplot(values, kde=True)
        plt.title(f'{modality} Uncertainty')
        plt.xlabel('Uncertainty Value')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    """Example of using adaptive ensemble learning"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path('outputs/adaptive_ensemble')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("Generating synthetic data...")
    train_data = generate_synthetic_data(1000)
    val_data = generate_synthetic_data(200)
    test_data = generate_synthetic_data(200)
    
    # Configure model
    model_config = {
        'ensemble_method': 'stacking',
        'n_base_models': 5,
        'base_model_config': {
            'image': {
                'input_dim': (3, 32, 32),
                'hidden_dim': 64
            },
            'sequence': {
                'input_dim': 20,
                'hidden_dim': 64,
                'num_layers': 2
            },
            'clinical': {
                'input_dim': 15,
                'hidden_dim': 64
            }
        },
        'adaptation_method': 'performance',
        'adaptation_frequency': 10,
        'adaptation_window': 50,
        'min_weight': 0.1,
        'weight_decay_factor': 0.99
    }
    
    # Initialize model
    print("Initializing adaptive ensemble...")
    model = AdaptiveEnsemble(model_config)
    
    # Train model
    print("Training model...")
    weight_history = []
    for epoch in range(5):
        history = model.train(train_data, val_data)
        weight_history.append(history['modality_weights'])
        print(f"Epoch {epoch + 1}/5 - Weights:", history['modality_weights'])
    
    # Plot weight evolution
    print("\nPlotting weight evolution...")
    plot_weight_evolution(weight_history, output_dir / 'weight_evolution.png')
    
    # Evaluate individual modality performance
    print("\nEvaluating individual modalities...")
    performances = {}
    uncertainties = {}
    
    for modality in ['image', 'sequence', 'clinical']:
        # Create single-modality batch
        single_modal_batch = {modality: test_data[modality]}
        
        # Get predictions
        predictions, uncertainty = model.predict(single_modal_batch)
        
        # Calculate accuracy
        accuracy = (predictions.argmax(dim=1) == test_data['labels']).float().mean()
        performances[modality] = accuracy.item()
        
        # Store uncertainties
        uncertainties[modality] = uncertainty.numpy()
    
    # Plot performance comparison
    print("Plotting performance comparison...")
    plot_performance_comparison(performances, output_dir / 'performance_comparison.png')
    
    # Plot uncertainty distributions
    print("Plotting uncertainty distributions...")
    plot_uncertainty_distribution(uncertainties, output_dir / 'uncertainty_distribution.png')
    
    # Final evaluation
    print("\nFinal Evaluation:")
    predictions, uncertainties = model.predict(test_data)
    accuracy = (predictions.argmax(dim=1) == test_data['labels']).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Analyze high uncertainty predictions
    uncertainty_threshold = uncertainties.mean() + uncertainties.std()
    high_uncertainty_mask = uncertainties > uncertainty_threshold
    high_uncertainty_acc = (predictions[high_uncertainty_mask].argmax(dim=1) == 
                          test_data['labels'][high_uncertainty_mask]).float().mean()
    
    print(f"High Uncertainty Predictions:")
    print(f"  Count: {high_uncertainty_mask.sum()}")
    print(f"  Accuracy: {high_uncertainty_acc:.4f}")
    
    # Save model
    print("\nSaving model...")
    model.save_ensemble(str(output_dir / 'adaptive_ensemble.pt'))

if __name__ == '__main__':
    main() 