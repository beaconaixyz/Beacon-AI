import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from beacon.models.ensemble import MultimodalEnsemble
from beacon.examples.multimodal_example import (
    generate_synthetic_data,
    MultimodalDataset,
    collate_fn
)

def plot_ensemble_predictions(predictions: torch.Tensor,
                            uncertainties: torch.Tensor,
                            true_labels: torch.Tensor,
                            title: str = "Ensemble Predictions"):
    """
    Plot ensemble predictions with uncertainty
    Args:
        predictions: Model predictions
        uncertainties: Prediction uncertainties
        true_labels: True labels
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by uncertainty
    sorted_indices = uncertainties.argsort()
    predictions = predictions[sorted_indices]
    uncertainties = uncertainties[sorted_indices]
    true_labels = true_labels[sorted_indices]
    
    # Plot predictions
    plt.errorbar(range(len(predictions)),
                predictions.argmax(dim=1).cpu(),
                yerr=uncertainties.cpu(),
                fmt='o',
                capsize=5,
                label='Predictions')
    
    # Plot true labels
    plt.scatter(range(len(true_labels)),
               true_labels.cpu(),
               c='red',
               marker='x',
               label='True Labels')
    
    plt.title(title)
    plt.xlabel('Sample Index (sorted by uncertainty)')
    plt.ylabel('Class')
    plt.legend()
    plt.show()

def plot_calibration_curve(ensemble: MultimodalEnsemble,
                          val_data: Dict[str, torch.Tensor],
                          n_bins: int = 10):
    """
    Plot calibration curve
    Args:
        ensemble: Trained ensemble
        val_data: Validation data
        n_bins: Number of bins for calibration
    """
    predictions, _ = ensemble.predict(val_data)
    confidences = predictions.max(dim=1)[0].cpu().numpy()
    true_labels = val_data['labels'].cpu().numpy()
    pred_labels = predictions.argmax(dim=1).cpu().numpy()
    
    # Calculate calibration curve
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_confidences.append(confidences[mask].mean())
            bin_accuracies.append((pred_labels[mask] == true_labels[mask]).mean())
    
    bin_confidences = np.array(bin_confidences)
    bin_accuracies = np.array(bin_accuracies)
    
    # Plot calibration curve
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model Calibration')
    plt.title('Calibration Curve')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importance(importance_scores: Dict[str, torch.Tensor]):
    """
    Plot feature importance scores
    Args:
        importance_scores: Dictionary of feature importance scores
    """
    plt.figure(figsize=(12, 6))
    
    n_modalities = len(importance_scores)
    for i, (modality, scores) in enumerate(importance_scores.items()):
        plt.subplot(1, n_modalities, i + 1)
        sns.heatmap(scores.cpu().numpy(),
                   cmap='viridis',
                   cbar_kws={'label': 'Importance'})
        plt.title(f'{modality} Feature Importance')
    
    plt.tight_layout()
    plt.show()

def plot_uncertainty_distribution(uncertainties: torch.Tensor,
                                method: str = "Entropy"):
    """
    Plot distribution of uncertainties
    Args:
        uncertainties: Model uncertainties
        method: Uncertainty estimation method
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(uncertainties.cpu().numpy(), bins=30)
    plt.title(f'Distribution of {method}-based Uncertainties')
    plt.xlabel('Uncertainty')
    plt.ylabel('Count')
    plt.show()

def main():
    """Example of using the multimodal ensemble"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    n_samples = 1000
    train_data, val_data = generate_synthetic_data(n_samples)
    
    # Create datasets
    train_dataset = MultimodalDataset(train_data)
    val_dataset = MultimodalDataset(val_data)
    
    # Model configuration
    model_config = {
        'n_models': 5,
        'model_config': {
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
                'input_dim': 20,
                'hidden_dim': 64
            },
            'fusion': {
                'method': 'attention',
                'hidden_dim': 128,
                'num_heads': 4
            },
            'output_dim': 2
        },
        'bootstrap_ratio': 0.8,
        'aggregation_method': 'mean',
        'uncertainty_method': 'entropy',
        'dropout_samples': 20,
        'temperature': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Initialize and train ensemble
    print("\nInitializing and training ensemble...")
    ensemble = MultimodalEnsemble(model_config)
    histories = ensemble.train(train_data)
    
    # Make predictions on validation set
    print("\nMaking predictions...")
    predictions, uncertainties = ensemble.predict(val_data)
    
    # Plot predictions with uncertainty
    print("\nPlotting predictions...")
    plot_ensemble_predictions(predictions, uncertainties, val_data['labels'])
    
    # Calibrate ensemble
    print("\nCalibrating ensemble...")
    optimal_temp = ensemble.calibrate(val_data)
    print(f"Optimal temperature: {optimal_temp:.3f}")
    
    # Plot calibration curve
    print("\nPlotting calibration curve...")
    plot_calibration_curve(ensemble, val_data)
    
    # Get feature importance
    print("\nCalculating feature importance...")
    importance_scores = ensemble.get_feature_importance(val_data)
    plot_feature_importance(importance_scores)
    
    # Perform Monte Carlo dropout
    print("\nPerforming Monte Carlo dropout...")
    mc_predictions, mc_uncertainties = ensemble.monte_carlo_dropout(val_data)
    
    # Plot uncertainty distributions
    print("\nPlotting uncertainty distributions...")
    plot_uncertainty_distribution(uncertainties, "Ensemble")
    plot_uncertainty_distribution(mc_uncertainties, "MC Dropout")
    
    # Save ensemble
    print("\nSaving ensemble...")
    ensemble.save_ensemble("ensemble_model")
    
    # Evaluate performance
    print("\nEvaluating performance...")
    pred_labels = predictions.argmax(dim=1)
    accuracy = (pred_labels == val_data['labels']).float().mean()
    print(f"Ensemble accuracy: {accuracy:.3f}")
    
    # Analyze high uncertainty predictions
    high_uncertainty_mask = uncertainties > uncertainties.median()
    high_uncertainty_acc = (pred_labels[high_uncertainty_mask] == 
                          val_data['labels'][high_uncertainty_mask]).float().mean()
    print(f"Accuracy on high uncertainty predictions: {high_uncertainty_acc:.3f}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 