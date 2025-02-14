import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, calibration_curve
from beacon.models.advanced_ensemble import AdvancedEnsemble
from beacon.examples.multimodal_example import generate_synthetic_data

def plot_ensemble_predictions(predictions, uncertainties, true_labels, save_path=None):
    """Plot model predictions with uncertainty."""
    plt.figure(figsize=(10, 6))
    
    # Sort by uncertainty
    sorted_indices = torch.argsort(uncertainties)
    predictions = predictions[sorted_indices]
    uncertainties = uncertainties[sorted_indices]
    true_labels = true_labels[sorted_indices]
    
    # Plot predictions and uncertainties
    plt.errorbar(range(len(predictions)), 
                predictions.cpu().numpy(),
                yerr=uncertainties.cpu().numpy(),
                fmt='o', 
                label='Predictions with Uncertainty')
    plt.scatter(range(len(true_labels)), 
               true_labels.cpu().numpy(), 
               c='red', 
               marker='x', 
               label='True Labels')
    
    plt.xlabel('Sample Index (sorted by uncertainty)')
    plt.ylabel('Prediction / True Label')
    plt.title('Ensemble Predictions with Uncertainty')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_calibration_curve(ensemble, val_data, save_path=None):
    """Plot calibration curve for ensemble."""
    predictions, _ = ensemble.predict(val_data)
    prob_true, prob_pred = calibration_curve(val_data['labels'].cpu().numpy(),
                                           predictions.cpu().numpy(),
                                           n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, 's-', label='Ensemble')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_importance(importance_scores, modalities, save_path=None):
    """Plot feature importance scores."""
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    x = range(len(modalities))
    plt.bar(x, importance_scores)
    plt.xticks(x, modalities)
    
    plt.xlabel('Modality')
    plt.ylabel('Feature Importance Score')
    plt.title('Feature Importance Across Modalities')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_uncertainty_distribution(uncertainties, method, save_path=None):
    """Plot distribution of uncertainties."""
    plt.figure(figsize=(10, 6))
    
    sns.histplot(uncertainties.cpu().numpy(), kde=True)
    plt.xlabel('Uncertainty Value')
    plt.ylabel('Count')
    plt.title(f'Distribution of Uncertainties ({method})')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory for plots
    output_dir = Path('outputs/advanced_ensemble')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    train_data = generate_synthetic_data(100)
    val_data = generate_synthetic_data(50)
    test_data = generate_synthetic_data(50)
    
    # Configure the ensemble model
    config = {
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
        'meta_model_config': {
            'hidden_dim': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        },
        'boosting_config': {
            'n_rounds': 3,
            'learning_rate': 0.1,
            'subsample_ratio': 0.8
        },
        'uncertainty_config': {
            'method': 'evidential',
            'prior_scale': 1.0,
            'n_samples': 5
        }
    }
    
    # Initialize and train ensemble
    print("Initializing and training ensemble...")
    ensemble = AdvancedEnsemble(config)
    history = ensemble.train(train_data, val_data)
    
    # Make predictions and get uncertainties
    print("Making predictions...")
    predictions, uncertainties = ensemble.predict(test_data)
    
    # Plot predictions with uncertainties
    print("Plotting predictions...")
    plot_ensemble_predictions(
        predictions,
        uncertainties,
        test_data['labels'],
        save_path=output_dir / 'predictions.png'
    )
    
    # Plot calibration curve
    print("Plotting calibration curve...")
    plot_calibration_curve(
        ensemble,
        val_data,
        save_path=output_dir / 'calibration.png'
    )
    
    # Get and plot feature importance
    print("Analyzing feature importance...")
    importance_scores = ensemble.get_feature_importance(val_data)
    plot_feature_importance(
        importance_scores,
        ['Image', 'Sequence', 'Clinical'],
        save_path=output_dir / 'feature_importance.png'
    )
    
    # Analyze uncertainty with different methods
    print("Analyzing uncertainties...")
    for method in ['evidential', 'bayesian', 'ensemble']:
        ensemble.config['uncertainty_config']['method'] = method
        _, uncertainties = ensemble.predict(test_data)
        plot_uncertainty_distribution(
            uncertainties,
            method,
            save_path=output_dir / f'uncertainty_{method}.png'
        )
    
    # Save the ensemble model
    print("Saving ensemble model...")
    ensemble.save_ensemble(str(output_dir / 'ensemble_model'))
    
    # Evaluate model performance
    print("\nModel Evaluation:")
    predictions, uncertainties = ensemble.predict(test_data)
    accuracy = (predictions.argmax(dim=1) == test_data['labels']).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Analyze high uncertainty predictions
    high_uncertainty_mask = uncertainties > uncertainties.mean() + uncertainties.std()
    high_uncertainty_acc = (predictions[high_uncertainty_mask].argmax(dim=1) == 
                          test_data['labels'][high_uncertainty_mask]).float().mean()
    print(f"High Uncertainty Predictions Accuracy: {high_uncertainty_acc:.4f}")
    print(f"Number of High Uncertainty Predictions: {high_uncertainty_mask.sum()}")

if __name__ == '__main__':
    main() 