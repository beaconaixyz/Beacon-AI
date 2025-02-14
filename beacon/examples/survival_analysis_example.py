import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from beacon.models.survival_model import SurvivalModel
from beacon.data.processor import DataProcessor
from typing import Tuple, Dict

def generate_synthetic_data(n_samples: int = 1000, 
                          n_features: int = 50) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic survival data
    Args:
        n_samples: Number of samples
        n_features: Number of features
    Returns:
        Dictionary containing features, survival times, and event indicators
    """
    # Generate features
    features = torch.randn(n_samples, n_features)
    
    # Generate true risk scores based on some features
    true_risks = 0.5 * features[:, 0] + 0.3 * features[:, 1] - 0.2 * features[:, 2]
    
    # Generate survival times using exponential distribution
    survival_time = torch.exp(-(true_risks - true_risks.mean()) + torch.randn(n_samples) * 0.5)
    
    # Generate censoring times
    censoring_time = torch.exp(torch.randn(n_samples) * 0.5)
    
    # Get observed time (minimum of survival and censoring time)
    observed_time = torch.minimum(survival_time, censoring_time)
    
    # Generate event indicators (1 if event observed, 0 if censored)
    event_indicator = (survival_time <= censoring_time).float()
    
    return {
        'features': features,
        'survival_time': observed_time,
        'event_indicator': event_indicator
    }

def plot_survival_curves(model: SurvivalModel, 
                        features: torch.Tensor,
                        time_points: torch.Tensor,
                        n_samples: int = 5):
    """
    Plot survival curves for selected samples
    Args:
        model: Trained survival model
        features: Input features
        time_points: Time points for survival function
        n_samples: Number of samples to plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get random samples
    indices = np.random.choice(len(features), n_samples, replace=False)
    selected_features = features[indices]
    
    # Predict survival functions
    survival_probs = model.predict_survival_function(selected_features, time_points)
    
    # Plot survival curves
    for i in range(n_samples):
        plt.plot(time_points.numpy(), survival_probs[i].numpy(), 
                label=f'Patient {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Predicted Survival Curves')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_risk_distribution(risk_scores: torch.Tensor):
    """
    Plot distribution of risk scores
    Args:
        risk_scores: Predicted risk scores
    """
    plt.figure(figsize=(8, 5))
    plt.hist(risk_scores.numpy(), bins=30, density=True)
    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.title('Distribution of Risk Scores')
    plt.grid(True)
    plt.show()

def main():
    """Example of using the survival analysis model"""
    
    # Generate synthetic data
    print("Generating synthetic survival data...")
    data = generate_synthetic_data()
    
    # Split data into train and test sets
    n_samples = len(data['features'])
    n_train = int(0.8 * n_samples)
    indices = torch.randperm(n_samples)
    
    train_data = {
        'features': data['features'][indices[:n_train]],
        'survival_time': data['survival_time'][indices[:n_train]],
        'event_indicator': data['event_indicator'][indices[:n_train]]
    }
    
    test_data = {
        'features': data['features'][indices[n_train:]],
        'survival_time': data['survival_time'][indices[n_train:]],
        'event_indicator': data['event_indicator'][indices[n_train:]]
    }
    
    # Model configuration
    model_config = {
        'input_dim': data['features'].shape[1],
        'hidden_dims': [64, 32, 16],
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    }
    
    # Initialize model
    print("\nInitializing survival analysis model...")
    model = SurvivalModel(model_config)
    
    # Train model
    print("\nTraining model...")
    history = model.train(train_data, epochs=50, batch_size=32)
    
    # Plot training history
    plt.figure(figsize=(8, 5))
    plt.plot([h['epoch'] for h in history], [h['loss'] for h in history])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.grid(True)
    plt.show()
    
    # Predict risk scores for test set
    print("\nPredicting risk scores...")
    risk_scores = model.predict_risk(test_data['features'])
    
    # Plot risk score distribution
    plot_risk_distribution(risk_scores)
    
    # Predict and plot survival curves
    print("\nGenerating survival curves...")
    time_points = torch.linspace(0, 10, 100)
    plot_survival_curves(model, test_data['features'], time_points)
    
    # Save model
    print("\nSaving model...")
    model.save("survival_model.pt")
    
    print("\nDone! Check the plots for visualization of the results.")

if __name__ == "__main__":
    main() 