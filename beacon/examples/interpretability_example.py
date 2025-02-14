import torch
import numpy as np
import matplotlib.pyplot as plt
from beacon.models.cancer_classifier import CancerClassifier
from beacon.utils.interpretability import ModelInterpreter
from beacon.data.processor import DataProcessor

def generate_synthetic_data(n_samples: int = 1000):
    """Generate synthetic patient data for demonstration"""
    # Generate features with known patterns
    age = np.random.normal(60, 10, n_samples)
    tumor_size = np.random.normal(3, 1, n_samples)
    marker_level = np.random.normal(100, 20, n_samples)
    
    # Create synthetic relationships
    risk_score = (
        0.3 * (age - 60) / 10 +  # Age contribution
        0.5 * (tumor_size - 3) +  # Tumor size contribution
        0.2 * (marker_level - 100) / 20  # Marker level contribution
    )
    
    # Generate labels based on risk score
    probabilities = 1 / (1 + np.exp(-risk_score))  # Sigmoid function
    labels = (probabilities > 0.5).astype(int)
    
    # Combine features
    features = np.column_stack([age, tumor_size, marker_level])
    
    return (
        torch.FloatTensor(features),
        torch.LongTensor(labels),
        ['Age', 'Tumor Size', 'Marker Level']
    )

def plot_feature_distributions(features: torch.Tensor, labels: torch.Tensor,
                             feature_names: list):
    """Plot feature distributions by class"""
    fig, axes = plt.subplots(1, len(feature_names), figsize=(15, 5))
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        for label in [0, 1]:
            mask = labels == label
            ax.hist(features[mask, i], bins=30, alpha=0.5,
                   label=f'Class {label}')
        ax.set_title(name)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Example of using model interpretability tools"""
    print("Generating synthetic data...")
    features, labels, feature_names = generate_synthetic_data()
    
    # Plot feature distributions
    print("\nPlotting feature distributions...")
    plot_feature_distributions(features, labels, feature_names)
    
    # Configure and train model
    print("\nTraining model...")
    model_config = {
        'input_dim': features.shape[1],
        'hidden_dim': 64,
        'output_dim': 2
    }
    
    model = CancerClassifier(model_config)
    model.train({
        'features': features,
        'labels': labels
    })
    
    # Initialize interpreter
    interpreter = ModelInterpreter({
        'method': 'integrated_gradients',
        'n_steps': 50,
        'feature_names': feature_names
    })
    
    # Get feature attributions
    print("\nCalculating feature attributions...")
    sample_idx = np.random.choice(len(features), 5)  # Select 5 random samples
    attributions, metadata = interpreter.explain_prediction(
        model,
        features[sample_idx]
    )
    
    # Visualize attributions
    print("\nVisualizing feature attributions...")
    interpreter.visualize_attributions(
        attributions,
        features[sample_idx],
        feature_names,
        'attributions'
    )
    
    # Analyze feature interactions
    print("\nAnalyzing feature interactions...")
    interactions = interpreter.analyze_feature_interactions(
        model,
        features[sample_idx],
        feature_names
    )
    
    print("\nFeature Interactions:")
    for interaction, strength in sorted(
        interactions.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{interaction}: {strength:.4f}")
    
    # Generate counterfactuals
    print("\nGenerating counterfactual examples...")
    target_class = 1 - labels[sample_idx]  # Opposite class
    counterfactuals, distances = interpreter.generate_counterfactuals(
        model,
        features[sample_idx],
        target_class
    )
    
    # Compare original samples with counterfactuals
    print("\nComparing original samples with counterfactuals:")
    for i in range(len(sample_idx)):
        print(f"\nSample {i + 1}:")
        print("Original features:")
        for name, value in zip(feature_names, features[sample_idx[i]]):
            print(f"  {name}: {value:.2f}")
        print("Counterfactual features:")
        for name, value in zip(feature_names, counterfactuals[i]):
            print(f"  {name}: {value:.2f}")
        print(f"Distance: {distances[i]:.2f}")
    
    # Try different interpretation methods
    print("\nComparing different interpretation methods...")
    methods = ['integrated_gradients', 'deep_lift', 'shap']
    
    for method in methods:
        print(f"\nUsing {method}...")
        interpreter.method = method
        attributions, metadata = interpreter.explain_prediction(
            model,
            features[sample_idx]
        )
        
        # Visualize attributions for each method
        interpreter.visualize_attributions(
            attributions,
            features[sample_idx],
            feature_names,
            f'attributions_{method}'
        )

if __name__ == "__main__":
    main() 