import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from beacon.models.genomic_model import GenomicModel
from beacon.data.processor import DataProcessor
from beacon.utils.metrics import Metrics

def load_genomic_data(data_path: str) -> pd.DataFrame:
    """
    Load genomic data from file
    Args:
        data_path: Path to data file
    Returns:
        DataFrame containing genomic data
    """
    # This is a placeholder for actual data loading
    # In real application, you would load real genomic data
    n_samples = 1000
    n_genes = 1000
    
    # Simulate gene expression data
    data = np.random.normal(0, 1, (n_samples, n_genes))
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    
    return pd.DataFrame(data, columns=gene_names)

def visualize_feature_importance(importance_scores: torch.Tensor, 
                               feature_names: list,
                               top_n: int = 20):
    """
    Visualize feature importance scores
    Args:
        importance_scores: Feature importance scores
        feature_names: Names of features
        top_n: Number of top features to show
    """
    # Convert to numpy and get top features
    scores = importance_scores.cpu().numpy()
    top_indices = np.argsort(scores)[-top_n:]
    top_scores = scores[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_scores)), top_scores)
    plt.yticks(range(len(top_scores)), top_features)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Important Features')
    plt.tight_layout()
    plt.show()

def plot_training_history(history: list):
    """
    Plot training history
    Args:
        history: Training history
    """
    plt.figure(figsize=(10, 5))
    plt.plot([h['epoch'] for h in history], [h['loss'] for h in history])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def main():
    """Example of using the genomic model"""
    
    # Load data
    print("Loading genomic data...")
    data = load_genomic_data("dummy_path")
    feature_names = data.columns.tolist()
    
    # Convert to tensor
    features = torch.FloatTensor(data.values)
    
    # Generate synthetic labels (e.g., cancer subtypes)
    labels = torch.randint(0, 2, (len(features),))
    
    # Model configuration
    model_config = {
        'input_dim': features.shape[1],
        'hidden_dims': [512, 256, 128],
        'output_dim': 2,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'task_type': 'classification',
        'l1_lambda': 0.01  # L1 regularization for feature selection
    }
    
    # Initialize model
    print("Initializing model...")
    model = GenomicModel(model_config)
    
    # Prepare data
    train_data = {
        'features': features,
        'labels': labels
    }
    
    # Train model
    print("Training model...")
    history = model.train(train_data, epochs=50, batch_size=32)
    
    # Plot training history
    plot_training_history(history)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(features)
    
    # Calculate metrics
    metrics = Metrics.calculate_all_metrics(
        labels.numpy(),
        torch.argmax(predictions, dim=1).numpy(),
        predictions[:, 1].numpy()
    )
    
    print("\nModel Performance:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Get and visualize feature importance
    print("\nCalculating feature importance...")
    importance_scores = model.get_feature_importance(features)
    visualize_feature_importance(importance_scores, feature_names)
    
    # Save model
    print("\nSaving model...")
    model.save("genomic_model.pt")

if __name__ == "__main__":
    main() 