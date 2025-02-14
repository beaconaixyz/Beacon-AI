import torch
import numpy as np
import pandas as pd
from beacon.models.cancer_classifier import CancerClassifier
from beacon.data.processor import DataProcessor
from beacon.utils.metrics import Metrics

def main():
    """Basic example of using the BEACON framework"""
    
    # Configuration
    model_config = {
        'input_dim': 100,
        'hidden_dim': 256,
        'output_dim': 2,
        'learning_rate': 0.001,
        'dropout': 0.3
    }
    
    data_config = {
        'normalization': 'standard',
        'image_augmentation': False
    }
    
    # Initialize model and processor
    model = CancerClassifier(model_config)
    processor = DataProcessor(data_config)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic clinical data
    clinical_data = pd.DataFrame({
        'age': np.random.normal(60, 10, n_samples),
        'tumor_size': np.random.normal(3, 1, n_samples),
        'marker_level': np.random.normal(100, 20, n_samples)
    })
    
    # Process clinical data
    processed_features = processor.process_clinical_data(clinical_data)
    
    # Generate synthetic labels
    labels = torch.randint(0, 2, (n_samples,))
    
    # Convert to PyTorch tensors
    features = torch.FloatTensor(processed_features)
    
    # Prepare data dictionary
    data = {
        'features': features,
        'labels': labels
    }
    
    # Train model
    print("Training model...")
    history = model.train(data, epochs=10, batch_size=32)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions_prob = model.predict(features)
    predictions = torch.argmax(predictions_prob, dim=1)
    
    # Calculate metrics
    metrics = Metrics.calculate_all_metrics(
        labels.numpy(),
        predictions.numpy(),
        predictions_prob[:, 1].numpy()
    )
    
    # Print results
    print("\nTraining completed. Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main() 