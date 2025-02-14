import pytest
import torch
import numpy as np
from beacon.models.genomic_model import GenomicModel
from beacon.data.processor import DataProcessor

@pytest.fixture
def sample_genomic_data():
    """Create sample genomic data for testing"""
    n_samples = 100
    n_features = 1000  # Number of genes/features
    
    # Generate synthetic genomic features (e.g., gene expression values)
    features = torch.randn(n_samples, n_features)
    
    # Generate synthetic labels (e.g., cancer subtypes)
    labels = torch.randint(0, 2, (n_samples,))
    
    return {
        'features': features,
        'labels': labels
    }

def test_genomic_model_classification(sample_genomic_data):
    """Test genomic model for classification task"""
    # Configure model
    model_config = {
        'input_dim': sample_genomic_data['features'].shape[1],
        'hidden_dims': [512, 256, 128],
        'output_dim': 2,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'task_type': 'classification',
        'l1_lambda': 0.01  # L1 regularization for feature selection
    }
    
    # Initialize model
    model = GenomicModel(model_config)
    
    # Verify model architecture
    assert isinstance(model.model, torch.nn.Module)
    
    # Train model
    history = model.train(sample_genomic_data, epochs=2, batch_size=32)
    
    # Check training history
    assert isinstance(history, list)
    assert len(history) == 2  # Two epochs
    assert 'loss' in history[0]
    
    # Test predictions
    with torch.no_grad():
        predictions = model.predict(sample_genomic_data['features'])
    
    # Check predictions shape and type
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(sample_genomic_data['features']), 2)
    assert torch.all(predictions >= 0) and torch.all(predictions <= 1)

def test_genomic_model_regression(sample_genomic_data):
    """Test genomic model for regression task"""
    # Create continuous labels for regression
    sample_genomic_data['labels'] = torch.randn(len(sample_genomic_data['features']))
    
    # Configure model for regression
    model_config = {
        'input_dim': sample_genomic_data['features'].shape[1],
        'hidden_dims': [512, 256, 128],
        'output_dim': 1,
        'task_type': 'regression'
    }
    
    # Initialize model
    model = GenomicModel(model_config)
    
    # Train model
    history = model.train(sample_genomic_data, epochs=2)
    
    # Test predictions
    with torch.no_grad():
        predictions = model.predict(sample_genomic_data['features'])
    
    # Check predictions shape
    assert predictions.shape == (len(sample_genomic_data['features']), 1)

def test_feature_importance(sample_genomic_data):
    """Test feature importance calculation"""
    model_config = {
        'input_dim': sample_genomic_data['features'].shape[1],
        'output_dim': 2,
        'task_type': 'classification'
    }
    
    model = GenomicModel(model_config)
    
    # Calculate feature importance
    importance_scores = model.get_feature_importance(sample_genomic_data['features'])
    
    # Check importance scores
    assert isinstance(importance_scores, torch.Tensor)
    assert importance_scores.shape[0] == sample_genomic_data['features'].shape[1]
    assert torch.all(importance_scores >= 0)  # Attention weights should be non-negative

def test_model_save_load(tmp_path, sample_genomic_data):
    """Test model save and load functionality"""
    model_config = {
        'input_dim': sample_genomic_data['features'].shape[1],
        'output_dim': 2,
        'task_type': 'classification'
    }
    
    # Initialize and train original model
    model = GenomicModel(model_config)
    model.train(sample_genomic_data, epochs=1)
    
    # Save model
    save_path = tmp_path / "genomic_model.pt"
    model.save(str(save_path))
    
    # Load model
    new_model = GenomicModel(model_config)
    new_model.load(str(save_path))
    
    # Compare predictions
    with torch.no_grad():
        original_preds = model.predict(sample_genomic_data['features'])
        loaded_preds = new_model.predict(sample_genomic_data['features'])
    
    # Predictions should be identical
    assert torch.allclose(original_preds, loaded_preds) 