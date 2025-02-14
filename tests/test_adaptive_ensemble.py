import pytest
import torch
import numpy as np
from beacon.models.adaptive_ensemble import AdaptiveEnsemble

@pytest.fixture
def sample_batch():
    """Create sample batch for testing"""
    batch_size = 10
    n_features = 20
    
    # Create synthetic multimodal data
    batch = {
        'images': torch.randn(batch_size, 3, 64, 64),
        'sequences': torch.randn(batch_size, 50, 128),
        'clinical': torch.randn(batch_size, n_features),
        'labels': torch.randint(0, 2, (batch_size,))
    }
    return batch

@pytest.fixture
def model_config():
    """Create model configuration"""
    return {
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
        'adaptation_frequency': 10,
        'min_weight': 0.1,
        'weight_decay': 0.99,
        'uncertainty_method': 'entropy'
    }

def test_initialization(model_config):
    """Test model initialization"""
    model = AdaptiveEnsemble(model_config)
    assert model.config['ensemble_method'] == 'stacking'
    assert model.config['adaptation_method'] == 'performance'
    assert len(model.modality_weights) == 3  # images, sequences, clinical

def test_weight_initialization(model_config):
    """Test weight initialization"""
    model = AdaptiveEnsemble(model_config)
    weights = model.modality_weights
    
    # Check initial weights
    assert all(w >= 0 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(w >= model.config['min_weight'] for w in weights.values())

def test_performance_based_adaptation(model_config, sample_batch):
    """Test performance-based weight adaptation"""
    model = AdaptiveEnsemble(model_config)
    model.config['adaptation_method'] = 'performance'
    
    # Get initial weights
    initial_weights = model.modality_weights.copy()
    
    # Update weights
    new_weights = model.update_weights(sample_batch, sample_batch['labels'])
    
    # Check weight properties
    assert len(new_weights) == len(initial_weights)
    assert abs(sum(new_weights.values()) - 1.0) < 1e-6
    assert all(w >= model.config['min_weight'] for w in new_weights.values())

def test_uncertainty_based_adaptation(model_config, sample_batch):
    """Test uncertainty-based weight adaptation"""
    model = AdaptiveEnsemble(model_config)
    model.config['adaptation_method'] = 'uncertainty'
    
    # Update weights
    new_weights = model.update_weights(sample_batch, sample_batch['labels'])
    
    # Check weight properties
    assert len(new_weights) == 3
    assert abs(sum(new_weights.values()) - 1.0) < 1e-6
    assert all(w >= 0 for w in new_weights.values())

def test_gradient_based_adaptation(model_config, sample_batch):
    """Test gradient-based weight adaptation"""
    model = AdaptiveEnsemble(model_config)
    model.config['adaptation_method'] = 'gradient'
    
    # Update weights
    new_weights = model.update_weights(sample_batch, sample_batch['labels'])
    
    # Check weight properties
    assert len(new_weights) == 3
    assert abs(sum(new_weights.values()) - 1.0) < 1e-6
    assert all(w >= model.config['min_weight'] for w in new_weights.values())

def test_weight_history(model_config, sample_batch):
    """Test weight history tracking"""
    model = AdaptiveEnsemble(model_config)
    
    # Train for a few steps
    for _ in range(3):
        model.train(sample_batch)
    
    # Check weight history
    assert hasattr(model, 'weight_history')
    assert len(model.weight_history) > 0
    assert all(isinstance(w, dict) for w in model.weight_history)

def test_prediction_with_weights(model_config, sample_batch):
    """Test predictions with weighted ensemble"""
    model = AdaptiveEnsemble(model_config)
    
    # Make predictions
    predictions, uncertainties = model.predict(sample_batch)
    
    # Check predictions
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape[0] == len(sample_batch['labels'])
    assert isinstance(uncertainties, torch.Tensor)
    assert uncertainties.shape[0] == len(sample_batch['labels'])

def test_training_with_adaptation(model_config, sample_batch):
    """Test training with weight adaptation"""
    model = AdaptiveEnsemble(model_config)
    
    # Train model
    history = model.train(sample_batch)
    
    # Check training history
    assert isinstance(history, dict)
    assert 'loss' in history
    assert 'weight_history' in history
    assert len(history['weight_history']) > 0

def test_minimum_weight_constraint(model_config, sample_batch):
    """Test minimum weight constraint"""
    model = AdaptiveEnsemble(model_config)
    model.config['min_weight'] = 0.2
    
    # Update weights
    new_weights = model.update_weights(sample_batch, sample_batch['labels'])
    
    # Check minimum weight constraint
    assert all(w >= model.config['min_weight'] for w in new_weights.values())

def test_weight_decay(model_config, sample_batch):
    """Test weight decay mechanism"""
    model = AdaptiveEnsemble(model_config)
    model.config['weight_decay'] = 0.9
    
    # Get initial weights
    initial_weights = model.modality_weights.copy()
    
    # Update weights multiple times
    for _ in range(3):
        new_weights = model.update_weights(sample_batch, sample_batch['labels'])
    
    # Check weight decay effect
    for modality in initial_weights:
        weight_diff = abs(new_weights[modality] - initial_weights[modality])
        assert weight_diff <= 1.0  # Weights should not change too drastically

def test_invalid_adaptation_method(model_config):
    """Test invalid adaptation method handling"""
    model_config['adaptation_method'] = 'invalid_method'
    with pytest.raises(ValueError):
        AdaptiveEnsemble(model_config)

def test_device_handling(model_config, sample_batch):
    """Test device handling"""
    model = AdaptiveEnsemble(model_config)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in sample_batch.items()}
    
    # Test predictions
    predictions, uncertainties = model.predict(batch)
    assert predictions.device == device
    assert uncertainties.device == device

def test_empty_batch_handling(model_config):
    """Test empty batch handling"""
    model = AdaptiveEnsemble(model_config)
    empty_batch = {
        'images': torch.empty(0, 3, 64, 64),
        'sequences': torch.empty(0, 50, 128),
        'clinical': torch.empty(0, 20),
        'labels': torch.empty(0)
    }
    
    with pytest.raises(ValueError):
        model.predict(empty_batch) 