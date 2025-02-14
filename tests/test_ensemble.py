import pytest
import torch
import numpy as np
from beacon.models.ensemble import MultimodalEnsemble

@pytest.fixture
def sample_batch():
    """Create sample batch for testing"""
    batch_size = 4
    return {
        'image': torch.randn(batch_size, 3, 32, 32),
        'sequence': torch.randn(batch_size, 10, 20),
        'clinical': torch.randn(batch_size, 20),
        'labels': torch.randint(0, 2, (batch_size,))
    }

@pytest.fixture
def model_config():
    """Create model configuration for testing"""
    return {
        'n_models': 3,
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
        'dropout_samples': 5,
        'temperature': 1.0
    }

@pytest.fixture
def ensemble(model_config):
    """Create ensemble for testing"""
    return MultimodalEnsemble(model_config)

def test_ensemble_initialization(ensemble, model_config):
    """Test ensemble initialization"""
    assert len(ensemble.models) == model_config['n_models']
    assert ensemble.config['bootstrap_ratio'] == model_config['bootstrap_ratio']
    assert ensemble.config['aggregation_method'] == model_config['aggregation_method']

def test_ensemble_training(ensemble, sample_batch):
    """Test ensemble training"""
    histories = ensemble.train(sample_batch)
    assert len(histories) == len(ensemble.models)
    assert all(isinstance(h, dict) for h in histories)

def test_ensemble_prediction(ensemble, sample_batch):
    """Test ensemble prediction"""
    predictions, uncertainties = ensemble.predict(sample_batch)
    
    assert isinstance(predictions, torch.Tensor)
    assert isinstance(uncertainties, torch.Tensor)
    assert predictions.shape[0] == len(sample_batch['image'])
    assert uncertainties.shape[0] == len(sample_batch['image'])

def test_monte_carlo_dropout(ensemble, sample_batch):
    """Test Monte Carlo dropout"""
    mean_pred, uncertainty = ensemble.monte_carlo_dropout(sample_batch)
    
    assert isinstance(mean_pred, torch.Tensor)
    assert isinstance(uncertainty, torch.Tensor)
    assert mean_pred.shape[0] == len(sample_batch['image'])
    assert uncertainty.shape[0] == len(sample_batch['image'])

def test_ensemble_calibration(ensemble, sample_batch):
    """Test ensemble calibration"""
    temperature = ensemble.calibrate(sample_batch)
    
    assert isinstance(temperature, float)
    assert temperature > 0
    assert ensemble.config['temperature'] == temperature

def test_feature_importance(ensemble, sample_batch):
    """Test feature importance calculation"""
    importance_scores = ensemble.get_feature_importance(sample_batch)
    
    assert isinstance(importance_scores, dict)
    assert set(importance_scores.keys()) == {'image', 'sequence', 'clinical'}
    assert all(isinstance(v, torch.Tensor) for v in importance_scores.values())

def test_save_load_ensemble(ensemble, sample_batch, tmp_path):
    """Test saving and loading ensemble"""
    # Train ensemble
    ensemble.train(sample_batch)
    
    # Get predictions before saving
    pred_before, _ = ensemble.predict(sample_batch)
    
    # Save ensemble
    save_path = str(tmp_path / "ensemble")
    ensemble.save_ensemble(save_path)
    
    # Create new ensemble and load
    new_ensemble = MultimodalEnsemble(ensemble.config)
    new_ensemble.load_ensemble(save_path)
    
    # Get predictions after loading
    pred_after, _ = new_ensemble.predict(sample_batch)
    
    assert torch.allclose(pred_before, pred_after)

def test_different_aggregation_methods(model_config, ensemble, sample_batch):
    """Test different aggregation methods"""
    # Test mean aggregation
    model_config['aggregation_method'] = 'mean'
    ensemble_mean = MultimodalEnsemble(model_config)
    pred_mean, _ = ensemble_mean.predict(sample_batch)
    
    # Test weighted mean aggregation
    model_config['aggregation_method'] = 'weighted_mean'
    ensemble_weighted = MultimodalEnsemble(model_config)
    pred_weighted, _ = ensemble_weighted.predict(sample_batch)
    
    # Test voting aggregation
    model_config['aggregation_method'] = 'voting'
    ensemble_voting = MultimodalEnsemble(model_config)
    pred_voting, _ = ensemble_voting.predict(sample_batch)
    
    assert pred_mean.shape == pred_weighted.shape
    assert len(pred_voting.shape) == 1  # Voting returns class indices

def test_different_uncertainty_methods(model_config, ensemble, sample_batch):
    """Test different uncertainty estimation methods"""
    # Test entropy-based uncertainty
    model_config['uncertainty_method'] = 'entropy'
    ensemble_entropy = MultimodalEnsemble(model_config)
    _, uncert_entropy = ensemble_entropy.predict(sample_batch)
    
    # Test variance-based uncertainty
    model_config['uncertainty_method'] = 'variance'
    ensemble_var = MultimodalEnsemble(model_config)
    _, uncert_var = ensemble_var.predict(sample_batch)
    
    assert uncert_entropy.shape == uncert_var.shape
    assert torch.all(uncert_entropy >= 0)
    assert torch.all(uncert_var >= 0)

def test_invalid_config(model_config):
    """Test handling of invalid configuration"""
    # Test invalid aggregation method
    invalid_config = model_config.copy()
    invalid_config['aggregation_method'] = 'invalid'
    
    with pytest.raises(ValueError):
        MultimodalEnsemble(invalid_config)
    
    # Test invalid uncertainty method
    invalid_config = model_config.copy()
    invalid_config['uncertainty_method'] = 'invalid'
    
    with pytest.raises(ValueError):
        MultimodalEnsemble(invalid_config)

def test_empty_batch(ensemble):
    """Test handling of empty batch"""
    empty_batch = {
        'image': torch.empty(0, 3, 32, 32),
        'sequence': torch.empty(0, 10, 20),
        'clinical': torch.empty(0, 20)
    }
    
    with pytest.raises(ValueError):
        ensemble.predict(empty_batch)

def test_device_handling(model_config, sample_batch):
    """Test handling of different devices"""
    if torch.cuda.is_available():
        model_config['device'] = 'cuda'
        ensemble = MultimodalEnsemble(model_config)
        
        # Check if models are on GPU
        assert all(next(m.parameters()).is_cuda for m in ensemble.models)
        
        # Test prediction with GPU tensors
        sample_batch_gpu = {
            k: v.cuda() for k, v in sample_batch.items()
            if isinstance(v, torch.Tensor)
        }
        pred, _ = ensemble.predict(sample_batch_gpu)
        assert pred.is_cuda 