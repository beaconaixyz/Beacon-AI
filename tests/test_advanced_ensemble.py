import pytest
import torch
import numpy as np
from beacon.models.advanced_ensemble import AdvancedEnsemble

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
def val_batch():
    """Create validation batch for testing"""
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
        'ensemble_method': 'stacking',
        'n_base_models': 3,
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

def test_stacking_initialization(model_config):
    """Test stacking ensemble initialization"""
    model_config['ensemble_method'] = 'stacking'
    ensemble = AdvancedEnsemble(model_config)
    
    assert len(ensemble.base_models) == model_config['n_base_models']
    assert ensemble.meta_model is not None

def test_boosting_initialization(model_config):
    """Test boosting ensemble initialization"""
    model_config['ensemble_method'] = 'boosting'
    ensemble = AdvancedEnsemble(model_config)
    
    assert len(ensemble.base_models) == 0  # Models are created during training
    assert hasattr(ensemble, 'model_weights')

def test_deep_ensemble_initialization(model_config):
    """Test deep ensemble initialization"""
    model_config['ensemble_method'] = 'deep_ensemble'
    ensemble = AdvancedEnsemble(model_config)
    
    assert len(ensemble.base_models) == model_config['n_base_models']

def test_stacking_training(model_config, sample_batch, val_batch):
    """Test stacking ensemble training"""
    model_config['ensemble_method'] = 'stacking'
    ensemble = AdvancedEnsemble(model_config)
    
    history = ensemble.train(sample_batch, val_batch)
    
    assert 'base_histories' in history
    assert 'meta_history' in history
    assert len(history['base_histories']) == model_config['n_base_models']

def test_boosting_training(model_config, sample_batch):
    """Test boosting ensemble training"""
    model_config['ensemble_method'] = 'boosting'
    ensemble = AdvancedEnsemble(model_config)
    
    history = ensemble.train(sample_batch)
    
    assert 'histories' in history
    assert 'model_weights' in history
    assert len(history['histories']) == model_config['boosting_config']['n_rounds']
    assert len(ensemble.model_weights) == model_config['boosting_config']['n_rounds']

def test_deep_ensemble_training(model_config, sample_batch):
    """Test deep ensemble training"""
    model_config['ensemble_method'] = 'deep_ensemble'
    ensemble = AdvancedEnsemble(model_config)
    
    history = ensemble.train(sample_batch)
    
    assert 'histories' in history
    assert len(history['histories']) == model_config['n_base_models']

def test_stacking_prediction(model_config, sample_batch, val_batch):
    """Test stacking ensemble prediction"""
    model_config['ensemble_method'] = 'stacking'
    ensemble = AdvancedEnsemble(model_config)
    
    # Train ensemble
    ensemble.train(sample_batch, val_batch)
    
    # Test prediction
    predictions, uncertainties = ensemble.predict(sample_batch)
    
    assert isinstance(predictions, torch.Tensor)
    assert isinstance(uncertainties, torch.Tensor)
    assert predictions.shape[0] == len(sample_batch['image'])
    assert uncertainties.shape[0] == len(sample_batch['image'])

def test_boosting_prediction(model_config, sample_batch):
    """Test boosting ensemble prediction"""
    model_config['ensemble_method'] = 'boosting'
    ensemble = AdvancedEnsemble(model_config)
    
    # Train ensemble
    ensemble.train(sample_batch)
    
    # Test prediction
    predictions, uncertainties = ensemble.predict(sample_batch)
    
    assert isinstance(predictions, torch.Tensor)
    assert isinstance(uncertainties, torch.Tensor)
    assert predictions.shape[0] == len(sample_batch['image'])
    assert uncertainties.shape[0] == len(sample_batch['image'])

def test_deep_ensemble_prediction(model_config, sample_batch):
    """Test deep ensemble prediction"""
    model_config['ensemble_method'] = 'deep_ensemble'
    ensemble = AdvancedEnsemble(model_config)
    
    # Train ensemble
    ensemble.train(sample_batch)
    
    # Test prediction with different uncertainty methods
    for method in ['evidential', 'bayesian', 'ensemble']:
        ensemble.config['uncertainty_config']['method'] = method
        predictions, uncertainties = ensemble.predict(sample_batch)
        
        assert isinstance(predictions, torch.Tensor)
        assert isinstance(uncertainties, torch.Tensor)
        assert predictions.shape[0] == len(sample_batch['image'])
        assert uncertainties.shape[0] == len(sample_batch['image'])

def test_save_load_stacking(model_config, sample_batch, val_batch, tmp_path):
    """Test saving and loading stacking ensemble"""
    model_config['ensemble_method'] = 'stacking'
    ensemble = AdvancedEnsemble(model_config)
    
    # Train ensemble
    ensemble.train(sample_batch, val_batch)
    
    # Get predictions before saving
    pred_before, _ = ensemble.predict(sample_batch)
    
    # Save ensemble
    save_path = str(tmp_path / "ensemble")
    ensemble.save_ensemble(save_path)
    
    # Create new ensemble and load
    new_ensemble = AdvancedEnsemble(model_config)
    new_ensemble.load_ensemble(save_path)
    
    # Get predictions after loading
    pred_after, _ = new_ensemble.predict(sample_batch)
    
    assert torch.allclose(pred_before, pred_after)

def test_save_load_boosting(model_config, sample_batch, tmp_path):
    """Test saving and loading boosting ensemble"""
    model_config['ensemble_method'] = 'boosting'
    ensemble = AdvancedEnsemble(model_config)
    
    # Train ensemble
    ensemble.train(sample_batch)
    
    # Get predictions before saving
    pred_before, _ = ensemble.predict(sample_batch)
    
    # Save ensemble
    save_path = str(tmp_path / "ensemble")
    ensemble.save_ensemble(save_path)
    
    # Create new ensemble and load
    new_ensemble = AdvancedEnsemble(model_config)
    new_ensemble.load_ensemble(save_path)
    
    # Get predictions after loading
    pred_after, _ = new_ensemble.predict(sample_batch)
    
    assert torch.allclose(pred_before, pred_after)
    assert len(new_ensemble.model_weights) == len(ensemble.model_weights)

def test_invalid_ensemble_method(model_config):
    """Test handling of invalid ensemble method"""
    model_config['ensemble_method'] = 'invalid'
    
    with pytest.raises(ValueError):
        AdvancedEnsemble(model_config)

def test_invalid_uncertainty_method(model_config, sample_batch):
    """Test handling of invalid uncertainty method"""
    model_config['ensemble_method'] = 'deep_ensemble'
    model_config['uncertainty_config']['method'] = 'invalid'
    ensemble = AdvancedEnsemble(model_config)
    
    # Train ensemble
    ensemble.train(sample_batch)
    
    with pytest.raises(ValueError):
        ensemble.predict(sample_batch)

def test_empty_batch(model_config):
    """Test handling of empty batch"""
    ensemble = AdvancedEnsemble(model_config)
    
    empty_batch = {
        'image': torch.empty(0, 3, 32, 32),
        'sequence': torch.empty(0, 10, 20),
        'clinical': torch.empty(0, 20),
        'labels': torch.empty(0)
    }
    
    with pytest.raises(ValueError):
        ensemble.train(empty_batch)

def test_device_handling(model_config, sample_batch):
    """Test handling of different devices"""
    if torch.cuda.is_available():
        model_config['device'] = 'cuda'
        ensemble = AdvancedEnsemble(model_config)
        
        # Check if models are on GPU
        assert all(next(m.parameters()).is_cuda for m in ensemble.base_models)
        if ensemble.meta_model is not None:
            assert next(ensemble.meta_model.parameters()).is_cuda
        
        # Test with GPU tensors
        sample_batch_gpu = {
            k: v.cuda() for k, v in sample_batch.items()
            if isinstance(v, torch.Tensor)
        }
        pred, _ = ensemble.predict(sample_batch_gpu)
        assert pred.is_cuda 