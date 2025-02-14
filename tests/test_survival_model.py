import pytest
import torch
import numpy as np
from beacon.models.survival_model import SurvivalModel

@pytest.fixture
def sample_survival_data():
    """Create sample survival data for testing"""
    n_samples = 100
    n_features = 50
    
    # Generate synthetic features
    features = torch.randn(n_samples, n_features)
    
    # Generate synthetic survival times (positive values)
    survival_time = torch.abs(torch.randn(n_samples))
    
    # Generate synthetic event indicators (0: censored, 1: event occurred)
    event_indicator = torch.randint(0, 2, (n_samples,))
    
    return {
        'features': features,
        'survival_time': survival_time,
        'event_indicator': event_indicator
    }

def test_model_initialization():
    """Test model initialization"""
    model_config = {
        'input_dim': 50,
        'hidden_dims': [32, 16],
        'dropout_rate': 0.2
    }
    
    model = SurvivalModel(model_config)
    
    # Verify model architecture
    assert isinstance(model.model, torch.nn.Module)
    assert len(model.model.network) > 0

def test_negative_log_likelihood(sample_survival_data):
    """Test negative log likelihood calculation"""
    model_config = {
        'input_dim': sample_survival_data['features'].shape[1],
        'hidden_dims': [32, 16]
    }
    
    model = SurvivalModel(model_config)
    
    # Generate some risk scores
    risk_scores = torch.randn(len(sample_survival_data['features']))
    
    # Calculate loss
    loss = model.negative_log_likelihood(
        risk_scores,
        sample_survival_data['survival_time'],
        sample_survival_data['event_indicator']
    )
    
    # Check loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_model_training(sample_survival_data):
    """Test model training"""
    model_config = {
        'input_dim': sample_survival_data['features'].shape[1],
        'hidden_dims': [32, 16],
        'learning_rate': 0.01
    }
    
    model = SurvivalModel(model_config)
    
    # Train model
    history = model.train(sample_survival_data, epochs=2, batch_size=32)
    
    # Check training history
    assert isinstance(history, list)
    assert len(history) == 2  # Two epochs
    assert 'loss' in history[0]
    assert history[0]['loss'] > 0

def test_risk_prediction(sample_survival_data):
    """Test risk score prediction"""
    model_config = {
        'input_dim': sample_survival_data['features'].shape[1],
        'hidden_dims': [32, 16]
    }
    
    model = SurvivalModel(model_config)
    
    # Predict risk scores
    risk_scores = model.predict_risk(sample_survival_data['features'])
    
    # Check predictions
    assert isinstance(risk_scores, torch.Tensor)
    assert risk_scores.shape == (len(sample_survival_data['features']), 1)
    assert torch.all(risk_scores >= 0)  # Risk scores should be non-negative

def test_survival_function_prediction(sample_survival_data):
    """Test survival function prediction"""
    model_config = {
        'input_dim': sample_survival_data['features'].shape[1],
        'hidden_dims': [32, 16]
    }
    
    model = SurvivalModel(model_config)
    
    # Create time points
    time_points = torch.linspace(0, 10, 100)
    
    # Predict survival function
    survival_probs = model.predict_survival_function(
        sample_survival_data['features'],
        time_points
    )
    
    # Check predictions
    assert isinstance(survival_probs, torch.Tensor)
    assert torch.all(survival_probs >= 0) and torch.all(survival_probs <= 1)

def test_model_save_load(tmp_path, sample_survival_data):
    """Test model save and load functionality"""
    model_config = {
        'input_dim': sample_survival_data['features'].shape[1],
        'hidden_dims': [32, 16]
    }
    
    # Initialize and train original model
    model = SurvivalModel(model_config)
    model.train(sample_survival_data, epochs=1)
    
    # Save model
    save_path = tmp_path / "survival_model.pt"
    model.save(str(save_path))
    
    # Load model
    new_model = SurvivalModel(model_config)
    new_model.load(str(save_path))
    
    # Compare predictions
    with torch.no_grad():
        original_preds = model.predict_risk(sample_survival_data['features'])
        loaded_preds = new_model.predict_risk(sample_survival_data['features'])
    
    # Predictions should be identical
    assert torch.allclose(original_preds, loaded_preds) 