import pytest
import torch
import numpy as np
from beacon.utils.interpretability import ModelInterpreter
from beacon.models.cancer_classifier import CancerClassifier

@pytest.fixture
def sample_tabular_data():
    """Create sample tabular data for testing"""
    n_samples = 100
    n_features = 10
    
    # Generate synthetic features
    features = torch.randn(n_samples, n_features)
    labels = torch.randint(0, 2, (n_samples,))
    
    return features, labels

@pytest.fixture
def sample_image_data():
    """Create sample image data for testing"""
    n_samples = 10
    channels = 1
    height = 64
    width = 64
    
    # Generate synthetic images
    images = torch.randn(n_samples, channels, height, width)
    labels = torch.randint(0, 2, (n_samples,))
    
    return images, labels

@pytest.fixture
def trained_model(sample_tabular_data):
    """Create and train a simple model for testing"""
    features, labels = sample_tabular_data
    
    model_config = {
        'input_dim': features.shape[1],
        'hidden_dim': 32,
        'output_dim': 2
    }
    
    model = CancerClassifier(model_config)
    model.train({
        'features': features,
        'labels': labels
    })
    
    return model

def test_interpreter_initialization():
    """Test interpreter initialization"""
    config = {
        'method': 'integrated_gradients',
        'n_steps': 50
    }
    
    interpreter = ModelInterpreter(config)
    assert interpreter.method == 'integrated_gradients'
    assert interpreter.config['n_steps'] == 50

def test_integrated_gradients(trained_model, sample_tabular_data):
    """Test Integrated Gradients attribution"""
    features, _ = sample_tabular_data
    
    interpreter = ModelInterpreter({'method': 'integrated_gradients'})
    attributions, metadata = interpreter.explain_prediction(
        trained_model,
        features[:5]  # Test with first 5 samples
    )
    
    # Check attribution shape and properties
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == features[:5].shape
    assert 'convergence_delta' in metadata

def test_deep_lift(trained_model, sample_tabular_data):
    """Test DeepLIFT attribution"""
    features, _ = sample_tabular_data
    
    interpreter = ModelInterpreter({'method': 'deep_lift'})
    attributions, metadata = interpreter.explain_prediction(
        trained_model,
        features[:5]
    )
    
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == features[:5].shape
    assert metadata['method'] == 'deep_lift'

def test_shap_values(trained_model, sample_tabular_data):
    """Test SHAP values calculation"""
    features, _ = sample_tabular_data
    
    interpreter = ModelInterpreter({'method': 'shap'})
    attributions, metadata = interpreter.explain_prediction(
        trained_model,
        features[:5]
    )
    
    assert isinstance(attributions, torch.Tensor)
    assert metadata['method'] == 'shap'

def test_feature_interactions(trained_model, sample_tabular_data):
    """Test feature interaction analysis"""
    features, _ = sample_tabular_data
    feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
    
    interpreter = ModelInterpreter({})
    interactions = interpreter.analyze_feature_interactions(
        trained_model,
        features[:5],
        feature_names
    )
    
    assert isinstance(interactions, dict)
    assert len(interactions) > 0
    
    # Check interaction values
    for value in interactions.values():
        assert isinstance(value, float)
        assert value >= 0  # Interaction strengths should be non-negative

def test_counterfactuals(trained_model, sample_tabular_data):
    """Test counterfactual generation"""
    features, _ = sample_tabular_data
    target = torch.ones(5)  # Target class 1 for first 5 samples
    
    interpreter = ModelInterpreter({})
    counterfactuals, distances = interpreter.generate_counterfactuals(
        trained_model,
        features[:5],
        target
    )
    
    assert isinstance(counterfactuals, torch.Tensor)
    assert isinstance(distances, torch.Tensor)
    assert counterfactuals.shape == features[:5].shape
    assert distances.shape[0] == 5
    assert torch.all(distances >= 0)  # Distances should be non-negative

def test_visualization(tmp_path, trained_model, sample_tabular_data):
    """Test attribution visualization"""
    features, _ = sample_tabular_data
    feature_names = [f"Feature_{i}" for i in range(features.shape[1])]
    
    interpreter = ModelInterpreter({})
    attributions, _ = interpreter.explain_prediction(
        trained_model,
        features[:5]
    )
    
    # Test visualization without saving
    interpreter.visualize_attributions(
        attributions,
        features[:5],
        feature_names
    )
    
    # Test visualization with saving
    save_path = tmp_path / "test_viz"
    interpreter.visualize_attributions(
        attributions,
        features[:5],
        feature_names,
        str(save_path)
    )
    
    assert (save_path.parent / f"{save_path.name}_feature_importance.png").exists()

def test_invalid_method():
    """Test invalid interpretation method"""
    interpreter = ModelInterpreter({'method': 'invalid_method'})
    
    with pytest.raises(ValueError):
        interpreter.explain_prediction(None, None) 