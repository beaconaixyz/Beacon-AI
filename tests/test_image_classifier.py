import pytest
import torch
import numpy as np
from beacon.models.image_classifier import MedicalImageCNN
from beacon.data.processor import DataProcessor

@pytest.fixture
def sample_image_data():
    """Create sample image data for testing"""
    # Create synthetic medical images (batch_size, channels, height, width)
    n_samples = 10
    image_size = 64
    images = torch.randn(n_samples, 1, image_size, image_size)
    labels = torch.randint(0, 2, (n_samples,))
    
    return {
        'images': images,
        'labels': labels
    }

def test_medical_image_cnn(sample_image_data):
    """Test medical image CNN functionality"""
    # Configure model
    model_config = {
        'in_channels': 1,
        'num_classes': 2,
        'learning_rate': 0.001
    }
    
    # Initialize model
    model = MedicalImageCNN(model_config)
    
    # Verify model architecture
    assert isinstance(model.model, torch.nn.Module)
    
    # Train model
    history = model.train(sample_image_data, epochs=2, batch_size=2)
    
    # Check training history
    assert isinstance(history, list)
    assert len(history) == 2  # Two epochs
    assert 'loss' in history[0]
    
    # Test predictions
    with torch.no_grad():
        predictions = model.predict(sample_image_data['images'])
        
    # Check predictions shape and type
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(sample_image_data['images']), 2)
    assert torch.all(predictions >= 0) and torch.all(predictions <= 1)  # Probabilities
    
    # Test forward pass with different image sizes
    different_size_image = torch.randn(1, 1, 128, 128)  # Different size
    with torch.no_grad():
        output = model.predict(different_size_image)
    assert output.shape == (1, 2)  # Should handle different input sizes

def test_model_save_load(tmp_path, sample_image_data):
    """Test model save and load functionality"""
    # Configure model
    model_config = {
        'in_channels': 1,
        'num_classes': 2
    }
    
    # Initialize model
    model = MedicalImageCNN(model_config)
    
    # Train model
    model.train(sample_image_data, epochs=1)
    
    # Save model
    save_path = tmp_path / "model.pt"
    model.save(str(save_path))
    
    # Load model
    new_model = MedicalImageCNN(model_config)
    new_model.load(str(save_path))
    
    # Compare predictions
    with torch.no_grad():
        original_preds = model.predict(sample_image_data['images'])
        loaded_preds = new_model.predict(sample_image_data['images'])
    
    # Predictions should be identical
    assert torch.allclose(original_preds, loaded_preds) 