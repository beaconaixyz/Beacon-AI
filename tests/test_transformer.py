import pytest
import torch
import numpy as np
from beacon.models.transformer import TransformerModel
import torch.nn as nn

@pytest.fixture
def sample_sequence_data():
    """Create sample sequence data for testing"""
    batch_size = 8
    seq_length = 50
    input_dim = 512
    
    # Generate random sequences
    sequences = torch.randn(batch_size, seq_length, input_dim)
    
    # Generate random lengths (for testing padding)
    lengths = torch.randint(10, seq_length + 1, (batch_size,))
    
    # Generate random labels
    labels = torch.randint(0, 2, (batch_size,))
    
    return {
        'sequences': sequences,
        'lengths': lengths,
        'targets': labels
    }

@pytest.fixture
def model_config():
    """Create model configuration for testing"""
    return {
        'input_dim': 512,
        'num_heads': 8,
        'num_layers': 2,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_seq_length': 100,
        'positional_encoding': 'sinusoidal',
        'task': 'classification',
        'output_dim': 2
    }

def test_model_initialization(model_config):
    """Test model initialization"""
    model = TransformerModel(model_config)
    
    # Check model components
    assert hasattr(model.model, 'embedding')
    assert hasattr(model.model, 'transformer')
    assert hasattr(model.model, 'output_head')
    assert hasattr(model.model, 'pos_encoding')
    
    # Check dimensions
    assert model.model.input_dim == model_config['input_dim']
    assert model.model.num_heads == model_config['num_heads']
    assert model.model.num_layers == model_config['num_layers']

def test_forward_pass(model_config, sample_sequence_data):
    """Test forward pass"""
    model = TransformerModel(model_config)
    
    # Forward pass without mask
    outputs = model.model(sample_sequence_data['sequences'])
    assert outputs.shape == (sample_sequence_data['sequences'].size(0), model_config['output_dim'])
    
    # Forward pass with mask
    mask = model.create_padding_mask(sample_sequence_data['lengths'])
    outputs_masked = model.model(sample_sequence_data['sequences'], mask)
    assert outputs_masked.shape == (sample_sequence_data['sequences'].size(0), model_config['output_dim'])

def test_padding_mask(model_config, sample_sequence_data):
    """Test padding mask creation"""
    model = TransformerModel(model_config)
    lengths = sample_sequence_data['lengths']
    
    mask = model.create_padding_mask(lengths)
    
    # Check mask shape
    assert mask.shape == (lengths.size(0), max(lengths))
    
    # Check mask values
    for i, length in enumerate(lengths):
        assert not mask[i, :length].any()  # Valid positions should be False
        assert mask[i, length:].all()      # Padding positions should be True

def test_training_step(model_config, sample_sequence_data):
    """Test training step"""
    model = TransformerModel(model_config)
    
    # Perform training step
    loss, predictions = model.train_step(sample_sequence_data)
    
    # Check outputs
    assert isinstance(loss, torch.Tensor)
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (sample_sequence_data['sequences'].size(0), model_config['output_dim'])
    assert not torch.isnan(loss)

def test_prediction_step(model_config, sample_sequence_data):
    """Test prediction step"""
    model = TransformerModel(model_config)
    
    # Perform prediction
    predictions = model.predict_step(sample_sequence_data)
    
    # Check predictions
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (sample_sequence_data['sequences'].size(0), model_config['output_dim'])
    
    if model_config['task'] == 'classification':
        # Check if probabilities sum to 1
        assert torch.allclose(predictions.sum(dim=1), torch.ones(predictions.size(0)))
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)

def test_attention_weights(model_config, sample_sequence_data):
    """Test attention weight extraction"""
    model = TransformerModel(model_config)
    
    # Get attention weights
    attention_weights = model.get_attention_weights(sample_sequence_data)
    
    # Check number of layers
    assert len(attention_weights) == model_config['num_layers']
    
    # Check attention weight shapes
    batch_size = sample_sequence_data['sequences'].size(0)
    seq_length = sample_sequence_data['sequences'].size(1)
    
    for weights in attention_weights:
        assert weights.shape == (batch_size, model_config['num_heads'], seq_length, seq_length)
        assert not torch.isnan(weights).any()

def test_regression_task(model_config, sample_sequence_data):
    """Test model with regression task"""
    # Modify config for regression
    model_config['task'] = 'regression'
    model_config['output_dim'] = 1
    
    # Modify targets for regression
    sample_sequence_data['targets'] = torch.randn(len(sample_sequence_data['sequences']))
    
    model = TransformerModel(model_config)
    
    # Test training step
    loss, predictions = model.train_step(sample_sequence_data)
    assert predictions.shape == (sample_sequence_data['sequences'].size(0), 1)
    
    # Test prediction step
    predictions = model.predict_step(sample_sequence_data)
    assert predictions.shape == (sample_sequence_data['sequences'].size(0), 1)

def test_long_sequences(model_config):
    """Test model with sequences longer than max_seq_length"""
    model = TransformerModel(model_config)
    
    # Create long sequence
    batch_size = 4
    seq_length = model_config['max_seq_length'] + 50
    sequences = torch.randn(batch_size, seq_length, model_config['input_dim'])
    
    # Forward pass should still work (with truncation)
    outputs = model.model(sequences)
    assert outputs.shape == (batch_size, model_config['output_dim'])

def test_different_positional_encoding(model_config):
    """Test model with learned positional encoding"""
    model_config['positional_encoding'] = 'learned'
    model = TransformerModel(model_config)
    
    # Check if positional encoding is a learnable parameter
    assert isinstance(model.model.pos_encoding, nn.Parameter)
    assert model.model.pos_encoding.requires_grad

def test_save_load(tmp_path, model_config, sample_sequence_data):
    """Test model save and load functionality"""
    model = TransformerModel(model_config)
    
    # Get predictions before saving
    original_predictions = model.predict_step(sample_sequence_data)
    
    # Save model
    save_path = tmp_path / "transformer.pt"
    model.save(str(save_path))
    
    # Load model
    new_model = TransformerModel(model_config)
    new_model.load(str(save_path))
    
    # Get predictions after loading
    loaded_predictions = new_model.predict_step(sample_sequence_data)
    
    # Compare predictions
    assert torch.allclose(original_predictions, loaded_predictions) 