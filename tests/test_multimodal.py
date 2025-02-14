import pytest
import torch
import numpy as np
from beacon.models.multimodal import MultimodalFusion

@pytest.fixture
def sample_batch():
    """Create sample batch with multiple modalities"""
    batch_size = 8
    
    # Create image data [B, C, H, W]
    image_data = torch.randn(batch_size, 1, 64, 64)
    
    # Create genomic data (graph data)
    genomic_data = {
        'x': torch.randn(batch_size * 10, 119),  # Node features
        'edge_index': torch.randint(0, batch_size * 10, (2, batch_size * 30)),  # Edges
        'batch': torch.repeat_interleave(torch.arange(batch_size), 10)  # Batch assignment
    }
    
    # Create clinical data
    clinical_data = torch.randn(batch_size, 32)
    
    # Create targets
    targets = torch.randint(0, 2, (batch_size,))
    
    return {
        'image': image_data,
        'genomic': genomic_data,
        'clinical': clinical_data,
        'target': targets
    }

@pytest.fixture
def model_config():
    """Create model configuration"""
    return {
        'image': {
            'enabled': True,
            'in_channels': 1,
            'base_filters': 32,
            'n_blocks': 3
        },
        'genomic': {
            'enabled': True,
            'conv_type': 'gat',
            'input_dim': 119,
            'hidden_dims': [64, 128],
            'num_heads': 4
        },
        'clinical': {
            'enabled': True,
            'input_dim': 32,
            'hidden_dims': [64, 32]
        },
        'fusion': {
            'method': 'attention',
            'hidden_dim': 256,
            'num_heads': 4,
            'dropout_rate': 0.3
        },
        'output_dim': 2,
        'task': 'classification',
        'learning_rate': 0.001
    }

def test_model_initialization(model_config):
    """Test model initialization"""
    model = MultimodalFusion(model_config)
    
    # Check model components
    assert hasattr(model.model, 'image_model')
    assert hasattr(model.model, 'genomic_model')
    assert hasattr(model.model, 'clinical_pathway')
    assert hasattr(model.model, 'fusion')
    assert hasattr(model.model, 'output')

def test_forward_pass(model_config, sample_batch):
    """Test forward pass"""
    model = MultimodalFusion(model_config)
    
    # Forward pass
    outputs = model.model(sample_batch)
    
    # Check output shape
    assert outputs.shape == (len(sample_batch['target']), model_config['output_dim'])
    assert not torch.isnan(outputs).any()

def test_training_step(model_config, sample_batch):
    """Test training step"""
    model = MultimodalFusion(model_config)
    
    # Training step
    loss, predictions = model.train_step(sample_batch)
    
    # Check outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)
    assert predictions.shape == (len(sample_batch['target']), model_config['output_dim'])

def test_prediction(model_config, sample_batch):
    """Test prediction"""
    model = MultimodalFusion(model_config)
    
    # Make predictions
    predictions = model.predict(sample_batch)
    
    # Check predictions
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(sample_batch['target']), model_config['output_dim'])
    if model_config['task'] == 'classification':
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)
        assert torch.allclose(predictions.sum(dim=1), torch.ones(len(predictions)))

def test_attention_weights(model_config, sample_batch):
    """Test attention weight extraction"""
    model = MultimodalFusion(model_config)
    
    # Get attention weights
    attention_weights = model.get_attention_weights(sample_batch)
    
    if model_config['fusion']['method'] == 'attention':
        assert isinstance(attention_weights, torch.Tensor)
        n_modalities = sum([
            model_config['image']['enabled'],
            model_config['genomic']['enabled'],
            model_config['clinical']['enabled']
        ])
        assert attention_weights.shape == (len(sample_batch['target']), n_modalities, n_modalities)
    else:
        assert attention_weights is None

def test_modality_embeddings(model_config, sample_batch):
    """Test modality embedding extraction"""
    model = MultimodalFusion(model_config)
    
    # Get embeddings
    embeddings = model.get_modality_embeddings(sample_batch)
    
    # Check embeddings
    assert isinstance(embeddings, dict)
    if model_config['image']['enabled']:
        assert 'image' in embeddings
        assert embeddings['image'].shape == (len(sample_batch['target']), 
                                           model_config['fusion']['hidden_dim'])
    if model_config['genomic']['enabled']:
        assert 'genomic' in embeddings
        assert embeddings['genomic'].shape == (len(sample_batch['target']),
                                             model_config['fusion']['hidden_dim'])
    if model_config['clinical']['enabled']:
        assert 'clinical' in embeddings
        assert embeddings['clinical'].shape == (len(sample_batch['target']),
                                              model_config['fusion']['hidden_dim'])

def test_disabled_modalities(model_config, sample_batch):
    """Test model with disabled modalities"""
    # Disable some modalities
    model_config['image']['enabled'] = False
    model_config['genomic']['enabled'] = True
    model_config['clinical']['enabled'] = True
    
    model = MultimodalFusion(model_config)
    
    # Remove disabled modality from batch
    batch = {k: v for k, v in sample_batch.items() if k != 'image'}
    
    # Forward pass should still work
    outputs = model.model(batch)
    assert outputs.shape == (len(batch['target']), model_config['output_dim'])

def test_concatenate_fusion(model_config, sample_batch):
    """Test concatenation fusion method"""
    model_config['fusion']['method'] = 'concatenate'
    model = MultimodalFusion(model_config)
    
    # Forward pass
    outputs = model.model(sample_batch)
    assert outputs.shape == (len(sample_batch['target']), model_config['output_dim'])

def test_regression_task(model_config, sample_batch):
    """Test regression task"""
    model_config['task'] = 'regression'
    model_config['output_dim'] = 1
    model = MultimodalFusion(model_config)
    
    # Change targets to continuous values
    sample_batch['target'] = torch.randn(len(sample_batch['target']))
    
    # Training step
    loss, predictions = model.train_step(sample_batch)
    assert isinstance(loss, torch.Tensor)
    assert predictions.shape == (len(sample_batch['target']), 1)

def test_save_load(tmp_path, model_config, sample_batch):
    """Test model save and load"""
    model = MultimodalFusion(model_config)
    
    # Train model
    model.train_step(sample_batch)
    
    # Save model
    save_path = tmp_path / "multimodal_model.pt"
    model.save(str(save_path))
    
    # Load model
    new_model = MultimodalFusion(model_config)
    new_model.load(str(save_path))
    
    # Compare predictions
    with torch.no_grad():
        original_preds = model.predict(sample_batch)
        loaded_preds = new_model.predict(sample_batch)
    
    assert torch.allclose(original_preds, loaded_preds) 