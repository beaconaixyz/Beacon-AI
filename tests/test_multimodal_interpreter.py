import pytest
import torch
import numpy as np
from beacon.interpretability.multimodal_interpreter import MultimodalInterpreter
from beacon.models.multimodal import MultimodalFusion

@pytest.fixture
def sample_batch():
    """Create sample batch data for testing"""
    batch_size = 4
    
    # Create image data (B, C, H, W)
    image_data = torch.randn(batch_size, 3, 64, 64)
    
    # Create clinical data (B, F)
    clinical_data = torch.randn(batch_size, 10)
    
    # Create genomic data (B, G)
    genomic_data = torch.randn(batch_size, 100)
    
    # Create target labels
    target = torch.randint(0, 2, (batch_size,))
    
    return {
        'image': image_data,
        'clinical': clinical_data,
        'genomic': genomic_data,
        'target': target
    }

@pytest.fixture
def model_config():
    """Create model configuration"""
    return {
        'image': {
            'input_dim': (3, 64, 64),
            'hidden_dim': 64,
            'num_layers': 3
        },
        'clinical': {
            'input_dim': 10,
            'hidden_dim': 32
        },
        'genomic': {
            'input_dim': 100,
            'hidden_dim': 64
        },
        'fusion': {
            'hidden_dim': 128,
            'num_heads': 4,
            'dropout_rate': 0.3
        },
        'output_dim': 2
    }

@pytest.fixture
def interpreter_config():
    """Create interpreter configuration"""
    return {
        'methods': ['integrated_gradients', 'deep_lift', 'occlusion'],
        'integrated_gradients': {
            'n_steps': 10,
            'internal_batch_size': 4
        },
        'deep_lift': {
            'multiply_by_inputs': True
        },
        'occlusion': {
            'sliding_window_shapes': {
                'image': (1, 8, 8),
                'clinical': (2,),
                'genomic': (5,)
            },
            'strides': {
                'image': (1, 4, 4),
                'clinical': (1,),
                'genomic': (2,)
            }
        }
    }

@pytest.fixture
def trained_model(model_config, sample_batch):
    """Create and train a model for testing"""
    model = MultimodalFusion(model_config)
    
    # Simulate training
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    for _ in range(2):  # Just a few iterations for testing
        optimizer.zero_grad()
        outputs = model(sample_batch)
        loss = criterion(outputs, sample_batch['target'])
        loss.backward()
        optimizer.step()
    
    return model

def test_interpreter_initialization(trained_model, interpreter_config):
    """Test interpreter initialization"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    assert interpreter.model is trained_model
    assert set(interpreter.interpreters.keys()).issubset(
        set(interpreter_config['methods'])
    )

def test_integrated_gradients(trained_model, interpreter_config, sample_batch):
    """Test Integrated Gradients interpretation"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    attributions = interpreter.interpret(
        sample_batch,
        sample_batch['target'],
        method='integrated_gradients'
    )
    
    # Check attributions
    for modality in ['image', 'clinical', 'genomic']:
        assert modality in attributions
        assert isinstance(attributions[modality], torch.Tensor)
        assert attributions[modality].shape == sample_batch[modality].shape

def test_deep_lift(trained_model, interpreter_config, sample_batch):
    """Test DeepLIFT interpretation"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    attributions = interpreter.interpret(
        sample_batch,
        sample_batch['target'],
        method='deep_lift'
    )
    
    # Check attributions
    for modality in ['image', 'clinical', 'genomic']:
        assert modality in attributions
        assert isinstance(attributions[modality], torch.Tensor)
        assert attributions[modality].shape == sample_batch[modality].shape

def test_occlusion(trained_model, interpreter_config, sample_batch):
    """Test Occlusion interpretation"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    attributions = interpreter.interpret(
        sample_batch,
        sample_batch['target'],
        method='occlusion'
    )
    
    # Check attributions
    for modality in ['image', 'clinical', 'genomic']:
        assert modality in attributions
        assert isinstance(attributions[modality], torch.Tensor)
        assert attributions[modality].shape == sample_batch[modality].shape

def test_noise_tunnel(trained_model, interpreter_config, sample_batch):
    """Test noise tunnel smoothing"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    # Get base attributions
    base_attr = interpreter.interpret(
        sample_batch,
        sample_batch['target']
    )
    
    # Apply noise tunnel
    smoothed_attr = interpreter.add_noise_tunnel(base_attr, sample_batch)
    
    # Check smoothed attributions
    for modality in base_attr:
        assert modality in smoothed_attr
        assert isinstance(smoothed_attr[modality], torch.Tensor)
        assert smoothed_attr[modality].shape == base_attr[modality].shape

def test_feature_interactions(trained_model, interpreter_config, sample_batch):
    """Test feature interaction analysis"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    interactions = interpreter.analyze_feature_interactions(
        sample_batch,
        sample_batch['target']
    )
    
    # Check interactions
    expected_pairs = ['image_clinical', 'image_genomic', 'clinical_genomic']
    for pair in expected_pairs:
        assert pair in interactions
        assert isinstance(interactions[pair], np.ndarray)

def test_interpretation_stats(trained_model, interpreter_config, sample_batch):
    """Test interpretation statistics"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    attributions = interpreter.interpret(
        sample_batch,
        sample_batch['target']
    )
    
    stats = interpreter.get_interpretation_stats(attributions)
    
    # Check statistics
    for modality in attributions:
        assert modality in stats
        assert isinstance(stats[modality], dict)
        assert all(metric in stats[modality] 
                  for metric in ['mean', 'std', 'max', 'min', 'sparsity'])

def test_method_comparison(trained_model, interpreter_config, sample_batch):
    """Test comparison of different interpretation methods"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    results = interpreter.compare_methods(
        sample_batch,
        sample_batch['target'],
        methods=['integrated_gradients', 'deep_lift']
    )
    
    # Check results
    assert set(results.keys()) == {'integrated_gradients', 'deep_lift'}
    for method_results in results.values():
        for modality in ['image', 'clinical', 'genomic']:
            assert modality in method_results
            assert isinstance(method_results[modality], torch.Tensor)

def test_invalid_method(trained_model, interpreter_config, sample_batch):
    """Test handling of invalid interpretation method"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    with pytest.raises(ValueError):
        interpreter.interpret(sample_batch, sample_batch['target'], 
                           method='invalid_method')

def test_missing_modality(trained_model, interpreter_config, sample_batch):
    """Test handling of missing modality"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    # Remove one modality
    batch_subset = {k: v for k, v in sample_batch.items() if k != 'image'}
    
    attributions = interpreter.interpret(
        batch_subset,
        sample_batch['target']
    )
    
    # Check that only available modalities have attributions
    assert 'image' not in attributions
    assert all(modality in attributions for modality in ['clinical', 'genomic'])

def test_device_handling(trained_model, interpreter_config, sample_batch):
    """Test handling of different devices"""
    interpreter = MultimodalInterpreter(trained_model, interpreter_config)
    
    # Move batch to different device if CUDA available
    if torch.cuda.is_available():
        batch_gpu = {k: v.cuda() for k, v in sample_batch.items()}
        attributions = interpreter.interpret(batch_gpu, batch_gpu['target'])
        
        # Check that attributions are on the same device
        for attr in attributions.values():
            assert attr.device.type == 'cuda' 