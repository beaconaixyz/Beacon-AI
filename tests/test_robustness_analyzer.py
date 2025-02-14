import pytest
import torch
import numpy as np
from beacon.interpretability.robustness_analyzer import MultimodalRobustnessAnalyzer
from beacon.models.multimodal import MultimodalFusion

@pytest.fixture
def sample_batch():
    """Create sample batch for testing"""
    batch_size = 4
    
    # Create synthetic data for each modality
    image_data = torch.randn(batch_size, 3, 64, 64)  # Image modality
    genomic_data = torch.randn(batch_size, 1000)  # Genomic modality
    clinical_data = torch.randn(batch_size, 10)  # Clinical modality
    
    # Create labels
    labels = torch.randint(0, 2, (batch_size,))
    
    return {
        'image': image_data,
        'genomic': genomic_data,
        'clinical': clinical_data,
        'target': labels
    }

@pytest.fixture
def model_config():
    """Create model configuration"""
    return {
        'image_encoder': {
            'input_dim': (3, 64, 64),
            'hidden_dim': 64
        },
        'genomic_encoder': {
            'input_dim': 1000,
            'hidden_dim': 64
        },
        'clinical_encoder': {
            'input_dim': 10,
            'hidden_dim': 64
        },
        'fusion': {
            'method': 'attention',
            'hidden_dim': 64,
            'num_heads': 4,
            'dropout_rate': 0.1
        },
        'output_dim': 2
    }

@pytest.fixture
def analyzer_config():
    """Create analyzer configuration"""
    return {
        'adversarial': {
            'epsilon': 0.1,
            'alpha': 0.01,
            'num_steps': 3,
            'random_start': True
        },
        'sensitivity': {
            'noise_types': ['gaussian'],
            'noise_levels': [0.01],
            'n_samples': 2
        },
        'feature_ablation': {
            'n_features': 5,
            'strategy': 'importance'
        },
        'cross_modality': {
            'enabled': True,
            'n_permutations': 2
        }
    }

@pytest.fixture
def trained_model(model_config, sample_batch):
    """Create and train a model"""
    model = MultimodalFusion(model_config)
    
    # Train for one step
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    outputs = model(sample_batch)
    loss = criterion(outputs, sample_batch['target'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return model

def test_analyzer_initialization(trained_model, analyzer_config):
    """Test analyzer initialization"""
    analyzer = MultimodalRobustnessAnalyzer(trained_model, analyzer_config)
    assert analyzer.model is trained_model
    assert analyzer.config == analyzer_config

def test_adversarial_examples(trained_model, analyzer_config, sample_batch):
    """Test adversarial example generation"""
    analyzer = MultimodalRobustnessAnalyzer(trained_model, analyzer_config)
    
    # Generate adversarial examples
    adv_batch = analyzer.generate_adversarial_examples(sample_batch)
    
    # Check output
    assert isinstance(adv_batch, dict)
    for modality in ['image', 'genomic', 'clinical']:
        assert modality in adv_batch
        assert isinstance(adv_batch[modality], torch.Tensor)
        assert adv_batch[modality].shape == sample_batch[modality].shape
        
        # Check perturbation magnitude
        delta = adv_batch[modality] - sample_batch[modality]
        assert torch.all(torch.abs(delta) <= analyzer_config['adversarial']['epsilon'])

def test_sensitivity_analysis(trained_model, analyzer_config, sample_batch):
    """Test sensitivity analysis"""
    analyzer = MultimodalRobustnessAnalyzer(trained_model, analyzer_config)
    
    # Test for each noise type
    for noise_type in analyzer_config['sensitivity']['noise_types']:
        sensitivity = analyzer.analyze_sensitivity(sample_batch, noise_type)
        
        # Check output
        assert isinstance(sensitivity, dict)
        for modality in ['image', 'genomic', 'clinical']:
            for noise_level in analyzer_config['sensitivity']['noise_levels']:
                key = f"{modality}_{noise_level}"
                assert key in sensitivity
                assert isinstance(sensitivity[key], np.ndarray)
                assert 0 <= sensitivity[key] <= 1

def test_feature_importance(trained_model, analyzer_config, sample_batch):
    """Test feature importance analysis"""
    analyzer = MultimodalRobustnessAnalyzer(trained_model, analyzer_config)
    
    # Get feature importance scores
    importance = analyzer.analyze_feature_importance(sample_batch)
    
    # Check output
    assert isinstance(importance, dict)
    for modality in ['image', 'genomic', 'clinical']:
        assert modality in importance
        assert isinstance(importance[modality], torch.Tensor)
        assert importance[modality].shape[0] == sample_batch[modality].shape[1]

def test_cross_modality_robustness(trained_model, analyzer_config, sample_batch):
    """Test cross-modality robustness analysis"""
    analyzer = MultimodalRobustnessAnalyzer(trained_model, analyzer_config)
    
    # Get robustness scores
    robustness = analyzer.analyze_cross_modality_robustness(sample_batch)
    
    # Check output
    assert isinstance(robustness, dict)
    expected_pairs = ['image_genomic', 'image_clinical', 'genomic_clinical']
    for pair in expected_pairs:
        assert pair in robustness
        assert isinstance(robustness[pair], float)
        assert 0 <= robustness[pair] <= 1

def test_robustness_metrics(trained_model, analyzer_config, sample_batch):
    """Test comprehensive robustness metrics"""
    analyzer = MultimodalRobustnessAnalyzer(trained_model, analyzer_config)
    
    # Get all metrics
    metrics = analyzer.get_robustness_metrics(sample_batch)
    
    # Check output structure
    assert isinstance(metrics, dict)
    assert 'adversarial' in metrics
    assert 'sensitivity_gaussian' in metrics
    assert 'feature_importance' in metrics
    assert 'cross_modality' in metrics
    
    # Check adversarial metrics
    assert 'accuracy_drop' in metrics['adversarial']
    assert 'confidence_drop' in metrics['adversarial']
    
    # Check feature importance
    for modality in ['image', 'genomic', 'clinical']:
        assert modality in metrics['feature_importance']

def test_invalid_noise_type(trained_model, analyzer_config, sample_batch):
    """Test handling of invalid noise type"""
    analyzer = MultimodalRobustnessAnalyzer(trained_model, analyzer_config)
    
    with pytest.raises(ValueError):
        analyzer.analyze_sensitivity(sample_batch, 'invalid_noise')

def test_disabled_cross_modality(trained_model, analyzer_config, sample_batch):
    """Test behavior when cross-modality analysis is disabled"""
    analyzer_config['cross_modality']['enabled'] = False
    analyzer = MultimodalRobustnessAnalyzer(trained_model, analyzer_config)
    
    robustness = analyzer.analyze_cross_modality_robustness(sample_batch)
    assert robustness == {} 