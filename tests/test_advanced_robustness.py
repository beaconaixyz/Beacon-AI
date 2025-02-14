import pytest
import torch
import numpy as np
from beacon.interpretability.advanced_robustness import AdvancedRobustnessAnalyzer
from beacon.models.multimodal import MultimodalFusion

@pytest.fixture
def sample_batch():
    """Create sample batch for testing"""
    batch_size = 4
    image_size = 32
    seq_length = 10
    feature_dim = 20
    
    return {
        'image': torch.randn(batch_size, 3, image_size, image_size),
        'sequence': torch.randn(batch_size, seq_length, feature_dim),
        'clinical': torch.randn(batch_size, feature_dim)
    }

@pytest.fixture
def model_config():
    """Create model configuration for testing"""
    return {
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
    }

@pytest.fixture
def analyzer_config():
    """Create analyzer configuration for testing"""
    return {
        'fgsm': {
            'epsilon': 0.1
        },
        'pgd': {
            'epsilon': 0.1,
            'alpha': 0.01,
            'num_steps': 5
        },
        'carlini_wagner': {
            'confidence': 0,
            'learning_rate': 0.01,
            'num_steps': 10,
            'binary_search_steps': 3
        },
        'deepfool': {
            'num_steps': 10,
            'overshoot': 0.02
        },
        'analysis': {
            'lipschitz_estimation': {
                'num_samples': 10,
                'radius': 0.1
            },
            'decision_boundary': {
                'num_points': 10,
                'radius': 1.0
            },
            'gradient_analysis': {
                'num_samples': 10,
                'step_size': 0.01
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
    
    for _ in range(2):  # Just a few steps
        outputs = model(sample_batch)
        labels = torch.randint(0, 2, (len(outputs),))
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def test_analyzer_initialization(trained_model, analyzer_config):
    """Test analyzer initialization"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    assert analyzer.model is trained_model
    assert analyzer.config == analyzer_config

def test_fgsm_attack(trained_model, analyzer_config, sample_batch):
    """Test FGSM attack"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    adversarial_batch = analyzer.fgsm_attack(sample_batch)
    
    # Check output structure
    assert isinstance(adversarial_batch, dict)
    assert set(adversarial_batch.keys()) == set(sample_batch.keys())
    
    # Check perturbation bounds
    for modality in adversarial_batch:
        assert torch.all(adversarial_batch[modality] >= 0)
        assert torch.all(adversarial_batch[modality] <= 1)
        
        # Check if perturbation is non-zero
        assert not torch.allclose(adversarial_batch[modality],
                                sample_batch[modality])

def test_carlini_wagner_attack(trained_model, analyzer_config, sample_batch):
    """Test Carlini & Wagner attack"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    adversarial_batch = analyzer.carlini_wagner_attack(sample_batch)
    
    # Check output structure
    assert isinstance(adversarial_batch, dict)
    assert set(adversarial_batch.keys()) == set(sample_batch.keys())
    
    # Check perturbation bounds
    for modality in adversarial_batch:
        assert torch.all(adversarial_batch[modality] >= 0)
        assert torch.all(adversarial_batch[modality] <= 1)
        
        # Check if perturbation is non-zero
        assert not torch.allclose(adversarial_batch[modality],
                                sample_batch[modality])

def test_deepfool_attack(trained_model, analyzer_config, sample_batch):
    """Test DeepFool attack"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    adversarial_batch = analyzer.deepfool_attack(sample_batch)
    
    # Check output structure
    assert isinstance(adversarial_batch, dict)
    assert set(adversarial_batch.keys()) == set(sample_batch.keys())
    
    # Check perturbation bounds
    for modality in adversarial_batch:
        assert torch.all(adversarial_batch[modality] >= 0)
        assert torch.all(adversarial_batch[modality] <= 1)
        
        # Check if perturbation is non-zero
        assert not torch.allclose(adversarial_batch[modality],
                                sample_batch[modality])

def test_lipschitz_estimation(trained_model, analyzer_config, sample_batch):
    """Test Lipschitz constant estimation"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    lipschitz = analyzer.estimate_lipschitz_constant(sample_batch)
    
    # Check output structure
    assert isinstance(lipschitz, dict)
    assert set(lipschitz.keys()) == set(sample_batch.keys())
    
    # Check Lipschitz constants
    for modality in lipschitz:
        assert isinstance(lipschitz[modality], float)
        assert lipschitz[modality] >= 0

def test_decision_boundary_analysis(trained_model, analyzer_config, sample_batch):
    """Test decision boundary analysis"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    results = analyzer.analyze_decision_boundary(sample_batch)
    
    # Check output structure
    assert isinstance(results, dict)
    assert set(results.keys()) == set(sample_batch.keys())
    
    # Check results format
    for modality in results:
        assert 'distances' in results[modality]
        assert 'directions' in results[modality]
        assert isinstance(results[modality]['distances'], np.ndarray)
        assert isinstance(results[modality]['directions'], np.ndarray)

def test_gradient_landscape_analysis(trained_model, analyzer_config, sample_batch):
    """Test gradient landscape analysis"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    results = analyzer.analyze_gradient_landscape(sample_batch)
    
    # Check output structure
    assert isinstance(results, dict)
    assert set(results.keys()) == set(sample_batch.keys())
    
    # Check results format
    for modality in results:
        assert 'gradients' in results[modality]
        assert 'points' in results[modality]
        assert isinstance(results[modality]['gradients'], np.ndarray)
        assert isinstance(results[modality]['points'], np.ndarray)

def test_invalid_batch(trained_model, analyzer_config):
    """Test handling of invalid batch data"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    empty_batch = {}
    
    with pytest.raises(ValueError):
        analyzer.fgsm_attack(empty_batch)

def test_device_handling(trained_model, analyzer_config, sample_batch):
    """Test handling of different devices"""
    if torch.cuda.is_available():
        trained_model = trained_model.cuda()
        analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
        
        # Test FGSM attack
        adversarial_batch = analyzer.fgsm_attack(sample_batch)
        assert all(t.is_cuda for t in adversarial_batch.values()
                  if isinstance(t, torch.Tensor))

def test_batch_size_handling(trained_model, analyzer_config):
    """Test handling of different batch sizes"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    
    # Test with batch size 1
    single_batch = {
        'image': torch.randn(1, 3, 32, 32),
        'sequence': torch.randn(1, 10, 20),
        'clinical': torch.randn(1, 20)
    }
    
    adversarial_batch = analyzer.fgsm_attack(single_batch)
    assert all(t.shape[0] == 1 for t in adversarial_batch.values()
              if isinstance(t, torch.Tensor))

def test_targeted_attack(trained_model, analyzer_config, sample_batch):
    """Test targeted attack"""
    analyzer = AdvancedRobustnessAnalyzer(trained_model, analyzer_config)
    target = torch.ones(len(next(iter(sample_batch.values()))), dtype=torch.long)
    
    # Test FGSM attack with target
    adversarial_batch = analyzer.fgsm_attack(sample_batch, target)
    
    # Check if attack changes prediction towards target
    with torch.no_grad():
        outputs = trained_model(adversarial_batch)
        pred = outputs.argmax(dim=1)
        assert torch.any(pred == target)  # At least some should be successful 