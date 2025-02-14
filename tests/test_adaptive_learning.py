import pytest
import torch
import numpy as np
from beacon.models.adaptive_learning import AdaptiveLearning

@pytest.fixture
def config():
    """Create test configuration"""
    return {
        'initial_lr': 0.001,
        'min_lr': 1e-6,
        'max_lr': 0.1,
        'adaptation_window': 5,
        'momentum': 0.9,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8
    }

@pytest.fixture
def modalities():
    """Create list of test modalities"""
    return ['image', 'sequence', 'clinical']

@pytest.fixture
def performance_metrics():
    """Create sample performance metrics"""
    return {
        'image': 0.8,
        'sequence': 0.7,
        'clinical': 0.6
    }

@pytest.fixture
def gradients():
    """Create sample gradients"""
    return {
        'image': torch.randn(10, requires_grad=True),
        'sequence': torch.randn(10, requires_grad=True),
        'clinical': torch.randn(10, requires_grad=True)
    }

def test_initialization(config):
    """Test initialization of AdaptiveLearning"""
    adaptive = AdaptiveLearning(config)
    
    assert adaptive.config['initial_lr'] == config['initial_lr']
    assert adaptive.config['min_lr'] == config['min_lr']
    assert adaptive.config['max_lr'] == config['max_lr']
    assert adaptive.step_count == 0
    assert len(adaptive.performance_history) == 0
    assert len(adaptive.lr_history) == 0

def test_learning_rate_initialization(config, modalities):
    """Test initialization of learning rates"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    for modality in modalities:
        assert modality in adaptive.learning_rates
        assert adaptive.learning_rates[modality] == config['initial_lr']
        assert adaptive.momentum_buffer[modality] == 0.0
        assert adaptive.first_moment[modality] == 0.0
        assert adaptive.second_moment[modality] == 0.0

def test_learning_rate_update(config, modalities, performance_metrics, gradients):
    """Test learning rate update mechanism"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    # Update learning rates
    updated_lrs = adaptive.update_learning_rates(performance_metrics, gradients)
    
    assert len(updated_lrs) == len(modalities)
    for modality in modalities:
        assert updated_lrs[modality] >= config['min_lr']
        assert updated_lrs[modality] <= config['max_lr']

def test_performance_history(config, modalities, performance_metrics, gradients):
    """Test performance history tracking"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    # Update multiple times
    for _ in range(10):
        adaptive.update_learning_rates(performance_metrics, gradients)
    
    assert len(adaptive.performance_history) == config['adaptation_window']
    assert len(adaptive.lr_history) == config['adaptation_window']

def test_learning_rate_bounds(config, modalities, performance_metrics, gradients):
    """Test learning rate bounds"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    # Update multiple times
    for _ in range(20):
        updated_lrs = adaptive.update_learning_rates(performance_metrics, gradients)
        
        for modality in modalities:
            assert updated_lrs[modality] >= config['min_lr']
            assert updated_lrs[modality] <= config['max_lr']

def test_momentum_update(config, modalities, performance_metrics, gradients):
    """Test momentum update mechanism"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    # Get initial learning rates
    initial_lrs = adaptive.learning_rates.copy()
    
    # Update with momentum
    updated_lrs = adaptive.update_learning_rates(performance_metrics, gradients)
    
    for modality in modalities:
        assert adaptive.momentum_buffer[modality] != 0.0
        assert updated_lrs[modality] != initial_lrs[modality]

def test_adaptive_factor(config, modalities, performance_metrics, gradients):
    """Test adaptive factor calculation"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    # Update multiple times to accumulate statistics
    for _ in range(5):
        adaptive.update_learning_rates(performance_metrics, gradients)
    
    for modality in modalities:
        assert adaptive.first_moment[modality] != 0.0
        assert adaptive.second_moment[modality] != 0.0

def test_performance_improvement(config, modalities, gradients):
    """Test learning rate adaptation based on performance improvement"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    # First update with lower performance
    perf_low = {m: 0.5 for m in modalities}
    adaptive.update_learning_rates(perf_low, gradients)
    
    # Second update with higher performance
    perf_high = {m: 0.8 for m in modalities}
    updated_lrs = adaptive.update_learning_rates(perf_high, gradients)
    
    # Learning rates should increase due to performance improvement
    for modality in modalities:
        assert updated_lrs[modality] > adaptive.config['initial_lr']

def test_learning_rate_stats(config, modalities, performance_metrics, gradients):
    """Test learning rate statistics calculation"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    # Update multiple times
    for _ in range(10):
        adaptive.update_learning_rates(performance_metrics, gradients)
    
    stats = adaptive.get_learning_rate_stats()
    
    for modality in modalities:
        assert 'mean' in stats[modality]
        assert 'std' in stats[modality]
        assert 'min' in stats[modality]
        assert 'max' in stats[modality]
        
        assert stats[modality]['min'] >= config['min_lr']
        assert stats[modality]['max'] <= config['max_lr']

def test_performance_stats(config, modalities, performance_metrics, gradients):
    """Test performance statistics calculation"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    # Update multiple times
    for _ in range(10):
        adaptive.update_learning_rates(performance_metrics, gradients)
    
    stats = adaptive.get_performance_stats()
    
    for modality in modalities:
        assert 'mean' in stats[modality]
        assert 'std' in stats[modality]
        assert 'min' in stats[modality]
        assert 'max' in stats[modality]

def test_reset(config, modalities, performance_metrics, gradients):
    """Test reset functionality"""
    adaptive = AdaptiveLearning(config)
    adaptive.initialize_learning_rates(modalities)
    
    # Update multiple times
    for _ in range(5):
        adaptive.update_learning_rates(performance_metrics, gradients)
    
    # Reset
    adaptive.reset()
    
    assert len(adaptive.learning_rates) == 0
    assert len(adaptive.momentum_buffer) == 0
    assert len(adaptive.first_moment) == 0
    assert len(adaptive.second_moment) == 0
    assert adaptive.step_count == 0
    assert len(adaptive.performance_history) == 0
    assert len(adaptive.lr_history) == 0 