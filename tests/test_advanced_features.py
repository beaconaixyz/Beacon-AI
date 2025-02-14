import pytest
import torch
import numpy as np
from beacon.models.feature_selector import AdaptiveFeatureSelector
from beacon.optimization.performance_optimizer import PerformanceOptimizer
from beacon.visualization.advanced_visualizer import AdvancedVisualizer

@pytest.fixture
def sample_multimodal_data():
    """Create sample multimodal data for testing"""
    n_samples = 100
    data = {
        'clinical': torch.randn(n_samples, 20),
        'genomic': torch.randn(n_samples, 1000),
        'imaging': torch.randn(n_samples, 50, 50)
    }
    return data

@pytest.fixture
def feature_selector():
    """Create feature selector instance"""
    config = {
        'n_features': {
            'clinical': 20,
            'genomic': 1000,
            'imaging': 2500
        },
        'selection_method': 'group_lasso',
        'group_size': 50,
        'modalities': ['clinical', 'genomic', 'imaging']
    }
    return AdaptiveFeatureSelector(config)

@pytest.fixture
def performance_optimizer():
    """Create performance optimizer instance"""
    config = {
        'batch_size_range': [32, 64, 128, 256],
        'max_memory_gb': 4.0
    }
    return PerformanceOptimizer(config)

@pytest.fixture
def advanced_visualizer():
    """Create advanced visualizer instance"""
    config = {
        'style': 'seaborn',
        'save_format': 'png',
        'dpi': 300
    }
    return AdvancedVisualizer(config)

def test_feature_selection_stability(feature_selector, sample_multimodal_data):
    """Test stability of feature selection"""
    n_runs = 5
    selected_features = []
    
    # Run feature selection multiple times
    for _ in range(n_runs):
        features = feature_selector.select_features(sample_multimodal_data)
        selected_features.append(features)
    
    # Check stability across runs
    for modality in sample_multimodal_data.keys():
        selections = torch.stack([f[modality] for f in selected_features])
        stability = torch.mean(torch.std(selections, dim=0))
        assert stability < 0.5  # Ensure reasonable stability

def test_performance_optimization(performance_optimizer, feature_selector, sample_multimodal_data):
    """Test performance optimization functionality"""
    # Test batch size optimization
    optimal_batch_size = performance_optimizer.optimize_batch_size(
        feature_selector,
        sample_multimodal_data
    )
    assert optimal_batch_size in [32, 64, 128, 256, 512]
    
    # Test memory optimization
    optimized_data = performance_optimizer.optimize_memory_usage(
        sample_multimodal_data
    )
    assert all(k in optimized_data for k in sample_multimodal_data.keys())
    
    # Test parallel processing
    parallel_results = performance_optimizer.parallelize_feature_selection(
        feature_selector,
        sample_multimodal_data
    )
    assert all(k in parallel_results for k in sample_multimodal_data.keys())

def test_visualization_functionality(advanced_visualizer, feature_selector, sample_multimodal_data):
    """Test visualization functionality"""
    # Get feature importance
    importance_scores = feature_selector.get_feature_importance()
    
    # Test feature importance evolution plot
    history = {
        modality: [torch.randn_like(scores) for _ in range(5)]
        for modality, scores in importance_scores.items()
    }
    advanced_visualizer.plot_feature_importance_evolution(history)
    
    # Test correlation network plot
    correlation_matrix = torch.corrcoef(
        sample_multimodal_data['clinical'].T
    )
    feature_names = [f'Feature_{i}' for i in range(20)]
    advanced_visualizer.plot_feature_correlation_network(
        correlation_matrix,
        feature_names
    )
    
    # Test feature embedding plot
    advanced_visualizer.plot_feature_embedding(
        sample_multimodal_data['clinical'],
        importance_scores['clinical']
    )

def test_group_feature_selection(feature_selector, sample_multimodal_data):
    """Test group-based feature selection"""
    # Select features with group constraint
    selected_features = feature_selector.select_features(
        sample_multimodal_data,
        use_groups=True
    )
    
    # Analyze group structure
    groups = feature_selector.analyze_feature_groups()
    
    # Check group properties
    for modality in sample_multimodal_data.keys():
        if modality in groups:
            # Check group sizes
            group_sizes = [len(g) for g in groups[modality]]
            assert all(s <= feature_selector.config['group_size'] for s in group_sizes)
            
            # Check group coherence
            coherence = feature_selector.compute_group_coherence(
                groups[modality],
                sample_multimodal_data[modality]
            )
            assert coherence > 0.5  # Ensure reasonable coherence

def test_temporal_feature_analysis(feature_selector):
    """Test temporal feature analysis"""
    # Create temporal data
    n_samples = 50
    n_timepoints = 5
    temporal_data = {
        'clinical': torch.randn(n_samples, n_timepoints, 20),
        'genomic': torch.randn(n_samples, n_timepoints, 100)
    }
    
    # Analyze temporal patterns
    patterns = feature_selector.analyze_temporal_patterns(temporal_data)
    
    # Check pattern properties
    for modality in temporal_data.keys():
        assert modality in patterns
        assert patterns[modality].shape[1] == n_timepoints
        
        # Check temporal consistency
        consistency = feature_selector.compute_temporal_consistency(
            patterns[modality]
        )
        assert consistency > 0.5  # Ensure reasonable consistency

def test_feature_interaction_analysis(feature_selector, sample_multimodal_data):
    """Test feature interaction analysis"""
    # Analyze feature interactions
    interactions = feature_selector.analyze_feature_interactions(
        sample_multimodal_data
    )
    
    # Check interaction properties
    for modality_pair, interaction_matrix in interactions.items():
        # Check matrix properties
        assert interaction_matrix.shape[0] == interaction_matrix.shape[1]
        assert torch.all(interaction_matrix >= -1) and torch.all(interaction_matrix <= 1)
        
        # Check symmetry
        assert torch.allclose(interaction_matrix, interaction_matrix.T)

def test_performance_impact_analysis(feature_selector, sample_multimodal_data):
    """Test analysis of feature selection impact on performance"""
    # Create synthetic labels
    labels = torch.randint(0, 2, (len(next(iter(sample_multimodal_data.values()))),))
    
    # Analyze performance impact
    impact_analysis = feature_selector.analyze_performance_impact(
        sample_multimodal_data,
        labels
    )
    
    # Check analysis results
    assert 'performance_metrics' in impact_analysis
    assert 'feature_counts' in impact_analysis
    
    metrics = impact_analysis['performance_metrics']
    for metric_name, values in metrics.items():
        assert len(values) > 0
        assert all(0 <= v <= 1 for v in values)  # Assuming normalized metrics

def test_advanced_visualization_dashboard(advanced_visualizer, feature_selector, sample_multimodal_data):
    """Test creation of comprehensive visualization dashboard"""
    # Generate necessary data
    importance_history = {
        modality: [torch.randn_like(data) for _ in range(5)]
        for modality, data in sample_multimodal_data.items()
    }
    
    performance_metrics = {
        'accuracy': [0.8, 0.85, 0.87, 0.89, 0.9],
        'f1_score': [0.75, 0.82, 0.85, 0.87, 0.89]
    }
    
    feature_counts = [10, 20, 30, 40, 50]
    
    stability_scores = {
        modality: torch.rand(100)
        for modality in sample_multimodal_data.keys()
    }
    
    interaction_matrix = torch.rand(10, 10)
    group_names = [f'Group_{i}' for i in range(10)]
    
    temporal_patterns = {
        modality: torch.randn(50, 5)
        for modality in sample_multimodal_data.keys()
    }
    
    # Create dashboard
    results = {
        'importance_history': importance_history,
        'performance_metrics': performance_metrics,
        'feature_counts': feature_counts,
        'stability_scores': stability_scores,
        'interaction_matrix': interaction_matrix,
        'group_names': group_names,
        'temporal_patterns': temporal_patterns
    }
    
    advanced_visualizer.create_interactive_dashboard(results)

def test_error_handling(feature_selector, performance_optimizer, advanced_visualizer):
    """Test error handling in advanced features"""
    # Test invalid data
    with pytest.raises(ValueError):
        feature_selector.select_features({'invalid_modality': torch.randn(10, 10)})
    
    # Test memory optimization with invalid input
    with pytest.raises(ValueError):
        performance_optimizer.optimize_memory_usage(
            {'clinical': torch.randn(1000000, 1000000)},  # Too large
            max_memory_gb=0.1
        )
    
    # Test visualization with invalid input
    with pytest.raises(ValueError):
        advanced_visualizer.plot_feature_correlation_network(
            torch.randn(10, 20),  # Non-square matrix
            ['Feature_1', 'Feature_2']  # Mismatched feature names
        ) 