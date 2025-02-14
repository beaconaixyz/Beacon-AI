import pytest
import torch
import numpy as np
from beacon.models.feature_selector import AdaptiveFeatureSelector

@pytest.fixture
def config():
    """Create test configuration"""
    return {
        'selection_method': 'ensemble',
        'update_frequency': 5,
        'selection_threshold': 0.5,
        'min_features': 0.2,
        'smoothing_factor': 0.9,
        'use_uncertainty': True,
        'modality_specific': True,
        'interaction_threshold': 0.3,
        'adaptive_threshold': True,
        'threshold_patience': 5,
        'threshold_delta': 0.05
    }

@pytest.fixture
def feature_dims():
    """Create feature dimensions for each modality"""
    return {
        'images': 64,
        'sequences': 128,
        'clinical': 20
    }

@pytest.fixture
def sample_batch(feature_dims):
    """Create sample batch for testing"""
    batch_size = 8
    return {
        'images': torch.randn(batch_size, feature_dims['images']),
        'sequences': torch.randn(batch_size, feature_dims['sequences']),
        'clinical': torch.randn(batch_size, feature_dims['clinical'])
    }

@pytest.fixture
def sample_labels(sample_batch):
    """Create sample labels for testing"""
    batch_size = len(next(iter(sample_batch.values())))
    return torch.randint(0, 2, (batch_size,))

class MockModel(torch.nn.Module):
    """Enhanced mock model for testing"""
    def __init__(self, feature_dims):
        super().__init__()
        self.feature_dims = feature_dims
        
    def forward(self, batch):
        batch_size = len(next(iter(batch.values())))
        return torch.randn(batch_size, 2)
    
    def get_attention_weights(self, batch):
        return {
            modality: torch.rand(dim)
            for modality, dim in self.feature_dims.items()
        }
    
    def get_feature_importance(self, batch):
        return {
            modality: torch.rand(dim)
            for modality, dim in self.feature_dims.items()
        }

def test_initialization(config):
    """Test initialization of feature selector"""
    selector = AdaptiveFeatureSelector(config)
    
    assert selector.config['selection_method'] == config['selection_method']
    assert selector.config['update_frequency'] == config['update_frequency']
    assert len(selector.feature_masks) == 0
    assert len(selector.importance_history) == 0
    assert len(selector.performance_history['original']) == 0
    assert len(selector.performance_history['selected']) == 0

def test_ensemble_importance_computation(config, feature_dims, sample_batch, sample_labels):
    """Test ensemble-based importance score computation"""
    selector = AdaptiveFeatureSelector(config)
    model = MockModel(feature_dims)
    
    # Test ensemble method
    selector.config['selection_method'] = 'ensemble'
    scores = selector._compute_importance_scores(sample_batch, model, sample_labels)
    
    for modality, score in scores.items():
        assert score.shape == (feature_dims[modality],)
        assert torch.all(torch.isfinite(score))  # Check for valid scores

def test_mutual_info_importance(config, feature_dims, sample_batch, sample_labels):
    """Test mutual information based importance computation"""
    selector = AdaptiveFeatureSelector(config)
    model = MockModel(feature_dims)
    
    selector.config['selection_method'] = 'mutual_info'
    scores = selector._compute_importance_scores(sample_batch, model, sample_labels)
    
    for modality, score in scores.items():
        assert score.shape == (feature_dims[modality],)
        assert torch.all(score >= 0)  # MI scores should be non-negative

def test_adaptive_thresholds(config, feature_dims, sample_batch, sample_labels):
    """Test adaptive threshold adjustment"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Record initial thresholds
    initial_thresholds = {m: selector.thresholds[m] for m in feature_dims}
    
    # Simulate improving performance
    selector.validate_selection(sample_batch, model, sample_labels)
    selector.validate_selection(sample_batch, model, sample_labels)
    
    # Check threshold changes
    for modality in feature_dims:
        assert selector.thresholds[modality] != initial_thresholds[modality]

def test_feature_interactions(config, feature_dims, sample_batch):
    """Test feature interaction computation"""
    selector = AdaptiveFeatureSelector(config)
    model = MockModel(feature_dims)
    
    interactions = selector._compute_feature_interactions(sample_batch, model)
    
    # Check interaction matrix properties
    for key, matrix in interactions.items():
        modality1, modality2 = key.split('_')
        assert matrix.shape == (feature_dims[modality1], feature_dims[modality2])
        assert torch.all(torch.isfinite(matrix))

def test_stability_computation(config, feature_dims, sample_batch, sample_labels):
    """Test stability score computation"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Update selection multiple times
    for step in range(3 * config['update_frequency']):
        selector.update_feature_selection(sample_batch, model, step, sample_labels)
    
    # Check stability scores
    for modality in feature_dims:
        assert len(selector.stability_scores[modality]) > 0
        assert 0 <= selector.stability_scores[modality][-1] <= 1

def test_reduction_ratio(config, feature_dims, sample_batch):
    """Test feature reduction ratio computation"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    
    for modality in feature_dims:
        ratio = selector._compute_reduction_ratio(modality)
        assert 0 <= ratio <= 1

def test_validation_metrics(config, feature_dims, sample_batch, sample_labels):
    """Test validation metrics computation"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    metrics = selector.validate_selection(sample_batch, model, sample_labels)
    
    assert 'original_performance' in metrics
    assert 'selected_performance' in metrics
    assert 'stability' in metrics
    assert 'reduction' in metrics
    
    for modality in feature_dims:
        assert modality in metrics['stability']
        assert modality in metrics['reduction']

def test_performance_history(config, feature_dims, sample_batch, sample_labels):
    """Test performance history tracking"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    n_validations = 3
    for _ in range(n_validations):
        selector.validate_selection(sample_batch, model, sample_labels)
    
    assert len(selector.performance_history['original']) == n_validations
    assert len(selector.performance_history['selected']) == n_validations

def test_cross_modal_correlation(config, feature_dims, sample_batch):
    """Test cross-modal correlation computation"""
    selector = AdaptiveFeatureSelector(config)
    
    # Test correlation between two modalities
    corr = selector._compute_cross_modal_correlation(
        sample_batch['images'],
        sample_batch['sequences']
    )
    
    assert corr.shape == (feature_dims['images'], feature_dims['sequences'])
    assert torch.all(torch.isfinite(corr))
    assert torch.all(corr >= -1) and torch.all(corr <= 1)

def test_unsupervised_validation(config, feature_dims, sample_batch):
    """Test validation without labels (unsupervised case)"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    metrics = selector.validate_selection(sample_batch, model)
    
    assert 'original_performance' in metrics
    assert 'selected_performance' in metrics
    assert torch.isfinite(torch.tensor(metrics['original_performance']))
    assert torch.isfinite(torch.tensor(metrics['selected_performance']))

def test_mask_initialization(config, feature_dims):
    """Test initialization of feature masks"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    
    for modality, dim in feature_dims.items():
        assert modality in selector.feature_masks
        assert selector.feature_masks[modality].shape == (dim,)
        assert torch.all(selector.feature_masks[modality] == 1.0)
        assert modality in selector.importance_history
        assert len(selector.importance_history[modality]) == 0

def test_feature_selection_update(config, feature_dims, sample_batch):
    """Test feature selection update"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Update at step that matches update frequency
    step = config['update_frequency']
    masks = selector.update_feature_selection(sample_batch, model, step)
    
    for modality, mask in masks.items():
        assert mask.shape == (feature_dims[modality],)
        assert torch.all(mask >= 0) and torch.all(mask <= 1)
        min_features = int(feature_dims[modality] * config['min_features'])
        assert mask.sum() >= min_features

def test_importance_score_computation(config, feature_dims, sample_batch):
    """Test importance score computation"""
    selector = AdaptiveFeatureSelector(config)
    model = MockModel(feature_dims)
    
    # Test attention-based importance
    selector.config['selection_method'] = 'attention'
    scores = selector._compute_importance_scores(sample_batch, model)
    for modality, score in scores.items():
        assert score.shape == (feature_dims[modality],)
    
    # Test gradient-based importance
    selector.config['selection_method'] = 'gradient'
    scores = selector._compute_importance_scores(sample_batch, model)
    for modality, score in scores.items():
        assert score.shape == (feature_dims[modality],)

def test_threshold_computation(config, feature_dims):
    """Test threshold computation"""
    selector = AdaptiveFeatureSelector(config)
    scores = torch.rand(feature_dims['images'])
    
    # Test modality-specific threshold
    selector.config['modality_specific'] = True
    threshold = selector._compute_threshold(scores, 'images')
    assert isinstance(threshold, torch.Tensor)
    
    # Test global threshold
    selector.config['modality_specific'] = False
    threshold = selector._compute_threshold(scores, 'images')
    assert threshold == config['selection_threshold']

def test_mask_application(config, feature_dims, sample_batch):
    """Test application of feature masks"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    
    # Create binary masks
    for modality in feature_dims:
        selector.feature_masks[modality] = torch.randint(0, 2, (feature_dims[modality],)).float()
    
    # Apply masks
    masked_batch = selector.apply_masks(sample_batch)
    
    for modality in feature_dims:
        assert torch.all(masked_batch[modality] == (
            sample_batch[modality] * selector.feature_masks[modality]))

def test_importance_history_tracking(config, feature_dims, sample_batch):
    """Test tracking of importance history"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Update multiple times
    n_updates = 3
    for step in range(n_updates * config['update_frequency']):
        selector.update_feature_selection(sample_batch, model, step)
    
    history = selector.get_importance_history()
    for modality in feature_dims:
        assert len(history[modality]) <= n_updates

def test_minimum_features_constraint(config, feature_dims, sample_batch):
    """Test minimum features constraint"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Set very high threshold to force minimum features selection
    selector.selection_threshold = 0.99
    
    masks = selector.update_feature_selection(sample_batch, model, 
                                           config['update_frequency'])
    
    for modality, mask in masks.items():
        min_features = int(feature_dims[modality] * config['min_features'])
        assert mask.sum() >= min_features

def test_device_handling(config, feature_dims, sample_batch):
    """Test handling of different devices"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    
    # Move batch to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_gpu = {k: v.to(device) for k, v in sample_batch.items()}
    
    # Apply masks
    masked_batch = selector.apply_masks(batch_gpu)
    
    for modality in feature_dims:
        assert masked_batch[modality].device == device 

def test_feature_redundancy(config, feature_dims, sample_batch):
    """Test feature redundancy computation"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Create highly correlated features
    batch_size = len(sample_batch['images'])
    base_features = torch.randn(batch_size, 10)
    correlated_features = torch.cat([
        base_features,
        base_features + 0.1 * torch.randn(batch_size, 10)  # Highly correlated
    ], dim=1)
    
    sample_batch['correlated'] = correlated_features
    feature_dims['correlated'] = correlated_features.shape[1]
    
    # Compute importance scores
    importance = torch.ones(correlated_features.shape[1])
    redundancy = selector._compute_feature_redundancy(correlated_features, importance)
    
    # Check redundancy properties
    assert isinstance(redundancy, torch.Tensor)
    assert redundancy.shape == importance.shape
    assert torch.all(redundancy >= 0)
    
    # Check that redundancy is higher for correlated features
    first_half = redundancy[:10]
    second_half = redundancy[10:]
    assert torch.mean(second_half) > torch.mean(first_half)

def test_ensemble_weights(config, feature_dims, sample_batch, sample_labels):
    """Test ensemble weighting mechanism"""
    config['selection_method'] = 'ensemble'
    config['ensemble_weights'] = {
        'attention': 0.4,
        'gradient': 0.3,
        'mutual_info': 0.3
    }
    
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Compute importance scores
    scores = selector._compute_importance_scores(sample_batch, model, sample_labels)
    
    # Check scores properties
    for modality in scores:
        assert isinstance(scores[modality], torch.Tensor)
        assert scores[modality].shape == (feature_dims[modality],)
        assert torch.all(scores[modality] >= 0)
        assert torch.all(scores[modality] <= 1)

def test_score_normalization(config):
    """Test importance score normalization"""
    selector = AdaptiveFeatureSelector(config)
    
    # Test with normal scores
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = selector._normalize_scores(scores)
    assert torch.all(normalized >= 0)
    assert torch.all(normalized <= 1)
    assert torch.isclose(normalized.min(), torch.tensor(0.0))
    assert torch.isclose(normalized.max(), torch.tensor(1.0))
    
    # Test with zero scores
    zero_scores = torch.zeros(5)
    normalized = selector._normalize_scores(zero_scores)
    assert torch.all(normalized == 0)
    
    # Test with negative scores
    neg_scores = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    normalized = selector._normalize_scores(neg_scores)
    assert torch.all(normalized >= 0)
    assert torch.all(normalized <= 1)

def test_adaptive_threshold_update(config, feature_dims, sample_batch, sample_labels):
    """Test adaptive threshold updating"""
    config['adaptive_threshold'] = True
    config['threshold_delta'] = 0.1
    
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Initialize thresholds
    initial_thresholds = {m: 0.5 for m in feature_dims}
    selector.thresholds = initial_thresholds.copy()
    
    # Test threshold increase with performance improvement
    selector._update_thresholds(0.2)  # Large improvement
    for modality in feature_dims:
        assert selector.thresholds[modality] > initial_thresholds[modality]
    
    # Test threshold decrease with performance degradation
    selector._update_thresholds(-0.2)  # Large degradation
    for modality in feature_dims:
        assert selector.thresholds[modality] < initial_thresholds[modality]
    
    # Test threshold bounds
    selector._update_thresholds(1.0)  # Try to increase beyond max
    for modality in feature_dims:
        assert selector.thresholds[modality] <= 0.9
    
    selector._update_thresholds(-1.0)  # Try to decrease beyond min
    for modality in feature_dims:
        assert selector.thresholds[modality] >= config['min_features']

def test_feature_selection_with_redundancy(config, feature_dims, sample_batch, sample_labels):
    """Test feature selection with redundancy consideration"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Create redundant features
    batch_size = len(sample_batch['images'])
    base_features = torch.randn(batch_size, 10)
    redundant_features = torch.cat([
        base_features,
        base_features + 0.1 * torch.randn(batch_size, 10)
    ], dim=1)
    
    sample_batch['redundant'] = redundant_features
    feature_dims['redundant'] = redundant_features.shape[1]
    
    # Update feature selection
    masks = selector.update_feature_selection(sample_batch, model, 0, sample_labels)
    
    # Check that redundant features are handled
    assert 'redundant' in masks
    redundant_mask = masks['redundant']
    assert redundant_mask.sum() >= int(len(redundant_mask) * config['min_features'])
    assert redundant_mask.sum() <= len(redundant_mask)

def test_validation_metrics(config, feature_dims, sample_batch, sample_labels):
    """Test validation metrics computation"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Run validation
    metrics = selector.validate_selection(sample_batch, model, sample_labels)
    
    # Check metric properties
    assert 'original_performance' in metrics
    assert 'selected_performance' in metrics
    
    for modality in feature_dims:
        assert f'{modality}_stability' in metrics
        assert f'{modality}_reduction' in metrics
        assert 0 <= metrics[f'{modality}_stability'] <= 1
        assert 0 <= metrics[f'{modality}_reduction'] <= 1

def test_performance_history_tracking(config, feature_dims, sample_batch, sample_labels):
    """Test performance history tracking"""
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    model = MockModel(feature_dims)
    
    # Run multiple validation steps
    n_steps = 3
    for _ in range(n_steps):
        selector.validate_selection(sample_batch, model, sample_labels)
    
    # Check history properties
    assert len(selector.performance_history['original']) == n_steps
    assert len(selector.performance_history['selected']) == n_steps
    
    # Check that values are reasonable
    for history in selector.performance_history.values():
        assert all(isinstance(x, float) for x in history)
        assert all(x >= 0 for x in history)

def test_lasso_importance(config, feature_dims, sample_batch, sample_labels):
    """Test LASSO-based importance computation"""
    selector = AdaptiveFeatureSelector(config)
    selector.config['selection_method'] = 'lasso'
    
    scores = selector._compute_importance_scores(sample_batch, selector.model, sample_labels)
    
    for modality, score in scores.items():
        assert score.shape == (feature_dims[modality],)
        assert torch.all(score >= 0)  # LASSO coefficients should be non-negative after abs

def test_elastic_net_importance(config, feature_dims, sample_batch, sample_labels):
    """Test Elastic Net based importance computation"""
    selector = AdaptiveFeatureSelector(config)
    selector.config['selection_method'] = 'elastic_net'
    
    scores = selector._compute_importance_scores(sample_batch, selector.model, sample_labels)
    
    for modality, score in scores.items():
        assert score.shape == (feature_dims[modality],)
        assert torch.all(score >= 0)  # Elastic Net coefficients should be non-negative after abs

def test_group_selection(config, feature_dims, sample_batch):
    """Test feature group selection"""
    config['group_selection']['enabled'] = True
    config['group_selection']['n_groups'] = 3
    selector = AdaptiveFeatureSelector(config)
    
    # Test group identification
    for modality, features in sample_batch.items():
        scores = torch.rand(feature_dims[modality])
        groups, group_scores = selector._identify_feature_groups(features, scores)
        
        # Check group properties
        assert len(groups) == config['group_selection']['n_groups']
        assert len(group_scores) == config['group_selection']['n_groups']
        
        # Check that all features are assigned to groups
        assigned_features = set()
        for group in groups:
            assigned_features.update(group)
        assert len(assigned_features) == feature_dims[modality]
        
        # Test group-based selection
        mask = selector._select_features_by_group(features, scores)
        assert mask.shape == (feature_dims[modality],)
        assert torch.all((mask == 0) | (mask == 1))  # Binary mask
        assert mask.sum() >= feature_dims[modality] * config['min_features']

def test_cross_modal_importance(config, feature_dims, sample_batch):
    """Test cross-modal importance computation"""
    config['cross_modal']['enabled'] = True
    selector = AdaptiveFeatureSelector(config)
    
    # Compute base scores
    base_scores = {
        modality: torch.rand(dim)
        for modality, dim in feature_dims.items()
    }
    
    # Test weighted sum fusion
    config['cross_modal']['fusion_method'] = 'weighted_sum'
    adjusted_scores = selector._compute_cross_modal_importance(sample_batch, base_scores)
    
    for modality in feature_dims:
        assert modality in adjusted_scores
        assert adjusted_scores[modality].shape == (feature_dims[modality],)
        assert torch.all(torch.isfinite(adjusted_scores[modality]))
    
    # Test attention fusion
    config['cross_modal']['fusion_method'] = 'attention'
    adjusted_scores = selector._compute_cross_modal_importance(sample_batch, base_scores)
    
    for modality in feature_dims:
        assert modality in adjusted_scores
        assert adjusted_scores[modality].shape == (feature_dims[modality],)
        assert torch.all(torch.isfinite(adjusted_scores[modality]))

def test_ensemble_method_weights(config, feature_dims, sample_batch, sample_labels):
    """Test ensemble method with different weight configurations"""
    config['selection_method'] = 'ensemble'
    config['ensemble_weights'] = {
        'attention': 0.2,
        'gradient': 0.2,
        'mutual_info': 0.2,
        'lasso': 0.2,
        'elastic_net': 0.2
    }
    
    selector = AdaptiveFeatureSelector(config)
    scores = selector._compute_importance_scores(sample_batch, selector.model, sample_labels)
    
    for modality, score in scores.items():
        assert score.shape == (feature_dims[modality],)
        assert torch.all(score >= 0)
        assert torch.all(score <= 1)  # Normalized scores

def test_group_stability(config, feature_dims, sample_batch):
    """Test stability of group selection"""
    config['group_selection']['enabled'] = True
    config['group_selection']['n_groups'] = 3
    selector = AdaptiveFeatureSelector(config)
    
    # Run multiple group selections
    n_runs = 3
    group_assignments = []
    
    for modality, features in sample_batch.items():
        scores = torch.rand(feature_dims[modality])
        
        for _ in range(n_runs):
            groups, _ = selector._identify_feature_groups(features, scores)
            group_assignments.append(groups)
        
        # Check consistency of group sizes
        group_sizes = [len(group) for group in groups]
        assert min(group_sizes) > 0  # No empty groups
        assert sum(group_sizes) == feature_dims[modality]  # All features assigned

def test_cross_modal_interaction_strength(config, feature_dims, sample_batch):
    """Test strength of cross-modal interactions"""
    config['cross_modal']['enabled'] = True
    selector = AdaptiveFeatureSelector(config)
    
    # Create correlated features across modalities
    batch_size = len(next(iter(sample_batch.values())))
    base_features = torch.randn(batch_size, 10)
    
    correlated_batch = {
        'mod1': torch.cat([base_features, torch.randn(batch_size, 10)], dim=1),
        'mod2': torch.cat([base_features + 0.1 * torch.randn(batch_size, 10),
                          torch.randn(batch_size, 10)], dim=1)
    }
    
    # Compute interactions
    interactions = selector._compute_feature_interactions(correlated_batch)
    
    # Check interaction properties
    interaction_key = 'mod1_mod2'
    assert interaction_key in interactions
    interaction_matrix = interactions[interaction_key]
    
    # Correlation should be stronger in the first 10 features
    first_block = interaction_matrix[:10, :10]
    other_block = interaction_matrix[10:, 10:]
    assert torch.mean(torch.abs(first_block)) > torch.mean(torch.abs(other_block))

def test_adaptive_group_selection(config, feature_dims, sample_batch, sample_labels):
    """Test adaptive adjustment of group selection"""
    config['group_selection']['enabled'] = True
    config['group_selection']['n_groups'] = 3
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    
    # Simulate multiple updates
    n_updates = 3
    for step in range(n_updates):
        selector.update_feature_selection(sample_batch, selector.model, step, sample_labels)
        
        # Validate group selection
        for modality in feature_dims:
            mask = selector.feature_masks[modality]
            assert torch.all((mask == 0) | (mask == 1))  # Binary mask
            assert mask.sum() >= feature_dims[modality] * config['min_features']
            
            # Check stability
            if step > 0:
                stability = selector.stability_scores[modality][-1]
                assert 0 <= stability <= 1

def test_method_comparison(config, feature_dims, sample_batch, sample_labels):
    """Test comparison of different feature selection methods"""
    methods = ['attention', 'gradient', 'lasso', 'elastic_net', 'ensemble']
    scores_by_method = {}
    
    for method in methods:
        config['selection_method'] = method
        selector = AdaptiveFeatureSelector(config)
        scores = selector._compute_importance_scores(sample_batch, selector.model, sample_labels)
        scores_by_method[method] = scores
    
    # Compare properties across methods
    for method, scores in scores_by_method.items():
        for modality, score in scores.items():
            assert score.shape == (feature_dims[modality],)
            assert torch.all(score >= 0)
            assert torch.all(torch.isfinite(score))

def test_group_selection_with_redundancy(config, feature_dims, sample_batch):
    """Test group selection with redundant features"""
    config['group_selection']['enabled'] = True
    selector = AdaptiveFeatureSelector(config)
    
    # Create redundant features
    batch_size = len(next(iter(sample_batch.values())))
    base_features = torch.randn(batch_size, 10)
    redundant_batch = {
        'redundant': torch.cat([
            base_features,
            base_features + 0.1 * torch.randn(batch_size, 10),
            torch.randn(batch_size, 10)
        ], dim=1)
    }
    
    # Test group identification
    scores = torch.rand(redundant_batch['redundant'].shape[1])
    groups, group_scores = selector._identify_feature_groups(
        redundant_batch['redundant'],
        scores
    )
    
    # Check that correlated features tend to be grouped together
    for group in groups:
        features_in_group = set(group)
        if any(i < 10 for i in features_in_group):
            # If group contains original features, it should also contain their redundant copies
            assert any(10 <= i < 20 for i in features_in_group)

def test_cross_modal_validation(config, feature_dims, sample_batch, sample_labels):
    """Test validation with cross-modal feature selection"""
    config['cross_modal']['enabled'] = True
    selector = AdaptiveFeatureSelector(config)
    selector.initialize_masks(feature_dims)
    
    # Run validation
    metrics = selector.validate_selection(sample_batch, selector.model, sample_labels)
    
    # Check metric properties
    assert 'original_performance' in metrics
    assert 'selected_performance' in metrics
    
    for modality in feature_dims:
        assert f'{modality}_stability' in metrics
        assert f'{modality}_reduction' in metrics
        
        # Check cross-modal metrics
        for other_modality in feature_dims:
            if modality != other_modality:
                interaction_key = f'{modality}_{other_modality}_interaction'
                if interaction_key in metrics:
                    assert 0 <= metrics[interaction_key] <= 1 