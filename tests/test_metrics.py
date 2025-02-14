import pytest
import numpy as np
from beacon.utils.metrics import Metrics

@pytest.fixture
def binary_classification_data():
    """Create sample binary classification data"""
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    y_prob = np.random.rand(10)
    return y_true, y_pred, y_prob

@pytest.fixture
def multiclass_classification_data():
    """Create sample multiclass classification data"""
    np.random.seed(42)
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 0, 2, 1])
    y_prob = np.random.rand(10, 3)
    # Normalize probabilities
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    return y_true, y_pred, y_prob

@pytest.fixture
def regression_data():
    """Create sample regression data"""
    np.random.seed(42)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
    return y_true, y_pred

def test_binary_classification_metrics(binary_classification_data):
    """Test binary classification metrics"""
    y_true, y_pred, y_prob = binary_classification_data
    
    # Calculate metrics
    metrics = Metrics.calculate_classification_metrics(
        y_true, y_pred, y_prob, average='binary'
    )
    
    # Check metric keys
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 'average_precision' in metrics
    
    # Check metric values
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1
    assert 0 <= metrics['average_precision'] <= 1

def test_multiclass_classification_metrics(multiclass_classification_data):
    """Test multiclass classification metrics"""
    y_true, y_pred, y_prob = multiclass_classification_data
    
    # Calculate metrics
    metrics = Metrics.calculate_classification_metrics(
        y_true, y_pred, y_prob, average='macro'
    )
    
    # Check metric keys
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 'average_precision' in metrics
    
    # Check metric values
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1
    assert 0 <= metrics['average_precision'] <= 1

def test_regression_metrics(regression_data):
    """Test regression metrics"""
    y_true, y_pred = regression_data
    
    # Calculate metrics
    metrics = Metrics.calculate_regression_metrics(y_true, y_pred)
    
    # Check metric keys
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    
    # Check metric values
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['r2'] <= 1

def test_confidence_intervals():
    """Test confidence interval calculation"""
    metric_values = [0.85, 0.87, 0.83, 0.89, 0.86]
    
    # Calculate confidence intervals
    ci = Metrics.calculate_confidence_intervals(metric_values)
    
    # Check keys
    assert 'mean' in ci
    assert 'lower_bound' in ci
    assert 'upper_bound' in ci
    assert 'std_error' in ci
    
    # Check values
    assert ci['lower_bound'] <= ci['mean'] <= ci['upper_bound']
    assert ci['std_error'] >= 0

def test_metric_formatting():
    """Test metric formatting"""
    metrics = {
        'accuracy': 0.8567,
        'precision': 0.9234,
        'recall': 0.7845
    }
    
    # Format metrics
    formatted = Metrics.format_metrics(metrics, precision=3)
    
    # Check formatting
    assert formatted['accuracy'] == '0.857'
    assert formatted['precision'] == '0.923'
    assert formatted['recall'] == '0.785'

def test_metric_descriptions():
    """Test metric descriptions"""
    # Check common metrics
    assert len(Metrics.get_metric_description('accuracy')) > 0
    assert len(Metrics.get_metric_description('precision')) > 0
    assert len(Metrics.get_metric_description('recall')) > 0
    
    # Check unknown metric
    assert Metrics.get_metric_description('unknown_metric') == 'No description available for this metric'

def test_edge_cases():
    """Test edge cases"""
    # Empty data
    with pytest.raises(ValueError):
        Metrics.calculate_confidence_intervals([])
    
    # Perfect predictions
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = Metrics.calculate_classification_metrics(y_true, y_pred)
    assert metrics['accuracy'] == 1.0
    
    # All wrong predictions
    y_pred = np.array([1, 0, 1, 0])
    metrics = Metrics.calculate_classification_metrics(y_true, y_pred)
    assert metrics['accuracy'] == 0.0 