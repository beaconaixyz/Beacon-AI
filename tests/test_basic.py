import pytest
import torch
import numpy as np
import pandas as pd
from beacon.models.cancer_classifier import CancerClassifier
from beacon.data.processor import DataProcessor
from beacon.utils.metrics import Metrics

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    clinical_data = pd.DataFrame({
        'age': np.random.normal(60, 10, n_samples),
        'tumor_size': np.random.normal(3, 1, n_samples),
        'marker_level': np.random.normal(100, 20, n_samples)
    })
    
    labels = torch.randint(0, 2, (n_samples,))
    
    return clinical_data, labels

def test_data_processor(sample_data):
    """Test data processor functionality"""
    clinical_data, _ = sample_data
    
    config = {'normalization': 'standard'}
    processor = DataProcessor(config)
    
    processed_data = processor.process_clinical_data(clinical_data)
    
    assert isinstance(processed_data, np.ndarray)
    assert processed_data.dtype == np.float32
    assert processed_data.shape[0] == len(clinical_data)

def test_cancer_classifier(sample_data):
    """Test cancer classifier functionality"""
    clinical_data, labels = sample_data
    
    # Configure model
    model_config = {
        'input_dim': 3,  # Three features in sample data
        'hidden_dim': 32,
        'output_dim': 2,
        'learning_rate': 0.001
    }
    
    model = CancerClassifier(model_config)
    
    # Process data
    processor = DataProcessor({'normalization': 'standard'})
    features = processor.process_clinical_data(clinical_data)
    features = torch.FloatTensor(features)
    
    # Train model
    data = {
        'features': features,
        'labels': labels
    }
    
    history = model.train(data, epochs=2, batch_size=32)
    
    assert isinstance(history, list)
    assert len(history) == 2  # Two epochs
    assert 'loss' in history[0]
    
    # Test predictions
    predictions = model.predict(features)
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(features), 2)

def test_metrics():
    """Test metrics calculations"""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
    
    metrics = Metrics.calculate_all_metrics(y_true, y_pred, y_prob)
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'auc_roc' in metrics
    assert 'average_precision' in metrics
    
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1 