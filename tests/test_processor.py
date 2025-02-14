import pytest
import numpy as np
from beacon.data.processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create synthetic data with known properties
    np.random.seed(42)
    n_samples, n_features = 100, 5
    data = np.random.randn(n_samples, n_features)
    
    # Add some missing values
    data[0, 0] = np.nan
    data[1, 1] = np.nan
    
    # Add some outliers
    data[2, 2] = 100.0  # Obvious outlier
    
    return data

@pytest.fixture
def processor_config():
    """Create sample processor configuration"""
    return {
        'scaling_method': 'standard',
        'handle_missing': True,
        'missing_strategy': 'mean',
        'remove_outliers': True,
        'outlier_threshold': 3.0
    }

def test_initialization(processor_config):
    """Test processor initialization"""
    processor = DataProcessor(processor_config)
    assert processor.config['scaling_method'] == 'standard'
    assert processor.scaler is not None

def test_data_validation(sample_data):
    """Test data validation"""
    processor = DataProcessor({})
    
    # Test valid data
    is_valid, error = processor.validate_data(sample_data)
    assert is_valid
    assert error is None
    
    # Test invalid data types
    is_valid, error = processor.validate_data([1, 2, 3])
    assert not is_valid
    assert "numpy array" in error
    
    # Test empty data
    is_valid, error = processor.validate_data(np.array([]))
    assert not is_valid
    assert "Empty" in error

def test_missing_value_handling(sample_data, processor_config):
    """Test missing value handling"""
    processor = DataProcessor(processor_config)
    
    # Test mean strategy
    processor.config['missing_strategy'] = 'mean'
    filled_data = processor._handle_missing_values(sample_data)
    assert not np.any(np.isnan(filled_data))
    
    # Test median strategy
    processor.config['missing_strategy'] = 'median'
    filled_data = processor._handle_missing_values(sample_data)
    assert not np.any(np.isnan(filled_data))
    
    # Test zero strategy
    processor.config['missing_strategy'] = 'zero'
    filled_data = processor._handle_missing_values(sample_data)
    assert not np.any(np.isnan(filled_data))

def test_outlier_removal(sample_data, processor_config):
    """Test outlier removal"""
    processor = DataProcessor(processor_config)
    cleaned_data = processor._remove_outliers(sample_data)
    
    # Should have removed at least one outlier
    assert len(cleaned_data) < len(sample_data)
    
    # Check if obvious outlier was removed
    max_value = np.max(np.abs(cleaned_data))
    assert max_value < 100.0

def test_scaling(sample_data, processor_config):
    """Test data scaling"""
    processor = DataProcessor(processor_config)
    
    # Test standard scaling
    processor.config['scaling_method'] = 'standard'
    processor._initialize_scaler()
    scaled_data = processor.fit_transform(sample_data)
    assert np.allclose(np.mean(scaled_data, axis=0), 0, atol=1e-10)
    assert np.allclose(np.std(scaled_data, axis=0), 1, atol=1e-10)
    
    # Test minmax scaling
    processor.config['scaling_method'] = 'minmax'
    processor._initialize_scaler()
    scaled_data = processor.fit_transform(sample_data)
    assert np.all(scaled_data >= 0)
    assert np.all(scaled_data <= 1)

def test_feature_stats(sample_data, processor_config):
    """Test feature statistics computation"""
    processor = DataProcessor(processor_config)
    stats = processor.get_feature_stats(sample_data)
    
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'missing_ratio' in stats
    
    assert stats['mean'].shape[0] == sample_data.shape[1]
    assert stats['missing_ratio'].shape[0] == sample_data.shape[1]

def test_transform_without_fit(sample_data, processor_config):
    """Test transform without fitting first"""
    processor = DataProcessor(processor_config)
    
    with pytest.raises(RuntimeError):
        processor.transform(sample_data)

def test_full_pipeline(sample_data, processor_config):
    """Test complete processing pipeline"""
    processor = DataProcessor(processor_config)
    
    # Process data through full pipeline
    processed_data = processor.fit_transform(sample_data)
    
    # Verify results
    assert not np.any(np.isnan(processed_data))  # No missing values
    assert len(processed_data) < len(sample_data)  # Outliers removed
    assert np.allclose(np.mean(processed_data, axis=0), 0, atol=1e-10)  # Standardized 