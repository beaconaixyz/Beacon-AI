import pytest
import numpy as np
import cv2
from beacon.data.image_processor import MedicalImageProcessor

@pytest.fixture
def sample_image():
    """Create sample medical image for testing"""
    # Create synthetic image with some patterns
    size = 128
    image = np.zeros((size, size), dtype=np.float32)
    
    # Add some circles and rectangles
    cv2.circle(image, (size//4, size//4), size//8, 1.0, -1)
    cv2.rectangle(image, (size//2, size//2), (3*size//4, 3*size//4), 0.5, -1)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    return image

@pytest.fixture
def processor_config():
    """Create sample processor configuration"""
    return {
        'normalize': True,
        'denoise': True,
        'enhance_contrast': True,
        'denoise_method': 'gaussian',
        'contrast_method': 'clahe',
        'use_augmentation': True,
        'extract_texture': True,
        'augmentation': {
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'rotation_limit': 15
        }
    }

def test_initialization(processor_config):
    """Test processor initialization"""
    processor = MedicalImageProcessor(processor_config)
    assert processor.config == processor_config
    assert processor.augmentation is not None

def test_preprocessing(sample_image, processor_config):
    """Test image preprocessing"""
    processor = MedicalImageProcessor(processor_config)
    processed = processor.preprocess(sample_image)
    
    # Check output properties
    assert isinstance(processed, np.ndarray)
    assert processed.shape == sample_image.shape
    assert processed.dtype == np.float32
    assert np.all(np.isfinite(processed))  # No NaN or inf values

def test_augmentation(sample_image, processor_config):
    """Test image augmentation"""
    processor = MedicalImageProcessor(processor_config)
    augmented = processor.augment(sample_image)
    
    # Check output properties
    assert isinstance(augmented, np.ndarray)
    assert augmented.shape == sample_image.shape
    assert augmented.dtype == np.float32
    assert np.all(np.isfinite(augmented))

def test_normalization(sample_image, processor_config):
    """Test image normalization"""
    processor = MedicalImageProcessor(processor_config)
    normalized = processor._normalize(sample_image)
    
    # Check normalization properties
    assert abs(np.mean(normalized)) < 1e-6  # Close to zero mean
    assert abs(np.std(normalized) - 1.0) < 1e-6  # Unit variance

def test_denoising(sample_image, processor_config):
    """Test image denoising"""
    processor = MedicalImageProcessor(processor_config)
    
    # Test all denoising methods
    for method in ['gaussian', 'median', 'bilateral']:
        processor.config['denoise_method'] = method
        denoised = processor._denoise(sample_image)
        
        assert isinstance(denoised, np.ndarray)
        assert denoised.shape == sample_image.shape
        assert denoised.dtype == np.float32
        assert np.all(np.isfinite(denoised))

def test_contrast_enhancement(sample_image, processor_config):
    """Test contrast enhancement"""
    processor = MedicalImageProcessor(processor_config)
    
    # Test all contrast enhancement methods
    for method in ['clahe', 'adaptive']:
        processor.config['contrast_method'] = method
        enhanced = processor._enhance_contrast(sample_image)
        
        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == np.float32
        assert np.all(np.isfinite(enhanced))

def test_tissue_segmentation(sample_image, processor_config):
    """Test tissue segmentation"""
    processor = MedicalImageProcessor(processor_config)
    segmented, mask = processor.segment_tissue(sample_image)
    
    # Check segmentation output
    assert isinstance(segmented, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert segmented.shape == sample_image.shape
    assert mask.shape == sample_image.shape
    assert mask.dtype == bool
    assert np.all(np.isfinite(segmented))

def test_feature_extraction(sample_image, processor_config):
    """Test feature extraction"""
    processor = MedicalImageProcessor(processor_config)
    features = processor.extract_features(sample_image)
    
    # Check extracted features
    assert isinstance(features, dict)
    assert 'mean_intensity' in features
    assert 'std_intensity' in features
    assert 'contrast' in features
    assert 'homogeneity' in features
    assert 'energy' in features
    assert 'correlation' in features
    
    # Check feature values
    for value in features.values():
        assert isinstance(value, float)
        assert np.isfinite(value)

def test_glcm_computation(sample_image, processor_config):
    """Test GLCM computation"""
    processor = MedicalImageProcessor(processor_config)
    glcm = processor._compute_glcm(sample_image)
    
    # Check GLCM properties
    assert isinstance(glcm, np.ndarray)
    assert glcm.shape == (processor.config.get('glcm_gray_levels', 16),) * 2
    assert np.all(glcm >= 0)
    assert np.allclose(np.sum(glcm), 1.0)  # Should be normalized

def test_texture_feature_computation(sample_image, processor_config):
    """Test texture feature computation"""
    processor = MedicalImageProcessor(processor_config)
    glcm = processor._compute_glcm(sample_image)
    features = processor._compute_texture_features(glcm)
    
    # Check texture features
    assert isinstance(features, dict)
    assert 'contrast' in features
    assert 'homogeneity' in features
    assert 'energy' in features
    assert 'correlation' in features
    
    # Check feature values
    for value in features.values():
        assert isinstance(value, float)
        assert np.isfinite(value)

def test_processing_pipeline(sample_image, processor_config):
    """Test complete processing pipeline"""
    processor = MedicalImageProcessor(processor_config)
    
    # Apply full pipeline
    processed = processor.preprocess(sample_image)
    augmented = processor.augment(processed)
    segmented, mask = processor.segment_tissue(augmented)
    features = processor.extract_features(segmented)
    
    # Check final outputs
    assert isinstance(processed, np.ndarray)
    assert isinstance(augmented, np.ndarray)
    assert isinstance(segmented, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert isinstance(features, dict)
    assert np.all(np.isfinite(processed))
    assert np.all(np.isfinite(augmented))
    assert np.all(np.isfinite(segmented)) 