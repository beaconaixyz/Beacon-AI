import pytest
import torch
import numpy as np
import os
import time
from beacon.utils.image_interpretability import ImageInterpreter
from beacon.models.image_classifier import MedicalImageCNN

@pytest.fixture
def sample_image():
    """Create sample medical image for testing"""
    # Create synthetic image with some patterns
    batch_size = 1
    channels = 1
    height = 64
    width = 64
    
    image = torch.zeros((batch_size, channels, height, width))
    
    # Add some circles and rectangles
    center_x = width // 4
    center_y = height // 4
    radius = min(width, height) // 8
    
    y, x = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing='ij'
    )
    
    # Add circle
    mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
    image[0, 0, mask] = 1.0
    
    # Add rectangle
    image[0, 0, height//2:3*height//4, width//2:3*width//4] = 0.5
    
    # Add some noise
    image += torch.randn_like(image) * 0.1
    image = torch.clamp(image, 0, 1)
    
    return image

@pytest.fixture
def trained_model(sample_image):
    """Create and train a simple CNN model for testing"""
    model_config = {
        'in_channels': 1,
        'num_classes': 2
    }
    
    model = MedicalImageCNN(model_config)
    
    # Train with synthetic data
    n_samples = 10
    images = torch.randn(n_samples, 1, 64, 64)
    labels = torch.randint(0, 2, (n_samples,))
    
    model.train({
        'images': images,
        'labels': labels
    })
    
    return model

def test_interpreter_initialization():
    """Test interpreter initialization"""
    config = {
        'method': 'guided_gradcam',
        'occlusion_window': 8
    }
    
    interpreter = ImageInterpreter(config)
    assert interpreter.method == 'guided_gradcam'
    assert interpreter.config['occlusion_window'] == 8

def test_guided_gradcam(trained_model, sample_image):
    """Test Guided GradCAM attribution"""
    interpreter = ImageInterpreter({'method': 'guided_gradcam'})
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == sample_image.shape
    assert metadata['method'] == 'guided_gradcam'

def test_occlusion(trained_model, sample_image):
    """Test Occlusion attribution"""
    interpreter = ImageInterpreter({
        'method': 'occlusion',
        'occlusion_window': 8,
        'occlusion_stride': 4
    })
    
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == sample_image.shape
    assert metadata['method'] == 'occlusion'
    assert metadata['window_size'] == 8
    assert metadata['stride'] == 4

def test_gradient(trained_model, sample_image):
    """Test Gradient attribution"""
    interpreter = ImageInterpreter({'method': 'gradient'})
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == sample_image.shape
    assert metadata['method'] == 'gradient'

def test_gradient_shap(trained_model, sample_image):
    """Test GradientSHAP attribution"""
    interpreter = ImageInterpreter({'method': 'gradshap'})
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == sample_image.shape
    assert metadata['method'] == 'gradshap'

def test_visualization(tmp_path, trained_model, sample_image):
    """Test attribution visualization"""
    interpreter = ImageInterpreter({})
    attribution, _ = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    
    # Test visualization without saving
    interpreter.visualize_attribution(
        sample_image,
        attribution
    )
    
    # Test visualization with saving
    save_path = tmp_path / "test_attribution.png"
    interpreter.visualize_attribution(
        sample_image,
        attribution,
        str(save_path)
    )
    
    assert save_path.exists()

def test_class_activation_map(trained_model, sample_image):
    """Test Class Activation Map generation"""
    interpreter = ImageInterpreter({})
    
    # Get last convolutional layer
    conv_layer = None
    for module in reversed(list(trained_model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            conv_layer = module
            break
    
    cam = interpreter.generate_class_activation_map(
        trained_model,
        sample_image,
        conv_layer
    )
    
    assert isinstance(cam, torch.Tensor)
    assert cam.shape[-2:] == sample_image.shape[-2:]

def test_activation_analysis(trained_model, sample_image):
    """Test activation pattern analysis"""
    interpreter = ImageInterpreter({})
    
    # Find a convolutional layer
    conv_layer_name = None
    for name, module in trained_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layer_name = name
            break
    
    stats = interpreter.analyze_activation_patterns(
        trained_model,
        sample_image,
        conv_layer_name
    )
    
    assert 'mean_activation' in stats
    assert 'max_activation' in stats
    assert 'channel_correlation' in stats
    
    # Test activation visualization
    activation = stats['mean_activation'].unsqueeze(-1).unsqueeze(-1)
    interpreter.visualize_layer_activations(
        activation.unsqueeze(0),
        n_channels=4
    )

def test_invalid_method():
    """Test invalid interpretation method"""
    interpreter = ImageInterpreter({'method': 'invalid_method'})
    
    with pytest.raises(ValueError):
        interpreter.explain_prediction(None, None)

def test_invalid_layer():
    """Test invalid layer name for activation analysis"""
    interpreter = ImageInterpreter({})
    
    with pytest.raises(ValueError):
        interpreter.analyze_activation_patterns(
            None,
            None,
            'invalid_layer'
        )

def test_feature_clustering(trained_model, sample_image):
    """Test feature clustering analysis"""
    interpreter = ImageInterpreter({})
    
    # Test clustering analysis
    clustering_results = interpreter.analyze_feature_clusters(
        trained_model,
        sample_image,
        'conv1'  # Test with first conv layer
    )
    
    # Check results structure
    assert isinstance(clustering_results, dict)
    assert 'features_2d' in clustering_results
    assert 'clusters' in clustering_results
    assert 'cluster_centers' in clustering_results
    assert 'cluster_stats' in clustering_results
    
    # Check shapes and types
    assert isinstance(clustering_results['features_2d'], np.ndarray)
    assert isinstance(clustering_results['clusters'], np.ndarray)
    assert isinstance(clustering_results['cluster_centers'], np.ndarray)
    assert isinstance(clustering_results['cluster_stats'], dict)
    
    # Test visualization
    interpreter.visualize_feature_clusters(clustering_results)

def test_interactive_visualization(tmp_path, trained_model, sample_image):
    """Test interactive visualization generation"""
    interpreter = ImageInterpreter({})
    
    # Generate attributions using different methods
    attribution_maps = {}
    methods = ['gradient', 'integrated_gradients', 'deep_lift']
    
    for method in methods:
        interpreter.method = method
        attribution, _ = interpreter.explain_prediction(trained_model, sample_image)
        attribution_maps[method] = attribution
    
    # Test interactive visualization
    save_path = tmp_path / "interactive_viz.html"
    interpreter.create_interactive_visualization(
        sample_image,
        attribution_maps,
        str(save_path)
    )
    
    # Check if file was created
    assert save_path.exists()
    assert save_path.stat().st_size > 0

def test_3d_visualization(tmp_path, trained_model, sample_image):
    """Test 3D visualization generation"""
    interpreter = ImageInterpreter({})
    
    # Generate attribution
    attribution, _ = interpreter.explain_prediction(trained_model, sample_image)
    
    # Test 3D visualization
    save_path = tmp_path / "3d_viz.png"
    interpreter.visualize_3d_attribution(
        sample_image,
        attribution,
        str(save_path)
    )
    
    # Check if file was created
    assert save_path.exists()
    assert save_path.stat().st_size > 0

def test_batch_processing(trained_model):
    """Test batch processing of images"""
    interpreter = ImageInterpreter({})
    
    # Create batch of images
    batch_size = 4
    images = torch.randn(batch_size, 1, 64, 64)
    
    # Test batch explanation
    attributions, metadata_list = interpreter.explain_batch(
        trained_model,
        images,
        batch_size=2  # Use smaller batch size to test batching
    )
    
    # Check results
    assert isinstance(attributions, torch.Tensor)
    assert attributions.shape == images.shape
    assert len(metadata_list) == batch_size
    
    # Check metadata for each image
    for metadata in metadata_list:
        assert isinstance(metadata, dict)
        assert 'method' in metadata
        assert 'processing_time' in metadata

def test_model_sensitivity(trained_model, sample_image):
    """Test model sensitivity analysis"""
    interpreter = ImageInterpreter({})
    
    # Test sensitivity analysis
    sensitivity_results = interpreter.analyze_model_sensitivity(
        trained_model,
        sample_image,
        n_scales=3
    )
    
    # Check results structure
    assert isinstance(sensitivity_results, dict)
    assert 'scale_sensitivity' in sensitivity_results
    assert 'rotation_sensitivity' in sensitivity_results
    assert 'noise_sensitivity' in sensitivity_results
    
    # Check values
    for key, value in sensitivity_results.items():
        assert isinstance(value, dict)
        assert 'values' in value
        assert 'mean' in value
        assert 'std' in value

def test_feature_interactions(trained_model, sample_image):
    """Test feature interaction analysis"""
    interpreter = ImageInterpreter({})
    
    # Test interaction analysis
    interaction_results = interpreter.analyze_feature_interactions(
        trained_model,
        sample_image,
        n_perturbations=5
    )
    
    # Check results
    assert isinstance(interaction_results, dict)
    assert len(interaction_results) > 0
    
    # Test visualization
    interpreter.visualize_feature_interactions(
        interaction_results,
        top_k=5
    )

def test_caching(tmp_path, trained_model, sample_image):
    """Test caching functionality"""
    cache_dir = str(tmp_path / "cache")
    
    interpreter = ImageInterpreter({
        'method': 'gradient',
        'caching': {
            'enabled': True,
            'directory': cache_dir,
            'max_size_gb': 1,
            'expiration_days': 1
        }
    })
    
    # First call should compute and cache
    start_time = time.time()
    attribution1, metadata1 = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    first_call_time = time.time() - start_time
    
    # Second call should use cache
    start_time = time.time()
    attribution2, metadata2 = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    second_call_time = time.time() - start_time
    
    # Check results are identical
    assert torch.allclose(attribution1, attribution2)
    assert metadata1 == metadata2
    
    # Second call should be faster
    assert second_call_time < first_call_time
    
    # Check cache file exists
    cache_files = os.listdir(cache_dir)
    assert len(cache_files) > 0
    assert any(file.endswith('.pkl') for file in cache_files)

def test_cache_cleanup(tmp_path, trained_model, sample_image):
    """Test cache cleanup functionality"""
    cache_dir = str(tmp_path / "cache")
    max_size = 1024 * 1024  # 1MB
    
    interpreter = ImageInterpreter({
        'caching': {
            'enabled': True,
            'directory': cache_dir,
            'max_size_gb': max_size / (1024 * 1024 * 1024),  # Convert to GB
            'expiration_days': 1
        }
    })
    
    # Generate multiple results to fill cache
    for _ in range(5):
        noisy_image = sample_image + torch.randn_like(sample_image) * 0.1
        interpreter.explain_prediction(trained_model, noisy_image)
    
    # Check cache size is maintained
    total_size = sum(
        os.path.getsize(os.path.join(cache_dir, f))
        for f in os.listdir(cache_dir)
    )
    assert total_size <= max_size

def test_cache_expiration(tmp_path, trained_model, sample_image):
    """Test cache expiration functionality"""
    cache_dir = str(tmp_path / "cache")
    
    interpreter = ImageInterpreter({
        'caching': {
            'enabled': True,
            'directory': cache_dir,
            'max_size_gb': 1,
            'expiration_days': 0  # Expire immediately
        }
    })
    
    # Generate result
    interpreter.explain_prediction(trained_model, sample_image)
    
    # Force cleanup
    interpreter._cleanup_cache()
    
    # Cache should be empty
    assert len(os.listdir(cache_dir)) == 0

def test_attribution_stability(trained_model, sample_image):
    """Test attribution stability analysis"""
    interpreter = ImageInterpreter({})
    
    # Test stability analysis
    stability_results = interpreter.analyze_attribution_stability(
        trained_model,
        sample_image,
        n_perturbations=3,
        noise_level=0.1
    )
    
    # Check results structure
    assert isinstance(stability_results, dict)
    for method in ['gradient', 'integrated_gradients', 'deep_lift']:
        assert method in stability_results
        assert 'mean_correlation' in stability_results[method]
        assert 'std_correlation' in stability_results[method]
        assert 'mean_rank_correlation' in stability_results[method]

def test_attribution_comparison(tmp_path, trained_model, sample_image):
    """Test attribution comparison visualization"""
    interpreter = ImageInterpreter({})
    
    # Generate attributions using different methods
    attribution_maps = {}
    methods = ['gradient', 'integrated_gradients', 'deep_lift']
    
    for method in methods:
        interpreter.method = method
        attribution, _ = interpreter.explain_prediction(trained_model, sample_image)
        attribution_maps[method] = attribution
    
    # Test comparison visualization
    save_path = tmp_path / "comparison.png"
    interpreter.visualize_attribution_comparison(
        sample_image,
        attribution_maps,
        str(save_path)
    )
    
    # Check if file was created
    assert save_path.exists()
    assert save_path.stat().st_size > 0

def test_feature_importance(trained_model, sample_image):
    """Test feature importance visualization"""
    interpreter = ImageInterpreter({})
    
    # Generate attributions using different methods
    attribution_maps = {}
    methods = ['gradient', 'integrated_gradients']
    
    for method in methods:
        interpreter.method = method
        attribution, _ = interpreter.explain_prediction(trained_model, sample_image)
        attribution_maps[method] = attribution
    
    # Test feature importance visualization
    interpreter.visualize_feature_importance(
        attribution_maps,
        top_k=3
    )

def test_gradcam_plus_plus(trained_model, sample_image):
    """Test Grad-CAM++ attribution"""
    interpreter = ImageInterpreter({'method': 'gradcam_plus_plus'})
    
    # Test attribution
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    
    # Check results
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == sample_image.shape
    assert metadata['method'] == 'gradcam_plus_plus'

def test_smooth_grad(trained_model, sample_image):
    """Test SmoothGrad attribution"""
    interpreter = ImageInterpreter({
        'method': 'smooth_grad',
        'n_samples': 3,
        'noise_level': 0.1
    })
    
    # Test attribution
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    
    # Check results
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == sample_image.shape
    assert metadata['method'] == 'smooth_grad'
    assert metadata['n_samples'] == 3
    assert metadata['noise_level'] == 0.1

def test_sensitivity_visualization(tmp_path, trained_model, sample_image):
    """Test sensitivity analysis visualization"""
    interpreter = ImageInterpreter({})
    
    # Generate sensitivity results
    sensitivity_results = interpreter.analyze_model_sensitivity(
        trained_model,
        sample_image,
        n_scales=3
    )
    
    # Test visualization
    save_path = tmp_path / "sensitivity.png"
    interpreter.visualize_sensitivity_analysis(
        sensitivity_results,
        str(save_path)
    )
    
    # Check if file was created
    assert save_path.exists()
    assert save_path.stat().st_size > 0

def test_invalid_config():
    """Test interpreter initialization with invalid config"""
    with pytest.raises(ValueError):
        ImageInterpreter({
            'method': 'invalid_method',
            'clustering': {
                'n_clusters': -1  # Invalid number of clusters
            }
        })

def test_empty_batch():
    """Test batch processing with empty batch"""
    interpreter = ImageInterpreter({})
    
    with pytest.raises(ValueError):
        interpreter.explain_batch(
            None,
            torch.Tensor([]),
            batch_size=1
        )

def test_invalid_save_path(trained_model, sample_image):
    """Test visualization with invalid save path"""
    interpreter = ImageInterpreter({})
    attribution, _ = interpreter.explain_prediction(trained_model, sample_image)
    
    with pytest.raises(Exception):
        interpreter.visualize_attribution(
            sample_image,
            attribution,
            save_path="/invalid/path/image.png"
        )

def test_zero_image(trained_model):
    """Test interpretation with zero image"""
    interpreter = ImageInterpreter({})
    
    # Create zero image
    zero_image = torch.zeros(1, 1, 64, 64)
    
    # Test attribution
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        zero_image
    )
    
    # Check results
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == zero_image.shape
    assert torch.allclose(attribution, torch.zeros_like(attribution))

def test_random_noise_image(trained_model):
    """Test interpretation with random noise image"""
    interpreter = ImageInterpreter({})
    
    # Create random noise image
    noise_image = torch.randn(1, 1, 64, 64)
    
    # Test attribution
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        noise_image
    )
    
    # Check results
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == noise_image.shape

def test_invalid_image_shape():
    """Test interpretation with invalid image shape"""
    interpreter = ImageInterpreter({})
    
    # Create image with invalid shape
    invalid_image = torch.randn(1, 1, 64)  # Missing one dimension
    
    with pytest.raises(ValueError):
        interpreter.explain_prediction(None, invalid_image)

def test_invalid_batch_size():
    """Test batch processing with invalid batch size"""
    interpreter = ImageInterpreter({})
    
    # Create batch of images
    images = torch.randn(4, 1, 64, 64)
    
    with pytest.raises(ValueError):
        interpreter.explain_batch(None, images, batch_size=0)
    
    with pytest.raises(ValueError):
        interpreter.explain_batch(None, images, batch_size=5)  # Larger than batch

def test_large_image(trained_model):
    """Test interpretation with large image"""
    interpreter = ImageInterpreter({})
    
    # Create large image
    large_image = torch.randn(1, 1, 512, 512)
    
    # Test attribution
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        large_image
    )
    
    # Check results
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == large_image.shape

def test_multiple_channels(trained_model):
    """Test interpretation with multi-channel image"""
    interpreter = ImageInterpreter({})
    
    # Create RGB image
    rgb_image = torch.randn(1, 3, 64, 64)
    
    # Test attribution
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        rgb_image
    )
    
    # Check results
    assert isinstance(attribution, torch.Tensor)
    assert attribution.shape == rgb_image.shape

def test_invalid_target():
    """Test interpretation with invalid target class"""
    interpreter = ImageInterpreter({})
    
    # Create image
    image = torch.randn(1, 1, 64, 64)
    
    with pytest.raises(ValueError):
        interpreter.explain_prediction(
            None,
            image,
            target=torch.tensor([100])  # Invalid class index
        )

def test_model_not_eval(trained_model, sample_image):
    """Test interpretation with model in training mode"""
    interpreter = ImageInterpreter({})
    
    # Set model to training mode
    trained_model.train()
    
    # Test attribution
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        sample_image
    )
    
    # Check if model was set back to eval mode
    assert not trained_model.training

def test_gpu_tensor_handling(trained_model, sample_image):
    """Test handling of GPU tensors if GPU is available"""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    interpreter = ImageInterpreter({})
    
    # Move image to GPU
    gpu_image = sample_image.cuda()
    
    # Test attribution
    attribution, metadata = interpreter.explain_prediction(
        trained_model,
        gpu_image
    )
    
    # Check results
    assert attribution.device == gpu_image.device

def test_batch_device_consistency(trained_model):
    """Test device consistency in batch processing"""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    interpreter = ImageInterpreter({})
    
    # Create batch of images on GPU
    images = torch.randn(4, 1, 64, 64).cuda()
    
    # Test batch processing
    attributions, metadata_list = interpreter.explain_batch(
        trained_model,
        images,
        batch_size=2
    )
    
    # Check device consistency
    assert attributions.device == images.device

def test_visualization_empty_attribution():
    """Test visualization with empty attribution map"""
    interpreter = ImageInterpreter({})
    
    # Create empty attribution map
    empty_attribution = torch.Tensor([])
    
    with pytest.raises(ValueError):
        interpreter.visualize_attribution(
            None,
            empty_attribution
        )

def test_feature_clustering_min_samples():
    """Test feature clustering with minimum number of samples"""
    interpreter = ImageInterpreter({})
    
    # Create minimal batch
    min_batch = torch.randn(2, 1, 64, 64)  # Only 2 samples
    
    with pytest.raises(ValueError):
        interpreter.analyze_feature_clusters(
            None,
            min_batch,
            'conv1'
        ) 