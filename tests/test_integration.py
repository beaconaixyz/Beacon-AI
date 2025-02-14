import pytest
import torch
import numpy as np
import os
from beacon.utils.image_interpretability import ImageInterpreter
from beacon.models.image_classifier import MedicalImageCNN
from beacon.data.processor import DataProcessor
from beacon.utils.metrics import Metrics

@pytest.fixture
def test_dataset():
    """Create a synthetic test dataset"""
    n_samples = 50
    image_size = 64
    
    # Create images with specific patterns
    images = []
    labels = []
    
    for i in range(n_samples):
        # Create base image
        image = torch.zeros(1, image_size, image_size)
        
        # Add pattern based on class
        if i % 2 == 0:  # Class 0
            # Add circle pattern
            center = image_size // 4
            radius = image_size // 8
            y, x = torch.meshgrid(
                torch.arange(image_size),
                torch.arange(image_size),
                indexing='ij'
            )
            mask = ((x - center)**2 + (y - center)**2 <= radius**2)
            image[0, mask] = 1.0
            labels.append(0)
        else:  # Class 1
            # Add rectangle pattern
            image[0, image_size//2:3*image_size//4,
                    image_size//2:3*image_size//4] = 1.0
            labels.append(1)
        
        # Add noise
        image += torch.randn_like(image) * 0.1
        images.append(image)
    
    return {
        'images': torch.stack(images),
        'labels': torch.tensor(labels)
    }

def test_end_to_end_pipeline(test_dataset, tmp_path):
    """Test complete pipeline from data processing to interpretation"""
    # 1. Data Processing
    processor = DataProcessor({
        'normalization': 'standard',
        'image_augmentation': True
    })
    
    processed_images = processor.process_imaging_data(
        test_dataset['images'].numpy()
    )
    processed_images = torch.FloatTensor(processed_images)
    
    # 2. Model Training
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Split data
    n_train = int(0.8 * len(processed_images))
    train_data = {
        'images': processed_images[:n_train],
        'labels': test_dataset['labels'][:n_train]
    }
    
    test_data = {
        'images': processed_images[n_train:],
        'labels': test_dataset['labels'][n_train:]
    }
    
    # Train model
    history = model.train(train_data, epochs=5)
    
    # 3. Model Evaluation
    predictions = model.predict(test_data['images'])
    metrics = Metrics.calculate_classification_metrics(
        test_data['labels'].numpy(),
        torch.argmax(predictions, dim=1).numpy(),
        predictions.numpy()
    )
    
    # 4. Model Interpretation
    interpreter = ImageInterpreter({
        'method': 'integrated_gradients',
        'n_steps': 50
    })
    
    # Test different interpretation methods
    methods = ['gradient', 'integrated_gradients', 'deep_lift']
    attribution_maps = {}
    
    for method in methods:
        interpreter.method = method
        attribution, metadata = interpreter.explain_prediction(
            model,
            test_data['images'][0].unsqueeze(0)
        )
        attribution_maps[method] = attribution
    
    # 5. Visualization and Analysis
    # Save visualizations
    save_dir = tmp_path / "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    
    # Attribution comparison
    interpreter.visualize_attribution_comparison(
        test_data['images'][0],
        attribution_maps,
        str(save_dir / "attribution_comparison.png")
    )
    
    # Feature importance
    interpreter.visualize_feature_importance(
        attribution_maps,
        top_k=3,
        save_path=str(save_dir / "feature_importance.png")
    )
    
    # Stability analysis
    stability_results = interpreter.analyze_attribution_stability(
        model,
        test_data['images'][0].unsqueeze(0)
    )
    
    # Assertions
    assert len(history) == 5  # Training completed
    assert all(key in metrics for key in ['accuracy', 'precision', 'recall'])
    assert metrics['accuracy'] > 0.5  # Better than random
    assert len(attribution_maps) == len(methods)
    assert len(stability_results) > 0
    assert os.path.exists(save_dir / "attribution_comparison.png")
    assert os.path.exists(save_dir / "feature_importance.png")

def test_cross_method_consistency(test_dataset):
    """Test consistency between different interpretation methods"""
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Train model
    model.train({
        'images': test_dataset['images'],
        'labels': test_dataset['labels']
    }, epochs=2)
    
    interpreter = ImageInterpreter({})
    methods = ['gradient', 'integrated_gradients', 'deep_lift']
    attributions = {}
    
    # Get attributions from different methods
    for method in methods:
        interpreter.method = method
        attribution, _ = interpreter.explain_prediction(
            model,
            test_dataset['images'][0].unsqueeze(0)
        )
        attributions[method] = attribution
    
    # Compare correlations between methods
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            correlation = torch.corrcoef(
                torch.stack([
                    attributions[method1].flatten(),
                    attributions[method2].flatten()
                ])
            )[0, 1]
            
            print(f"\nCorrelation between {method1} and {method2}: {correlation:.4f}")
            assert not torch.isnan(correlation)
            assert correlation > -1 and correlation < 1

def test_data_processor_integration(test_dataset):
    """Test integration between data processor and interpreter"""
    processor = DataProcessor({
        'normalization': 'standard',
        'image_augmentation': True
    })
    
    interpreter = ImageInterpreter({'method': 'gradient'})
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Process data
    processed_images = processor.process_imaging_data(
        test_dataset['images'].numpy()
    )
    processed_images = torch.FloatTensor(processed_images)
    
    # Train model with processed data
    model.train({
        'images': processed_images,
        'labels': test_dataset['labels']
    }, epochs=2)
    
    # Get interpretation for both raw and processed images
    raw_attribution, _ = interpreter.explain_prediction(
        model,
        test_dataset['images'][0].unsqueeze(0)
    )
    
    processed_attribution, _ = interpreter.explain_prediction(
        model,
        processed_images[0].unsqueeze(0)
    )
    
    # Check shapes and values
    assert raw_attribution.shape == processed_attribution.shape
    assert not torch.allclose(raw_attribution, processed_attribution)

def test_batch_integration(test_dataset):
    """Test integration of batch processing with other components"""
    processor = DataProcessor({
        'normalization': 'standard'
    })
    
    interpreter = ImageInterpreter({
        'method': 'gradient',
        'batch_size': 4
    })
    
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Process and train
    processed_images = processor.process_imaging_data(
        test_dataset['images'].numpy()
    )
    processed_images = torch.FloatTensor(processed_images)
    
    model.train({
        'images': processed_images,
        'labels': test_dataset['labels']
    }, epochs=2)
    
    # Test batch interpretation
    batch_size = 4
    attributions, metadata = interpreter.explain_batch(
        model,
        processed_images[:batch_size]
    )
    
    # Verify results
    assert attributions.shape == processed_images[:batch_size].shape
    assert len(metadata) == batch_size

def test_error_propagation():
    """Test error handling and propagation across components"""
    processor = DataProcessor({
        'normalization': 'invalid_method'  # Invalid configuration
    })
    
    interpreter = ImageInterpreter({
        'method': 'invalid_method'  # Invalid method
    })
    
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Test processor error
    with pytest.raises(ValueError):
        processor.process_imaging_data(np.random.randn(10, 1, 64, 64))
    
    # Test interpreter error
    with pytest.raises(ValueError):
        interpreter.explain_prediction(
            model,
            torch.randn(1, 1, 64, 64)
        )
    
    # Test model error
    with pytest.raises(RuntimeError):
        model.predict(torch.randn(1, 2, 64, 64))  # Wrong number of channels 

def test_model_save_load(test_dataset, tmp_path):
    """Test model saving and loading functionality"""
    # Train original model
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    model.train({
        'images': test_dataset['images'],
        'labels': test_dataset['labels']
    }, epochs=2)
    
    # Save model
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    # Create new model and load weights
    loaded_model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    loaded_model.load_state_dict(torch.load(save_path))
    
    # Compare predictions
    model.eval()
    loaded_model.eval()
    
    with torch.no_grad():
        original_preds = model(test_dataset['images'])
        loaded_preds = loaded_model(test_dataset['images'])
    
    assert torch.allclose(original_preds, loaded_preds)

def test_multi_gpu_training():
    """Test multi-GPU training if available"""
    if not torch.cuda.device_count() > 1:
        pytest.skip("Multiple GPUs not available")
    
    # Create larger dataset
    n_samples = 200
    image_size = 128
    images = torch.randn(n_samples, 1, image_size, image_size)
    labels = torch.randint(0, 2, (n_samples,))
    
    # Create model
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Wrap model with DataParallel
    model = torch.nn.DataParallel(model)
    
    # Train model
    history = model.module.train({
        'images': images.cuda(),
        'labels': labels.cuda()
    }, epochs=2)
    
    assert len(history) == 2
    assert all(isinstance(loss, float) for loss in history)

def test_distributed_processing(test_dataset):
    """Test distributed data processing"""
    processor = DataProcessor({
        'normalization': 'standard',
        'image_augmentation': True,
        'num_workers': 4
    })
    
    # Process data in parallel
    processed_images = processor.process_imaging_data(
        test_dataset['images'].numpy(),
        parallel=True
    )
    
    assert isinstance(processed_images, np.ndarray)
    assert processed_images.shape == test_dataset['images'].shape

def test_model_ensemble(test_dataset):
    """Test ensemble of multiple models"""
    n_models = 3
    models = []
    
    # Train multiple models
    for _ in range(n_models):
        model = MedicalImageCNN({
            'in_channels': 1,
            'num_classes': 2
        })
        
        model.train({
            'images': test_dataset['images'],
            'labels': test_dataset['labels']
        }, epochs=2)
        
        models.append(model)
    
    # Get ensemble predictions
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = torch.softmax(model(test_dataset['images']), dim=1)
            predictions.append(pred)
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    
    # Evaluate ensemble
    metrics = Metrics.calculate_classification_metrics(
        test_dataset['labels'].numpy(),
        torch.argmax(ensemble_pred, dim=1).numpy(),
        ensemble_pred.numpy()
    )
    
    assert metrics['accuracy'] > 0.5

def test_interpretation_stability(test_dataset):
    """Test stability of interpretations across different model initializations"""
    n_models = 3
    interpreter = ImageInterpreter({'method': 'integrated_gradients'})
    attributions = []
    
    # Get attributions from multiple models
    for _ in range(n_models):
        model = MedicalImageCNN({
            'in_channels': 1,
            'num_classes': 2
        })
        
        model.train({
            'images': test_dataset['images'],
            'labels': test_dataset['labels']
        }, epochs=2)
        
        attribution, _ = interpreter.explain_prediction(
            model,
            test_dataset['images'][0].unsqueeze(0)
        )
        attributions.append(attribution)
    
    # Calculate correlation between attributions
    attributions = torch.stack(attributions)
    correlations = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            corr = torch.corrcoef(torch.stack([
                attributions[i].flatten(),
                attributions[j].flatten()
            ]))[0, 1]
            correlations.append(corr)
    
    # Check correlation strength
    mean_correlation = torch.tensor(correlations).mean()
    assert mean_correlation > 0.5  # Strong positive correlation expected

def test_adversarial_robustness(test_dataset):
    """Test model and interpretation robustness against adversarial examples"""
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Train model
    model.train({
        'images': test_dataset['images'],
        'labels': test_dataset['labels']
    }, epochs=2)
    
    interpreter = ImageInterpreter({'method': 'integrated_gradients'})
    
    # Generate adversarial example using FGSM
    epsilon = 0.1
    image = test_dataset['images'][0].unsqueeze(0)
    image.requires_grad = True
    
    output = model(image)
    loss = torch.nn.functional.cross_entropy(output, test_dataset['labels'][0:1])
    loss.backward()
    
    perturbed_image = image + epsilon * image.grad.sign()
    
    # Get interpretations for both original and perturbed images
    orig_attribution, _ = interpreter.explain_prediction(model, image)
    pert_attribution, _ = interpreter.explain_prediction(model, perturbed_image)
    
    # Calculate attribution similarity
    similarity = torch.cosine_similarity(
        orig_attribution.flatten(),
        pert_attribution.flatten(),
        dim=0
    )
    
    assert similarity > 0  # Some positive correlation should remain 