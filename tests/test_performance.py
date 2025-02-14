import pytest
import torch
import torch.distributed as dist
import numpy as np
import time
import psutil
import os
from beacon.models.image_classifier import MedicalImageCNN
from beacon.data.processor import DataProcessor
from beacon.utils.image_interpretability import ImageInterpreter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

@pytest.fixture
def large_dataset():
    """Create a large synthetic dataset for performance testing"""
    n_samples = 1000
    image_size = 128
    
    # Generate synthetic images
    images = torch.randn(n_samples, 1, image_size, image_size)
    labels = torch.randint(0, 2, (n_samples,))
    
    return {
        'images': images,
        'labels': labels
    }

def test_training_speed(large_dataset):
    """Test model training speed"""
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Measure training time
    start_time = time.time()
    history = model.train(large_dataset, epochs=5)
    training_time = time.time() - start_time
    
    # Calculate training speed
    n_samples = len(large_dataset['images'])
    samples_per_second = (n_samples * 5) / training_time
    
    print(f"\nTraining Speed: {samples_per_second:.2f} samples/second")
    assert samples_per_second > 0

def test_inference_speed(large_dataset):
    """Test model inference speed"""
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Train model
    model.train(large_dataset, epochs=1)
    model.eval()
    
    # Warm up
    with torch.no_grad():
        _ = model(large_dataset['images'][:10])
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        _ = model(large_dataset['images'])
    inference_time = time.time() - start_time
    
    # Calculate inference speed
    n_samples = len(large_dataset['images'])
    samples_per_second = n_samples / inference_time
    
    print(f"\nInference Speed: {samples_per_second:.2f} samples/second")
    assert samples_per_second > 0

def test_memory_usage(large_dataset):
    """Test memory usage during training and inference"""
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    initial_memory = get_memory_usage()
    
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    # Measure memory during training
    model.train(large_dataset, epochs=1)
    training_memory = get_memory_usage()
    
    # Measure memory during inference
    model.eval()
    with torch.no_grad():
        _ = model(large_dataset['images'])
    inference_memory = get_memory_usage()
    
    print(f"\nMemory Usage:")
    print(f"Initial: {initial_memory:.2f} MB")
    print(f"Training: {training_memory:.2f} MB")
    print(f"Inference: {inference_memory:.2f} MB")
    
    assert training_memory > initial_memory
    assert inference_memory > initial_memory

def test_gpu_memory_usage(large_dataset):
    """Test GPU memory usage if available"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    def get_gpu_memory():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    # Move data to GPU
    cuda_dataset = {
        'images': large_dataset['images'].cuda(),
        'labels': large_dataset['labels'].cuda()
    }
    
    initial_memory = get_gpu_memory()
    
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    }).cuda()
    
    # Measure GPU memory during training
    model.train(cuda_dataset, epochs=1)
    training_memory = get_gpu_memory()
    
    # Measure GPU memory during inference
    model.eval()
    with torch.no_grad():
        _ = model(cuda_dataset['images'])
    inference_memory = get_gpu_memory()
    
    print(f"\nGPU Memory Usage:")
    print(f"Initial: {initial_memory:.2f} MB")
    print(f"Training: {training_memory:.2f} MB")
    print(f"Inference: {inference_memory:.2f} MB")
    
    assert training_memory > initial_memory
    assert inference_memory > initial_memory

def setup_ddp(rank, world_size, dataset):
    """Setup for distributed data parallel training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Create model and move to GPU
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    }).to(rank)
    
    # Wrap model
    model = DDP(model, device_ids=[rank])
    
    # Create data sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset['images'],
        num_replicas=world_size,
        rank=rank
    )
    
    return model, train_sampler

def cleanup_ddp():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def test_distributed_training(large_dataset):
    """Test distributed training performance"""
    if not torch.cuda.device_count() > 1:
        pytest.skip("Multiple GPUs not available")
    
    world_size = torch.cuda.device_count()
    
    mp.spawn(
        setup_ddp,
        args=(world_size, large_dataset),
        nprocs=world_size,
        join=True
    )

def test_batch_size_scaling(large_dataset):
    """Test performance scaling with different batch sizes"""
    batch_sizes = [16, 32, 64, 128]
    training_times = []
    
    for batch_size in batch_sizes:
        model = MedicalImageCNN({
            'in_channels': 1,
            'num_classes': 2
        })
        
        start_time = time.time()
        model.train(large_dataset, epochs=1, batch_size=batch_size)
        training_times.append(time.time() - start_time)
    
    print("\nBatch Size Scaling:")
    for bs, t in zip(batch_sizes, training_times):
        print(f"Batch Size {bs}: {t:.2f} seconds")
    
    # Training time should generally decrease with larger batch sizes
    assert training_times[0] > training_times[-1]

def test_data_loading_speed(large_dataset):
    """Test data loading and preprocessing speed"""
    processor = DataProcessor({
        'normalization': 'standard',
        'image_augmentation': True
    })
    
    # Measure data loading time
    start_time = time.time()
    processed_images = processor.process_imaging_data(
        large_dataset['images'].numpy()
    )
    loading_time = time.time() - start_time
    
    samples_per_second = len(large_dataset['images']) / loading_time
    print(f"\nData Loading Speed: {samples_per_second:.2f} samples/second")
    assert samples_per_second > 0

def test_interpretation_speed(large_dataset):
    """Test interpretation speed for different methods"""
    model = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    model.train(large_dataset, epochs=1)
    
    methods = ['gradient', 'integrated_gradients', 'deep_lift']
    interpretation_times = {}
    
    for method in methods:
        interpreter = ImageInterpreter({'method': method})
        
        start_time = time.time()
        _ = interpreter.explain_prediction(
            model,
            large_dataset['images'][:10]
        )
        interpretation_times[method] = (time.time() - start_time) / 10
    
    print("\nInterpretation Speed (seconds per sample):")
    for method, time_taken in interpretation_times.items():
        print(f"{method}: {time_taken:.4f}")
    
    assert all(t > 0 for t in interpretation_times.values()) 