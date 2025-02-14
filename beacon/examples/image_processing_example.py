import numpy as np
import matplotlib.pyplot as plt
from beacon.data.image_processor import MedicalImageProcessor
import cv2

def create_synthetic_medical_image(size: int = 256) -> np.ndarray:
    """
    Create synthetic medical image for demonstration
    Args:
        size: Image size
    Returns:
        Synthetic medical image
    """
    # Create base image
    image = np.zeros((size, size), dtype=np.float32)
    
    # Add tissue-like structures
    for _ in range(5):
        center = (
            np.random.randint(size//4, 3*size//4),
            np.random.randint(size//4, 3*size//4)
        )
        radius = np.random.randint(size//8, size//4)
        intensity = np.random.uniform(0.3, 1.0)
        cv2.circle(image, center, radius, intensity, -1)
    
    # Add some texture
    texture = np.random.normal(0, 0.1, image.shape)
    image = np.clip(image + texture, 0, 1)
    
    # Add some noise
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    return image

def plot_image_grid(images: list, titles: list, figsize: tuple = (15, 5)):
    """
    Plot grid of images
    Args:
        images: List of images to plot
        titles: List of titles for each image
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(features: dict):
    """
    Plot feature importance
    Args:
        features: Dictionary of features
    """
    plt.figure(figsize=(10, 5))
    plt.bar(features.keys(), features.values())
    plt.xticks(rotation=45)
    plt.title('Image Features')
    plt.tight_layout()
    plt.show()

def main():
    """Example of using the medical image processor"""
    
    # Create synthetic medical image
    print("Creating synthetic medical image...")
    original_image = create_synthetic_medical_image()
    
    # Configure image processor
    processor_config = {
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
            'rotation_limit': 15,
            'noise_prob': 0.3
        }
    }
    
    # Initialize processor
    print("\nInitializing image processor...")
    processor = MedicalImageProcessor(processor_config)
    
    # Preprocess image
    print("Preprocessing image...")
    processed_image = processor.preprocess(original_image)
    
    # Apply data augmentation
    print("Applying data augmentation...")
    augmented_image = processor.augment(processed_image)
    
    # Segment tissue
    print("Segmenting tissue...")
    segmented_image, mask = processor.segment_tissue(processed_image)
    
    # Extract features
    print("Extracting image features...")
    features = processor.extract_features(processed_image)
    
    # Visualize results
    print("\nVisualizing results...")
    
    # Plot original and processed images
    plot_image_grid(
        [original_image, processed_image, augmented_image, segmented_image],
        ['Original', 'Processed', 'Augmented', 'Segmented']
    )
    
    # Plot segmentation mask
    plt.figure(figsize=(5, 5))
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    plt.show()
    
    # Plot extracted features
    plot_feature_importance(features)
    
    # Print feature values
    print("\nExtracted Features:")
    for name, value in features.items():
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main() 