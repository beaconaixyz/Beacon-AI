import torch
import numpy as np
from beacon.models.image_classifier import MedicalImageCNN
from beacon.data.processor import DataProcessor
from beacon.utils.metrics import Metrics
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path: str, processor: DataProcessor) -> torch.Tensor:
    """
    Load and preprocess a medical image
    Args:
        image_path: Path to the image file
        processor: Data processor instance
    Returns:
        Preprocessed image tensor
    """
    # This is a placeholder for actual image loading
    # In real application, you would use libraries like PIL or OpenCV
    image = np.random.randn(64, 64)  # Simulated image
    
    # Process image
    processed_image = processor.process_imaging_data(image)
    return torch.FloatTensor(processed_image)

def visualize_results(image: torch.Tensor, prediction: torch.Tensor, 
                     class_names: list = ['Benign', 'Malignant']):
    """
    Visualize image and prediction
    Args:
        image: Input image tensor
        prediction: Model prediction
        class_names: List of class names
    """
    plt.figure(figsize=(10, 5))
    
    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    # Plot prediction probabilities
    plt.subplot(1, 2, 2)
    probs = prediction.squeeze().numpy()
    plt.bar(class_names, probs)
    plt.title('Prediction Probabilities')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def main():
    """Example of using the medical image CNN"""
    
    # Model configuration
    model_config = {
        'in_channels': 1,
        'num_classes': 2,
        'learning_rate': 0.001
    }
    
    # Data processor configuration
    data_config = {
        'normalization': 'standard',
        'image_augmentation': True
    }
    
    # Initialize model and processor
    model = MedicalImageCNN(model_config)
    processor = DataProcessor(data_config)
    
    # Generate sample training data
    n_samples = 100
    image_size = 64
    train_images = torch.randn(n_samples, 1, image_size, image_size)
    train_labels = torch.randint(0, 2, (n_samples,))
    
    # Prepare training data
    train_data = {
        'images': train_images,
        'labels': train_labels
    }
    
    # Train model
    print("Training model...")
    history = model.train(train_data, epochs=10, batch_size=32)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot([h['epoch'] for h in history], [h['loss'] for h in history])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # Simulate new image prediction
    print("\nProcessing new image...")
    new_image = load_and_preprocess_image("dummy_path", processor)
    new_image = new_image.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    print("Making prediction...")
    prediction = model.predict(new_image)
    
    # Visualize results
    visualize_results(new_image[0], prediction[0])
    
    # Print prediction probabilities
    print("\nPrediction probabilities:")
    print(f"Benign: {prediction[0][0]:.4f}")
    print(f"Malignant: {prediction[0][1]:.4f}")

if __name__ == "__main__":
    main() 