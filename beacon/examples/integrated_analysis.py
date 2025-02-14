import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

from beacon.models.cancer_classifier import CancerClassifier
from beacon.models.image_classifier import MedicalImageCNN
from beacon.models.genomic_model import GenomicModel
from beacon.models.survival_model import SurvivalModel
from beacon.data.processor import DataProcessor
from beacon.data.image_processor import MedicalImageProcessor
from beacon.utils.metrics import Metrics

def generate_synthetic_patient_data(n_samples: int = 100) -> Dict[str, Any]:
    """
    Generate synthetic patient data including clinical, imaging, and genomic features
    Args:
        n_samples: Number of patients
    Returns:
        Dictionary containing different types of patient data
    """
    # Generate clinical data
    clinical_data = pd.DataFrame({
        'age': np.random.normal(60, 10, n_samples),
        'tumor_size': np.random.normal(3, 1, n_samples),
        'lymph_nodes': np.random.randint(0, 10, n_samples),
        'marker_level': np.random.normal(100, 20, n_samples)
    })
    
    # Generate imaging data (simplified 64x64 images)
    images = np.zeros((n_samples, 1, 64, 64))
    for i in range(n_samples):
        # Create synthetic tumor-like patterns
        center_x = np.random.randint(20, 44)
        center_y = np.random.randint(20, 44)
        radius = np.random.randint(5, 15)
        intensity = np.random.uniform(0.5, 1.0)
        
        x, y = np.ogrid[:64, :64]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        images[i, 0, mask] = intensity
    
    # Add noise to images
    images += np.random.normal(0, 0.1, images.shape)
    images = np.clip(images, 0, 1)
    
    # Generate genomic data (simplified gene expression values)
    n_genes = 1000
    genomic_data = np.random.normal(0, 1, (n_samples, n_genes))
    
    # Generate survival data
    survival_times = np.random.exponential(24, n_samples)  # months
    event_indicators = np.random.binomial(1, 0.7, n_samples)
    
    # Generate labels (cancer subtypes)
    labels = np.random.randint(0, 2, n_samples)
    
    return {
        'clinical_data': clinical_data,
        'images': torch.FloatTensor(images),
        'genomic_data': torch.FloatTensor(genomic_data),
        'survival_times': torch.FloatTensor(survival_times),
        'event_indicators': torch.FloatTensor(event_indicators),
        'labels': torch.LongTensor(labels)
    }

def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process different types of patient data
    Args:
        data: Raw patient data
    Returns:
        Processed data
    """
    # Process clinical data
    clinical_processor = DataProcessor({
        'scaling_method': 'standard',
        'handle_missing': True
    })
    processed_clinical = clinical_processor.fit_transform(
        data['clinical_data'].values
    )
    
    # Process images
    image_processor = MedicalImageProcessor({
        'normalize': True,
        'denoise': True,
        'enhance_contrast': True
    })
    processed_images = torch.stack([
        torch.FloatTensor(image_processor.preprocess(img.squeeze().numpy()))
        for img in data['images']
    ])
    
    # Process genomic data
    genomic_processor = DataProcessor({
        'scaling_method': 'standard',
        'handle_missing': True,
        'remove_outliers': True
    })
    processed_genomic = genomic_processor.fit_transform(
        data['genomic_data'].numpy()
    )
    
    return {
        'clinical_features': torch.FloatTensor(processed_clinical),
        'images': processed_images.unsqueeze(1),  # Add channel dimension
        'genomic_features': torch.FloatTensor(processed_genomic),
        'survival_times': data['survival_times'],
        'event_indicators': data['event_indicators'],
        'labels': data['labels']
    }

def train_models(data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Train different models on processed data
    Args:
        data: Processed patient data
    Returns:
        Dictionary of trained models
    """
    # Initialize models
    cancer_classifier = CancerClassifier({
        'input_dim': data['clinical_features'].shape[1],
        'hidden_dim': 64,
        'output_dim': 2
    })
    
    image_classifier = MedicalImageCNN({
        'in_channels': 1,
        'num_classes': 2
    })
    
    genomic_model = GenomicModel({
        'input_dim': data['genomic_features'].shape[1],
        'hidden_dims': [512, 256, 128],
        'output_dim': 2,
        'task_type': 'classification'
    })
    
    survival_model = SurvivalModel({
        'input_dim': data['clinical_features'].shape[1],
        'hidden_dims': [64, 32]
    })
    
    # Train cancer classifier
    cancer_classifier.train({
        'features': data['clinical_features'],
        'labels': data['labels']
    })
    
    # Train image classifier
    image_classifier.train({
        'images': data['images'],
        'labels': data['labels']
    })
    
    # Train genomic model
    genomic_model.train({
        'features': data['genomic_features'],
        'labels': data['labels']
    })
    
    # Train survival model
    survival_model.train({
        'features': data['clinical_features'],
        'survival_time': data['survival_times'],
        'event_indicator': data['event_indicators']
    })
    
    return {
        'cancer_classifier': cancer_classifier,
        'image_classifier': image_classifier,
        'genomic_model': genomic_model,
        'survival_model': survival_model
    }

def evaluate_models(models: Dict[str, Any], data: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all models
    Args:
        models: Dictionary of trained models
        data: Test data
    Returns:
        Dictionary of evaluation metrics for each model
    """
    results = {}
    
    # Evaluate cancer classifier
    cancer_preds = models['cancer_classifier'].predict(data['clinical_features'])
    results['cancer_classifier'] = Metrics.calculate_classification_metrics(
        data['labels'].numpy(),
        torch.argmax(cancer_preds, dim=1).numpy(),
        cancer_preds.numpy()
    )
    
    # Evaluate image classifier
    image_preds = models['image_classifier'].predict(data['images'])
    results['image_classifier'] = Metrics.calculate_classification_metrics(
        data['labels'].numpy(),
        torch.argmax(image_preds, dim=1).numpy(),
        image_preds.numpy()
    )
    
    # Evaluate genomic model
    genomic_preds = models['genomic_model'].predict(data['genomic_features'])
    results['genomic_model'] = Metrics.calculate_classification_metrics(
        data['labels'].numpy(),
        torch.argmax(genomic_preds, dim=1).numpy(),
        genomic_preds.numpy()
    )
    
    # Calculate risk scores from survival model
    risk_scores = models['survival_model'].predict_risk(data['clinical_features'])
    
    return results

def visualize_results(results: Dict[str, Dict[str, float]]):
    """
    Visualize model performance
    Args:
        results: Dictionary of evaluation metrics
    """
    # Plot performance comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, model in enumerate(models):
        performance = [results[model][metric] for metric in metrics]
        ax.bar(x + i * width, performance, width, label=model)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function demonstrating integrated analysis"""
    print("Generating synthetic patient data...")
    data = generate_synthetic_patient_data(n_samples=500)
    
    print("\nProcessing data...")
    processed_data = process_data(data)
    
    print("\nTraining models...")
    models = train_models(processed_data)
    
    print("\nEvaluating models...")
    results = evaluate_models(models, processed_data)
    
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    print("\nVisualizing results...")
    visualize_results(results)
    
    print("\nSaving models...")
    for name, model in models.items():
        model.save(f"{name}.pt")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 