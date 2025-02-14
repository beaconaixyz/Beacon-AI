#!/usr/bin/env python3

"""
Model Visualization Script for BEACON

This script handles model visualization, feature importance analysis, and interpretability plots.
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from captum.attr import (
    IntegratedGradients,
    GuidedGradCam,
    DeepLift,
    GradientShap,
    Occlusion
)

from beacon.models import BeaconModel
from beacon.data import BeaconDataset
from beacon.utils.logger import setup_logger
from beacon.utils.visualization import (
    plot_feature_importance,
    plot_attention_weights,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)

class ModelVisualizer:
    """Handles model visualization and interpretability analysis"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the model visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.model_path = Path(config['model_path'])
        self.data_dir = Path(config['data_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            name="visualizer",
            log_file=self.output_dir / "visualization.log"
        )
        
        # Load model
        self.logger.info(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = BeaconModel(checkpoint['config']['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set style for static plots
        plt.style.use('seaborn')
        
        # Create directories for different visualization types
        self.plots_dir = self.output_dir / 'plots'
        self.feature_dir = self.plots_dir / 'feature_analysis'
        self.attention_dir = self.plots_dir / 'attention_maps'
        self.attribution_dir = self.plots_dir / 'attributions'
        self.embedding_dir = self.plots_dir / 'embeddings'
        
        for directory in [self.plots_dir, self.feature_dir, self.attention_dir,
                         self.attribution_dir, self.embedding_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> torch.utils.data.DataLoader:
        """Load data for visualization.

        Returns:
            DataLoader containing the data
        """
        dataset = BeaconDataset(
            data_dir=self.data_dir,
            transform=self.config['data'].get('transform', None)
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['visualization']['batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 4),
            pin_memory=True
        )
        
        return dataloader

    def visualize_feature_importance(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Generate feature importance visualizations.

        Args:
            dataloader: DataLoader containing the data
        """
        self.logger.info("Generating feature importance visualizations...")
        
        # Collect features and importance scores
        features = []
        importance_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                
                if hasattr(self.model, 'get_features'):
                    batch_features = self.model.get_features(inputs)
                    features.append(batch_features.cpu().numpy())
                
                if hasattr(self.model, 'get_feature_importance'):
                    importance = self.model.get_feature_importance(inputs)
                    importance_scores.append(importance.cpu().numpy())
        
        if features:
            features = np.concatenate(features)
            
            # Generate feature correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                np.corrcoef(features.T),
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f'
            )
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(self.feature_dir / 'feature_correlation.png')
            plt.close()
        
        if importance_scores:
            importance_scores = np.concatenate(importance_scores)
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            plot_feature_importance(
                importance_scores,
                feature_names=self.config['data'].get('feature_names', None)
            )
            plt.title('Feature Importance Scores')
            plt.tight_layout()
            plt.savefig(self.feature_dir / 'feature_importance.png')
            plt.close()

    def visualize_attention(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 5
    ) -> None:
        """Generate attention visualization plots.

        Args:
            dataloader: DataLoader containing the data
            num_samples: Number of samples to visualize
        """
        if not hasattr(self.model, 'get_attention_weights'):
            self.logger.info("Model does not support attention visualization")
            return
        
        self.logger.info("Generating attention visualizations...")
        
        samples_processed = 0
        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= num_samples:
                    break
                
                inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                attention_weights = self.model.get_attention_weights(inputs)
                
                for i in range(min(len(attention_weights), num_samples - samples_processed)):
                    plt.figure(figsize=(10, 8))
                    plot_attention_weights(
                        attention_weights[i].cpu().numpy(),
                        token_labels=self.config['data'].get('token_labels', None)
                    )
                    plt.title(f'Attention Weights - Sample {samples_processed + i + 1}')
                    plt.tight_layout()
                    plt.savefig(self.attention_dir / f'attention_sample_{samples_processed + i + 1}.png')
                    plt.close()
                
                samples_processed += len(attention_weights)

    def compute_attributions(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 5
    ) -> None:
        """Generate attribution visualizations using various methods.

        Args:
            dataloader: DataLoader containing the data
            num_samples: Number of samples to visualize
        """
        self.logger.info("Computing and visualizing attributions...")
        
        # Initialize attribution methods
        ig = IntegratedGradients(self.model)
        guided_gradcam = GuidedGradCam(self.model, self.model.features)
        deeplift = DeepLift(self.model)
        gradient_shap = GradientShap(self.model)
        occlusion = Occlusion(self.model)
        
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
            
            # Compute attributions using different methods
            attribution_methods = {
                'integrated_gradients': ig,
                'guided_gradcam': guided_gradcam,
                'deeplift': deeplift,
                'gradient_shap': gradient_shap,
                'occlusion': occlusion
            }
            
            for method_name, method in attribution_methods.items():
                attributions = method.attribute(inputs)
                
                for i in range(min(len(attributions), num_samples - samples_processed)):
                    plt.figure(figsize=(10, 6))
                    plt.imshow(attributions[i].cpu().numpy(), cmap='seismic', center=0)
                    plt.colorbar(label='Attribution Score')
                    plt.title(f'{method_name.replace("_", " ").title()} - Sample {samples_processed + i + 1}')
                    plt.tight_layout()
                    plt.savefig(
                        self.attribution_dir / 
                        f'{method_name}_sample_{samples_processed + i + 1}.png'
                    )
                    plt.close()
            
            samples_processed += len(attributions)

    def visualize_embeddings(
        self,
        dataloader: torch.utils.data.DataLoader,
        method: str = 'tsne'
    ) -> None:
        """Generate embedding visualizations using dimensionality reduction.

        Args:
            dataloader: DataLoader containing the data
            method: Dimensionality reduction method ('tsne' or 'pca')
        """
        if not hasattr(self.model, 'get_embeddings'):
            self.logger.info("Model does not support embedding visualization")
            return
        
        self.logger.info(f"Generating {method.upper()} embedding visualizations...")
        
        # Collect embeddings and labels
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                batch_embeddings = self.model.get_embeddings(inputs)
                embeddings.append(batch_embeddings.cpu().numpy())
                labels.append(batch['targets'].numpy())
        
        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # pca
            reducer = PCA(n_components=2)
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Create interactive plot using plotly
        fig = px.scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            color=labels,
            title=f'{method.upper()} Visualization of Model Embeddings',
            labels={'color': 'Class'},
            template='plotly_white'
        )
        
        fig.write_html(self.embedding_dir / f'{method}_embeddings.html')
        
        # Also save a static version
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=labels,
            cmap='viridis'
        )
        plt.colorbar(scatter, label='Class')
        plt.title(f'{method.upper()} Visualization of Model Embeddings')
        plt.tight_layout()
        plt.savefig(self.embedding_dir / f'{method}_embeddings.png')
        plt.close()

    def visualize_predictions(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> None:
        """Generate prediction-related visualizations.

        Args:
            dataloader: DataLoader containing the data
        """
        self.logger.info("Generating prediction visualizations...")
        
        # Collect predictions and targets
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                targets.append(batch['targets'].numpy())
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plot_confusion_matrix(
            targets,
            (predictions > 0.5).astype(int),
            normalize=True
        )
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png')
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plot_roc_curve(targets, predictions)
        plt.title('ROC Curve')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'roc_curve.png')
        plt.close()
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plot_precision_recall_curve(targets, predictions)
        plt.title('Precision-Recall Curve')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'precision_recall_curve.png')
        plt.close()
        
        # Create prediction distribution plot
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(['Negative', 'Positive']):
            mask = targets == i
            plt.hist(
                predictions[mask],
                bins=50,
                alpha=0.5,
                label=label,
                density=True
            )
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.title('Prediction Score Distribution by Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'prediction_distribution.png')
        plt.close()

    def generate_visualization_report(self) -> None:
        """Generate an HTML report combining all visualizations."""
        self.logger.info("Generating visualization report...")
        
        report_path = self.output_dir / 'visualization_report.html'
        
        # Create HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>BEACON Model Visualization Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2 { color: #333; }",
            ".section { margin-bottom: 40px; }",
            ".plot-container { margin: 20px 0; }",
            "img { max-width: 100%; border: 1px solid #ddd; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>BEACON Model Visualization Report</h1>"
        ]
        
        # Add sections for each visualization type
        sections = {
            'Feature Analysis': self.feature_dir,
            'Attention Maps': self.attention_dir,
            'Attribution Analysis': self.attribution_dir,
            'Embedding Visualizations': self.embedding_dir,
            'Prediction Analysis': self.plots_dir
        }
        
        for section_name, directory in sections.items():
            html_content.extend([
                f"<div class='section'>",
                f"<h2>{section_name}</h2>"
            ])
            
            # Add plots from the directory
            if directory.exists():
                for plot_file in directory.glob('*'):
                    if plot_file.suffix in ['.png', '.html']:
                        if plot_file.suffix == '.html':
                            # For interactive plots, embed them
                            with open(plot_file) as f:
                                html_content.extend([
                                    "<div class='plot-container'>",
                                    f.read(),
                                    "</div>"
                                ])
                        else:
                            # For static plots, add as images
                            html_content.extend([
                                "<div class='plot-container'>",
                                f"<img src='{plot_file.relative_to(self.output_dir)}' />",
                                "</div>"
                            ])
            
            html_content.append("</div>")
        
        html_content.extend([
            "</body>",
            "</html>"
        ])
        
        # Write the report
        with open(report_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        self.logger.info(f"Visualization report generated at {report_path}")

    def run_visualization(self) -> None:
        """Run the complete visualization pipeline."""
        self.logger.info("Starting visualization pipeline...")
        
        # Load data
        dataloader = self.load_data()
        
        # Generate different types of visualizations
        self.visualize_feature_importance(dataloader)
        self.visualize_attention(dataloader)
        self.compute_attributions(dataloader)
        self.visualize_embeddings(dataloader, method='tsne')
        self.visualize_embeddings(dataloader, method='pca')
        self.visualize_predictions(dataloader)
        
        # Generate comprehensive report
        self.generate_visualization_report()
        
        self.logger.info("Visualization pipeline completed!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate BEACON model visualizations")
    parser.add_argument("--config", required=True, help="Path to visualization config file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Run visualization
    visualizer = ModelVisualizer(config)
    visualizer.run_visualization()

if __name__ == "__main__":
    main() 