#!/usr/bin/env python3

"""
Evaluation Script for BEACON

This script implements model evaluation functionality with support for different metrics
and data types.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score
)
from scripts.training.train import BeaconDataset, BaseModel

class ModelEvaluator:
    """Handle model evaluation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = Path(config.get('data_dir', 'processed_data'))
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.output_dir = Path(config.get('output_dir', 'evaluation'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self, data_type: str) -> nn.Module:
        """Load trained model.

        Args:
            data_type: Type of data the model was trained on

        Returns:
            Loaded model
        """
        # Create dataset to get input/output dimensions
        dataset = BeaconDataset(self.data_dir, data_type)
        sample_features, sample_labels = dataset[0]
        input_dim = sample_features.shape[0]
        output_dim = sample_labels.shape[0]

        # Create and load model
        model = BaseModel(input_dim, output_dim).to(self.device)
        model_path = self.model_dir / f'{data_type}_model.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found at {model_path}")
            
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model

    def evaluate_classification(self, predictions: np.ndarray, targets: np.ndarray,
                              threshold: float = 0.5) -> Dict[str, float]:
        """Calculate classification metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            threshold: Classification threshold for binary predictions

        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to binary predictions
        binary_preds = (predictions >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(targets, binary_preds),
            'precision': precision_score(targets, binary_preds, average='weighted'),
            'recall': recall_score(targets, binary_preds, average='weighted'),
            'f1': f1_score(targets, binary_preds, average='weighted'),
            'auc_roc': roc_auc_score(targets, predictions, average='weighted'),
            'auc_pr': average_precision_score(targets, predictions, average='weighted')
        }

        return metrics

    def evaluate_regression(self, predictions: np.ndarray,
                          targets: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth values

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions)
        }

        return metrics

    def evaluate(self, data_type: str) -> Dict[str, float]:
        """Evaluate model on specified data type.

        Args:
            data_type: Type of data to evaluate on

        Returns:
            Dictionary of evaluation metrics
        """
        # Load model and data
        model = self.load_model(data_type)
        dataset = BeaconDataset(self.data_dir, data_type)
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                predictions = model(features)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate batches
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)

        # Calculate metrics based on data type
        if data_type in ['clinical', 'imaging', 'genomic']:
            metrics = self.evaluate_classification(predictions, targets)
        else:  # survival
            metrics = self.evaluate_regression(predictions, targets)

        # Log results
        self.logger.info(f"\nEvaluation results for {data_type} model:")
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")

        # Save results
        results_file = self.output_dir / f'{data_type}_evaluation.csv'
        pd.DataFrame([metrics]).to_csv(results_file, index=False)
        self.logger.info(f"\nResults saved to {results_file}")

        return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate BEACON models")
    parser.add_argument("--data-dir", default="processed_data",
                      help="Directory containing processed data")
    parser.add_argument("--model-dir", default="models",
                      help="Directory containing trained models")
    parser.add_argument("--output-dir", default="evaluation",
                      help="Output directory for evaluation results")
    parser.add_argument("--data-type", required=True,
                      choices=['clinical', 'imaging', 'genomic', 'survival'],
                      help="Type of data to evaluate")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for evaluation")
    args = parser.parse_args()

    config = vars(args)
    evaluator = ModelEvaluator(config)
    evaluator.evaluate(args.data_type)

if __name__ == "__main__":
    main() 