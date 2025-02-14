#!/usr/bin/env python3

"""
Inference Script for BEACON

This script implements model inference functionality for making predictions
on new data.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Union, List
import logging
from scripts.training.train import BaseModel

class ModelPredictor:
    """Handle model inference"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize predictor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.output_dir = Path(config.get('output_dir', 'predictions'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self, data_type: str, input_dim: int, output_dim: int) -> nn.Module:
        """Load trained model.

        Args:
            data_type: Type of data the model was trained on
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            Loaded model
        """
        model = BaseModel(input_dim, output_dim).to(self.device)
        model_path = self.model_dir / f'{data_type}_model.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model found at {model_path}")
            
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model

    def preprocess_clinical_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Preprocess clinical data.

        Args:
            data: Raw clinical data

        Returns:
            Preprocessed data tensor
        """
        # Ensure all required columns are present
        required_cols = [
            'age', 'weight', 'height', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'heart_rate', 'temperature',
            'glucose', 'cholesterol', 'smoking'
        ]
        
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert categorical variables
        smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
        data['smoking'] = data['smoking'].map(smoking_map)

        # Convert to tensor
        return torch.FloatTensor(data[required_cols].values)

    def preprocess_imaging_data(self, data: np.ndarray) -> torch.Tensor:
        """Preprocess imaging data.

        Args:
            data: Raw imaging data

        Returns:
            Preprocessed data tensor
        """
        # Normalize to [0, 1] range
        data = (data - data.min()) / (data.max() - data.min())

        # Add channel dimension if needed
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=1)

        return torch.FloatTensor(data)

    def preprocess_genomic_data(self, data: Dict[str, np.ndarray]) -> torch.Tensor:
        """Preprocess genomic data.

        Args:
            data: Dictionary containing different types of genomic data

        Returns:
            Preprocessed data tensor
        """
        # Ensure all required keys are present
        required_keys = ['expression', 'mutations', 'cnv']
        missing_keys = set(required_keys) - set(data.keys())
        if missing_keys:
            raise ValueError(f"Missing required genomic data types: {missing_keys}")

        # Concatenate different types of data
        processed_data = np.concatenate([
            data['expression'],
            data['mutations'],
            data['cnv']
        ], axis=1)

        return torch.FloatTensor(processed_data)

    def preprocess_survival_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Preprocess survival data.

        Args:
            data: Raw survival data

        Returns:
            Preprocessed data tensor
        """
        # Ensure all required columns are present
        required_cols = ['age', 'stage', 'grade']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert categorical variables
        stage_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
        grade_map = {'Low': 0, 'Medium': 1, 'High': 2}
        
        data['stage'] = data['stage'].map(stage_map)
        data['grade'] = data['grade'].map(grade_map)

        return torch.FloatTensor(data[required_cols].values)

    def predict(self, data_type: str, input_data: Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]) -> np.ndarray:
        """Make predictions on new data.

        Args:
            data_type: Type of data to make predictions for
            input_data: Input data for predictions

        Returns:
            Model predictions
        """
        # Preprocess data based on type
        if data_type == 'clinical':
            processed_data = self.preprocess_clinical_data(input_data)
            output_dim = 2  # diabetes and hypertension
        elif data_type == 'imaging':
            processed_data = self.preprocess_imaging_data(input_data)
            output_dim = 1  # binary classification
        elif data_type == 'genomic':
            processed_data = self.preprocess_genomic_data(input_data)
            output_dim = 1  # binary classification
        elif data_type == 'survival':
            processed_data = self.preprocess_survival_data(input_data)
            output_dim = 2  # time and event
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Load model
        input_dim = processed_data.shape[1]
        model = self.load_model(data_type, input_dim, output_dim)

        # Make predictions
        with torch.no_grad():
            predictions = model(processed_data.to(self.device))
            predictions = predictions.cpu().numpy()

        return predictions

    def save_predictions(self, predictions: np.ndarray, data_type: str) -> None:
        """Save model predictions.

        Args:
            predictions: Model predictions
            data_type: Type of data
        """
        # Create appropriate column names based on data type
        if data_type == 'clinical':
            columns = ['diabetes_prob', 'hypertension_prob']
        elif data_type == 'imaging':
            columns = ['abnormality_prob']
        elif data_type == 'genomic':
            columns = ['expression_high_prob']
        elif data_type == 'survival':
            columns = ['predicted_time', 'event_prob']
        else:
            columns = [f'prediction_{i}' for i in range(predictions.shape[1])]

        # Save predictions
        predictions_df = pd.DataFrame(predictions, columns=columns)
        output_file = self.output_dir / f'{data_type}_predictions.csv'
        predictions_df.to_csv(output_file, index=False)
        self.logger.info(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Make predictions using BEACON models")
    parser.add_argument("--model-dir", default="models",
                      help="Directory containing trained models")
    parser.add_argument("--output-dir", default="predictions",
                      help="Output directory for predictions")
    parser.add_argument("--data-type", required=True,
                      choices=['clinical', 'imaging', 'genomic', 'survival'],
                      help="Type of data to make predictions for")
    parser.add_argument("--input-file", required=True,
                      help="Path to input data file")
    args = parser.parse_args()

    # Load input data based on type
    if args.data_type == 'clinical' or args.data_type == 'survival':
        input_data = pd.read_csv(args.input_file)
    elif args.data_type == 'imaging':
        input_data = np.load(args.input_file)
    elif args.data_type == 'genomic':
        input_data = dict(np.load(args.input_file))
    else:
        raise ValueError(f"Unsupported data type: {args.data_type}")

    config = vars(args)
    predictor = ModelPredictor(config)
    predictions = predictor.predict(args.data_type, input_data)
    predictor.save_predictions(predictions, args.data_type)

if __name__ == "__main__":
    main() 