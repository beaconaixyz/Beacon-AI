#!/usr/bin/env python3

"""
Data Transformation Script for BEACON

This script handles data transformations and feature engineering.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

from beacon.data import DataTransformer
from beacon.utils.metrics import Metrics

class DataTransformationRunner:
    """Run data transformations and feature engineering"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the transformation runner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.input_dir = Path(config['input_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.transformer = DataTransformer(config.get('transformer_config', {}))
        self.label_encoders = {}
        self.scalers = {}

    def transform_clinical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform clinical data.

        Args:
            data: Input clinical data

        Returns:
            Transformed DataFrame
        """
        # Numeric features
        numeric_features = ['age', 'weight', 'height', 'blood_pressure_systolic',
                          'blood_pressure_diastolic', 'heart_rate', 'temperature']
        
        # Categorical features
        categorical_features = ['sex', 'smoking_status', 'diagnosis']
        
        # Date features
        date_features = ['admission_date', 'discharge_date']
        
        transformed_data = data.copy()
        
        # Transform numeric features
        for feature in numeric_features:
            if feature in data.columns:
                if feature not in self.scalers:
                    self.scalers[feature] = StandardScaler()
                    transformed_data[feature] = self.scalers[feature].fit_transform(
                        data[feature].values.reshape(-1, 1)
                    )
                else:
                    transformed_data[feature] = self.scalers[feature].transform(
                        data[feature].values.reshape(-1, 1)
                    )
        
        # Transform categorical features
        for feature in categorical_features:
            if feature in data.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    transformed_data[feature] = self.label_encoders[feature].fit_transform(
                        data[feature]
                    )
                else:
                    transformed_data[feature] = self.label_encoders[feature].transform(
                        data[feature]
                    )
        
        # Transform date features
        for feature in date_features:
            if feature in data.columns:
                transformed_data[feature] = pd.to_datetime(data[feature])
                transformed_data[f'{feature}_year'] = transformed_data[feature].dt.year
                transformed_data[f'{feature}_month'] = transformed_data[feature].dt.month
                transformed_data[f'{feature}_day'] = transformed_data[feature].dt.day
                transformed_data = transformed_data.drop(columns=[feature])
        
        # Feature engineering
        if 'admission_date' in data.columns and 'discharge_date' in data.columns:
            transformed_data['length_of_stay'] = (
                pd.to_datetime(data['discharge_date']) - 
                pd.to_datetime(data['admission_date'])
            ).dt.days
        
        if 'height' in data.columns and 'weight' in data.columns:
            height_m = data['height'] / 100  # Convert cm to m
            transformed_data['bmi'] = data['weight'] / (height_m ** 2)
        
        return transformed_data

    def transform_imaging_data(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform imaging data.

        Args:
            images: Dictionary of images

        Returns:
            Dictionary of transformed images
        """
        transformed_images = {}
        
        for patient_id, image in images.items():
            # Normalize to [0, 1]
            normalized = (image - image.min()) / (image.max() - image.min())
            
            # Apply transformations based on configuration
            if self.config.get('apply_clahe', False):
                normalized = self._apply_clahe(normalized)
            
            if self.config.get('apply_gaussian_blur', False):
                normalized = self._apply_gaussian_blur(normalized)
            
            transformed_images[patient_id] = normalized
        
        return transformed_images

    def transform_genomic_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform genomic data.

        Args:
            data: Dictionary of genomic data arrays

        Returns:
            Dictionary of transformed arrays
        """
        transformed_data = {}
        
        # Transform expression data
        if 'expression' in data:
            expression = data['expression']
            # Log transform
            expression = np.log2(expression + 1)
            # Z-score normalization
            expression = (expression - expression.mean(axis=0)) / expression.std(axis=0)
            transformed_data['expression'] = expression
        
        # Transform mutation data (already binary)
        if 'mutations' in data:
            transformed_data['mutations'] = data['mutations']
        
        # Transform CNV data
        if 'cnv' in data:
            cnv = data['cnv']
            # Scale to [-1, 1]
            cnv = cnv / 2
            transformed_data['cnv'] = cnv
        
        return transformed_data

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        from skimage import exposure
        return exposure.equalize_adapthist(image)

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=1)

    def save_transformed_data(self, data: Any, name: str) -> None:
        """Save transformed data.

        Args:
            data: Data to save
            name: Name of the dataset
        """
        output_path = self.output_dir / f'transformed_{name}'
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(f'{output_path}.csv', index=False)
        elif isinstance(data, np.ndarray):
            np.save(f'{output_path}.npy', data)
        elif isinstance(data, dict):
            if all(isinstance(v, np.ndarray) for v in data.values()):
                np.savez(f'{output_path}.npz', **data)
            else:
                pd.DataFrame(data).to_csv(f'{output_path}.csv', index=False)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def save_transformers(self) -> None:
        """Save fitted transformers."""
        import joblib
        
        # Save scalers
        scaler_path = self.output_dir / 'scalers'
        scaler_path.mkdir(exist_ok=True)
        for feature, scaler in self.scalers.items():
            joblib.dump(scaler, scaler_path / f'{feature}_scaler.joblib')
        
        # Save label encoders
        encoder_path = self.output_dir / 'encoders'
        encoder_path.mkdir(exist_ok=True)
        for feature, encoder in self.label_encoders.items():
            joblib.dump(encoder, encoder_path / f'{feature}_encoder.joblib')

    def transform_all(self) -> None:
        """Transform all data types."""
        # Transform clinical data
        clinical_file = self.input_dir / 'clinical.csv'
        if clinical_file.exists():
            clinical_data = pd.read_csv(clinical_file)
            transformed_clinical = self.transform_clinical_data(clinical_data)
            self.save_transformed_data(transformed_clinical, 'clinical')
            print(f"Transformed clinical data shape: {transformed_clinical.shape}")

        # Transform imaging data
        image_file = self.input_dir / 'processed_images.npz'
        if image_file.exists():
            images = dict(np.load(image_file))
            transformed_images = self.transform_imaging_data(images)
            self.save_transformed_data(transformed_images, 'images')
            print(f"Transformed {len(transformed_images)} images")

        # Transform genomic data
        genomic_file = self.input_dir / 'processed_genomic.npz'
        if genomic_file.exists():
            genomic_data = dict(np.load(genomic_file))
            transformed_genomic = self.transform_genomic_data(genomic_data)
            self.save_transformed_data(transformed_genomic, 'genomic')
            print(f"Transformed genomic data with shapes:")
            for key, value in transformed_genomic.items():
                print(f"  {key}: {value.shape}")

        # Save transformers
        self.save_transformers()
        print("\nTransformation completed!")
        print(f"Transformed data and transformers saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Transform data for BEACON")
    parser.add_argument("--input-dir", required=True, help="Input data directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--config", help="Configuration file path")
    args = parser.parse_args()

    # Load configuration
    config = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir
    }

    if args.config:
        import yaml
        with open(args.config) as f:
            config.update(yaml.safe_load(f))

    # Run transformations
    transformer = DataTransformationRunner(config)
    transformer.transform_all()

if __name__ == "__main__":
    main() 