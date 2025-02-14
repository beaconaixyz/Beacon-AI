#!/usr/bin/env python3

"""
Data Processor implementations for BEACON framework.

This module provides data processing classes for different types of medical data,
including preprocessing, feature extraction, and data augmentation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import cv2
from PIL import Image

from beacon.core.base import BeaconBase


class DataProcessor(BeaconBase):
    """Base class for data processing in BEACON framework."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the data processor.

        Args:
            config: Configuration dictionary containing processing parameters
        """
        super().__init__(config)
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            'clinical': {
                'scaling': 'standard',  # or 'minmax'
                'imputation': 'mean',   # or 'median', 'constant'
                'categorical_encoding': 'label',  # or 'onehot'
                'feature_selection': None
            },
            'imaging': {
                'resize': (224, 224),
                'normalize': True,
                'augmentation': False,
                'color_mode': 'rgb'  # or 'grayscale'
            },
            'genomic': {
                'normalization': 'log2',  # or 'zscore'
                'feature_selection': None,
                'mutation_encoding': 'binary'
            }
        }

    def process_clinical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process clinical data.

        Args:
            data: Input clinical data DataFrame

        Returns:
            Processed DataFrame
        """
        processed_data = data.copy()
        config = self.config['clinical']

        # Handle missing values
        for column in processed_data.columns:
            if processed_data[column].isnull().any():
                if column not in self.imputers:
                    self.imputers[column] = SimpleImputer(
                        strategy=config['imputation']
                    )
                    processed_data[column] = self.imputers[column].fit_transform(
                        processed_data[[column]]
                    )
                else:
                    processed_data[column] = self.imputers[column].transform(
                        processed_data[[column]]
                    )

        # Process numeric features
        numeric_columns = processed_data.select_dtypes(
            include=['int64', 'float64']
        ).columns
        
        if config['scaling'] == 'standard':
            scaler_class = StandardScaler
        else:  # minmax
            scaler_class = MinMaxScaler

        for column in numeric_columns:
            if column not in self.scalers:
                self.scalers[column] = scaler_class()
                processed_data[column] = self.scalers[column].fit_transform(
                    processed_data[[column]]
                )
            else:
                processed_data[column] = self.scalers[column].transform(
                    processed_data[[column]]
                )

        # Process categorical features
        categorical_columns = processed_data.select_dtypes(
            include=['object', 'category']
        ).columns

        if config['categorical_encoding'] == 'label':
            for column in categorical_columns:
                if column not in self.encoders:
                    self.encoders[column] = LabelEncoder()
                    processed_data[column] = self.encoders[column].fit_transform(
                        processed_data[column]
                    )
                else:
                    processed_data[column] = self.encoders[column].transform(
                        processed_data[column]
                    )
        else:  # onehot
            processed_data = pd.get_dummies(
                processed_data,
                columns=categorical_columns,
                prefix=categorical_columns
            )

        return processed_data

    def process_imaging_data(self, image: np.ndarray) -> np.ndarray:
        """Process medical imaging data.

        Args:
            image: Input image array

        Returns:
            Processed image array
        """
        config = self.config['imaging']

        # Convert to RGB if needed
        if len(image.shape) == 2 and config['color_mode'] == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and config['color_mode'] == 'grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize image
        if config['resize']:
            image = cv2.resize(image, config['resize'])

        # Normalize pixel values
        if config['normalize']:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image /= 255.0

        # Apply augmentation if enabled
        if config['augmentation']:
            image = self._augment_image(image)

        return image

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image.

        Args:
            image: Input image array

        Returns:
            Augmented image array
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image)

        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image)

        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-30, 30)
            image = Image.fromarray((image * 255).astype(np.uint8))
            image = image.rotate(angle)
            image = np.array(image).astype(np.float32) / 255.0

        # Random brightness adjustment
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        return image

    def process_genetic_data(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Process genomic data.

        Args:
            data: Dictionary containing genomic data arrays

        Returns:
            Processed feature array
        """
        config = self.config['genomic']
        processed_features = []

        # Process expression data
        if 'expression' in data:
            expression = data['expression']
            if config['normalization'] == 'log2':
                expression = np.log2(expression + 1)
            else:  # zscore
                expression = (expression - expression.mean(axis=0)) / (
                    expression.std(axis=0) + 1e-8
                )
            processed_features.append(expression)

        # Process mutation data
        if 'mutations' in data:
            mutations = data['mutations']
            if config['mutation_encoding'] == 'binary':
                mutations = (mutations > 0).astype(np.float32)
            processed_features.append(mutations)

        # Process CNV data if available
        if 'cnv' in data:
            cnv = data['cnv']
            cnv = (cnv - cnv.mean(axis=0)) / (cnv.std(axis=0) + 1e-8)
            processed_features.append(cnv)

        # Combine all features
        combined_features = np.concatenate(processed_features, axis=1)

        # Apply feature selection if specified
        if config['feature_selection']:
            combined_features = self._select_features(
                combined_features,
                config['feature_selection']
            )

        return combined_features

    def _select_features(
        self,
        features: np.ndarray,
        selection_config: Dict[str, Any]
    ) -> np.ndarray:
        """Select relevant features.

        Args:
            features: Input feature array
            selection_config: Feature selection configuration

        Returns:
            Selected features array
        """
        method = selection_config.get('method', 'variance')
        n_features = selection_config.get('n_features', features.shape[1])

        if method == 'variance':
            # Select features with highest variance
            variances = np.var(features, axis=0)
            selected_indices = np.argsort(variances)[-n_features:]
            return features[:, selected_indices]

        elif method == 'correlation':
            # Select features with lowest correlation
            corr_matrix = np.corrcoef(features.T)
            np.fill_diagonal(corr_matrix, 0)
            mean_corr = np.mean(np.abs(corr_matrix), axis=0)
            selected_indices = np.argsort(mean_corr)[:n_features]
            return features[:, selected_indices]

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

    def save_state(self, path: str) -> None:
        """Save processor state.

        Args:
            path: Path to save state
        """
        import joblib
        state = {
            'config': self.config,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers
        }
        joblib.dump(state, path)

    def load_state(self, path: str) -> None:
        """Load processor state.

        Args:
            path: Path to load state from
        """
        import joblib
        state = joblib.load(path)
        self.config = state['config']
        self.scalers = state['scalers']
        self.encoders = state['encoders']
        self.imputers = state['imputers'] 