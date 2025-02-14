#!/usr/bin/env python3

"""
Data Preprocessing Script for BEACON

This script handles data preprocessing for various data types including clinical,
imaging, genomic, and survival data.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

class DataPreprocessor:
    """Preprocess different types of data for BEACON"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the preprocessor.

        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.input_dir = Path(config.get('input_dir', 'test_data'))
        self.output_dir = Path(config.get('output_dir', 'processed_data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_clinical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess clinical data.

        Args:
            data: Raw clinical data

        Returns:
            Preprocessed clinical data
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()

        # Handle missing values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Fill numeric missing values with median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # Fill categorical missing values with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Normalize numeric features
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        return df

    def preprocess_imaging_data(self, images: np.ndarray) -> np.ndarray:
        """Preprocess imaging data.

        Args:
            images: Raw imaging data

        Returns:
            Preprocessed imaging data
        """
        # Normalize to [0, 1] range
        images = (images - images.min()) / (images.max() - images.min())

        # Add channel dimension if needed
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=1)

        return images

    def preprocess_genomic_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess genomic data.

        Args:
            data: Dictionary containing different types of genomic data

        Returns:
            Preprocessed genomic data
        """
        processed_data = {}

        # Process expression data
        if 'expression' in data:
            # Log transform and normalize expression data
            expression = np.log2(data['expression'] + 1)
            scaler = StandardScaler()
            processed_data['expression'] = scaler.fit_transform(expression)

        # Process mutation data
        if 'mutations' in data:
            # Keep binary values as is
            processed_data['mutations'] = data['mutations']

        # Process copy number variations
        if 'cnv' in data:
            # Normalize CNV data
            scaler = StandardScaler()
            processed_data['cnv'] = scaler.fit_transform(data['cnv'])

        return processed_data

    def preprocess_survival_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess survival data.

        Args:
            data: Raw survival data

        Returns:
            Preprocessed survival data
        """
        df = data.copy()

        # Encode categorical variables
        categorical_cols = ['stage', 'grade']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Normalize numeric features
        numeric_cols = ['time', 'age']
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        return df

    def load_and_preprocess(self, data_type: str) -> Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        """Load and preprocess data of specified type.

        Args:
            data_type: Type of data to process ('clinical', 'imaging', 'genomic', or 'survival')

        Returns:
            Preprocessed data
        """
        try:
            if data_type == 'clinical':
                data = pd.read_csv(self.input_dir / 'clinical.csv')
                return self.preprocess_clinical_data(data)
            
            elif data_type == 'imaging':
                images = np.load(self.input_dir / 'imaging_0.npy')
                return self.preprocess_imaging_data(images)
            
            elif data_type == 'genomic':
                data = dict(np.load(self.input_dir / 'genomic.npz'))
                return self.preprocess_genomic_data(data)
            
            elif data_type == 'survival':
                data = pd.read_csv(self.input_dir / 'survival.csv')
                return self.preprocess_survival_data(data)
            
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
        
        except Exception as e:
            self.logger.error(f"Error processing {data_type} data: {str(e)}")
            raise

    def save_processed_data(self, data: Any, name: str) -> None:
        """Save preprocessed data.

        Args:
            data: Preprocessed data to save
            name: Name of the dataset
        """
        path = self.output_dir / name
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{path}_processed.csv", index=False)
        elif isinstance(data, np.ndarray):
            np.save(f"{path}_processed.npy", data)
        elif isinstance(data, dict):
            np.savez(f"{path}_processed.npz", **data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for BEACON")
    parser.add_argument("--input-dir", default="test_data", help="Input directory containing raw data")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory for processed data")
    parser.add_argument("--data-types", nargs='+', 
                        default=['clinical', 'imaging', 'genomic', 'survival'],
                        help="Types of data to process")
    args = parser.parse_args()

    config = vars(args)
    preprocessor = DataPreprocessor(config)

    for data_type in args.data_types:
        try:
            processed_data = preprocessor.load_and_preprocess(data_type)
            preprocessor.save_processed_data(processed_data, data_type)
            print(f"Successfully processed {data_type} data")
        except Exception as e:
            print(f"Failed to process {data_type} data: {str(e)}")

if __name__ == "__main__":
    main() 