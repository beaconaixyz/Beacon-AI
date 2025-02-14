#!/usr/bin/env python3

"""
Test Data Generator for BEACON

This script generates synthetic test data for various BEACON components.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, List

class TestDataGenerator:
    """Generate synthetic test data for BEACON"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the data generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'test_data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(config.get('seed', 42))

    def generate_clinical_data(self) -> pd.DataFrame:
        """Generate synthetic clinical data.

        Returns:
            DataFrame containing clinical data
        """
        n_samples = self.config.get('n_samples', 1000)
        
        data = {
            'age': self.rng.normal(60, 10, n_samples),
            'weight': self.rng.normal(70, 15, n_samples),
            'height': self.rng.normal(170, 10, n_samples),
            'blood_pressure_systolic': self.rng.normal(120, 15, n_samples),
            'blood_pressure_diastolic': self.rng.normal(80, 10, n_samples),
            'heart_rate': self.rng.normal(75, 12, n_samples),
            'temperature': self.rng.normal(37, 0.5, n_samples),
            'glucose': self.rng.normal(100, 20, n_samples),
            'cholesterol': self.rng.normal(200, 40, n_samples),
            'smoking': self.rng.choice(['Never', 'Former', 'Current'], n_samples),
            'diabetes': self.rng.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'hypertension': self.rng.choice([0, 1], n_samples, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)

    def generate_imaging_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic medical imaging data.

        Returns:
            Tuple of (images, labels)
        """
        n_samples = self.config.get('n_samples', 100)
        image_size = self.config.get('image_size', 64)
        
        # Generate random images
        images = self.rng.normal(0, 1, (n_samples, 1, image_size, image_size))
        
        # Add some structure to images
        for i in range(n_samples):
            # Add random circles
            center_x = self.rng.integers(image_size//4, 3*image_size//4)
            center_y = self.rng.integers(image_size//4, 3*image_size//4)
            radius = self.rng.integers(5, 15)
            
            y, x = np.ogrid[-center_y:image_size-center_y, -center_x:image_size-center_x]
            mask = x*x + y*y <= radius*radius
            images[i, 0, mask] += 2.0
        
        # Generate labels
        labels = self.rng.choice([0, 1], n_samples)
        
        return images, labels

    def generate_genomic_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic genomic data.

        Returns:
            Dictionary containing genomic data
        """
        n_samples = self.config.get('n_samples', 100)
        n_genes = self.config.get('n_genes', 1000)
        
        # Generate expression data
        expression = self.rng.normal(0, 1, (n_samples, n_genes))
        
        # Generate mutation data (sparse)
        mutations = self.rng.choice([0, 1], (n_samples, n_genes), p=[0.99, 0.01])
        
        # Generate copy number variations
        cnv = self.rng.choice([-2, -1, 0, 1, 2], (n_samples, n_genes), p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        return {
            'expression': expression,
            'mutations': mutations,
            'cnv': cnv
        }

    def generate_survival_data(self) -> pd.DataFrame:
        """Generate synthetic survival data.

        Returns:
            DataFrame containing survival data
        """
        n_samples = self.config.get('n_samples', 1000)
        
        data = {
            'time': self.rng.exponential(500, n_samples),
            'event': self.rng.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'age': self.rng.normal(60, 10, n_samples),
            'stage': self.rng.choice(['I', 'II', 'III', 'IV'], n_samples),
            'grade': self.rng.choice(['Low', 'Medium', 'High'], n_samples)
        }
        
        return pd.DataFrame(data)

    def save_data(self, data: Any, name: str) -> None:
        """Save generated data.

        Args:
            data: Data to save
            name: Name of the dataset
        """
        path = self.output_dir / name
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{path}.csv", index=False)
        elif isinstance(data, np.ndarray):
            np.save(f"{path}.npy", data)
        elif isinstance(data, dict):
            np.savez(f"{path}.npz", **data)
        elif isinstance(data, tuple):
            for i, item in enumerate(data):
                np.save(f"{path}_{i}.npy", item)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data for BEACON")
    parser.add_argument("--output-dir", default="test_data", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image-size", type=int, default=64, help="Image size")
    parser.add_argument("--n-genes", type=int, default=1000, help="Number of genes")
    args = parser.parse_args()

    config = vars(args)
    generator = TestDataGenerator(config)

    # Generate and save clinical data
    clinical_data = generator.generate_clinical_data()
    generator.save_data(clinical_data, "clinical")
    print(f"Generated clinical data: {len(clinical_data)} samples")

    # Generate and save imaging data
    images, labels = generator.generate_imaging_data()
    generator.save_data((images, labels), "imaging")
    print(f"Generated imaging data: {len(images)} samples")

    # Generate and save genomic data
    genomic_data = generator.generate_genomic_data()
    generator.save_data(genomic_data, "genomic")
    print(f"Generated genomic data: {genomic_data['expression'].shape[0]} samples")

    # Generate and save survival data
    survival_data = generator.generate_survival_data()
    generator.save_data(survival_data, "survival")
    print(f"Generated survival data: {len(survival_data)} samples")

if __name__ == "__main__":
    main() 