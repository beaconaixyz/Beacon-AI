#!/usr/bin/env python3

"""
Dataset implementations for BEACON framework.

This module provides dataset classes for handling different types of medical data,
including clinical, imaging, and genomic data.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from beacon.core.base import BeaconBase
from beacon.data.processor import DataProcessor
from beacon.data.cache import DataCache


class BeaconDataset(Dataset, BeaconBase):
    """Base dataset class for BEACON framework."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset.

        Args:
            config: Configuration dictionary containing dataset parameters
        """
        BeaconBase.__init__(self, config)
        self.data_dir = Path(config.get('data_dir', '.'))
        self.cache_dir = Path(config.get('cache_dir', '.cache'))
        self.processor = DataProcessor(config.get('processor', {}))
        self.cache = DataCache(config.get('cache', {}))
        self.transform = None
        self.data = None
        self.labels = None

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            'data_dir': '.',
            'cache_dir': '.cache',
            'use_cache': True,
            'cache_ttl': 3600,  # 1 hour
            'processor': {},
            'transform': None
        }

    def load_data(self) -> None:
        """Load dataset. Must be implemented by subclasses."""
        raise NotImplementedError

    def setup(self) -> None:
        """Setup the dataset."""
        self.load_data()
        if self.config.get('transform'):
            self.transform = self._build_transform()

    def _build_transform(self) -> Optional[Any]:
        """Build data transformation pipeline.

        Returns:
            Transformation pipeline or None
        """
        return None

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples in the dataset
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing the sample data and label
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)

        item = {'data': sample}
        if self.labels is not None:
            item['label'] = self.labels[idx]

        return item


class ClinicalDataset(BeaconDataset):
    """Dataset for clinical data."""

    def load_data(self) -> None:
        """Load clinical data from CSV file."""
        data_file = self.data_dir / 'clinical_data.csv'
        
        if self.config['use_cache']:
            cached_data = self.cache.load(data_file)
            if cached_data is not None:
                self.data, self.labels = cached_data
                return

        df = pd.read_csv(data_file)
        
        # Split features and labels
        if 'label' in df.columns:
            self.labels = torch.tensor(df.pop('label').values, dtype=torch.float32)
        
        # Process features
        self.data = torch.tensor(
            self.processor.process_clinical_data(df).values,
            dtype=torch.float32
        )
        
        if self.config['use_cache']:
            self.cache.save((self.data, self.labels), data_file)


class ImageDataset(BeaconDataset):
    """Dataset for medical images."""

    def load_data(self) -> None:
        """Load medical images from directory."""
        image_dir = self.data_dir / 'images'
        label_file = self.data_dir / 'image_labels.csv'
        
        if self.config['use_cache']:
            cached_data = self.cache.load(image_dir)
            if cached_data is not None:
                self.data, self.labels = cached_data
                return

        # Load image paths
        image_paths = sorted(image_dir.glob('*.png'))
        
        # Load images
        images = []
        for img_path in image_paths:
            image = Image.open(img_path)
            image = np.array(image)
            images.append(self.processor.process_imaging_data(image))
        
        self.data = torch.tensor(np.stack(images), dtype=torch.float32)
        
        # Load labels if available
        if label_file.exists():
            labels_df = pd.read_csv(label_file)
            self.labels = torch.tensor(labels_df['label'].values, dtype=torch.float32)
        
        if self.config['use_cache']:
            self.cache.save((self.data, self.labels), image_dir)

    def _build_transform(self) -> Optional[Any]:
        """Build image transformation pipeline."""
        if not self.config.get('transform'):
            return None
            
        transforms_list = []
        transform_config = self.config['transform']
        
        if transform_config.get('normalize', True):
            transforms_list.append(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        
        if transform_config.get('augment', False):
            # Add data augmentation transforms
            pass
        
        if transforms_list:
            return lambda x: torch.stack([t(x) for t in transforms_list])
        return None


class GenomicDataset(BeaconDataset):
    """Dataset for genomic data."""

    def load_data(self) -> None:
        """Load genomic data."""
        data_file = self.data_dir / 'genomic_data.npz'
        
        if self.config['use_cache']:
            cached_data = self.cache.load(data_file)
            if cached_data is not None:
                self.data, self.labels = cached_data
                return

        # Load genomic data
        genomic_data = np.load(data_file)
        
        # Process features
        features = self.processor.process_genetic_data({
            'expression': genomic_data['expression'],
            'mutations': genomic_data['mutations']
        })
        
        self.data = torch.tensor(features, dtype=torch.float32)
        
        if 'labels' in genomic_data:
            self.labels = torch.tensor(
                genomic_data['labels'],
                dtype=torch.float32
            )
        
        if self.config['use_cache']:
            self.cache.save((self.data, self.labels), data_file)


class MultimodalDataset(BeaconDataset):
    """Dataset for multimodal medical data."""

    def load_data(self) -> None:
        """Load multimodal data."""
        if self.config['use_cache']:
            cached_data = self.cache.load(self.data_dir)
            if cached_data is not None:
                self.data, self.labels = cached_data
                return

        # Load different modalities
        clinical_data = pd.read_csv(self.data_dir / 'clinical_data.csv')
        clinical_features = self.processor.process_clinical_data(clinical_data)
        
        image_dir = self.data_dir / 'images'
        image_paths = sorted(image_dir.glob('*.png'))
        images = []
        for img_path in image_paths:
            image = Image.open(img_path)
            image = np.array(image)
            images.append(self.processor.process_imaging_data(image))
        
        genomic_data = np.load(self.data_dir / 'genomic_data.npz')
        genomic_features = self.processor.process_genetic_data({
            'expression': genomic_data['expression'],
            'mutations': genomic_data['mutations']
        })
        
        # Combine modalities
        self.data = {
            'clinical': torch.tensor(clinical_features.values, dtype=torch.float32),
            'imaging': torch.tensor(np.stack(images), dtype=torch.float32),
            'genomic': torch.tensor(genomic_features, dtype=torch.float32)
        }
        
        # Load labels
        label_file = self.data_dir / 'labels.csv'
        if label_file.exists():
            labels_df = pd.read_csv(label_file)
            self.labels = torch.tensor(labels_df['label'].values, dtype=torch.float32)
        
        if self.config['use_cache']:
            self.cache.save((self.data, self.labels), self.data_dir)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a multimodal sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing data from all modalities and label
        """
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        sample = {
            modality: data[idx] 
            for modality, data in self.data.items()
        }

        if self.transform:
            sample = self.transform(sample)

        item = {'data': sample}
        if self.labels is not None:
            item['label'] = self.labels[idx]

        return item 