"""
BEACON Data Module

This module provides data handling functionality for the BEACON framework,
including dataset implementations, data processing, and caching.
"""

from beacon.data.dataset import (
    BeaconDataset,
    ClinicalDataset,
    ImageDataset,
    GenomicDataset,
    MultimodalDataset
)

from beacon.data.processor import DataProcessor
from beacon.data.cache import DataCache

__all__ = [
    'BeaconDataset',
    'ClinicalDataset',
    'ImageDataset',
    'GenomicDataset',
    'MultimodalDataset',
    'DataProcessor',
    'DataCache'
]
