#!/usr/bin/env python3

"""
Data Cache implementation for BEACON framework.

This module provides caching functionality for processed data to improve performance.
"""

import os
import time
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import torch

from beacon.core.base import BeaconBase


class DataCache(BeaconBase):
    """Cache handler for processed data."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the cache handler.

        Args:
            config: Configuration dictionary containing cache parameters
        """
        super().__init__(config)
        self.cache_dir = Path(config.get('cache_dir', '.cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_expired()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            'cache_dir': '.cache',
            'ttl': 3600,  # Time to live in seconds (1 hour)
            'max_size': 1e9,  # Maximum cache size in bytes (1GB)
            'cleanup_interval': 86400  # Cleanup interval in seconds (1 day)
        }

    def _get_cache_path(self, key: Union[str, Path]) -> Path:
        """Get cache file path for a key.

        Args:
            key: Cache key (usually a file path)

        Returns:
            Path to cache file
        """
        # Create hash of the key
        if isinstance(key, Path):
            key = str(key.absolute())
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _get_metadata_path(self, cache_path: Path) -> Path:
        """Get metadata file path for a cache file.

        Args:
            cache_path: Path to cache file

        Returns:
            Path to metadata file
        """
        return cache_path.with_suffix('.meta')

    def _save_metadata(self, path: Path, metadata: Dict[str, Any]) -> None:
        """Save metadata for a cache entry.

        Args:
            path: Path to metadata file
            metadata: Metadata dictionary
        """
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)

    def _load_metadata(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load metadata for a cache entry.

        Args:
            path: Path to metadata file

        Returns:
            Metadata dictionary or None if file doesn't exist
        """
        if not path.exists():
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            return None

    def save(
        self,
        data: Any,
        key: Union[str, Path],
        ttl: Optional[int] = None
    ) -> None:
        """Save data to cache.

        Args:
            data: Data to cache
            key: Cache key
            ttl: Time to live in seconds (optional)
        """
        cache_path = self._get_cache_path(key)
        metadata_path = self._get_metadata_path(cache_path)

        # Check cache size before saving
        if self._get_cache_size() > self.config['max_size']:
            self._cleanup_oldest()

        # Save data
        try:
            if isinstance(data, (tuple, list)):
                # Handle multiple tensors/arrays
                tensors = []
                arrays = []
                others = []
                for item in data:
                    if isinstance(item, torch.Tensor):
                        tensors.append(item)
                    elif isinstance(item, np.ndarray):
                        arrays.append(item)
                    else:
                        others.append(item)
                
                torch.save(tensors, cache_path)
                if arrays or others:
                    with open(cache_path.with_suffix('.npz'), 'wb') as f:
                        np.savez(f, *arrays)
                    if others:
                        with open(cache_path.with_suffix('.pkl'), 'wb') as f:
                            pickle.dump(others, f)
            
            elif isinstance(data, torch.Tensor):
                torch.save(data, cache_path)
            
            elif isinstance(data, np.ndarray):
                np.save(cache_path, data)
            
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)

            # Save metadata
            metadata = {
                'timestamp': time.time(),
                'ttl': ttl or self.config['ttl'],
                'type': type(data).__name__,
                'size': os.path.getsize(cache_path)
            }
            self._save_metadata(metadata_path, metadata)

        except Exception as e:
            self.logger.error(f"Error saving to cache: {str(e)}")
            if cache_path.exists():
                cache_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

    def load(self, key: Union[str, Path]) -> Optional[Any]:
        """Load data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found or expired
        """
        cache_path = self._get_cache_path(key)
        metadata_path = self._get_metadata_path(cache_path)

        # Check if cache exists
        if not cache_path.exists():
            return None

        # Load metadata
        metadata = self._load_metadata(metadata_path)
        if metadata is None:
            return None

        # Check if cache is expired
        if time.time() - metadata['timestamp'] > metadata['ttl']:
            self._remove_cache_entry(cache_path)
            return None

        # Load data
        try:
            if metadata['type'] == 'tuple' or metadata['type'] == 'list':
                # Load multiple items
                tensors = torch.load(cache_path) if cache_path.exists() else []
                arrays = []
                others = []
                
                if cache_path.with_suffix('.npz').exists():
                    with np.load(cache_path.with_suffix('.npz')) as data:
                        arrays = [data[key] for key in data.files]
                
                if cache_path.with_suffix('.pkl').exists():
                    with open(cache_path.with_suffix('.pkl'), 'rb') as f:
                        others = pickle.load(f)
                
                # Combine all items in the original order
                return tuple(tensors + arrays + others)
            
            elif metadata['type'] == 'Tensor':
                return torch.load(cache_path)
            
            elif metadata['type'] == 'ndarray':
                return np.load(cache_path)
            
            else:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

        except Exception as e:
            self.logger.error(f"Error loading from cache: {str(e)}")
            self._remove_cache_entry(cache_path)
            return None

    def clear(self) -> None:
        """Clear all cache entries."""
        for path in self.cache_dir.glob('*'):
            path.unlink()

    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        for metadata_path in self.cache_dir.glob('*.meta'):
            metadata = self._load_metadata(metadata_path)
            if metadata and current_time - metadata['timestamp'] > metadata['ttl']:
                cache_path = metadata_path.with_suffix('.cache')
                self._remove_cache_entry(cache_path)

    def _cleanup_oldest(self) -> None:
        """Remove oldest cache entries until size is under limit."""
        entries = []
        total_size = 0

        # Collect all cache entries and their metadata
        for metadata_path in self.cache_dir.glob('*.meta'):
            metadata = self._load_metadata(metadata_path)
            if metadata:
                cache_path = metadata_path.with_suffix('.cache')
                entries.append((
                    cache_path,
                    metadata['timestamp'],
                    metadata['size']
                ))
                total_size += metadata['size']

        # Sort by timestamp (oldest first)
        entries.sort(key=lambda x: x[1])

        # Remove oldest entries until under size limit
        while total_size > self.config['max_size'] and entries:
            cache_path, _, size = entries.pop(0)
            self._remove_cache_entry(cache_path)
            total_size -= size

    def _remove_cache_entry(self, cache_path: Path) -> None:
        """Remove a cache entry and its metadata.

        Args:
            cache_path: Path to cache file
        """
        # Remove all related files
        if cache_path.exists():
            cache_path.unlink()
        if cache_path.with_suffix('.meta').exists():
            cache_path.with_suffix('.meta').unlink()
        if cache_path.with_suffix('.npz').exists():
            cache_path.with_suffix('.npz').unlink()
        if cache_path.with_suffix('.pkl').exists():
            cache_path.with_suffix('.pkl').unlink()

    def _get_cache_size(self) -> int:
        """Get total size of cache in bytes.

        Returns:
            Total cache size in bytes
        """
        total_size = 0
        for path in self.cache_dir.glob('*'):
            total_size += path.stat().st_size
        return total_size

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        stats = {
            'total_size': self._get_cache_size(),
            'num_entries': len(list(self.cache_dir.glob('*.cache'))),
            'max_size': self.config['max_size'],
            'ttl': self.config['ttl']
        }
        return stats 