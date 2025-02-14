"""
Data validation and integration module for BEACON

This module implements standardized data collection, quality validation,
and version control functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass

@dataclass
class DataVersion:
    """Data version information."""
    version_id: str
    timestamp: datetime
    description: str
    source: str
    checksum: str
    metadata: Dict[str, Any]

class DataValidator:
    """Data validation and quality control."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define validation schemas
        self.schemas = {
            'clinical': {
                'required_columns': [
                    'age', 'weight', 'height', 'blood_pressure_systolic',
                    'blood_pressure_diastolic', 'heart_rate', 'temperature',
                    'glucose', 'cholesterol', 'smoking'
                ],
                'data_types': {
                    'age': 'numeric',
                    'weight': 'numeric',
                    'height': 'numeric',
                    'blood_pressure_systolic': 'numeric',
                    'blood_pressure_diastolic': 'numeric',
                    'heart_rate': 'numeric',
                    'temperature': 'numeric',
                    'glucose': 'numeric',
                    'cholesterol': 'numeric',
                    'smoking': ['Never', 'Former', 'Current']
                },
                'value_ranges': {
                    'age': (0, 120),
                    'weight': (0, 500),
                    'height': (0, 300),
                    'blood_pressure_systolic': (60, 250),
                    'blood_pressure_diastolic': (40, 150),
                    'heart_rate': (30, 200),
                    'temperature': (30, 45),
                    'glucose': (0, 500),
                    'cholesterol': (0, 1000)
                }
            },
            'imaging': {
                'required_dimensions': 4,  # (samples, channels, height, width)
                'value_range': (-1000, 1000),
                'min_resolution': (64, 64),
                'allowed_formats': ['.npy', '.npz', '.dicom']
            },
            'genomic': {
                'expression': {
                    'min_genes': 1000,
                    'value_range': (0, float('inf')),
                    'missing_threshold': 0.2
                },
                'mutations': {
                    'allowed_values': [0, 1],
                    'sparsity_threshold': 0.01
                },
                'cnv': {
                    'allowed_values': [-2, -1, 0, 1, 2],
                    'missing_threshold': 0.1
                }
            },
            'survival': {
                'required_columns': ['time', 'event', 'age', 'stage', 'grade'],
                'data_types': {
                    'time': 'numeric',
                    'event': [0, 1],
                    'age': 'numeric',
                    'stage': ['I', 'II', 'III', 'IV'],
                    'grade': ['Low', 'Medium', 'High']
                },
                'value_ranges': {
                    'time': (0, float('inf')),
                    'age': (0, 120)
                }
            }
        }

    def validate_clinical_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate clinical data.

        Args:
            data: Clinical data DataFrame

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        schema = self.schemas['clinical']
        
        # Check required columns
        missing_cols = set(schema['required_columns']) - set(data.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types and values
        for col, dtype in schema['data_types'].items():
            if col in data.columns:
                if dtype == 'numeric':
                    if not pd.to_numeric(data[col], errors='coerce').notna().all():
                        errors.append(f"Column {col} must contain numeric values")
                    else:
                        # Check value ranges
                        value_range = schema['value_ranges'].get(col)
                        if value_range:
                            if not data[col].between(*value_range).all():
                                errors.append(f"Values in {col} must be between {value_range[0]} and {value_range[1]}")
                else:
                    if not data[col].isin(dtype).all():
                        errors.append(f"Values in {col} must be one of {dtype}")
        
        return len(errors) == 0, errors

    def validate_imaging_data(self, data: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate imaging data.

        Args:
            data: Imaging data array

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        schema = self.schemas['imaging']
        
        # Check dimensions
        if len(data.shape) != schema['required_dimensions']:
            errors.append(f"Data must be {schema['required_dimensions']}-dimensional")
        
        # Check resolution
        if data.shape[-2:] < schema['min_resolution']:
            errors.append(f"Image resolution must be at least {schema['min_resolution']}")
        
        # Check value range
        if not np.all((data >= schema['value_range'][0]) & (data <= schema['value_range'][1])):
            errors.append(f"Values must be between {schema['value_range']}")
        
        return len(errors) == 0, errors

    def validate_genomic_data(self, data: Dict[str, np.ndarray]) -> Tuple[bool, List[str]]:
        """Validate genomic data.

        Args:
            data: Dictionary of genomic data arrays

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        schema = self.schemas['genomic']
        
        # Validate expression data
        if 'expression' in data:
            expr_schema = schema['expression']
            expr_data = data['expression']
            
            if expr_data.shape[1] < expr_schema['min_genes']:
                errors.append(f"Expression data must have at least {expr_schema['min_genes']} genes")
            
            missing_rate = np.isnan(expr_data).mean()
            if missing_rate > expr_schema['missing_threshold']:
                errors.append(f"Expression data missing rate ({missing_rate:.2f}) exceeds threshold")
        
        # Validate mutation data
        if 'mutations' in data:
            mut_schema = schema['mutations']
            mut_data = data['mutations']
            
            if not np.all(np.isin(mut_data, mut_schema['allowed_values'])):
                errors.append("Mutation data must contain only binary values")
            
            sparsity = (mut_data != 0).mean()
            if sparsity < mut_schema['sparsity_threshold']:
                errors.append(f"Mutation data sparsity ({sparsity:.4f}) below threshold")
        
        # Validate CNV data
        if 'cnv' in data:
            cnv_schema = schema['cnv']
            cnv_data = data['cnv']
            
            if not np.all(np.isin(cnv_data, cnv_schema['allowed_values'])):
                errors.append("CNV data must contain only allowed values: -2, -1, 0, 1, 2")
            
            missing_rate = np.isnan(cnv_data).mean()
            if missing_rate > cnv_schema['missing_threshold']:
                errors.append(f"CNV data missing rate ({missing_rate:.2f}) exceeds threshold")
        
        return len(errors) == 0, errors

    def validate_survival_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate survival data.

        Args:
            data: Survival data DataFrame

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        schema = self.schemas['survival']
        
        # Check required columns
        missing_cols = set(schema['required_columns']) - set(data.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types and values
        for col, dtype in schema['data_types'].items():
            if col in data.columns:
                if dtype == 'numeric':
                    if not pd.to_numeric(data[col], errors='coerce').notna().all():
                        errors.append(f"Column {col} must contain numeric values")
                    else:
                        # Check value ranges
                        value_range = schema['value_ranges'].get(col)
                        if value_range:
                            if not data[col].between(*value_range).all():
                                errors.append(f"Values in {col} must be between {value_range[0]} and {value_range[1]}")
                else:
                    if not data[col].isin(dtype).all():
                        errors.append(f"Values in {col} must be one of {dtype}")
        
        return len(errors) == 0, errors

class DataVersionControl:
    """Data version control system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize version control system.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.version_dir = Path(config.get('version_dir', 'versions'))
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        # Load version history
        self.version_history = self._load_version_history()

    def _load_version_history(self) -> Dict[str, DataVersion]:
        """Load version history from disk.

        Returns:
            Dictionary of version information
        """
        history = {}
        history_file = self.version_dir / 'version_history.json'
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
                for version_id, data in history_data.items():
                    history[version_id] = DataVersion(
                        version_id=version_id,
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        description=data['description'],
                        source=data['source'],
                        checksum=data['checksum'],
                        metadata=data['metadata']
                    )
        
        return history

    def _save_version_history(self) -> None:
        """Save version history to disk."""
        history_data = {
            version_id: {
                'timestamp': version.timestamp.isoformat(),
                'description': version.description,
                'source': version.source,
                'checksum': version.checksum,
                'metadata': version.metadata
            }
            for version_id, version in self.version_history.items()
        }
        
        with open(self.version_dir / 'version_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)

    def create_version(self, data: Any, description: str, source: str,
                      metadata: Dict[str, Any]) -> DataVersion:
        """Create new data version.

        Args:
            data: Data to version
            description: Version description
            source: Data source
            metadata: Additional metadata

        Returns:
            Created version information
        """
        # Generate version ID and checksum
        timestamp = datetime.now()
        version_id = timestamp.strftime('%Y%m%d_%H%M%S')
        checksum = self._compute_checksum(data)
        
        # Create version info
        version = DataVersion(
            version_id=version_id,
            timestamp=timestamp,
            description=description,
            source=source,
            checksum=checksum,
            metadata=metadata
        )
        
        # Save version info
        self.version_history[version_id] = version
        self._save_version_history()
        
        # Save data
        self._save_version_data(data, version_id)
        
        return version

    def get_version(self, version_id: str) -> Optional[Any]:
        """Retrieve specific data version.

        Args:
            version_id: Version ID to retrieve

        Returns:
            Retrieved data or None if not found
        """
        if version_id not in self.version_history:
            return None
        
        return self._load_version_data(version_id)

    def list_versions(self) -> List[DataVersion]:
        """List all available versions.

        Returns:
            List of version information
        """
        return list(self.version_history.values())

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two data versions.

        Args:
            version_id1: First version ID
            version_id2: Second version ID

        Returns:
            Dictionary containing comparison results
        """
        if version_id1 not in self.version_history or version_id2 not in self.version_history:
            return {}
        
        v1 = self.version_history[version_id1]
        v2 = self.version_history[version_id2]
        
        comparison = {
            'timestamp_diff': (v2.timestamp - v1.timestamp).total_seconds(),
            'source_changed': v1.source != v2.source,
            'checksum_changed': v1.checksum != v2.checksum,
            'metadata_diff': self._compare_metadata(v1.metadata, v2.metadata)
        }
        
        return comparison

    @staticmethod
    def _compute_checksum(data: Any) -> str:
        """Compute data checksum.

        Args:
            data: Data to compute checksum for

        Returns:
            Computed checksum
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data_bytes = data.to_csv().encode()
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, (dict, list)):
            data_bytes = json.dumps(data, sort_keys=True).encode()
        else:
            data_bytes = str(data).encode()
        
        return hashlib.sha256(data_bytes).hexdigest()

    def _save_version_data(self, data: Any, version_id: str) -> None:
        """Save versioned data to disk.

        Args:
            data: Data to save
            version_id: Version ID
        """
        data_path = self.version_dir / f'data_{version_id}'
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{data_path}.csv", index=False)
        elif isinstance(data, np.ndarray):
            np.save(f"{data_path}.npy", data)
        elif isinstance(data, dict):
            np.savez(f"{data_path}.npz", **data)
        else:
            with open(f"{data_path}.pkl", 'wb') as f:
                import pickle
                pickle.dump(data, f)

    def _load_version_data(self, version_id: str) -> Optional[Any]:
        """Load versioned data from disk.

        Args:
            version_id: Version ID to load

        Returns:
            Loaded data or None if not found
        """
        # Try different file extensions
        for ext in ['.csv', '.npy', '.npz', '.pkl']:
            data_path = self.version_dir / f'data_{version_id}{ext}'
            if data_path.exists():
                if ext == '.csv':
                    return pd.read_csv(data_path)
                elif ext == '.npy':
                    return np.load(data_path)
                elif ext == '.npz':
                    return dict(np.load(data_path))
                else:
                    with open(data_path, 'rb') as f:
                        import pickle
                        return pickle.load(f)
        
        return None

    @staticmethod
    def _compare_metadata(metadata1: Dict[str, Any],
                         metadata2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two metadata dictionaries.

        Args:
            metadata1: First metadata dictionary
            metadata2: Second metadata dictionary

        Returns:
            Dictionary containing metadata differences
        """
        all_keys = set(metadata1.keys()) | set(metadata2.keys())
        differences = {}
        
        for key in all_keys:
            if key not in metadata1:
                differences[key] = ('missing', metadata2[key])
            elif key not in metadata2:
                differences[key] = (metadata1[key], 'missing')
            elif metadata1[key] != metadata2[key]:
                differences[key] = (metadata1[key], metadata2[key])
        
        return differences 