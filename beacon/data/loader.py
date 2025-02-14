import os
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from ..core.base import BeaconBase

class DataLoader(BeaconBase):
    """Base class for data loading"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.data_path = Path(self.config['data_path'])
        self._validate_data_path()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'data_path': './data',
            'file_pattern': '*.csv',
            'batch_size': 32,
            'shuffle': True,
            'validation_split': 0.2,
            'test_split': 0.1,
            'random_seed': 42
        }
    
    def _validate_data_path(self) -> None:
        """Validate data path exists"""
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
            
        if not self.data_path.is_dir():
            raise ValueError(f"Data path is not a directory: {self.data_path}")
    
    def load_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load data from CSV file"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded data from {filepath}: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def load_batch(self, data: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Load a batch of data"""
        if batch_size is None:
            batch_size = self.config['batch_size']
            
        if self.config['shuffle']:
            indices = np.random.permutation(len(data))
            data = data[indices]
            
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def train_val_test_split(
        self, 
        data: np.ndarray,
        val_split: Optional[float] = None,
        test_split: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation and test sets"""
        if val_split is None:
            val_split = self.config['validation_split']
        if test_split is None:
            test_split = self.config['test_split']
            
        if not 0 <= val_split <= 1 or not 0 <= test_split <= 1:
            raise ValueError("Split ratios must be between 0 and 1")
            
        if val_split + test_split >= 1:
            raise ValueError("Sum of validation and test splits must be less than 1")
            
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        
        # Calculate split indices
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        
        test_size = int(test_split * n_samples)
        val_size = int(val_split * n_samples)
        train_size = n_samples - val_size - test_size
        
        # Split data
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return data[train_indices], data[val_indices], data[test_indices]
    
    def get_file_list(self, pattern: Optional[str] = None) -> List[Path]:
        """Get list of files matching pattern"""
        if pattern is None:
            pattern = self.config['file_pattern']
            
        files = list(self.data_path.glob(pattern))
        if not files:
            self.logger.warning(f"No files found matching pattern: {pattern}")
            
        return files
    
    def validate_file(self, filepath: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Validate input file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            return False, f"File does not exist: {filepath}"
            
        if not filepath.is_file():
            return False, f"Not a file: {filepath}"
            
        if filepath.stat().st_size == 0:
            return False, f"Empty file: {filepath}"
            
        return True, None
    
    def validate_data_structure(self) -> None:
        """Validate data directory structure and files"""
        # Check if data directory exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        # Check required files
        required_files = self.get_file_list()
        
        for file in required_files:
            if not file.exists():
                self.logger.warning(f"Required file not found: {file}")
        
        # Check image directory
        image_dir = self.data_path / 'images'
        if not image_dir.exists():
            self.logger.warning(f"Image directory not found: {image_dir}")
    
    def load_clinical_data(self) -> pd.DataFrame:
        """
        Load clinical data
        Returns:
            DataFrame containing clinical data
        """
        file_path = self.data_path / 'clinical.csv'
        try:
            data = pd.read_csv(file_path)
            self.logger.info(f"Loaded clinical data: {len(data)} records")
            return data
        except Exception as e:
            self.logger.error(f"Error loading clinical data: {str(e)}")
            raise
    
    def load_images(self, patient_ids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load medical images
        Args:
            patient_ids: Optional list of patient IDs to load
        Returns:
            Dictionary mapping patient IDs to image arrays
        """
        image_dir = self.data_path / 'images'
        images = {}
        
        if patient_ids is None:
            # Load all images in directory
            pattern = f"*{self.config['image_format']}"
            image_files = list(image_dir.glob(pattern))
        else:
            # Load specific patient images
            image_files = [
                image_dir / f"{pid}{self.config['image_format']}"
                for pid in patient_ids
            ]
        
        for img_path in image_files:
            try:
                # Read image
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                
                # Convert to float32 and normalize
                img = img.astype(np.float32) / 255.0
                
                # Store with patient ID as key
                patient_id = img_path.stem
                images[patient_id] = img
                
            except Exception as e:
                self.logger.error(f"Error loading image {img_path}: {str(e)}")
                continue
        
        self.logger.info(f"Loaded {len(images)} images")
        return images
    
    def load_genomic_data(self) -> pd.DataFrame:
        """
        Load genomic data
        Returns:
            DataFrame containing genomic data
        """
        file_path = self.data_path / 'genomic.csv'
        try:
            data = pd.read_csv(file_path)
            self.logger.info(f"Loaded genomic data: {len(data)} records")
            return data
        except Exception as e:
            self.logger.error(f"Error loading genomic data: {str(e)}")
            raise
    
    def load_survival_data(self) -> pd.DataFrame:
        """
        Load survival data
        Returns:
            DataFrame containing survival times and event indicators
        """
        file_path = self.data_path / 'survival.csv'
        try:
            data = pd.read_csv(file_path)
            self.logger.info(f"Loaded survival data: {len(data)} records")
            return data
        except Exception as e:
            self.logger.error(f"Error loading survival data: {str(e)}")
            raise
    
    def load_labels(self) -> pd.DataFrame:
        """
        Load labels
        Returns:
            DataFrame containing labels
        """
        file_path = self.data_path / 'labels.csv'
        try:
            data = pd.read_csv(file_path)
            self.logger.info(f"Loaded labels: {len(data)} records")
            return data
        except Exception as e:
            self.logger.error(f"Error loading labels: {str(e)}")
            raise
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all data types
        Returns:
            Dictionary containing all data
        """
        # Load all data types
        clinical_data = self.load_clinical_data()
        patient_ids = clinical_data['patient_id'].tolist()
        
        images = self.load_images(patient_ids)
        genomic_data = self.load_genomic_data()
        survival_data = self.load_survival_data()
        labels = self.load_labels()
        
        # Convert images to tensor
        image_tensors = []
        for pid in patient_ids:
            if pid in images:
                image_tensors.append(torch.FloatTensor(images[pid]))
        
        if image_tensors:
            image_tensor = torch.stack(image_tensors)
        else:
            image_tensor = torch.empty(0)
        
        # Ensure all data is aligned by patient ID
        merged_data = clinical_data.merge(
            genomic_data,
            on='patient_id',
            how='left'
        ).merge(
            survival_data,
            on='patient_id',
            how='left'
        ).merge(
            labels,
            on='patient_id',
            how='left'
        )
        
        return {
            'clinical_data': merged_data[clinical_data.columns],
            'images': image_tensor,
            'genomic_data': torch.FloatTensor(
                merged_data.drop(columns=['patient_id']).values
            ),
            'survival_times': torch.FloatTensor(
                survival_data['survival_time'].values
            ),
            'event_indicators': torch.FloatTensor(
                survival_data['event_indicator'].values
            ),
            'labels': torch.LongTensor(labels['label'].values)
        }
    
    def validate_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate loaded data
        Args:
            data: Dictionary containing loaded data
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if all required keys are present
            required_keys = [
                'clinical_data', 'images', 'genomic_data',
                'survival_times', 'event_indicators', 'labels'
            ]
            for key in required_keys:
                if key not in data:
                    return False, f"Missing required data: {key}"
            
            # Check data sizes
            n_samples = len(data['clinical_data'])
            if len(data['labels']) != n_samples:
                return False, "Inconsistent number of samples"
            
            if len(data['survival_times']) != n_samples:
                return False, "Inconsistent number of survival times"
            
            # Check data types
            if not isinstance(data['images'], torch.Tensor):
                return False, "Images must be a torch.Tensor"
            
            if not isinstance(data['labels'], torch.Tensor):
                return False, "Labels must be a torch.Tensor"
            
            return True, ""
            
        except Exception as e:
            return False, str(e)
    
    def save_predictions(self, predictions: Dict[str, np.ndarray], 
                        output_dir: str) -> None:
        """
        Save model predictions
        Args:
            predictions: Dictionary of predictions
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, preds in predictions.items():
            file_path = output_path / f"{name}_predictions.csv"
            pd.DataFrame(preds).to_csv(file_path, index=False)
            self.logger.info(f"Saved predictions to {file_path}") 