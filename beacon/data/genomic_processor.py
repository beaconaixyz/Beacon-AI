import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from ..core.base import BeaconBase
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class GenomicProcessor(BeaconBase):
    """Processor for genomic data analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize genomic processor
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.scaler = None
        self._initialize_scaler()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'normalization': 'standard',
            'handle_missing': True,
            'missing_strategy': 'mean',
            'min_expression': 0.1,
            'max_missing_ratio': 0.3
        }
    
    def _initialize_scaler(self):
        """Initialize data scaler"""
        if self.config['normalization'] == 'standard':
            self.scaler = StandardScaler()
        elif self.config['normalization'] == 'minmax':
            self.scaler = MinMaxScaler()
    
    def process_expression_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Process gene expression data
        Args:
            data: Gene expression data matrix
        Returns:
            Processed expression data
        """
        # Handle missing values
        if self.config['handle_missing']:
            data = self._handle_missing_values(data)
        
        # Filter low expression genes
        data = self._filter_low_expression(data)
        
        # Normalize data
        if self.scaler is not None:
            data = self.scaler.fit_transform(data)
        
        return data
    
    def process_mutation_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Process mutation data
        Args:
            data: Mutation data matrix
        Returns:
            Processed mutation data
        """
        # Convert mutation types to numeric
        data = self._encode_mutation_types(data)
        
        # Handle missing values
        if self.config['handle_missing']:
            data = self._handle_missing_values(data)
        
        return data.values
    
    def process_cnv_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Process copy number variation data
        Args:
            data: CNV data matrix
        Returns:
            Processed CNV data
        """
        # Handle missing values
        if self.config['handle_missing']:
            data = self._handle_missing_values(data)
        
        # Normalize data
        if self.scaler is not None:
            data = self.scaler.fit_transform(data)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data"""
        if self.config['missing_strategy'] == 'mean':
            return data.fillna(data.mean())
        elif self.config['missing_strategy'] == 'median':
            return data.fillna(data.median())
        elif self.config['missing_strategy'] == 'zero':
            return data.fillna(0)
        else:
            raise ValueError(f"Unknown missing value strategy: {self.config['missing_strategy']}")
    
    def _filter_low_expression(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out low expression genes"""
        min_expr = self.config['min_expression']
        return data.loc[:, data.mean() >= min_expr]
    
    def _encode_mutation_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode mutation types to numeric values"""
        mutation_types = {
            'missense': 1,
            'nonsense': 2,
            'silent': 0,
            'frameshift': 3,
            'splice': 4
        }
        return data.replace(mutation_types)
    
    def get_feature_importance(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate feature importance scores
        Args:
            data: Input data matrix
        Returns:
            Feature importance scores
        """
        # Calculate variance of each feature
        importance = data.var()
        
        # Normalize scores
        importance = (importance - importance.min()) / (importance.max() - importance.min())
        
        return importance
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate input data
        Args:
            data: Input data matrix
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for empty data
        if data.empty:
            return False, "Empty data matrix"
        
        # Check missing ratio
        missing_ratio = data.isnull().mean().max()
        if missing_ratio > self.config['max_missing_ratio']:
            return False, f"Too many missing values: {missing_ratio:.2%}"
        
        return True, None 