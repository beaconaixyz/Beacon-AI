import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from ..core.base import BeaconBase

class DataValidator(BeaconBase):
    """Base class for data validation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validators = {}
        self._initialize_validators()
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': True,
            'raise_on_error': False,
            'log_errors': True,
            'validators': {
                'missing_values': {
                    'enabled': True,
                    'threshold': 0.5  # Max allowed missing ratio
                },
                'outliers': {
                    'enabled': True,
                    'method': 'zscore',  # 'zscore' or 'iqr'
                    'threshold': 3.0
                },
                'data_types': {
                    'enabled': True,
                    'numeric_columns': [],
                    'categorical_columns': [],
                    'datetime_columns': []
                },
                'value_range': {
                    'enabled': True,
                    'ranges': {}  # column_name: (min, max)
                },
                'unique_values': {
                    'enabled': True,
                    'columns': []  # Columns that should have unique values
                },
                'correlations': {
                    'enabled': True,
                    'threshold': 0.95  # Max allowed correlation
                }
            }
        }
    
    def _initialize_validators(self) -> None:
        """Initialize validation functions"""
        if not self.config['enabled']:
            return
            
        validator_config = self.config['validators']
        
        if validator_config['missing_values']['enabled']:
            self.validators['missing_values'] = self._validate_missing_values
            
        if validator_config['outliers']['enabled']:
            self.validators['outliers'] = self._validate_outliers
            
        if validator_config['data_types']['enabled']:
            self.validators['data_types'] = self._validate_data_types
            
        if validator_config['value_range']['enabled']:
            self.validators['value_range'] = self._validate_value_ranges
            
        if validator_config['unique_values']['enabled']:
            self.validators['unique_values'] = self._validate_unique_values
            
        if validator_config['correlations']['enabled']:
            self.validators['correlations'] = self._validate_correlations
    
    def validate(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[bool, List[str]]:
        """Validate data using all enabled validators"""
        if not self.config['enabled']:
            return True, []
            
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            msg = f"Invalid data type: {type(data)}"
            if self.config['raise_on_error']:
                raise ValueError(msg)
            return False, [msg]
        
        errors = []
        for name, validator in self.validators.items():
            try:
                valid, message = validator(data)
                if not valid:
                    errors.append(f"{name}: {message}")
                    if self.config['log_errors']:
                        self.logger.error(f"Validation error - {name}: {message}")
            except Exception as e:
                msg = f"Error in {name} validator: {str(e)}"
                errors.append(msg)
                if self.config['log_errors']:
                    self.logger.error(msg)
        
        is_valid = len(errors) == 0
        if not is_valid and self.config['raise_on_error']:
            raise ValueError("\n".join(errors))
            
        return is_valid, errors
    
    def _validate_missing_values(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[bool, Optional[str]]:
        """Validate missing values"""
        config = self.config['validators']['missing_values']
        
        if isinstance(data, np.ndarray):
            missing_ratio = np.mean(np.isnan(data))
        else:
            missing_ratio = data.isnull().mean().mean()
            
        if missing_ratio > config['threshold']:
            return False, f"Missing value ratio ({missing_ratio:.2%}) exceeds threshold ({config['threshold']:.2%})"
            
        return True, None
    
    def _validate_outliers(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[bool, Optional[str]]:
        """Validate outliers"""
        config = self.config['validators']['outliers']
        
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data
            
        if config['method'] == 'zscore':
            z_scores = np.abs((numeric_data - np.mean(numeric_data)) / np.std(numeric_data))
            outliers = np.any(z_scores > config['threshold'])
        elif config['method'] == 'iqr':
            Q1 = np.percentile(numeric_data, 25, axis=0)
            Q3 = np.percentile(numeric_data, 75, axis=0)
            IQR = Q3 - Q1
            outliers = np.any((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR)))
        else:
            return False, f"Unknown outlier detection method: {config['method']}"
            
        if outliers:
            return False, f"Outliers detected using {config['method']} method"
            
        return True, None
    
    def _validate_data_types(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[bool, Optional[str]]:
        """Validate data types"""
        config = self.config['validators']['data_types']
        
        if not isinstance(data, pd.DataFrame):
            return True, None  # Skip for numpy arrays
            
        errors = []
        
        # Check numeric columns
        for col in config['numeric_columns']:
            if col in data.columns and not np.issubdtype(data[col].dtype, np.number):
                errors.append(f"Column {col} should be numeric")
        
        # Check categorical columns
        for col in config['categorical_columns']:
            if col in data.columns and not pd.api.types.is_categorical_dtype(data[col]):
                errors.append(f"Column {col} should be categorical")
        
        # Check datetime columns
        for col in config['datetime_columns']:
            if col in data.columns and not pd.api.types.is_datetime64_any_dtype(data[col]):
                errors.append(f"Column {col} should be datetime")
        
        if errors:
            return False, "; ".join(errors)
            
        return True, None
    
    def _validate_value_ranges(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[bool, Optional[str]]:
        """Validate value ranges"""
        config = self.config['validators']['value_range']
        
        if not isinstance(data, pd.DataFrame):
            return True, None  # Skip for numpy arrays
            
        errors = []
        
        for col, (min_val, max_val) in config['ranges'].items():
            if col in data.columns:
                if data[col].min() < min_val:
                    errors.append(f"Column {col} has values below minimum {min_val}")
                if data[col].max() > max_val:
                    errors.append(f"Column {col} has values above maximum {max_val}")
        
        if errors:
            return False, "; ".join(errors)
            
        return True, None
    
    def _validate_unique_values(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[bool, Optional[str]]:
        """Validate unique values"""
        config = self.config['validators']['unique_values']
        
        if not isinstance(data, pd.DataFrame):
            return True, None  # Skip for numpy arrays
            
        errors = []
        
        for col in config['columns']:
            if col in data.columns and data[col].nunique() != len(data):
                errors.append(f"Column {col} contains duplicate values")
        
        if errors:
            return False, "; ".join(errors)
            
        return True, None
    
    def _validate_correlations(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[bool, Optional[str]]:
        """Validate correlations"""
        config = self.config['validators']['correlations']
        
        if isinstance(data, pd.DataFrame):
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data
            
        if numeric_data.shape[1] < 2:
            return True, None  # Skip if less than 2 features
            
        corr_matrix = np.corrcoef(numeric_data.T)
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlations
        
        if np.any(np.abs(corr_matrix) > config['threshold']):
            return False, f"High correlations detected (threshold: {config['threshold']})"
            
        return True, None
    
    def add_validator(self, name: str, validator: Callable) -> None:
        """Add custom validator function"""
        if not self.config['enabled']:
            return
            
        self.validators[name] = validator
    
    def remove_validator(self, name: str) -> None:
        """Remove validator by name"""
        if name in self.validators:
            del self.validators[name]
    
    def get_validator_names(self) -> List[str]:
        """Get names of enabled validators"""
        return list(self.validators.keys())
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate configuration"""
        if not isinstance(self.config['enabled'], bool):
            return False, "enabled must be a boolean"
            
        if not isinstance(self.config['raise_on_error'], bool):
            return False, "raise_on_error must be a boolean"
            
        if not isinstance(self.config['log_errors'], bool):
            return False, "log_errors must be a boolean"
            
        # Validate validator configs
        for name, config in self.config['validators'].items():
            if not isinstance(config['enabled'], bool):
                return False, f"{name}.enabled must be a boolean"
        
        return True, None 