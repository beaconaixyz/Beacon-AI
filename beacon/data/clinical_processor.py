import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from ..core.base import BeaconBase
from sklearn.preprocessing import StandardScaler, LabelEncoder

class ClinicalProcessor(BeaconBase):
    """Processor for clinical data analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clinical processor
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.numeric_scaler = None
        self.categorical_encoders = {}
        self._initialize_processors()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'numeric_scaling': 'standard',
            'handle_missing': True,
            'missing_strategy': 'mean',
            'categorical_encoding': 'label',
            'max_missing_ratio': 0.3,
            'date_format': '%Y-%m-%d'
        }
    
    def _initialize_processors(self):
        """Initialize data processors"""
        if self.config['numeric_scaling'] == 'standard':
            self.numeric_scaler = StandardScaler()
        elif self.config['numeric_scaling'] == 'minmax':
            self.numeric_scaler = MinMaxScaler()
    
    def process_patient_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process patient clinical data
        Args:
            data: Patient clinical data
        Returns:
            Processed clinical data
        """
        # Handle missing values
        if self.config['handle_missing']:
            data = self._handle_missing_values(data)
        
        # Process different types of columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        date_cols = self._identify_date_columns(data)
        
        # Process each type
        processed_data = pd.DataFrame()
        
        # Process numeric columns
        if len(numeric_cols) > 0:
            processed_data[numeric_cols] = self._process_numeric(data[numeric_cols])
        
        # Process categorical columns
        if len(categorical_cols) > 0:
            processed_cats = self._process_categorical(data[categorical_cols])
            processed_data = pd.concat([processed_data, processed_cats], axis=1)
        
        # Process date columns
        if len(date_cols) > 0:
            processed_dates = self._process_dates(data[date_cols])
            processed_data = pd.concat([processed_data, processed_dates], axis=1)
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data"""
        # Handle numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if self.config['missing_strategy'] == 'mean':
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        elif self.config['missing_strategy'] == 'median':
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        elif self.config['missing_strategy'] == 'zero':
            data[numeric_cols] = data[numeric_cols].fillna(0)
        
        # Handle categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[categorical_cols] = data[categorical_cols].fillna('unknown')
        
        return data
    
    def _process_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process numeric columns"""
        if self.numeric_scaler is not None:
            return pd.DataFrame(
                self.numeric_scaler.fit_transform(data),
                columns=data.columns,
                index=data.index
            )
        return data
    
    def _process_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process categorical columns"""
        processed_data = pd.DataFrame()
        
        for col in data.columns:
            if col not in self.categorical_encoders:
                self.categorical_encoders[col] = LabelEncoder()
            
            processed_data[col] = self.categorical_encoders[col].fit_transform(data[col])
        
        return processed_data
    
    def _process_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process date columns"""
        processed_data = pd.DataFrame()
        reference_date = pd.Timestamp.now()
        
        for col in data.columns:
            dates = pd.to_datetime(data[col], format=self.config['date_format'])
            processed_data[f"{col}_days"] = (dates - reference_date).dt.days
        
        # Normalize days
        if self.numeric_scaler is not None:
            processed_data = pd.DataFrame(
                self.numeric_scaler.fit_transform(processed_data),
                columns=processed_data.columns,
                index=processed_data.index
            )
        
        return processed_data
    
    def _identify_date_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify columns containing dates"""
        date_cols = []
        for col in data.columns:
            try:
                pd.to_datetime(data[col], format=self.config['date_format'])
                date_cols.append(col)
            except:
                continue
        return date_cols
    
    def get_feature_stats(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate feature statistics
        Args:
            data: Input data matrix
        Returns:
            Dictionary of feature statistics
        """
        stats = {
            'missing_ratio': data.isnull().mean(),
            'unique_values': data.nunique(),
        }
        
        # Add numeric statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats.update({
                'mean': data[numeric_cols].mean(),
                'std': data[numeric_cols].std(),
                'min': data[numeric_cols].min(),
                'max': data[numeric_cols].max()
            })
        
        return stats
    
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
        
        # Check date format for date columns
        date_cols = self._identify_date_columns(data)
        for col in date_cols:
            try:
                pd.to_datetime(data[col], format=self.config['date_format'])
            except:
                return False, f"Invalid date format in column {col}"
        
        return True, None 