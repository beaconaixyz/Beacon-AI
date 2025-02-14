import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from ..core.base import BeaconBase
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DataTransformer(BeaconBase):
    """Base class for data transformation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.label_encoders = {}
        self.onehot_encoders = {}
        self._initialize_encoders()
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'categorical_columns': [],
            'numerical_columns': [],
            'label_columns': [],
            'onehot_columns': [],
            'drop_columns': [],
            'date_columns': [],
            'date_format': '%Y-%m-%d',
            'fill_missing': True,
            'missing_strategy': 'mean'
        }
    
    def _initialize_encoders(self):
        """Initialize encoders for categorical variables"""
        for col in self.config['label_columns']:
            self.label_encoders[col] = LabelEncoder()
            
        for col in self.config['onehot_columns']:
            self.onehot_encoders[col] = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fit transformer to data"""
        # Fit label encoders
        for col in self.config['label_columns']:
            if col in df.columns:
                self.label_encoders[col].fit(df[col].astype(str))
        
        # Fit onehot encoders
        for col in self.config['onehot_columns']:
            if col in df.columns:
                self.onehot_encoders[col].fit(df[[col]])
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        df = df.copy()
        
        # Handle missing values
        if self.config['fill_missing']:
            df = self._fill_missing_values(df)
        
        # Transform date columns
        for col in self.config['date_columns']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=self.config['date_format'])
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df = df.drop(columns=[col])
        
        # Transform categorical columns
        for col in self.config['label_columns']:
            if col in df.columns:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Apply one-hot encoding
        for col in self.config['onehot_columns']:
            if col in df.columns:
                onehot = self.onehot_encoders[col].transform(df[[col]])
                onehot_df = pd.DataFrame(
                    onehot,
                    columns=[f'{col}_{cat}' for cat in self.onehot_encoders[col].categories_[0]],
                    index=df.index
                )
                df = pd.concat([df, onehot_df], axis=1)
                df = df.drop(columns=[col])
        
        # Drop specified columns
        df = df.drop(columns=self.config['drop_columns'], errors='ignore')
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data"""
        self.fit(df)
        return self.transform(df)
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in dataframe"""
        df = df.copy()
        
        if self.config['missing_strategy'] == 'mean':
            for col in self.config['numerical_columns']:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())
        elif self.config['missing_strategy'] == 'median':
            for col in self.config['numerical_columns']:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
        elif self.config['missing_strategy'] == 'mode':
            for col in self.config['categorical_columns']:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])
        elif self.config['missing_strategy'] == 'zero':
            df = df.fillna(0)
        
        return df
    
    def inverse_transform_labels(self, labels: np.ndarray, column: str) -> np.ndarray:
        """Inverse transform encoded labels"""
        if column not in self.label_encoders:
            raise ValueError(f"No label encoder found for column: {column}")
            
        return self.label_encoders[column].inverse_transform(labels)
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature names after transformation"""
        features = []
        
        # Add numerical columns
        features.extend([col for col in self.config['numerical_columns'] if col in df.columns])
        
        # Add transformed date columns
        for col in self.config['date_columns']:
            if col in df.columns:
                features.extend([f'{col}_year', f'{col}_month', f'{col}_day'])
        
        # Add label encoded columns
        features.extend([col for col in self.config['label_columns'] if col in df.columns])
        
        # Add one-hot encoded columns
        for col in self.config['onehot_columns']:
            if col in df.columns:
                features.extend([
                    f'{col}_{cat}' for cat in self.onehot_encoders[col].categories_[0]
                ])
        
        return features
    
    def validate_columns(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Validate column configuration"""
        all_cols = (
            self.config['categorical_columns'] +
            self.config['numerical_columns'] +
            self.config['label_columns'] +
            self.config['onehot_columns'] +
            self.config['date_columns']
        )
        
        # Check for duplicate columns
        if len(all_cols) != len(set(all_cols)):
            return False, "Duplicate columns in configuration"
        
        # Check if all configured columns exist in dataframe
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns in dataframe: {missing_cols}"
        
        return True, None 