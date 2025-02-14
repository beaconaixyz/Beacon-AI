import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from ..core.base import BeaconBase
from .processor import DataProcessor
from .transformer import DataTransformer
from .validator import DataValidator
from .augmentation import DataAugmentation
from .cache import DataCache

class DataPipeline(BeaconBase):
    """Base class for data processing pipelines"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._initialize_components()
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'enabled': True,
            'cache_enabled': True,
            'validate_data': True,
            'processor_config': {},
            'transformer_config': {},
            'validator_config': {},
            'augmentation_config': {},
            'cache_config': {},
            'steps': [
                'validate',
                'process',
                'transform',
                'augment'
            ]
        }
    
    def _initialize_components(self) -> None:
        """Initialize pipeline components"""
        if not self.config['enabled']:
            return
            
        # Initialize components
        self.processor = DataProcessor(self.config['processor_config'])
        self.transformer = DataTransformer(self.config['transformer_config'])
        self.validator = DataValidator(self.config['validator_config'])
        self.augmentation = DataAugmentation(self.config['augmentation_config'])
        
        if self.config['cache_enabled']:
            self.cache = DataCache(self.config['cache_config'])
        else:
            self.cache = None
    
    def run(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Run data through pipeline"""
        if not self.config['enabled']:
            return data
            
        # Try to load from cache first
        if self.cache is not None:
            cached_data = self.cache.load(data, prefix='pipeline')
            if cached_data is not None:
                return cached_data
        
        # Process data through pipeline steps
        processed_data = data
        for step in self.config['steps']:
            processed_data = self._run_step(step, processed_data)
        
        # Cache results
        if self.cache is not None:
            self.cache.save(processed_data, prefix='pipeline')
        
        return processed_data
    
    def _run_step(
        self,
        step: str,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Run single pipeline step"""
        if step == 'validate' and self.config['validate_data']:
            valid, errors = self.validator.validate(data)
            if not valid:
                raise ValueError(f"Data validation failed:\n{'\n'.join(errors)}")
                
        elif step == 'process':
            data = self.processor.fit_transform(data)
            
        elif step == 'transform':
            data = self.transformer.fit_transform(data)
            
        elif step == 'augment':
            data = self.augmentation.augment(data)
            
        else:
            raise ValueError(f"Unknown pipeline step: {step}")
            
        return data
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> None:
        """Fit pipeline components to data"""
        if not self.config['enabled']:
            return
            
        # Validate data if enabled
        if self.config['validate_data']:
            valid, errors = self.validator.validate(data)
            if not valid:
                raise ValueError(f"Data validation failed:\n{'\n'.join(errors)}")
        
        # Fit components in order
        if 'process' in self.config['steps']:
            self.processor.fit(data)
            data = self.processor.transform(data)
            
        if 'transform' in self.config['steps']:
            self.transformer.fit(data)
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data using fitted pipeline"""
        if not self.config['enabled']:
            return data
            
        # Try to load from cache first
        if self.cache is not None:
            cached_data = self.cache.load(data, prefix='transform')
            if cached_data is not None:
                return cached_data
        
        # Transform data through pipeline steps
        transformed_data = data
        for step in self.config['steps']:
            if step == 'process':
                transformed_data = self.processor.transform(transformed_data)
            elif step == 'transform':
                transformed_data = self.transformer.transform(transformed_data)
            elif step == 'augment':
                transformed_data = self.augmentation.augment(transformed_data)
        
        # Cache results
        if self.cache is not None:
            self.cache.save(transformed_data, prefix='transform')
        
        return transformed_data
    
    def fit_transform(
        self,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit pipeline and transform data"""
        self.fit(data)
        return self.transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation"""
        if isinstance(self.transformer, DataTransformer):
            return self.transformer.get_feature_names()
        return []
    
    def get_step_names(self) -> List[str]:
        """Get names of enabled pipeline steps"""
        return self.config['steps']
    
    def add_step(self, step: str, position: Optional[int] = None) -> None:
        """Add step to pipeline"""
        if step not in ['validate', 'process', 'transform', 'augment']:
            raise ValueError(f"Invalid step name: {step}")
            
        if position is None:
            self.config['steps'].append(step)
        else:
            self.config['steps'].insert(position, step)
    
    def remove_step(self, step: str) -> None:
        """Remove step from pipeline"""
        if step in self.config['steps']:
            self.config['steps'].remove(step)
    
    def clear_cache(self) -> None:
        """Clear pipeline cache"""
        if self.cache is not None:
            self.cache.clear()
    
    def validate_config(self) -> Tuple[bool, Optional[str]]:
        """Validate pipeline configuration"""
        if not isinstance(self.config['enabled'], bool):
            return False, "enabled must be a boolean"
            
        if not isinstance(self.config['cache_enabled'], bool):
            return False, "cache_enabled must be a boolean"
            
        if not isinstance(self.config['validate_data'], bool):
            return False, "validate_data must be a boolean"
            
        if not isinstance(self.config['steps'], list):
            return False, "steps must be a list"
            
        valid_steps = {'validate', 'process', 'transform', 'augment'}
        invalid_steps = set(self.config['steps']) - valid_steps
        if invalid_steps:
            return False, f"Invalid steps: {invalid_steps}"
        
        return True, None 