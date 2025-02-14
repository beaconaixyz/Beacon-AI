from typing import Dict, List, Optional, Union, Any, Type
from ..core.base import BeaconBase
from .processor import DataProcessor
from .transformer import DataTransformer
from .validator import DataValidator
from .augmentation import DataAugmentation
from .cache import DataCache
from .dataset import BeaconDataset
from .iterator import BeaconDataIterator
from .sampler import BeaconSampler
from .pipeline import DataPipeline

class DataFactory(BeaconBase):
    """Factory class for creating data-related objects"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._register_defaults()
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'registered_classes': {
                'processor': DataProcessor,
                'transformer': DataTransformer,
                'validator': DataValidator,
                'augmentation': DataAugmentation,
                'cache': DataCache,
                'dataset': BeaconDataset,
                'iterator': BeaconDataIterator,
                'sampler': BeaconSampler,
                'pipeline': DataPipeline
            },
            'default_configs': {
                'processor': {},
                'transformer': {},
                'validator': {},
                'augmentation': {},
                'cache': {},
                'dataset': {},
                'iterator': {},
                'sampler': {},
                'pipeline': {}
            }
        }
    
    def _register_defaults(self) -> None:
        """Register default classes"""
        self.registered_classes = self.config['registered_classes'].copy()
        self.default_configs = self.config['default_configs'].copy()
    
    def register_class(
        self,
        name: str,
        class_type: Type,
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register new class type"""
        if not issubclass(class_type, BeaconBase):
            raise TypeError(f"Class must inherit from BeaconBase: {class_type}")
            
        self.registered_classes[name] = class_type
        if default_config is not None:
            self.default_configs[name] = default_config
    
    def unregister_class(self, name: str) -> None:
        """Unregister class type"""
        if name in self.registered_classes:
            del self.registered_classes[name]
            if name in self.default_configs:
                del self.default_configs[name]
    
    def create(
        self,
        class_type: str,
        config: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> BeaconBase:
        """Create instance of registered class"""
        if class_type not in self.registered_classes:
            raise ValueError(f"Unknown class type: {class_type}")
            
        # Get class and default config
        cls = self.registered_classes[class_type]
        default_config = self.default_configs.get(class_type, {})
        
        # Merge configs
        if config is not None:
            merged_config = {**default_config, **config}
        else:
            merged_config = default_config
        
        # Create instance
        return cls(merged_config, *args, **kwargs)
    
    def create_processor(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> DataProcessor:
        """Create data processor"""
        return self.create('processor', config)
    
    def create_transformer(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> DataTransformer:
        """Create data transformer"""
        return self.create('transformer', config)
    
    def create_validator(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> DataValidator:
        """Create data validator"""
        return self.create('validator', config)
    
    def create_augmentation(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> DataAugmentation:
        """Create data augmentation"""
        return self.create('augmentation', config)
    
    def create_cache(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> DataCache:
        """Create data cache"""
        return self.create('cache', config)
    
    def create_dataset(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> BeaconDataset:
        """Create dataset"""
        return self.create('dataset', config)
    
    def create_iterator(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> BeaconDataIterator:
        """Create data iterator"""
        return self.create('iterator', config)
    
    def create_sampler(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> BeaconSampler:
        """Create data sampler"""
        return self.create('sampler', config)
    
    def create_pipeline(
        self,
        config: Optional[Dict[str, Any]] = None
    ) -> DataPipeline:
        """Create data pipeline"""
        return self.create('pipeline', config)
    
    def get_registered_classes(self) -> Dict[str, Type]:
        """Get dictionary of registered classes"""
        return self.registered_classes.copy()
    
    def get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of default configurations"""
        return self.default_configs.copy()
    
    def validate_registration(self, name: str, class_type: Type) -> None:
        """Validate class registration"""
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
            
        if not isinstance(class_type, type):
            raise TypeError("Class type must be a class")
            
        if not issubclass(class_type, BeaconBase):
            raise TypeError(f"Class must inherit from BeaconBase: {class_type}")
            
        if name in self.registered_classes:
            raise ValueError(f"Class already registered: {name}")
    
    def create_all(
        self,
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, BeaconBase]:
        """Create instances of all registered classes"""
        instances = {}
        configs = configs or {}
        
        for name in self.registered_classes:
            config = configs.get(name)
            instances[name] = self.create(name, config)
            
        return instances 