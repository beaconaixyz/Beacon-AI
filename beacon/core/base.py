import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import os
import json
import yaml
from pathlib import Path

class BeaconBase(ABC):
    """Base class for all Beacon components"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base component
        Args:
            config: Configuration dictionary
        """
        self.config = self._validate_config(config)
        self.logger = self._setup_logger()
        
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration dictionary
        Args:
            config: Input configuration
        Returns:
            Validated configuration
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Create a copy to avoid modifying the input
        validated_config = config.copy()
        
        # Add default values if not present
        defaults = self._get_default_config()
        for key, value in defaults.items():
            if key not in validated_config:
                validated_config[key] = value
                
        return validated_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        Returns:
            Dictionary of default configuration values
        """
        return {
            'log_level': 'INFO',
            'save_format': 'torch'
        }
    
    def _setup_logger(self) -> logging.Logger:
        """
        Setup logging for the component
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.__class__.__name__)
        
        # Set log level from config
        log_level = getattr(logging, self.config['log_level'].upper())
        logger.setLevel(log_level)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler if log_file is specified
            if 'log_file' in self.config:
                file_handler = logging.FileHandler(self.config['log_file'])
                file_handler.setFormatter(console_formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def save_config(self, path: str) -> None:
        """
        Save configuration to file
        Args:
            path: Path to save configuration
        """
        path = Path(path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=4)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def load_config(self, path: str) -> None:
        """
        Load configuration from file
        Args:
            path: Path to configuration file
        """
        path = Path(path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        # Load based on file extension
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config = json.load(f)
        elif path.suffix in ['.yml', '.yaml']:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Update configuration
        self.config.update(config)
        self._validate_config(self.config)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration
        Returns:
            Configuration dictionary
        """
        return self.config
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set configuration
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        self._validate_config(self.config)
    
    def reset_config(self):
        """Reset configuration to default"""
        self.config = self._get_default_config()
        self._validate_config(self.config)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema
        Returns:
            Schema dictionary
        """
        return {
            'type': 'object',
            'properties': {},
            'required': []
        }
    
    def validate_config_schema(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema
        Args:
            config: Configuration to validate
        Returns:
            True if valid
        """
        try:
            import jsonschema
            schema = self.get_config_schema()
            jsonschema.validate(config, schema)
            return True
        except ImportError:
            print("jsonschema package not found. Schema validation skipped.")
            return True
        except jsonschema.exceptions.ValidationError:
            return False
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configurations
        Args:
            configs: Configurations to merge
        Returns:
            Merged configuration
        """
        merged = self._get_default_config()
        for config in configs:
            merged.update(config)
        return merged
    
    def get_nested_config(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value
        Args:
            key_path: Dot-separated path to key
            default: Default value if key not found
        Returns:
            Configuration value
        """
        current = self.config
        for key in key_path.split('.'):
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                return default
        return current
    
    def set_nested_config(self, key_path: str, value: Any):
        """
        Set nested configuration value
        Args:
            key_path: Dot-separated path to key
            value: Value to set
        """
        keys = key_path.split('.')
        current = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        self._validate_config(self.config)
    
    def delete_config_key(self, key_path: str):
        """
        Delete configuration key
        Args:
            key_path: Dot-separated path to key
        """
        keys = key_path.split('.')
        current = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                return
            current = current[key]
        
        # Delete the key if it exists
        if keys[-1] in current:
            del current[keys[-1]]
            self._validate_config(self.config)
    
    def get_config_diff(self, other_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get difference between current and other configuration
        Args:
            other_config: Configuration to compare with
        Returns:
            Dictionary of differences
        """
        def get_diff(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
            diff = {}
            for k in set(d1.keys()) | set(d2.keys()):
                if k not in d1:
                    diff[k] = ('added', d2[k])
                elif k not in d2:
                    diff[k] = ('removed', d1[k])
                elif d1[k] != d2[k]:
                    if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                        nested_diff = get_diff(d1[k], d2[k])
                        if nested_diff:
                            diff[k] = ('modified', nested_diff)
                    else:
                        diff[k] = ('modified', (d1[k], d2[k]))
            return diff
        
        return get_diff(self.config, other_config)

class BeaconModel(BeaconBase):
    """Base class for all cancer analysis models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.device = self._setup_device()
        self.model = self._build_model()
        self.optimizer = None
        self.scheduler = None
    
    def _setup_device(self) -> torch.device:
        """
        Setup computation device
        Returns:
            PyTorch device
        """
        if self.config.get('device') == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        return device
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        config = super()._get_default_config()
        config.update({
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 10,
            'device': 'cuda',
            'optimizer': 'adam',
            'scheduler': None,
            'early_stopping_patience': None,
            'model_dir': 'models'
        })
        return config
    
    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Make predictions"""
        pass
    
    def save(self, path: str) -> None:
        """
        Save model state
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare state dict
        state_dict = {
            'model_state': self.model.state_dict(),
            'config': self.config
        }
        
        if self.optimizer is not None:
            state_dict['optimizer_state'] = self.optimizer.state_dict()
        
        if self.scheduler is not None:
            state_dict['scheduler_state'] = self.scheduler.state_dict()
        
        # Save the state
        torch.save(state_dict, path)
        self.logger.info(f"Model saved to {path}")
        
        # Save config separately for easy access
        config_path = Path(path).with_suffix('.json')
        self.save_config(str(config_path))
    
    def load(self, path: str) -> None:
        """
        Load model state
        Args:
            path: Path to model file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        state_dict = torch.load(path, map_location=self.device)
        
        # Load configuration
        self.config = self._validate_config(state_dict['config'])
        
        # Rebuild model with loaded config
        self.model = self._build_model()
        self.model.load_state_dict(state_dict['model_state'])
        
        # Load optimizer state if available
        if 'optimizer_state' in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
        
        # Load scheduler state if available
        if 'scheduler_state' in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler_state'])
        
        self.logger.info(f"Model loaded from {path}")
    
    def to(self, device: Union[str, torch.device]) -> None:
        """
        Move model to specified device
        Args:
            device: Target device
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model.to(device)
        self.logger.info(f"Model moved to {device}")
