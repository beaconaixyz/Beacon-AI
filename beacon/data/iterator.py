import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from torch.utils.data import DataLoader, Sampler
from ..core.base import BeaconBase
from .dataset import BeaconDataset

class BeaconDataIterator(BeaconBase):
    """Base class for data iteration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 4,
            'pin_memory': True,
            'drop_last': False,
            'sampler': None,
            'collate_fn': None
        }
    
    def setup(self, dataset: BeaconDataset) -> None:
        """Setup data loaders"""
        self.dataset = dataset
        
        # Create data loaders for each split
        self.train_loader = self._create_loader('train')
        self.val_loader = self._create_loader('val')
        self.test_loader = self._create_loader('test')
    
    def _create_loader(self, split: str) -> DataLoader:
        """Create data loader for specified split"""
        if self.dataset is None:
            raise ValueError("No dataset provided. Call setup with a dataset first")
            
        # Get indices for the split
        indices = self.dataset.get_split_indices(split)
        
        # Create sampler if specified
        sampler = None
        if self.config['sampler'] is not None:
            sampler = self.config['sampler'](indices)
        
        # Create loader
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=self.config['shuffle'] and sampler is None,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            drop_last=self.config['drop_last'],
            sampler=sampler,
            collate_fn=self.config['collate_fn']
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader"""
        if self.train_loader is None:
            raise ValueError("Train loader not created. Call setup first")
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader"""
        if self.val_loader is None:
            raise ValueError("Validation loader not created. Call setup first")
        return self.val_loader
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader"""
        if self.test_loader is None:
            raise ValueError("Test loader not created. Call setup first")
        return self.test_loader
    
    def get_all_loaders(self) -> Dict[str, DataLoader]:
        """Get all data loaders"""
        return {
            'train': self.get_train_loader(),
            'val': self.get_val_loader(),
            'test': self.get_test_loader()
        }
    
    def get_num_batches(self, split: str) -> int:
        """Get number of batches for specified split"""
        loader = self.get_loader(split)
        return len(loader)
    
    def get_loader(self, split: str) -> DataLoader:
        """Get loader for specified split"""
        if split == 'train':
            return self.get_train_loader()
        elif split == 'val':
            return self.get_val_loader()
        elif split == 'test':
            return self.get_test_loader()
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def iterate(self, split: str) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over data in specified split"""
        loader = self.get_loader(split)
        for batch in loader:
            yield batch
    
    def validate_loader(self, loader: DataLoader) -> Tuple[bool, Optional[str]]:
        """Validate data loader"""
        if loader is None:
            return False, "Loader is None"
            
        if not isinstance(loader, DataLoader):
            return False, "Invalid loader type"
            
        if len(loader) == 0:
            return False, "Empty loader"
            
        return True, None
    
    def validate_setup(self) -> Tuple[bool, Optional[str]]:
        """Validate iterator setup"""
        if self.dataset is None:
            return False, "No dataset provided"
            
        # Validate all loaders
        for split, loader in self.get_all_loaders().items():
            valid, message = self.validate_loader(loader)
            if not valid:
                return False, f"Invalid {split} loader: {message}"
        
        return True, None 