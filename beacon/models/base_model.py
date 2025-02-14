import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from tqdm import tqdm
from ..core.base import BeaconBase

class BaseModel(BeaconBase):
    """Base class for all models"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()
        self.device = torch.device(self.config['device'])
        self.model.to(self.device)
        self.history = {'train_loss': [], 'val_loss': [], 
                       'train_acc': [], 'val_acc': []}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'criterion': 'cross_entropy',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_epochs': 100,
            'early_stopping_patience': 10,
            'model_save_path': 'model.pth'
        }
    
    def _build_model(self) -> nn.Module:
        """Build model architecture"""
        raise NotImplementedError("Subclass must implement _build_model")
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer"""
        if self.config['optimizer'].lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'].lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
    
    def _build_criterion(self) -> nn.Module:
        """Build loss criterion"""
        if self.config['criterion'].lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.config['criterion'].lower() == 'bce':
            return nn.BCEWithLogitsLoss()
        elif self.config['criterion'].lower() == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion: {self.config['criterion']}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train one epoch
        Args:
            train_loader: Training data loader
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            # Get batch data
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch
            else:
                inputs = batch
                targets = None
            
            # Move to device
            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            if targets is not None:
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy for classification
                if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.BCEWithLogitsLoss)):
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model
        Args:
            val_loader: Validation data loader
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Get batch data
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None
                
                # Move to device
                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # Calculate accuracy for classification
                    if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.BCEWithLogitsLoss)):
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def fit(self, train_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None,
            callbacks: Optional[List[Any]] = None) -> Dict[str, List[float]]:
        """
        Train model
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            callbacks: Optional list of callbacks
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate if validation loader is provided
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(self.config['model_save_path'])
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if val_loader is not None:
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Execute callbacks
            if callbacks is not None:
                for callback in callbacks:
                    callback(self, epoch)
        
        return self.history
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make predictions
        Args:
            inputs: Input tensor
        Returns:
            Predictions
        """
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            if isinstance(self.criterion, (nn.CrossEntropyLoss, nn.BCEWithLogitsLoss)):
                outputs = torch.softmax(outputs, dim=1)
        return outputs
    
    def save_model(self, path: str):
        """
        Save model state
        Args:
            path: Save path
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    def load_model(self, path: str):
        """
        Load model state
        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config.update(checkpoint['config'])
        self.history = checkpoint['history']
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_structure': str(self.model),
            'optimizer': str(self.optimizer),
            'criterion': str(self.criterion),
            'device': str(self.device),
            'config': self.config
        } 