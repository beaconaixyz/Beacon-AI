import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BeaconModel
from typing import Dict, Any, List

class MedicalImageCNN(BeaconModel):
    """CNN model for medical image analysis"""
    
    def _build_model(self) -> nn.Module:
        """
        Build CNN architecture for medical image analysis
        Returns:
            PyTorch CNN model
        """
        class CNNArchitecture(nn.Module):
            def __init__(self, in_channels, num_classes):
                super().__init__()
                
                # First convolutional block
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                # Second convolutional block
                self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                # Third convolutional block
                self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                # Adaptive pooling to handle different input sizes
                self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
                
                # Fully connected layers
                self.fc = nn.Sequential(
                    nn.Linear(128 * 4 * 4, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                # Convolutional layers
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                
                # Adaptive pooling
                x = self.adaptive_pool(x)
                
                # Flatten
                x = x.view(x.size(0), -1)
                
                # Fully connected layers
                x = self.fc(x)
                return x
        
        model = CNNArchitecture(
            in_channels=self.config.get('in_channels', 1),
            num_classes=self.config['num_classes']
        ).to(self.device)
        
        self.logger.info(f"Created CNN model with architecture:\n{model}")
        return model
    
    def train(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        """
        Train the model
        Args:
            data: Dictionary containing images and labels
            **kwargs: Additional training parameters
        Returns:
            Training history
        """
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        criterion = nn.CrossEntropyLoss()
        
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = self._create_batches(data, batch_size)
            
            for batch in batches:
                optimizer.zero_grad()
                outputs = self.model(batch['images'])
                loss = criterion(outputs, batch['labels'])
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(batches)
            history.append({
                'epoch': epoch + 1,
                'loss': avg_loss
            })
            
            self.logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return history
    
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions
        Args:
            data: Input images
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data.to(self.device))
            return F.softmax(predictions, dim=1)
    
    def _create_batches(self, data: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Create batches from data
        Args:
            data: Input data dictionary
            batch_size: Size of each batch
        Returns:
            List of batch dictionaries
        """
        n_samples = len(data['images'])
        indices = torch.randperm(n_samples)
        batches = []
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = {
                'images': data['images'][batch_indices].to(self.device),
                'labels': data['labels'][batch_indices].to(self.device)
            }
            batches.append(batch)
        
        return batches 