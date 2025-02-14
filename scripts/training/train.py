#!/usr/bin/env python3

"""
Training Script for BEACON

This script implements model training functionality with support for different model architectures
and data types.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split

class BeaconDataset(Dataset):
    """Custom dataset for BEACON data"""

    def __init__(self, data_dir: Path, data_type: str, mode: str = 'train'):
        """Initialize dataset.

        Args:
            data_dir: Directory containing processed data
            data_type: Type of data ('clinical', 'imaging', 'genomic', or 'survival')
            mode: 'train' or 'val'
        """
        self.data_dir = data_dir
        self.data_type = data_type
        self.mode = mode
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load processed data.

        Returns:
            Dictionary containing features and labels
        """
        if self.data_type == 'clinical':
            data = pd.read_csv(self.data_dir / 'clinical_processed.csv')
            features = data.drop(['diabetes', 'hypertension'], axis=1).values
            labels = data[['diabetes', 'hypertension']].values
            
        elif self.data_type == 'imaging':
            features = np.load(self.data_dir / 'imaging_processed.npy')
            labels = np.load(self.data_dir / 'imaging_1.npy')
            
        elif self.data_type == 'genomic':
            data = dict(np.load(self.data_dir / 'genomic_processed.npz'))
            features = np.concatenate([
                data['expression'],
                data['mutations'],
                data['cnv']
            ], axis=1)
            # For demonstration, we'll use expression values > threshold as labels
            labels = (data['expression'].mean(axis=1) > 0).astype(int)
            
        elif self.data_type == 'survival':
            data = pd.read_csv(self.data_dir / 'survival_processed.csv')
            features = data.drop(['time', 'event'], axis=1).values
            labels = data[['time', 'event']].values
            
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

        return {'features': features, 'labels': labels}

    def __len__(self) -> int:
        return len(self.data['features'])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.FloatTensor(self.data['features'][idx])
        labels = torch.FloatTensor(self.data['labels'][idx])
        return features, labels

class BaseModel(nn.Module):
    """Base model class"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ModelTrainer:
    """Handle model training and evaluation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = Path(config.get('data_dir', 'processed_data'))
        self.output_dir = Path(config.get('output_dir', 'models'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_dataloaders(self, data_type: str) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders.

        Args:
            data_type: Type of data to use

        Returns:
            Tuple of (train_loader, val_loader)
        """
        dataset = BeaconDataset(self.data_dir, data_type)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

        return train_loader, val_loader

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """Train for one epoch.

        Args:
            model: Model to train
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer

        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0.0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                self.logger.info(f'Training batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        return total_loss / len(train_loader)

    def validate(self, model: nn.Module, val_loader: DataLoader,
                criterion: nn.Module) -> float:
        """Validate the model.

        Args:
            model: Model to validate
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Validation loss
        """
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, data_type: str) -> None:
        """Train model for specified data type.

        Args:
            data_type: Type of data to train on
        """
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(data_type)
        
        # Get input and output dimensions
        sample_features, sample_labels = next(iter(train_loader))
        input_dim = sample_features.shape[1]
        output_dim = sample_labels.shape[1]

        # Create model
        model = BaseModel(input_dim, output_dim).to(self.device)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )

        # Training loop
        n_epochs = self.config.get('n_epochs', 100)
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self.validate(model, val_loader, criterion)
            
            self.logger.info(f'Epoch {epoch+1}/{n_epochs}:')
            self.logger.info(f'Training Loss: {train_loss:.4f}')
            self.logger.info(f'Validation Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.output_dir / f'{data_type}_model.pt')
                self.logger.info('Saved new best model')

def main():
    parser = argparse.ArgumentParser(description="Train models for BEACON")
    parser.add_argument("--data-dir", default="processed_data",
                      help="Directory containing processed data")
    parser.add_argument("--output-dir", default="models",
                      help="Output directory for trained models")
    parser.add_argument("--data-type", required=True,
                      choices=['clinical', 'imaging', 'genomic', 'survival'],
                      help="Type of data to train on")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--n-epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                      help="Learning rate")
    args = parser.parse_args()

    config = vars(args)
    trainer = ModelTrainer(config)
    trainer.train(args.data_type)

if __name__ == "__main__":
    main() 