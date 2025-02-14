"""
Model optimization and version management for BEACON

This module implements model tuning, evaluation metrics, and version management
functionality for cancer-specific models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    brier_score_loss
)
from lifelines.utils import concordance_index
import optuna
from copy import deepcopy
import hashlib
import pickle

@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    timestamp: datetime
    model_type: str
    cancer_type: str
    architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    checksum: str

class ModelOptimizer:
    """Model optimization and tuning class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def optimize_hyperparameters(self, model_class: type,
                               train_data: Dict[str, torch.Tensor],
                               val_data: Dict[str, torch.Tensor],
                               n_trials: int = 50) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna.

        Args:
            model_class: Model class to optimize
            train_data: Training data dictionary
            val_data: Validation data dictionary
            n_trials: Number of optimization trials

        Returns:
            Dictionary of optimal hyperparameters
        """
        def objective(trial):
            # Define hyperparameter search space
            hp = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'hidden_dims': trial.suggest_categorical('hidden_dims', [
                    [128, 64], [256, 128], [512, 256], [1024, 512]
                ])
            }
            
            # Create model with trial hyperparameters
            model = model_class(hp).to(self.device)
            
            # Train model
            train_loader = self._create_data_loader(train_data, hp['batch_size'])
            val_loader = self._create_data_loader(val_data, hp['batch_size'])
            
            optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(100):  # Max 100 epochs
                # Training
                model.train()
                train_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = self._compute_loss(model, batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        loss = self._compute_loss(model, batch)
                        val_loss += loss.item()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            return best_val_loss
        
        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

    def train_model(self, model: nn.Module, train_data: Dict[str, torch.Tensor],
                   val_data: Dict[str, torch.Tensor],
                   hyperparameters: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Train model with given hyperparameters.

        Args:
            model: Model to train
            train_data: Training data dictionary
            val_data: Validation data dictionary
            hyperparameters: Hyperparameters dictionary

        Returns:
            Tuple of (trained model, training history)
        """
        model = model.to(self.device)
        train_loader = self._create_data_loader(train_data, hyperparameters['batch_size'])
        val_loader = self._create_data_loader(val_data, hyperparameters['batch_size'])
        
        optimizer = torch.optim.Adam(model.parameters(),
                                   lr=hyperparameters['learning_rate'])
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        best_val_loss = float('inf')
        best_model = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):  # Max 200 epochs
            # Training
            model.train()
            train_loss = 0
            train_predictions = []
            train_targets = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                loss, preds, targets = self._forward_pass(model, batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_predictions.extend(preds.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
            
            # Validation
            model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    loss, preds, targets = self._forward_pass(model, batch)
                    val_loss += loss.item()
                    val_predictions.extend(preds.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(train_predictions, train_targets)
            val_metrics = self._calculate_metrics(val_predictions, val_targets)
            
            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['train_metrics'].append(train_metrics)
            history['val_metrics'].append(val_metrics)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_model, history

    def evaluate_model(self, model: nn.Module,
                      test_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            model: Model to evaluate
            test_data: Test data dictionary

        Returns:
            Dictionary of performance metrics
        """
        model = model.to(self.device)
        model.eval()
        
        test_loader = self._create_data_loader(test_data, batch_size=32)
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                _, preds, true_vals = self._forward_pass(model, batch)
                predictions.extend(preds.cpu().numpy())
                targets.extend(true_vals.cpu().numpy())
        
        return self._calculate_metrics(predictions, targets)

    def _create_data_loader(self, data: Dict[str, torch.Tensor],
                          batch_size: int) -> DataLoader:
        """Create PyTorch DataLoader.

        Args:
            data: Data dictionary
            batch_size: Batch size

        Returns:
            DataLoader instance
        """
        # Convert all data to tensors
        tensors = [torch.as_tensor(v).float() for v in data.values()]
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _compute_loss(self, model: nn.Module, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute loss for a batch.

        Args:
            model: Model instance
            batch: Tuple of input tensors

        Returns:
            Loss tensor
        """
        *inputs, targets = batch
        outputs = model(*inputs)
        
        if isinstance(outputs, tuple):
            # Multiple outputs (e.g., cancer type, stage, grade)
            loss = 0
            for output, target in zip(outputs, targets):
                loss += nn.CrossEntropyLoss()(output, target)
            return loss
        else:
            # Single output
            return nn.CrossEntropyLoss()(outputs, targets)

    def _forward_pass(self, model: nn.Module,
                     batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass.

        Args:
            model: Model instance
            batch: Tuple of input tensors

        Returns:
            Tuple of (loss, predictions, targets)
        """
        *inputs, targets = batch
        outputs = model(*inputs)
        
        if isinstance(outputs, tuple):
            # Multiple outputs
            loss = 0
            predictions = []
            for output, target in zip(outputs, targets):
                loss += nn.CrossEntropyLoss()(output, target)
                predictions.append(output.argmax(dim=1))
            predictions = torch.stack(predictions)
            return loss, predictions, targets
        else:
            # Single output
            loss = nn.CrossEntropyLoss()(outputs, targets)
            predictions = outputs.argmax(dim=1)
            return loss, predictions, targets

    def _calculate_metrics(self, predictions: List[float],
                         targets: List[float]) -> Dict[str, float]:
        """Calculate performance metrics.

        Args:
            predictions: List of predictions
            targets: List of true values

        Returns:
            Dictionary of metrics
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='weighted'),
            'recall': recall_score(targets, predictions, average='weighted'),
            'f1': f1_score(targets, predictions, average='weighted'),
            'auc_roc': roc_auc_score(targets, predictions, average='weighted'),
            'average_precision': average_precision_score(targets, predictions),
            'brier_score': brier_score_loss(targets, predictions)
        }
        
        # Add concordance index for survival prediction
        if predictions.ndim == 2 and predictions.shape[1] == 2:  # time and event
            metrics['c_index'] = concordance_index(
                targets[:, 0],  # time
                predictions[:, 0],  # predicted time
                targets[:, 1]  # event
            )
        
        return metrics

class ModelVersionManager:
    """Model version management class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model version manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path(config.get('model_dir', 'models'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load version history
        self.version_history = self._load_version_history()

    def _load_version_history(self) -> Dict[str, ModelVersion]:
        """Load version history from disk.

        Returns:
            Dictionary of version information
        """
        history = {}
        history_file = self.model_dir / 'version_history.json'
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
                for version_id, data in history_data.items():
                    history[version_id] = ModelVersion(
                        version_id=version_id,
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        model_type=data['model_type'],
                        cancer_type=data['cancer_type'],
                        architecture=data['architecture'],
                        hyperparameters=data['hyperparameters'],
                        performance_metrics=data['performance_metrics'],
                        training_history=data['training_history'],
                        checksum=data['checksum']
                    )
        
        return history

    def _save_version_history(self) -> None:
        """Save version history to disk."""
        history_data = {
            version_id: {
                'timestamp': version.timestamp.isoformat(),
                'model_type': version.model_type,
                'cancer_type': version.cancer_type,
                'architecture': version.architecture,
                'hyperparameters': version.hyperparameters,
                'performance_metrics': version.performance_metrics,
                'training_history': version.training_history,
                'checksum': version.checksum
            }
            for version_id, version in self.version_history.items()
        }
        
        with open(self.model_dir / 'version_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)

    def save_model_version(self, model: nn.Module, model_info: Dict[str, Any]) -> ModelVersion:
        """Save new model version.

        Args:
            model: Trained model
            model_info: Model information dictionary

        Returns:
            Created version information
        """
        # Generate version ID and checksum
        timestamp = datetime.now()
        version_id = f"{model_info['model_type']}_{model_info['cancer_type']}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create version info
        version = ModelVersion(
            version_id=version_id,
            timestamp=timestamp,
            model_type=model_info['model_type'],
            cancer_type=model_info['cancer_type'],
            architecture=model_info['architecture'],
            hyperparameters=model_info['hyperparameters'],
            performance_metrics=model_info['performance_metrics'],
            training_history=model_info['training_history'],
            checksum=self._compute_model_checksum(model)
        )
        
        # Save version info
        self.version_history[version_id] = version
        self._save_version_history()
        
        # Save model state
        torch.save(model.state_dict(), self.model_dir / f"{version_id}.pt")
        
        return version

    def load_model_version(self, version_id: str, model_class: type) -> Optional[nn.Module]:
        """Load specific model version.

        Args:
            version_id: Version ID to load
            model_class: Model class to instantiate

        Returns:
            Loaded model or None if not found
        """
        if version_id not in self.version_history:
            return None
        
        version = self.version_history[version_id]
        model_path = self.model_dir / f"{version_id}.pt"
        
        if not model_path.exists():
            return None
        
        try:
            # Create model instance with saved architecture
            model = model_class(version.architecture)
            
            # Load state dict
            model.load_state_dict(torch.load(model_path))
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model version {version_id}: {str(e)}")
            return None

    def list_versions(self, model_type: Optional[str] = None,
                     cancer_type: Optional[str] = None) -> List[ModelVersion]:
        """List available model versions.

        Args:
            model_type: Optional model type filter
            cancer_type: Optional cancer type filter

        Returns:
            List of version information
        """
        versions = list(self.version_history.values())
        
        if model_type:
            versions = [v for v in versions if v.model_type == model_type]
        if cancer_type:
            versions = [v for v in versions if v.cancer_type == cancer_type]
        
        return versions

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two model versions.

        Args:
            version_id1: First version ID
            version_id2: Second version ID

        Returns:
            Dictionary containing comparison results
        """
        if version_id1 not in self.version_history or version_id2 not in self.version_history:
            return {}
        
        v1 = self.version_history[version_id1]
        v2 = self.version_history[version_id2]
        
        # Compare metrics
        metric_diff = {}
        for metric in v1.performance_metrics:
            if metric in v2.performance_metrics:
                metric_diff[metric] = v2.performance_metrics[metric] - v1.performance_metrics[metric]
        
        comparison = {
            'timestamp_diff': (v2.timestamp - v1.timestamp).total_seconds(),
            'architecture_changed': v1.architecture != v2.architecture,
            'hyperparameters_changed': v1.hyperparameters != v2.hyperparameters,
            'metric_differences': metric_diff,
            'checksum_changed': v1.checksum != v2.checksum
        }
        
        return comparison

    @staticmethod
    def _compute_model_checksum(model: nn.Module) -> str:
        """Compute model checksum.

        Args:
            model: Model to compute checksum for

        Returns:
            Computed checksum
        """
        # Get model state dict as bytes
        state_bytes = pickle.dumps(model.state_dict())
        return hashlib.sha256(state_bytes).hexdigest() 