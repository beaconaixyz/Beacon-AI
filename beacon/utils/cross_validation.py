import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Callable
from sklearn.model_selection import KFold, StratifiedKFold
from ..core.base import BeaconBase
import logging
from pathlib import Path
import json

class CrossValidator(BeaconBase):
    """Cross validation for model evaluation and ensemble learning"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cross validator
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.cv = self._setup_cv()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'n_splits': 5,
            'shuffle': True,
            'stratified': True,
            'random_state': 42,
            'save_fold_models': True,
            'ensemble_method': 'voting'  # 'voting' or 'averaging'
        }
    
    def _setup_cv(self) -> KFold:
        """Setup cross validation splitter"""
        if self.config['stratified']:
            return StratifiedKFold(
                n_splits=self.config['n_splits'],
                shuffle=self.config['shuffle'],
                random_state=self.config['random_state']
            )
        else:
            return KFold(
                n_splits=self.config['n_splits'],
                shuffle=self.config['shuffle'],
                random_state=self.config['random_state']
            )
    
    def split_data(self, data: Dict[str, torch.Tensor]) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Split data into training and validation sets
        Args:
            data: Dictionary containing all data
        Returns:
            List of (train_data, val_data) tuples
        """
        # Get indices for stratification
        if self.config['stratified']:
            split_indices = self.cv.split(
                np.zeros(len(data['labels'])),
                data['labels'].numpy()
            )
        else:
            split_indices = self.cv.split(np.zeros(len(data['labels'])))
        
        # Create splits
        splits = []
        for train_idx, val_idx in split_indices:
            train_data = {
                key: value[train_idx] if isinstance(value, torch.Tensor)
                else value.iloc[train_idx] for key, value in data.items()
            }
            val_data = {
                key: value[val_idx] if isinstance(value, torch.Tensor)
                else value.iloc[val_idx] for key, value in data.items()
            }
            splits.append((train_data, val_data))
        
        return splits
    
    def cross_validate(self, 
                      data: Dict[str, torch.Tensor],
                      model_builder: Callable,
                      model_config: Dict[str, Any],
                      output_dir: str) -> Tuple[Dict[str, float], List[Any]]:
        """
        Perform cross validation
        Args:
            data: Dictionary containing all data
            model_builder: Function to build model
            model_config: Model configuration
            output_dir: Output directory for saving models
        Returns:
            Tuple of (mean metrics, list of trained models)
        """
        splits = self.split_data(data)
        fold_metrics = []
        fold_models = []
        
        for fold_idx, (train_data, val_data) in enumerate(splits):
            self.logger.info(f"Training fold {fold_idx + 1}/{self.config['n_splits']}")
            
            # Build and train model
            model = model_builder(model_config)
            history = model.train(train_data)
            
            # Evaluate model
            metrics = model.evaluate(val_data)
            fold_metrics.append(metrics)
            
            # Save fold model if configured
            if self.config['save_fold_models']:
                save_path = Path(output_dir) / f"fold_{fold_idx + 1}.pt"
                model.save(str(save_path))
            
            fold_models.append(model)
        
        # Calculate mean metrics
        mean_metrics = {}
        for metric in fold_metrics[0].keys():
            values = [m[metric] for m in fold_metrics]
            mean_metrics[metric] = float(np.mean(values))
            std = float(np.std(values))
            self.logger.info(f"{metric}: {mean_metrics[metric]:.4f} ± {std:.4f}")
        
        # Save cross validation results
        results = {
            'mean_metrics': mean_metrics,
            'fold_metrics': fold_metrics
        }
        results_path = Path(output_dir) / "cv_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return mean_metrics, fold_models
    
    def ensemble_predict(self, 
                        models: List[Any],
                        data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Make predictions using ensemble of models
        Args:
            models: List of trained models
            data: Input data
        Returns:
            Ensemble predictions
        """
        predictions = []
        for model in models:
            pred = model.predict(data)
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(predictions)
        
        # Apply ensemble method
        if self.config['ensemble_method'] == 'voting':
            # For classification, take mode of predictions
            ensemble_preds = torch.mode(torch.argmax(stacked_preds, dim=2), dim=0)[0]
        else:  # averaging
            # For regression or probability outputs, take mean
            ensemble_preds = torch.mean(stacked_preds, dim=0)
        
        return ensemble_preds
    
    def evaluate_ensemble(self, 
                         models: List[Any],
                         data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate ensemble model
        Args:
            models: List of trained models
            data: Test data
        Returns:
            Dictionary of evaluation metrics
        """
        # Get ensemble predictions
        ensemble_preds = self.ensemble_predict(models, data)
        
        # Calculate metrics
        if self.config['ensemble_method'] == 'voting':
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            y_true = data['labels'].numpy()
            y_pred = ensemble_preds.numpy()
            
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='macro')),
                'recall': float(recall_score(y_true, y_pred, average='macro')),
                'f1': float(f1_score(y_true, y_pred, average='macro'))
            }
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, r2_score
            y_true = data['labels'].numpy()
            y_pred = ensemble_preds.numpy()
            
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2': float(r2_score(y_true, y_pred))
            }
        
        return metrics 