import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from ..core.base import BeaconBase

class Metrics(BeaconBase):
    """Evaluation metrics for medical image analysis"""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
            'average': 'macro',
            'threshold': 0.5
        }
    
    def calculate_classification_metrics(self,
                                      y_true: Union[np.ndarray, torch.Tensor],
                                      y_pred: Union[np.ndarray, torch.Tensor],
                                      y_score: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Calculate classification metrics
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_score: Prediction scores (probabilities)
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_score, torch.Tensor):
            y_score = y_score.cpu().numpy()
        
        metrics = {}
        
        # Basic metrics
        if 'accuracy' in self.config['metrics']:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in self.config['metrics']:
            metrics['precision'] = precision_score(
                y_true, y_pred, average=self.config['average']
            )
        
        if 'recall' in self.config['metrics']:
            metrics['recall'] = recall_score(
                y_true, y_pred, average=self.config['average']
            )
        
        if 'f1' in self.config['metrics']:
            metrics['f1'] = f1_score(
                y_true, y_pred, average=self.config['average']
            )
        
        # Confusion matrix
        if 'confusion_matrix' in self.config['metrics']:
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Metrics requiring probability scores
        if y_score is not None:
            if 'auc' in self.config['metrics']:
                if y_score.shape[1] == 2:  # Binary classification
                    metrics['auc'] = roc_auc_score(y_true, y_score[:, 1])
                else:  # Multi-class
                    metrics['auc'] = roc_auc_score(
                        y_true, y_score, multi_class='ovr',
                        average=self.config['average']
                    )
            
            if 'average_precision' in self.config['metrics']:
                if y_score.shape[1] == 2:  # Binary classification
                    metrics['average_precision'] = average_precision_score(
                        y_true, y_score[:, 1]
                    )
                else:  # Multi-class
                    metrics['average_precision'] = average_precision_score(
                        y_true, y_score, average=self.config['average']
                    )
        
        return metrics
    
    def calculate_regression_metrics(self,
                                   y_true: Union[np.ndarray, torch.Tensor],
                                   y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Calculate regression metrics
        Args:
            y_true: True values
            y_pred: Predicted values
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        metrics = {}
        
        if 'mse' in self.config['metrics']:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        
        if 'mae' in self.config['metrics']:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        if 'r2' in self.config['metrics']:
            metrics['r2'] = r2_score(y_true, y_pred)
        
        if 'explained_variance' in self.config['metrics']:
            metrics['explained_variance'] = np.var(y_pred) / np.var(y_true)
        
        return metrics
    
    def calculate_segmentation_metrics(self,
                                     y_true: Union[np.ndarray, torch.Tensor],
                                     y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Calculate segmentation metrics
        Args:
            y_true: True masks
            y_pred: Predicted masks
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        metrics = {}
        
        # Dice coefficient
        if 'dice' in self.config['metrics']:
            intersection = np.sum(y_true * y_pred)
            union = np.sum(y_true) + np.sum(y_pred)
            metrics['dice'] = 2.0 * intersection / (union + 1e-6)
        
        # IoU (Jaccard index)
        if 'iou' in self.config['metrics']:
            intersection = np.sum(y_true * y_pred)
            union = np.sum(y_true) + np.sum(y_pred) - intersection
            metrics['iou'] = intersection / (union + 1e-6)
        
        # Pixel accuracy
        if 'pixel_accuracy' in self.config['metrics']:
            metrics['pixel_accuracy'] = np.mean(y_true == y_pred)
        
        # Sensitivity and specificity
        if 'sensitivity' in self.config['metrics'] or 'specificity' in self.config['metrics']:
            true_positive = np.sum((y_true == 1) & (y_pred == 1))
            true_negative = np.sum((y_true == 0) & (y_pred == 0))
            false_positive = np.sum((y_true == 0) & (y_pred == 1))
            false_negative = np.sum((y_true == 1) & (y_pred == 0))
            
            if 'sensitivity' in self.config['metrics']:
                metrics['sensitivity'] = true_positive / (true_positive + false_negative + 1e-6)
            
            if 'specificity' in self.config['metrics']:
                metrics['specificity'] = true_negative / (true_negative + false_positive + 1e-6)
        
        return metrics
    
    def calculate_interpretation_metrics(self,
                                      attribution: Union[np.ndarray, torch.Tensor],
                                      reference: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Calculate interpretation metrics
        Args:
            attribution: Attribution map
            reference: Reference attribution (ground truth)
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if needed
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.cpu().numpy()
        if isinstance(reference, torch.Tensor):
            reference = reference.cpu().numpy()
        
        metrics = {}
        
        # Sparsity
        if 'sparsity' in self.config['metrics']:
            metrics['sparsity'] = np.mean(attribution == 0)
        
        # Attribution statistics
        if 'attribution_stats' in self.config['metrics']:
            metrics.update({
                'mean_attribution': float(np.mean(attribution)),
                'std_attribution': float(np.std(attribution)),
                'max_attribution': float(np.max(attribution)),
                'min_attribution': float(np.min(attribution))
            })
        
        # Metrics requiring reference attribution
        if reference is not None:
            # Correlation
            if 'correlation' in self.config['metrics']:
                metrics['correlation'] = float(
                    np.corrcoef(attribution.flatten(), reference.flatten())[0, 1]
                )
            
            # Mean squared error
            if 'mse' in self.config['metrics']:
                metrics['mse'] = float(
                    mean_squared_error(reference.flatten(), attribution.flatten())
                )
        
        return metrics
    
    def aggregate_metrics(self, metric_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics from multiple runs
        Args:
            metric_list: List of metric dictionaries
        Returns:
            Dictionary of aggregated metrics
        """
        if not metric_list:
            return {}
        
        # Initialize aggregated metrics
        aggregated = {}
        
        # Get all metric names
        metric_names = set()
        for metrics in metric_list:
            metric_names.update(metrics.keys())
        
        # Calculate statistics for each metric
        for name in metric_names:
            values = [m[name] for m in metric_list if name in m]
            if values:
                aggregated[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return aggregated 