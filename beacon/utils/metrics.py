import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)

class Metrics:
    """Basic evaluation metrics for cancer analysis"""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        Calculate classification metrics
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            average: Averaging strategy for multiclass metrics
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_positives'] = cm[1, 1] if average == 'binary' else cm.diagonal()
        metrics['false_positives'] = cm[0, 1] if average == 'binary' else cm.sum(axis=0) - cm.diagonal()
        metrics['false_negatives'] = cm[1, 0] if average == 'binary' else cm.sum(axis=1) - cm.diagonal()
        
        # Advanced metrics if probabilities are provided
        if y_prob is not None:
            if average == 'binary':
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                metrics['average_precision'] = average_precision_score(y_true, y_prob)
            else:
                # For multiclass, calculate macro average
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
                metrics['average_precision'] = average_precision_score(
                    y_true, y_prob, average='macro'
                )
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Mean squared error
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        
        # Root mean squared error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Mean absolute error
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # R-squared score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
        
        return metrics
    
    @staticmethod
    def calculate_confidence_intervals(
        metric_values: List[float],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence intervals for metrics
        Args:
            metric_values: List of metric values from different runs/folds
            confidence_level: Confidence level (default: 0.95)
        Returns:
            Dictionary with mean, lower and upper bounds
        """
        if not metric_values:
            raise ValueError("Empty metric values list")
        
        # Calculate mean
        mean_value = np.mean(metric_values)
        
        # Calculate standard error
        std_error = np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))
        
        # Calculate z-score for given confidence level
        z_score = -np.sqrt(2) * np.log((1 - confidence_level) / 2)
        
        # Calculate confidence interval
        margin_of_error = z_score * std_error
        lower_bound = mean_value - margin_of_error
        upper_bound = mean_value + margin_of_error
        
        return {
            'mean': mean_value,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std_error': std_error
        }
    
    @staticmethod
    def format_metrics(
        metrics: Dict[str, float],
        precision: int = 4
    ) -> Dict[str, str]:
        """
        Format metric values for display
        Args:
            metrics: Dictionary of metric values
            precision: Number of decimal places
        Returns:
            Dictionary of formatted metric strings
        """
        return {
            key: f"{value:.{precision}f}"
            for key, value in metrics.items()
        }
    
    @staticmethod
    def get_metric_description(metric_name: str) -> str:
        """
        Get description of a metric
        Args:
            metric_name: Name of the metric
        Returns:
            Description string
        """
        descriptions = {
            'accuracy': 'Proportion of correct predictions among total predictions',
            'precision': 'Proportion of true positives among positive predictions',
            'recall': 'Proportion of true positives among actual positives',
            'f1': 'Harmonic mean of precision and recall',
            'roc_auc': 'Area under the ROC curve',
            'average_precision': 'Area under the precision-recall curve',
            'mse': 'Mean squared error between predictions and true values',
            'rmse': 'Root mean squared error between predictions and true values',
            'mae': 'Mean absolute error between predictions and true values',
            'r2': 'Proportion of variance in target explained by predictions'
        }
        
        return descriptions.get(
            metric_name,
            'No description available for this metric'
        ) 