#!/usr/bin/env python3

"""
Monitoring Script for BEACON

This script implements monitoring functionality for model performance and system health.
"""

import os
import argparse
import numpy as np
import pandas as pd
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import logging
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

class SystemMonitor:
    """Monitor system resources"""

    def __init__(self):
        """Initialize system monitor."""
        self.start_time = time.time()

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics.

        Returns:
            Dictionary of system metrics
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024 ** 3),
            'memory_total_gb': memory.total / (1024 ** 3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024 ** 3),
            'disk_total_gb': disk.total / (1024 ** 3),
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
        
        return metrics

class ModelMonitor:
    """Monitor model performance"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize model monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_dir = Path(config.get('metrics_dir', 'metrics'))
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def calculate_drift_metrics(self, reference_data: pd.DataFrame,
                              current_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data drift metrics.

        Args:
            reference_data: Reference data distribution
            current_data: Current data distribution

        Returns:
            Dictionary of drift metrics
        """
        drift_metrics = {}
        
        # Calculate basic statistics
        for column in reference_data.columns:
            if reference_data[column].dtype in ['int64', 'float64']:
                # Calculate KL divergence for numeric columns
                ref_hist = np.histogram(reference_data[column], bins=50, density=True)[0]
                cur_hist = np.histogram(current_data[column], bins=50, density=True)[0]
                
                # Add small epsilon to avoid division by zero
                epsilon = 1e-10
                ref_hist = ref_hist + epsilon
                cur_hist = cur_hist + epsilon
                
                kl_div = np.sum(ref_hist * np.log(ref_hist / cur_hist))
                drift_metrics[f'{column}_kl_divergence'] = kl_div
                
                # Calculate distribution statistics
                drift_metrics.update({
                    f'{column}_mean_diff': abs(reference_data[column].mean() - current_data[column].mean()),
                    f'{column}_std_diff': abs(reference_data[column].std() - current_data[column].std())
                })
            else:
                # Calculate distribution difference for categorical columns
                ref_dist = reference_data[column].value_counts(normalize=True)
                cur_dist = current_data[column].value_counts(normalize=True)
                
                # Calculate Jensen-Shannon divergence
                m = 0.5 * (ref_dist + cur_dist)
                js_div = 0.5 * (
                    (ref_dist * np.log(ref_dist / m)).sum() +
                    (cur_dist * np.log(cur_dist / m)).sum()
                )
                drift_metrics[f'{column}_js_divergence'] = js_div

        return drift_metrics

    def calculate_performance_metrics(self, predictions: np.ndarray,
                                   targets: np.ndarray,
                                   task_type: str) -> Dict[str, float]:
        """Calculate model performance metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            task_type: Type of task ('classification' or 'regression')

        Returns:
            Dictionary of performance metrics
        """
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(targets, predictions),
                'precision': precision_score(targets, predictions, average='weighted'),
                'recall': recall_score(targets, predictions, average='weighted'),
                'f1': f1_score(targets, predictions, average='weighted')
            }
        else:  # regression
            metrics = {
                'mse': mean_squared_error(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions)),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions)
            }
        
        return metrics

    def log_metrics(self, metrics: Dict[str, float], metric_type: str) -> None:
        """Log metrics to file.

        Args:
            metrics: Dictionary of metrics to log
            metric_type: Type of metrics ('system', 'drift', or 'performance')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = self.metrics_dir / f'{metric_type}_metrics_{timestamp}.json'
        
        # Add timestamp to metrics
        metrics['timestamp'] = timestamp
        
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        self.logger.info(f"Saved {metric_type} metrics to {metrics_file}")

    def analyze_metrics_history(self, metric_type: str) -> Dict[str, Any]:
        """Analyze historical metrics.

        Args:
            metric_type: Type of metrics to analyze

        Returns:
            Dictionary containing metric analysis
        """
        # Load all metric files
        metric_files = list(self.metrics_dir.glob(f'{metric_type}_metrics_*.json'))
        all_metrics = []
        
        for file in metric_files:
            with open(file) as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Convert to DataFrame for analysis
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        metrics_df = metrics_df.sort_values('timestamp')
        
        # Calculate statistics
        analysis = {}
        for column in metrics_df.columns:
            if column != 'timestamp' and isinstance(metrics_df[column].iloc[0], (int, float)):
                analysis[f'{column}_mean'] = metrics_df[column].mean()
                analysis[f'{column}_std'] = metrics_df[column].std()
                analysis[f'{column}_trend'] = metrics_df[column].diff().mean()
                analysis[f'{column}_last_value'] = metrics_df[column].iloc[-1]
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description="Monitor BEACON system and models")
    parser.add_argument("--metrics-dir", default="metrics",
                      help="Directory for storing metrics")
    parser.add_argument("--reference-data", required=True,
                      help="Path to reference data file")
    parser.add_argument("--current-data", required=True,
                      help="Path to current data file")
    parser.add_argument("--task-type", choices=['classification', 'regression'],
                      required=True, help="Type of task")
    args = parser.parse_args()

    # Initialize monitors
    system_monitor = SystemMonitor()
    model_monitor = ModelMonitor({'metrics_dir': args.metrics_dir})

    # Monitor system metrics
    system_metrics = system_monitor.get_system_metrics()
    model_monitor.log_metrics(system_metrics, 'system')

    # Load data and calculate drift metrics
    reference_data = pd.read_csv(args.reference_data)
    current_data = pd.read_csv(args.current_data)
    drift_metrics = model_monitor.calculate_drift_metrics(reference_data, current_data)
    model_monitor.log_metrics(drift_metrics, 'drift')

    # Analyze metrics history
    system_analysis = model_monitor.analyze_metrics_history('system')
    drift_analysis = model_monitor.analyze_metrics_history('drift')
    performance_analysis = model_monitor.analyze_metrics_history('performance')

    # Print analysis results
    print("\nSystem Metrics Analysis:")
    print(json.dumps(system_analysis, indent=2))
    print("\nDrift Metrics Analysis:")
    print(json.dumps(drift_analysis, indent=2))
    print("\nPerformance Metrics Analysis:")
    print(json.dumps(performance_analysis, indent=2))

if __name__ == "__main__":
    main()