"""
Clinical validation module for BEACON

This module implements validation protocols, real-world data collection,
and system performance evaluation functionality.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import json

@dataclass
class ValidationProtocol:
    """Validation protocol information."""
    protocol_id: str
    cancer_type: str
    description: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    endpoints: List[str]
    metrics: List[str]
    required_sample_size: int
    duration: int  # months
    follow_up_schedule: List[Dict[str, Any]]

@dataclass
class ValidationResult:
    """Validation result information."""
    protocol_id: str
    start_date: datetime
    end_date: Optional[datetime]
    sample_size: int
    patient_characteristics: Dict[str, Any]
    performance_metrics: Dict[str, float]
    survival_analysis: Dict[str, Any]
    adverse_events: List[Dict[str, Any]]
    conclusions: List[str]

class ClinicalValidator:
    """Clinical validation system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize clinical validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_dir = Path(config.get('validation_dir', 'validations'))
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Load protocols and results
        self.protocols = self._load_protocols()
        self.results = self._load_results()

    def _load_protocols(self) -> Dict[str, ValidationProtocol]:
        """Load validation protocols from disk.

        Returns:
            Dictionary of validation protocols
        """
        protocols = {}
        protocol_file = self.validation_dir / 'protocols.json'
        
        if protocol_file.exists():
            with open(protocol_file, 'r') as f:
                protocol_data = json.load(f)
                for protocol_id, data in protocol_data.items():
                    protocols[protocol_id] = ValidationProtocol(
                        protocol_id=protocol_id,
                        cancer_type=data['cancer_type'],
                        description=data['description'],
                        inclusion_criteria=data['inclusion_criteria'],
                        exclusion_criteria=data['exclusion_criteria'],
                        endpoints=data['endpoints'],
                        metrics=data['metrics'],
                        required_sample_size=data['required_sample_size'],
                        duration=data['duration'],
                        follow_up_schedule=data['follow_up_schedule']
                    )
        
        return protocols

    def _load_results(self) -> Dict[str, List[ValidationResult]]:
        """Load validation results from disk.

        Returns:
            Dictionary of validation results by protocol
        """
        results = {}
        results_file = self.validation_dir / 'results.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results_data = json.load(f)
                for protocol_id, protocol_results in results_data.items():
                    results[protocol_id] = []
                    for result_data in protocol_results:
                        results[protocol_id].append(ValidationResult(
                            protocol_id=protocol_id,
                            start_date=datetime.fromisoformat(result_data['start_date']),
                            end_date=datetime.fromisoformat(result_data['end_date'])
                                    if result_data['end_date'] else None,
                            sample_size=result_data['sample_size'],
                            patient_characteristics=result_data['patient_characteristics'],
                            performance_metrics=result_data['performance_metrics'],
                            survival_analysis=result_data['survival_analysis'],
                            adverse_events=result_data['adverse_events'],
                            conclusions=result_data['conclusions']
                        ))
        
        return results

    def create_validation_protocol(self, protocol_data: Dict[str, Any]) -> ValidationProtocol:
        """Create new validation protocol.

        Args:
            protocol_data: Protocol information dictionary

        Returns:
            Created validation protocol
        """
        # Generate protocol ID
        protocol_id = f"VAL_{protocol_data['cancer_type']}_{datetime.now().strftime('%Y%m%d')}"
        
        # Create protocol
        protocol = ValidationProtocol(
            protocol_id=protocol_id,
            cancer_type=protocol_data['cancer_type'],
            description=protocol_data['description'],
            inclusion_criteria=protocol_data['inclusion_criteria'],
            exclusion_criteria=protocol_data['exclusion_criteria'],
            endpoints=protocol_data['endpoints'],
            metrics=protocol_data['metrics'],
            required_sample_size=protocol_data['required_sample_size'],
            duration=protocol_data['duration'],
            follow_up_schedule=protocol_data['follow_up_schedule']
        )
        
        # Save protocol
        self.protocols[protocol_id] = protocol
        self._save_protocols()
        
        return protocol

    def record_validation_result(self, result_data: Dict[str, Any]) -> ValidationResult:
        """Record validation result.

        Args:
            result_data: Result information dictionary

        Returns:
            Created validation result
        """
        protocol_id = result_data['protocol_id']
        if protocol_id not in self.protocols:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        # Create result
        result = ValidationResult(
            protocol_id=protocol_id,
            start_date=result_data['start_date'],
            end_date=result_data.get('end_date'),
            sample_size=result_data['sample_size'],
            patient_characteristics=result_data['patient_characteristics'],
            performance_metrics=result_data['performance_metrics'],
            survival_analysis=result_data['survival_analysis'],
            adverse_events=result_data['adverse_events'],
            conclusions=result_data['conclusions']
        )
        
        # Save result
        if protocol_id not in self.results:
            self.results[protocol_id] = []
        self.results[protocol_id].append(result)
        self._save_results()
        
        return result

    def evaluate_system_performance(self, protocol_id: str,
                                  predictions: Dict[str, Any],
                                  ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate system performance for a validation protocol.

        Args:
            protocol_id: Validation protocol ID
            predictions: Model predictions
            ground_truth: Ground truth data

        Returns:
            Dictionary containing evaluation results
        """
        if protocol_id not in self.protocols:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        protocol = self.protocols[protocol_id]
        evaluation = {}
        
        # Calculate metrics specified in protocol
        for metric in protocol.metrics:
            if metric == 'accuracy':
                evaluation[metric] = accuracy_score(
                    ground_truth['labels'],
                    predictions['labels']
                )
            elif metric == 'precision':
                evaluation[metric] = precision_score(
                    ground_truth['labels'],
                    predictions['labels'],
                    average='weighted'
                )
            elif metric == 'recall':
                evaluation[metric] = recall_score(
                    ground_truth['labels'],
                    predictions['labels'],
                    average='weighted'
                )
            elif metric == 'f1':
                evaluation[metric] = f1_score(
                    ground_truth['labels'],
                    predictions['labels'],
                    average='weighted'
                )
            elif metric == 'auc_roc':
                evaluation[metric] = roc_auc_score(
                    ground_truth['labels'],
                    predictions['probabilities'],
                    average='weighted'
                )
            elif metric == 'survival_analysis':
                evaluation[metric] = self._perform_survival_analysis(
                    predictions['survival_times'],
                    ground_truth['survival_times'],
                    predictions['events'],
                    ground_truth['events']
                )
        
        # Add confusion matrix
        evaluation['confusion_matrix'] = confusion_matrix(
            ground_truth['labels'],
            predictions['labels']
        ).tolist()
        
        return evaluation

    def analyze_real_world_performance(self, protocol_id: str) -> Dict[str, Any]:
        """Analyze real-world performance for a validation protocol.

        Args:
            protocol_id: Validation protocol ID

        Returns:
            Dictionary containing analysis results
        """
        if protocol_id not in self.protocols or protocol_id not in self.results:
            raise ValueError(f"Protocol {protocol_id} not found or has no results")
        
        protocol = self.protocols[protocol_id]
        results = self.results[protocol_id]
        
        analysis = {
            'protocol_summary': {
                'cancer_type': protocol.cancer_type,
                'description': protocol.description,
                'required_sample_size': protocol.required_sample_size,
                'actual_sample_size': sum(r.sample_size for r in results)
            },
            'performance_metrics': self._aggregate_performance_metrics(results),
            'patient_demographics': self._analyze_patient_demographics(results),
            'adverse_events': self._analyze_adverse_events(results),
            'survival_analysis': self._aggregate_survival_analysis(results),
            'conclusions': self._generate_conclusions(protocol, results)
        }
        
        return analysis

    def _save_protocols(self) -> None:
        """Save validation protocols to disk."""
        protocol_data = {
            protocol_id: {
                'cancer_type': protocol.cancer_type,
                'description': protocol.description,
                'inclusion_criteria': protocol.inclusion_criteria,
                'exclusion_criteria': protocol.exclusion_criteria,
                'endpoints': protocol.endpoints,
                'metrics': protocol.metrics,
                'required_sample_size': protocol.required_sample_size,
                'duration': protocol.duration,
                'follow_up_schedule': protocol.follow_up_schedule
            }
            for protocol_id, protocol in self.protocols.items()
        }
        
        with open(self.validation_dir / 'protocols.json', 'w') as f:
            json.dump(protocol_data, f, indent=2)

    def _save_results(self) -> None:
        """Save validation results to disk."""
        results_data = {
            protocol_id: [
                {
                    'start_date': result.start_date.isoformat(),
                    'end_date': result.end_date.isoformat() if result.end_date else None,
                    'sample_size': result.sample_size,
                    'patient_characteristics': result.patient_characteristics,
                    'performance_metrics': result.performance_metrics,
                    'survival_analysis': result.survival_analysis,
                    'adverse_events': result.adverse_events,
                    'conclusions': result.conclusions
                }
                for result in protocol_results
            ]
            for protocol_id, protocol_results in self.results.items()
        }
        
        with open(self.validation_dir / 'results.json', 'w') as f:
            json.dump(results_data, f, indent=2)

    def _perform_survival_analysis(self, pred_times: np.ndarray,
                                 true_times: np.ndarray,
                                 pred_events: np.ndarray,
                                 true_events: np.ndarray) -> Dict[str, Any]:
        """Perform survival analysis.

        Args:
            pred_times: Predicted survival times
            true_times: True survival times
            pred_events: Predicted events
            true_events: True events

        Returns:
            Dictionary containing survival analysis results
        """
        # Fit Kaplan-Meier models
        kmf_true = KaplanMeierFitter()
        kmf_pred = KaplanMeierFitter()
        
        kmf_true.fit(true_times, true_events)
        kmf_pred.fit(pred_times, pred_events)
        
        # Perform log-rank test
        log_rank_result = logrank_test(true_times, pred_times,
                                     true_events, pred_events)
        
        analysis = {
            'true_survival': {
                'times': kmf_true.survival_function_.index.tolist(),
                'probabilities': kmf_true.survival_function_['KM_estimate'].tolist()
            },
            'predicted_survival': {
                'times': kmf_pred.survival_function_.index.tolist(),
                'probabilities': kmf_pred.survival_function_['KM_estimate'].tolist()
            },
            'log_rank_test': {
                'statistic': log_rank_result.test_statistic,
                'p_value': log_rank_result.p_value
            }
        }
        
        return analysis

    def _aggregate_performance_metrics(self,
                                    results: List[ValidationResult]) -> Dict[str, Dict[str, float]]:
        """Aggregate performance metrics across validation results.

        Args:
            results: List of validation results

        Returns:
            Dictionary of aggregated metrics
        """
        metrics = {}
        
        for result in results:
            for metric, value in result.performance_metrics.items():
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(value)
        
        return {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            for metric, values in metrics.items()
        }

    def _analyze_patient_demographics(self,
                                   results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze patient demographics across validation results.

        Args:
            results: List of validation results

        Returns:
            Dictionary containing demographic analysis
        """
        demographics = {}
        
        for result in results:
            for key, value in result.patient_characteristics.items():
                if key not in demographics:
                    demographics[key] = []
                demographics[key].append(value)
        
        analysis = {}
        for key, values in demographics.items():
            if isinstance(values[0], (int, float)):
                analysis[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                value_counts = pd.Series(values).value_counts()
                analysis[key] = {
                    'distribution': {
                        str(k): int(v) for k, v in value_counts.items()
                    }
                }
        
        return analysis

    def _analyze_adverse_events(self,
                              results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze adverse events across validation results.

        Args:
            results: List of validation results

        Returns:
            Dictionary containing adverse event analysis
        """
        all_events = []
        for result in results:
            all_events.extend(result.adverse_events)
        
        event_types = pd.DataFrame(all_events)
        
        analysis = {
            'total_events': len(all_events),
            'events_per_patient': len(all_events) / sum(r.sample_size for r in results),
            'severity_distribution': event_types['severity'].value_counts().to_dict(),
            'type_distribution': event_types['type'].value_counts().to_dict(),
            'serious_events': len(event_types[event_types['severity'] == 'serious']),
            'related_events': len(event_types[event_types['related'] == True])
        }
        
        return analysis

    def _aggregate_survival_analysis(self,
                                  results: List[ValidationResult]) -> Dict[str, Any]:
        """Aggregate survival analysis across validation results.

        Args:
            results: List of validation results

        Returns:
            Dictionary containing aggregated survival analysis
        """
        # Combine survival data from all results
        times = []
        events = []
        predictions = []
        
        for result in results:
            if 'survival_analysis' in result.performance_metrics:
                analysis = result.survival_analysis
                times.extend(analysis['true_survival']['times'])
                events.extend([1] * len(analysis['true_survival']['times']))
                predictions.extend(analysis['predicted_survival']['probabilities'])
        
        if not times:
            return {}
        
        # Fit combined Kaplan-Meier model
        kmf = KaplanMeierFitter()
        kmf.fit(times, events)
        
        analysis = {
            'overall_survival': {
                'times': kmf.survival_function_.index.tolist(),
                'probabilities': kmf.survival_function_['KM_estimate'].tolist()
            },
            'median_survival': float(kmf.median_survival_time_),
            'prediction_accuracy': float(np.mean(np.abs(
                np.array(predictions) - kmf.survival_function_['KM_estimate']
            )))
        }
        
        return analysis

    def _generate_conclusions(self, protocol: ValidationProtocol,
                            results: List[ValidationResult]) -> List[str]:
        """Generate conclusions from validation results.

        Args:
            protocol: Validation protocol
            results: List of validation results

        Returns:
            List of conclusion statements
        """
        conclusions = []
        
        # Sample size conclusion
        total_samples = sum(r.sample_size for r in results)
        if total_samples >= protocol.required_sample_size:
            conclusions.append(
                f"Required sample size of {protocol.required_sample_size} was met "
                f"with {total_samples} patients"
            )
        else:
            conclusions.append(
                f"Sample size requirement not met: {total_samples}/{protocol.required_sample_size} "
                "patients enrolled"
            )
        
        # Performance metrics conclusions
        metrics = self._aggregate_performance_metrics(results)
        for metric, stats in metrics.items():
            conclusions.append(
                f"Average {metric}: {stats['mean']:.3f} (±{stats['std']:.3f})"
            )
        
        # Adverse events conclusion
        events = self._analyze_adverse_events(results)
        conclusions.append(
            f"Observed {events['total_events']} adverse events "
            f"({events['events_per_patient']:.2f} per patient)"
        )
        
        # Survival analysis conclusion
        survival = self._aggregate_survival_analysis(results)
        if 'median_survival' in survival:
            conclusions.append(
                f"Median survival time: {survival['median_survival']:.1f} months"
            )
        
        return conclusions 