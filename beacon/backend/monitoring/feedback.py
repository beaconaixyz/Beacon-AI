"""
Monitoring and feedback system for BEACON

This module implements monitoring and feedback mechanisms for tracking treatment
outcomes, patient responses, and model performance.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

@dataclass
class PatientFeedback:
    """Patient feedback class."""
    patient_id: str
    timestamp: datetime
    treatment_phase: str
    symptoms: List[Dict[str, Any]]
    side_effects: List[Dict[str, Any]]
    quality_of_life: int  # Scale 1-10
    comments: str

@dataclass
class TreatmentOutcome:
    """Treatment outcome class."""
    patient_id: str
    treatment_id: str
    start_date: datetime
    end_date: Optional[datetime]
    response_type: str  # Complete, Partial, Stable, Progressive
    survival_time: Optional[int]  # Days
    progression_free_survival: Optional[int]  # Days
    biomarker_changes: Dict[str, float]
    complications: List[str]

@dataclass
class ModelPrediction:
    """Model prediction record class."""
    timestamp: datetime
    model_version: str
    input_data: Dict[str, Any]
    predictions: Dict[str, Any]
    ground_truth: Optional[Dict[str, Any]]
    confidence_scores: Dict[str, float]

class MonitoringSystem:
    """System monitoring and feedback collection class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize monitoring system.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self.feedback_store = pd.DataFrame()
        self.outcomes_store = pd.DataFrame()
        self.predictions_store = pd.DataFrame()
        
        # Load historical data if available
        self._load_historical_data()

    def _load_historical_data(self) -> None:
        """Load historical monitoring data."""
        try:
            feedback_path = Path(self.config['feedback_store_path'])
            outcomes_path = Path(self.config['outcomes_store_path'])
            predictions_path = Path(self.config['predictions_store_path'])
            
            if feedback_path.exists():
                self.feedback_store = pd.read_csv(feedback_path)
            if outcomes_path.exists():
                self.outcomes_store = pd.read_csv(outcomes_path)
            if predictions_path.exists():
                self.predictions_store = pd.read_csv(predictions_path)
                
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")

    def record_patient_feedback(self, feedback: PatientFeedback) -> None:
        """Record patient feedback.

        Args:
            feedback: Patient feedback instance
        """
        try:
            feedback_dict = {
                'patient_id': feedback.patient_id,
                'timestamp': feedback.timestamp,
                'treatment_phase': feedback.treatment_phase,
                'symptoms': str(feedback.symptoms),
                'side_effects': str(feedback.side_effects),
                'quality_of_life': feedback.quality_of_life,
                'comments': feedback.comments
            }
            
            self.feedback_store = pd.concat([
                self.feedback_store,
                pd.DataFrame([feedback_dict])
            ], ignore_index=True)
            
            # Save updated data
            self.feedback_store.to_csv(
                Path(self.config['feedback_store_path']),
                index=False
            )
            
        except Exception as e:
            self.logger.error(f"Error recording patient feedback: {str(e)}")

    def record_treatment_outcome(self, outcome: TreatmentOutcome) -> None:
        """Record treatment outcome.

        Args:
            outcome: Treatment outcome instance
        """
        try:
            outcome_dict = {
                'patient_id': outcome.patient_id,
                'treatment_id': outcome.treatment_id,
                'start_date': outcome.start_date,
                'end_date': outcome.end_date,
                'response_type': outcome.response_type,
                'survival_time': outcome.survival_time,
                'progression_free_survival': outcome.progression_free_survival,
                'biomarker_changes': str(outcome.biomarker_changes),
                'complications': str(outcome.complications)
            }
            
            self.outcomes_store = pd.concat([
                self.outcomes_store,
                pd.DataFrame([outcome_dict])
            ], ignore_index=True)
            
            # Save updated data
            self.outcomes_store.to_csv(
                Path(self.config['outcomes_store_path']),
                index=False
            )
            
        except Exception as e:
            self.logger.error(f"Error recording treatment outcome: {str(e)}")

    def record_model_prediction(self, prediction: ModelPrediction) -> None:
        """Record model prediction.

        Args:
            prediction: Model prediction instance
        """
        try:
            prediction_dict = {
                'timestamp': prediction.timestamp,
                'model_version': prediction.model_version,
                'input_data': str(prediction.input_data),
                'predictions': str(prediction.predictions),
                'ground_truth': str(prediction.ground_truth),
                'confidence_scores': str(prediction.confidence_scores)
            }
            
            self.predictions_store = pd.concat([
                self.predictions_store,
                pd.DataFrame([prediction_dict])
            ], ignore_index=True)
            
            # Save updated data
            self.predictions_store.to_csv(
                Path(self.config['predictions_store_path']),
                index=False
            )
            
        except Exception as e:
            self.logger.error(f"Error recording model prediction: {str(e)}")

class PerformanceAnalyzer:
    """Model performance analysis class."""
    
    def __init__(self, monitoring_system: MonitoringSystem):
        """Initialize performance analyzer.

        Args:
            monitoring_system: Monitoring system instance
        """
        self.monitoring = monitoring_system
        self.logger = logging.getLogger(__name__)

    def analyze_model_performance(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Analyze model performance metrics.

        Args:
            time_window: Optional time window for analysis

        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Filter predictions by time window if specified
            predictions = self.monitoring.predictions_store
            if time_window:
                cutoff_time = datetime.now() - time_window
                predictions = predictions[
                    pd.to_datetime(predictions['timestamp']) > cutoff_time
                ]
            
            # Calculate performance metrics
            metrics = {
                'overall': self._calculate_overall_metrics(predictions),
                'by_cancer_type': self._calculate_metrics_by_group(predictions, 'cancer_type'),
                'by_stage': self._calculate_metrics_by_group(predictions, 'stage'),
                'confidence_analysis': self._analyze_confidence_scores(predictions),
                'error_analysis': self._analyze_error_patterns(predictions)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing model performance: {str(e)}")
            return {}

    def analyze_treatment_outcomes(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Analyze treatment outcomes.

        Args:
            time_window: Optional time window for analysis

        Returns:
            Dictionary containing outcome analysis
        """
        try:
            # Filter outcomes by time window if specified
            outcomes = self.monitoring.outcomes_store
            if time_window:
                cutoff_time = datetime.now() - time_window
                outcomes = outcomes[
                    pd.to_datetime(outcomes['start_date']) > cutoff_time
                ]
            
            # Calculate outcome metrics
            analysis = {
                'response_rates': self._calculate_response_rates(outcomes),
                'survival_analysis': self._analyze_survival(outcomes),
                'complication_rates': self._analyze_complications(outcomes),
                'biomarker_trends': self._analyze_biomarker_trends(outcomes)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing treatment outcomes: {str(e)}")
            return {}

    def analyze_patient_feedback(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Analyze patient feedback.

        Args:
            time_window: Optional time window for analysis

        Returns:
            Dictionary containing feedback analysis
        """
        try:
            # Filter feedback by time window if specified
            feedback = self.monitoring.feedback_store
            if time_window:
                cutoff_time = datetime.now() - time_window
                feedback = feedback[
                    pd.to_datetime(feedback['timestamp']) > cutoff_time
                ]
            
            # Calculate feedback metrics
            analysis = {
                'quality_of_life_trends': self._analyze_qol_trends(feedback),
                'common_side_effects': self._analyze_side_effects(feedback),
                'symptom_patterns': self._analyze_symptom_patterns(feedback),
                'sentiment_analysis': self._analyze_feedback_sentiment(feedback)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing patient feedback: {str(e)}")
            return {}

    def generate_performance_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report.

        Args:
            time_window: Optional time window for report

        Returns:
            Dictionary containing performance report
        """
        report = {
            'model_performance': self.analyze_model_performance(time_window),
            'treatment_outcomes': self.analyze_treatment_outcomes(time_window),
            'patient_feedback': self.analyze_patient_feedback(time_window),
            'recommendations': self._generate_recommendations()
        }
        
        return report

    def _calculate_overall_metrics(self, predictions: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall model performance metrics.

        Args:
            predictions: DataFrame of model predictions

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        try:
            # Extract true labels and predictions
            y_true = []
            y_pred = []
            
            for _, row in predictions.iterrows():
                truth = eval(row['ground_truth'])
                preds = eval(row['predictions'])
                
                if truth and preds:
                    y_true.append(list(truth.values())[0])  # Assuming single prediction
                    y_pred.append(list(preds.values())[0])
            
            if y_true and y_pred:
                metrics.update({
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted'),
                    'recall': recall_score(y_true, y_pred, average='weighted'),
                    'f1': f1_score(y_true, y_pred, average='weighted'),
                    'auc_roc': roc_auc_score(y_true, y_pred, average='weighted')
                })
                
        except Exception as e:
            self.logger.error(f"Error calculating overall metrics: {str(e)}")
            
        return metrics

    def _calculate_metrics_by_group(self, predictions: pd.DataFrame,
                                  group_key: str) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by group.

        Args:
            predictions: DataFrame of model predictions
            group_key: Key to group by

        Returns:
            Dictionary of metrics by group
        """
        group_metrics = {}
        
        try:
            # Group predictions
            for _, row in predictions.iterrows():
                input_data = eval(row['input_data'])
                if group_key in input_data:
                    group = input_data[group_key]
                    if group not in group_metrics:
                        group_metrics[group] = {'y_true': [], 'y_pred': []}
                    
                    truth = eval(row['ground_truth'])
                    preds = eval(row['predictions'])
                    
                    if truth and preds:
                        group_metrics[group]['y_true'].append(list(truth.values())[0])
                        group_metrics[group]['y_pred'].append(list(preds.values())[0])
            
            # Calculate metrics for each group
            results = {}
            for group, data in group_metrics.items():
                if data['y_true'] and data['y_pred']:
                    results[group] = {
                        'accuracy': accuracy_score(data['y_true'], data['y_pred']),
                        'precision': precision_score(data['y_true'], data['y_pred'],
                                                  average='weighted'),
                        'recall': recall_score(data['y_true'], data['y_pred'],
                                            average='weighted'),
                        'f1': f1_score(data['y_true'], data['y_pred'],
                                     average='weighted')
                    }
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics by group: {str(e)}")
            return {}

    def _analyze_confidence_scores(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model confidence scores.

        Args:
            predictions: DataFrame of model predictions

        Returns:
            Dictionary containing confidence score analysis
        """
        analysis = {}
        
        try:
            confidence_scores = []
            correct_predictions = []
            
            for _, row in predictions.iterrows():
                scores = eval(row['confidence_scores'])
                truth = eval(row['ground_truth'])
                preds = eval(row['predictions'])
                
                if scores and truth and preds:
                    confidence = list(scores.values())[0]
                    correct = list(truth.values())[0] == list(preds.values())[0]
                    
                    confidence_scores.append(confidence)
                    correct_predictions.append(correct)
            
            if confidence_scores and correct_predictions:
                analysis.update({
                    'mean_confidence': np.mean(confidence_scores),
                    'confidence_std': np.std(confidence_scores),
                    'confidence_by_correctness': {
                        'correct': np.mean([s for s, c in zip(confidence_scores, correct_predictions) if c]),
                        'incorrect': np.mean([s for s, c in zip(confidence_scores, correct_predictions) if not c])
                    }
                })
                
        except Exception as e:
            self.logger.error(f"Error analyzing confidence scores: {str(e)}")
            
        return analysis

    def _analyze_error_patterns(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze error patterns in predictions.

        Args:
            predictions: DataFrame of model predictions

        Returns:
            Dictionary containing error pattern analysis
        """
        analysis = {}
        
        try:
            error_cases = []
            
            for _, row in predictions.iterrows():
                truth = eval(row['ground_truth'])
                preds = eval(row['predictions'])
                input_data = eval(row['input_data'])
                
                if truth and preds:
                    true_val = list(truth.values())[0]
                    pred_val = list(preds.values())[0]
                    
                    if true_val != pred_val:
                        error_cases.append({
                            'true_value': true_val,
                            'predicted_value': pred_val,
                            'input_features': input_data
                        })
            
            if error_cases:
                # Analyze common patterns in error cases
                analysis.update({
                    'total_errors': len(error_cases),
                    'error_rate': len(error_cases) / len(predictions),
                    'common_misclassifications': self._find_common_misclassifications(error_cases),
                    'feature_correlations': self._analyze_error_features(error_cases)
                })
                
        except Exception as e:
            self.logger.error(f"Error analyzing error patterns: {str(e)}")
            
        return analysis

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis.

        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Analyze recent performance trends
            performance = self.analyze_model_performance(timedelta(days=30))
            outcomes = self.analyze_treatment_outcomes(timedelta(days=30))
            feedback = self.analyze_patient_feedback(timedelta(days=30))
            
            # Generate model-related recommendations
            if performance.get('overall', {}).get('accuracy', 1.0) < 0.8:
                recommendations.append({
                    'type': 'model',
                    'priority': 'high',
                    'description': 'Model accuracy below threshold, consider retraining',
                    'suggested_action': 'Retrain model with recent data'
                })
            
            # Generate treatment-related recommendations
            response_rates = outcomes.get('response_rates', {})
            if response_rates.get('complete', 0) < 0.3:
                recommendations.append({
                    'type': 'treatment',
                    'priority': 'medium',
                    'description': 'Low complete response rate observed',
                    'suggested_action': 'Review treatment protocols'
                })
            
            # Generate patient care recommendations
            qol_trends = feedback.get('quality_of_life_trends', {})
            if qol_trends.get('trend', 0) < 0:
                recommendations.append({
                    'type': 'patient_care',
                    'priority': 'high',
                    'description': 'Declining quality of life scores',
                    'suggested_action': 'Review supportive care measures'
                })
                
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            
        return recommendations

    @staticmethod
    def _find_common_misclassifications(error_cases: List[Dict[str, Any]]) -> Dict[str, int]:
        """Find common misclassification patterns.

        Args:
            error_cases: List of error cases

        Returns:
            Dictionary of misclassification patterns and counts
        """
        patterns = {}
        for case in error_cases:
            pattern = f"{case['true_value']}->{case['predicted_value']}"
            patterns[pattern] = patterns.get(pattern, 0) + 1
        return patterns

    @staticmethod
    def _analyze_error_features(error_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze feature correlations with errors.

        Args:
            error_cases: List of error cases

        Returns:
            Dictionary of feature correlations
        """
        feature_counts = {}
        for case in error_cases:
            for feature, value in case['input_features'].items():
                if isinstance(value, (int, float)):
                    if feature not in feature_counts:
                        feature_counts[feature] = []
                    feature_counts[feature].append(value)
        
        correlations = {}
        for feature, values in feature_counts.items():
            if len(values) > 1:
                correlations[feature] = np.mean(values)
        return correlations 