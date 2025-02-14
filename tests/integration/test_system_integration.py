"""
System integration tests for BEACON

This module implements comprehensive integration tests for the BEACON system,
testing component interactions, data flow, and error handling.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from beacon.backend.models.cancer import (
    CancerDiagnosisModel,
    TreatmentRecommendationModel
)
from beacon.backend.knowledge.medical import (
    MedicalKnowledgeBase,
    TreatmentAdvisor
)
from beacon.backend.monitoring.feedback import (
    MonitoringSystem,
    PerformanceAnalyzer
)
from beacon.backend.data.validation import (
    DataValidator,
    DataVersionControl
)

class TestSystemIntegration:
    """System integration tests."""

    @pytest.fixture
    def setup_test_data(self):
        """Set up test data."""
        # Create test directories
        test_dir = Path("test_data")
        test_dir.mkdir(exist_ok=True)
        
        # Generate synthetic data
        clinical_data = pd.DataFrame({
            'age': np.random.normal(60, 10, 100),
            'weight': np.random.normal(70, 15, 100),
            'height': np.random.normal(170, 10, 100),
            'blood_pressure': np.random.normal(120, 15, 100),
            'cancer_type': np.random.choice(['A', 'B', 'C'], 100),
            'stage': np.random.choice(['I', 'II', 'III', 'IV'], 100)
        })
        
        imaging_data = np.random.normal(0, 1, (100, 1, 64, 64))
        
        genomic_data = {
            'expression': np.random.normal(0, 1, (100, 1000)),
            'mutations': np.random.choice([0, 1], (100, 1000), p=[0.99, 0.01]),
            'cnv': np.random.choice([-2, -1, 0, 1, 2], (100, 1000))
        }
        
        return {
            'clinical': clinical_data,
            'imaging': imaging_data,
            'genomic': genomic_data,
            'test_dir': test_dir
        }

    def test_data_processing_pipeline(self, setup_test_data):
        """Test the complete data processing pipeline."""
        data = setup_test_data
        
        # Initialize components
        validator = DataValidator({'test_dir': data['test_dir']})
        version_control = DataVersionControl({'test_dir': data['test_dir']})
        
        # Test clinical data validation and versioning
        is_valid, messages = validator.validate_clinical_data(data['clinical'])
        assert is_valid, f"Clinical data validation failed: {messages}"
        
        version = version_control.create_version(
            data['clinical'],
            "Test clinical data",
            "Integration test",
            {'timestamp': datetime.now().isoformat()}
        )
        assert version is not None
        
        # Test imaging data validation and versioning
        is_valid, messages = validator.validate_imaging_data(data['imaging'])
        assert is_valid, f"Imaging data validation failed: {messages}"
        
        version = version_control.create_version(
            data['imaging'],
            "Test imaging data",
            "Integration test",
            {'timestamp': datetime.now().isoformat()}
        )
        assert version is not None
        
        # Test genomic data validation and versioning
        is_valid, messages = validator.validate_genomic_data(data['genomic'])
        assert is_valid, f"Genomic data validation failed: {messages}"
        
        version = version_control.create_version(
            data['genomic'],
            "Test genomic data",
            "Integration test",
            {'timestamp': datetime.now().isoformat()}
        )
        assert version is not None

    def test_model_pipeline(self, setup_test_data):
        """Test the model training and inference pipeline."""
        data = setup_test_data
        
        # Initialize models
        diagnosis_config = {
            'clinical_dim': data['clinical'].shape[1],
            'image_channels': 1,
            'image_size': 64,
            'genomic_dim': 1000,
            'num_cancer_types': 3
        }
        
        diagnosis_model = CancerDiagnosisModel(diagnosis_config)
        treatment_model = TreatmentRecommendationModel(diagnosis_config)
        
        # Test forward pass
        clinical_tensor = torch.FloatTensor(data['clinical'].values)
        imaging_tensor = torch.FloatTensor(data['imaging'])
        genomic_tensor = torch.FloatTensor(data['genomic']['expression'])
        
        cancer_probs, stage_probs, grade_probs = diagnosis_model(
            clinical_tensor,
            imaging_tensor,
            genomic_tensor
        )
        assert cancer_probs.shape == (100, 3)
        assert stage_probs.shape == (100, 4)
        assert grade_probs.shape == (100, 3)
        
        treatment_probs, drug_probs, dosage = treatment_model(
            clinical_tensor,
            imaging_tensor,
            genomic_tensor
        )
        assert treatment_probs.shape[0] == 100
        assert drug_probs.shape[0] == 100
        assert dosage.shape[0] == 100

    def test_knowledge_integration(self, setup_test_data):
        """Test medical knowledge base integration."""
        # Initialize components
        kb_config = {
            'drug_database_path': 'test_data/drugs.csv',
            'guidelines_path': 'test_data/guidelines.csv',
            'trials_path': 'test_data/trials.csv'
        }
        
        kb = MedicalKnowledgeBase(kb_config)
        advisor = TreatmentAdvisor(kb)
        
        # Test treatment recommendations
        patient_data = {
            'cancer_type': 'A',
            'stage': 'II',
            'grade': 'Medium',
            'age': 65,
            'conditions': ['diabetes']
        }
        
        recommendations = advisor.get_treatment_recommendations(patient_data)
        assert 'treatment_plan' in recommendations
        assert 'drug_recommendations' in recommendations
        assert 'monitoring_plan' in recommendations

    def test_monitoring_system(self, setup_test_data):
        """Test the monitoring and feedback system."""
        # Initialize components
        monitoring_config = {
            'feedback_path': 'test_data/feedback.csv',
            'outcomes_path': 'test_data/outcomes.csv',
            'predictions_path': 'test_data/predictions.csv'
        }
        
        monitoring = MonitoringSystem(monitoring_config)
        analyzer = PerformanceAnalyzer(monitoring)
        
        # Record test feedback
        feedback = PatientFeedback(
            patient_id="TEST001",
            timestamp=datetime.now(),
            treatment_phase="Phase 1",
            symptoms=[{"type": "pain", "severity": 3}],
            side_effects=[{"type": "nausea", "severity": 2}],
            quality_of_life=7,
            comments="Test feedback"
        )
        monitoring.record_patient_feedback(feedback)
        
        # Test analysis
        performance_report = analyzer.generate_performance_report()
        assert 'model_performance' in performance_report
        assert 'treatment_outcomes' in performance_report
        assert 'patient_feedback' in performance_report

    def test_error_handling(self, setup_test_data):
        """Test system error handling."""
        data = setup_test_data
        validator = DataValidator({'test_dir': data['test_dir']})
        
        # Test invalid clinical data
        invalid_clinical = data['clinical'].copy()
        invalid_clinical['age'] = 'invalid'
        is_valid, messages = validator.validate_clinical_data(invalid_clinical)
        assert not is_valid
        assert len(messages) > 0
        
        # Test invalid imaging data
        invalid_imaging = np.random.normal(0, 1, (100, 1, 64))  # Wrong shape
        is_valid, messages = validator.validate_imaging_data(invalid_imaging)
        assert not is_valid
        assert len(messages) > 0
        
        # Test invalid genomic data
        invalid_genomic = {
            'expression': np.random.normal(0, 1, (100, 1000)),
            'mutations': 'invalid'
        }
        is_valid, messages = validator.validate_genomic_data(invalid_genomic)
        assert not is_valid
        assert len(messages) > 0

if __name__ == "__main__":
    pytest.main([__file__]) 