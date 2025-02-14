"""
Medical knowledge base and treatment guidelines for BEACON

This module implements medical knowledge integration, including treatment guidelines,
drug interactions, and clinical trial matching.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import dataclass

@dataclass
class DrugInfo:
    """Drug information class."""
    name: str
    category: str
    mechanism: str
    indications: List[str]
    contraindications: List[str]
    side_effects: List[str]
    interactions: List[str]
    dosage_range: Dict[str, float]
    monitoring_requirements: List[str]

@dataclass
class TreatmentGuideline:
    """Treatment guideline class."""
    cancer_type: str
    stage: str
    grade: str
    patient_criteria: Dict[str, Any]
    recommended_treatments: List[str]
    treatment_sequence: List[str]
    evidence_level: str
    source: str
    last_updated: datetime

@dataclass
class ClinicalTrial:
    """Clinical trial information class."""
    trial_id: str
    title: str
    description: str
    phase: int
    status: str
    criteria: Dict[str, Any]
    locations: List[str]
    contact_info: Dict[str, str]
    start_date: datetime
    end_date: Optional[datetime]

class MedicalKnowledgeBase:
    """Medical knowledge base class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize medical knowledge base.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load knowledge bases
        self.drugs_db = self._load_drug_database()
        self.guidelines_db = self._load_treatment_guidelines()
        self.trials_db = self._load_clinical_trials()

    def _load_drug_database(self) -> Dict[str, DrugInfo]:
        """Load drug database.

        Returns:
            Dictionary of drug information
        """
        try:
            drug_data = pd.read_csv(Path(self.config['drug_database_path']))
            
            drugs_db = {}
            for _, row in drug_data.iterrows():
                drug = DrugInfo(
                    name=row['name'],
                    category=row['category'],
                    mechanism=row['mechanism'],
                    indications=row['indications'].split(';'),
                    contraindications=row['contraindications'].split(';'),
                    side_effects=row['side_effects'].split(';'),
                    interactions=row['interactions'].split(';'),
                    dosage_range={
                        'min': row['min_dosage'],
                        'max': row['max_dosage'],
                        'unit': row['dosage_unit']
                    },
                    monitoring_requirements=row['monitoring'].split(';')
                )
                drugs_db[row['name']] = drug
                
            return drugs_db
            
        except Exception as e:
            self.logger.error(f"Error loading drug database: {str(e)}")
            return {}

    def _load_treatment_guidelines(self) -> Dict[str, List[TreatmentGuideline]]:
        """Load treatment guidelines.

        Returns:
            Dictionary of treatment guidelines by cancer type
        """
        try:
            guidelines_data = pd.read_csv(Path(self.config['guidelines_path']))
            
            guidelines_db = {}
            for _, row in guidelines_data.iterrows():
                guideline = TreatmentGuideline(
                    cancer_type=row['cancer_type'],
                    stage=row['stage'],
                    grade=row['grade'],
                    patient_criteria=eval(row['patient_criteria']),
                    recommended_treatments=row['recommended_treatments'].split(';'),
                    treatment_sequence=row['treatment_sequence'].split(';'),
                    evidence_level=row['evidence_level'],
                    source=row['source'],
                    last_updated=datetime.strptime(row['last_updated'], '%Y-%m-%d')
                )
                
                if row['cancer_type'] not in guidelines_db:
                    guidelines_db[row['cancer_type']] = []
                guidelines_db[row['cancer_type']].append(guideline)
                
            return guidelines_db
            
        except Exception as e:
            self.logger.error(f"Error loading treatment guidelines: {str(e)}")
            return {}

    def _load_clinical_trials(self) -> List[ClinicalTrial]:
        """Load clinical trials database.

        Returns:
            List of clinical trials
        """
        try:
            trials_data = pd.read_csv(Path(self.config['trials_path']))
            
            trials = []
            for _, row in trials_data.iterrows():
                trial = ClinicalTrial(
                    trial_id=row['trial_id'],
                    title=row['title'],
                    description=row['description'],
                    phase=row['phase'],
                    status=row['status'],
                    criteria=eval(row['criteria']),
                    locations=row['locations'].split(';'),
                    contact_info=eval(row['contact_info']),
                    start_date=datetime.strptime(row['start_date'], '%Y-%m-%d'),
                    end_date=datetime.strptime(row['end_date'], '%Y-%m-%d') if pd.notna(row['end_date']) else None
                )
                trials.append(trial)
                
            return trials
            
        except Exception as e:
            self.logger.error(f"Error loading clinical trials: {str(e)}")
            return []

class TreatmentAdvisor:
    """Treatment advisor class."""
    
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        """Initialize treatment advisor.

        Args:
            knowledge_base: Medical knowledge base instance
        """
        self.kb = knowledge_base
        self.logger = logging.getLogger(__name__)

    def get_treatment_recommendations(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get treatment recommendations for a patient.

        Args:
            patient_data: Dictionary containing patient information

        Returns:
            Dictionary containing treatment recommendations
        """
        try:
            # Get relevant guidelines
            guidelines = self._find_matching_guidelines(patient_data)
            
            # Get drug recommendations
            drug_recommendations = self._get_drug_recommendations(patient_data, guidelines)
            
            # Check drug interactions
            drug_interactions = self._check_drug_interactions(drug_recommendations)
            
            # Find matching clinical trials
            matching_trials = self._find_matching_trials(patient_data)
            
            recommendations = {
                'treatment_plan': self._create_treatment_plan(guidelines),
                'drug_recommendations': drug_recommendations,
                'drug_interactions': drug_interactions,
                'clinical_trials': matching_trials,
                'monitoring_plan': self._create_monitoring_plan(drug_recommendations)
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating treatment recommendations: {str(e)}")
            return {}

    def _find_matching_guidelines(self, patient_data: Dict[str, Any]) -> List[TreatmentGuideline]:
        """Find treatment guidelines matching patient criteria.

        Args:
            patient_data: Dictionary containing patient information

        Returns:
            List of matching treatment guidelines
        """
        matching_guidelines = []
        
        if patient_data['cancer_type'] in self.kb.guidelines_db:
            for guideline in self.kb.guidelines_db[patient_data['cancer_type']]:
                if (guideline.stage == patient_data['stage'] and
                    guideline.grade == patient_data['grade'] and
                    self._check_patient_criteria(patient_data, guideline.patient_criteria)):
                    matching_guidelines.append(guideline)
        
        return matching_guidelines

    def _get_drug_recommendations(self, patient_data: Dict[str, Any],
                                guidelines: List[TreatmentGuideline]) -> List[DrugInfo]:
        """Get drug recommendations based on guidelines and patient data.

        Args:
            patient_data: Dictionary containing patient information
            guidelines: List of matching treatment guidelines

        Returns:
            List of recommended drugs
        """
        recommended_drugs = []
        
        for guideline in guidelines:
            for treatment in guideline.recommended_treatments:
                if treatment in self.kb.drugs_db:
                    drug = self.kb.drugs_db[treatment]
                    if self._check_drug_suitability(drug, patient_data):
                        recommended_drugs.append(drug)
        
        return recommended_drugs

    def _check_drug_interactions(self, drugs: List[DrugInfo]) -> List[Dict[str, Any]]:
        """Check for interactions between recommended drugs.

        Args:
            drugs: List of recommended drugs

        Returns:
            List of drug interactions
        """
        interactions = []
        
        for i, drug1 in enumerate(drugs):
            for drug2 in drugs[i+1:]:
                if drug2.name in drug1.interactions:
                    interactions.append({
                        'drug1': drug1.name,
                        'drug2': drug2.name,
                        'severity': 'high',
                        'description': f"Interaction between {drug1.name} and {drug2.name}"
                    })
        
        return interactions

    def _find_matching_trials(self, patient_data: Dict[str, Any]) -> List[ClinicalTrial]:
        """Find clinical trials matching patient criteria.

        Args:
            patient_data: Dictionary containing patient information

        Returns:
            List of matching clinical trials
        """
        matching_trials = []
        
        for trial in self.kb.trials_db:
            if (trial.status == 'Recruiting' and
                self._check_trial_eligibility(patient_data, trial.criteria)):
                matching_trials.append(trial)
        
        return matching_trials

    def _create_treatment_plan(self, guidelines: List[TreatmentGuideline]) -> Dict[str, Any]:
        """Create structured treatment plan from guidelines.

        Args:
            guidelines: List of treatment guidelines

        Returns:
            Dictionary containing treatment plan
        """
        if not guidelines:
            return {}
            
        # Use the most recent guideline
        guideline = max(guidelines, key=lambda g: g.last_updated)
        
        return {
            'phases': [
                {
                    'phase': i + 1,
                    'treatment': treatment,
                    'duration': '2-3 weeks',  # This should be customized
                    'description': f"Phase {i + 1}: {treatment}"
                }
                for i, treatment in enumerate(guideline.treatment_sequence)
            ],
            'evidence_level': guideline.evidence_level,
            'source': guideline.source
        }

    def _create_monitoring_plan(self, drugs: List[DrugInfo]) -> Dict[str, Any]:
        """Create monitoring plan based on drug requirements.

        Args:
            drugs: List of recommended drugs

        Returns:
            Dictionary containing monitoring plan
        """
        monitoring_plan = {
            'regular_tests': set(),
            'frequency': {},
            'parameters': {}
        }
        
        for drug in drugs:
            for requirement in drug.monitoring_requirements:
                monitoring_plan['regular_tests'].add(requirement)
                monitoring_plan['frequency'][requirement] = '2 weeks'  # This should be customized
                monitoring_plan['parameters'][requirement] = {
                    'normal_range': '0-100',  # This should be customized
                    'critical_values': '> 100'  # This should be customized
                }
        
        monitoring_plan['regular_tests'] = list(monitoring_plan['regular_tests'])
        return monitoring_plan

    @staticmethod
    def _check_patient_criteria(patient_data: Dict[str, Any],
                              criteria: Dict[str, Any]) -> bool:
        """Check if patient meets specific criteria.

        Args:
            patient_data: Dictionary containing patient information
            criteria: Dictionary of criteria to check

        Returns:
            True if patient meets criteria
        """
        for key, value in criteria.items():
            if key not in patient_data:
                return False
            if isinstance(value, (list, tuple)):
                if patient_data[key] not in value:
                    return False
            elif patient_data[key] != value:
                return False
        return True

    @staticmethod
    def _check_drug_suitability(drug: DrugInfo, patient_data: Dict[str, Any]) -> bool:
        """Check if drug is suitable for patient.

        Args:
            drug: Drug information
            patient_data: Dictionary containing patient information

        Returns:
            True if drug is suitable
        """
        # Check contraindications
        for contraindication in drug.contraindications:
            if contraindication in patient_data.get('conditions', []):
                return False
        
        # Check if cancer type is in indications
        if patient_data['cancer_type'] not in drug.indications:
            return False
            
        return True

    @staticmethod
    def _check_trial_eligibility(patient_data: Dict[str, Any],
                               criteria: Dict[str, Any]) -> bool:
        """Check if patient is eligible for clinical trial.

        Args:
            patient_data: Dictionary containing patient information
            criteria: Dictionary of eligibility criteria

        Returns:
            True if patient is eligible
        """
        for key, value in criteria.items():
            if key not in patient_data:
                return False
            if isinstance(value, dict):
                if 'min' in value and patient_data[key] < value['min']:
                    return False
                if 'max' in value and patient_data[key] > value['max']:
                    return False
            elif isinstance(value, (list, tuple)):
                if patient_data[key] not in value:
                    return False
            elif patient_data[key] != value:
                return False
        return True 