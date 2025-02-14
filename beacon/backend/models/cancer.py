"""
Cancer diagnosis and treatment models for BEACON

This module implements specialized models for cancer diagnosis, prognosis,
and treatment recommendation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import logging

class MultiModalEncoder(nn.Module):
    """Multi-modal data encoder for cancer diagnosis."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Clinical data encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(config['clinical_dim'], 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Imaging data encoder (using CNN)
        self.imaging_encoder = nn.Sequential(
            nn.Conv2d(config['image_channels'], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (config['image_size']//4) * (config['image_size']//4), 128)
        )
        
        # Genomic data encoder
        self.genomic_encoder = nn.Sequential(
            nn.Linear(config['genomic_dim'], 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(384, 256),  # 384 = 128 * 3 (clinical + imaging + genomic)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, clinical: torch.Tensor, imaging: torch.Tensor, 
                genomic: torch.Tensor) -> torch.Tensor:
        """Forward pass through the multi-modal encoder.

        Args:
            clinical: Clinical data tensor
            imaging: Imaging data tensor
            genomic: Genomic data tensor

        Returns:
            Fused representation tensor
        """
        clinical_features = self.clinical_encoder(clinical)
        imaging_features = self.imaging_encoder(imaging)
        genomic_features = self.genomic_encoder(genomic)
        
        # Concatenate features
        combined = torch.cat([clinical_features, imaging_features, genomic_features], dim=1)
        
        # Fuse features
        fused = self.fusion(combined)
        
        return fused

class CancerDiagnosisModel(nn.Module):
    """Cancer diagnosis model combining multi-modal data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.encoder = MultiModalEncoder(config)
        
        # Diagnosis head
        self.diagnosis_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, config['num_cancer_types'])
        )
        
        # Stage prediction head
        self.stage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 4)  # 4 stages: I, II, III, IV
        )
        
        # Grade prediction head
        self.grade_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # 3 grades: Low, Medium, High
        )

    def forward(self, clinical: torch.Tensor, imaging: torch.Tensor,
                genomic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the cancer diagnosis model.

        Args:
            clinical: Clinical data tensor
            imaging: Imaging data tensor
            genomic: Genomic data tensor

        Returns:
            Tuple of (cancer_type_probs, stage_probs, grade_probs)
        """
        # Get fused features
        features = self.encoder(clinical, imaging, genomic)
        
        # Get predictions
        cancer_type_logits = self.diagnosis_head(features)
        stage_logits = self.stage_head(features)
        grade_logits = self.grade_head(features)
        
        # Apply softmax
        cancer_type_probs = F.softmax(cancer_type_logits, dim=1)
        stage_probs = F.softmax(stage_logits, dim=1)
        grade_probs = F.softmax(grade_logits, dim=1)
        
        return cancer_type_probs, stage_probs, grade_probs

class TreatmentRecommendationModel(nn.Module):
    """Treatment recommendation model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.encoder = MultiModalEncoder(config)
        
        # Treatment recommendation head
        self.treatment_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, config['num_treatments'])
        )
        
        # Drug combination head
        self.drug_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, config['num_drugs'])
        )
        
        # Dosage optimization head
        self.dosage_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, config['num_drugs'])  # One dosage per drug
        )

    def forward(self, clinical: torch.Tensor, imaging: torch.Tensor,
                genomic: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the treatment recommendation model.

        Args:
            clinical: Clinical data tensor
            imaging: Imaging data tensor
            genomic: Genomic data tensor

        Returns:
            Tuple of (treatment_probs, drug_probs, dosage_predictions)
        """
        # Get fused features
        features = self.encoder(clinical, imaging, genomic)
        
        # Get predictions
        treatment_logits = self.treatment_head(features)
        drug_logits = self.drug_head(features)
        dosage_predictions = self.dosage_head(features)
        
        # Apply activation functions
        treatment_probs = F.softmax(treatment_logits, dim=1)
        drug_probs = torch.sigmoid(drug_logits)  # Multi-label drug selection
        
        return treatment_probs, drug_probs, dosage_predictions

class ModelExplainer:
    """Model explainability class."""
    
    def __init__(self, model: nn.Module):
        """Initialize model explainer.

        Args:
            model: PyTorch model to explain
        """
        self.model = model
        self.logger = logging.getLogger(__name__)

    def compute_feature_importance(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Compute feature importance using integrated gradients.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Dictionary of feature importance scores
        """
        importances = {}
        for name, tensor in inputs.items():
            # Create baseline (zeros)
            baseline = torch.zeros_like(tensor)
            
            # Compute gradients
            tensor.requires_grad = True
            outputs = self.model(**{name: tensor})
            
            if isinstance(outputs, tuple):
                output = outputs[0]  # Use first output for importance
            else:
                output = outputs
                
            gradients = torch.autograd.grad(output.sum(), tensor)[0]
            
            # Compute importance scores
            importance = (tensor - baseline) * gradients
            importances[name] = importance.detach().cpu().numpy()
            
        return importances

    def explain_decision(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Generate explanation for model decision.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Dictionary containing explanation components
        """
        explanation = {
            'feature_importance': self.compute_feature_importance(inputs),
            'confidence_scores': self._compute_confidence_scores(inputs),
            'decision_factors': self._identify_key_factors(inputs)
        }
        
        return explanation

    def _compute_confidence_scores(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute confidence scores for predictions.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Dictionary of confidence scores
        """
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if isinstance(outputs, tuple):
                scores = {}
                for i, output in enumerate(outputs):
                    probs = F.softmax(output, dim=1)
                    confidence = probs.max(dim=1)[0]
                    scores[f'output_{i}_confidence'] = confidence.mean().item()
            else:
                probs = F.softmax(outputs, dim=1)
                scores = {'confidence': probs.max(dim=1)[0].mean().item()}
                
        return scores

    def _identify_key_factors(self, inputs: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Identify key factors influencing the decision.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            List of key factors with their importance
        """
        # Get feature importance
        importance_scores = self.compute_feature_importance(inputs)
        
        key_factors = []
        for name, scores in importance_scores.items():
            # Find top influential features
            flat_scores = scores.reshape(scores.shape[0], -1)
            top_indices = np.argsort(-np.abs(flat_scores), axis=1)[:, :5]  # Top 5 features
            
            for i in range(scores.shape[0]):
                for idx in top_indices[i]:
                    key_factors.append({
                        'input_type': name,
                        'feature_index': idx,
                        'importance_score': float(flat_scores[i, idx])
                    })
        
        return key_factors 