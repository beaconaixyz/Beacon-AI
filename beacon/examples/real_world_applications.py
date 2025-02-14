import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from beacon.models.feature_selector import AdaptiveFeatureSelector
from beacon.models.multimodal import MultimodalFusion
from beacon.data.processor import DataProcessor
from beacon.utils.metrics import Metrics
from beacon.visualization.visualizer import Visualizer

def clinical_risk_stratification_example():
    """
    Example of using feature selection for clinical risk stratification
    - Demonstrates feature selection on clinical data
    - Shows how to identify key risk factors
    - Includes visualization of risk groups
    """
    # Generate synthetic clinical data
    n_samples = 1000
    n_features = 50
    clinical_data = torch.randn(n_samples, n_features)
    risk_scores = torch.sigmoid(clinical_data[:, 0] * 0.5 + clinical_data[:, 1] * 0.3)
    
    # Configure feature selector
    selector_config = {
        'n_features': n_features,
        'selection_method': 'lasso',
        'adaptive_threshold': True,
        'modalities': ['clinical']
    }
    
    # Initialize selector
    selector = AdaptiveFeatureSelector(selector_config)
    
    # Select features
    selected_features = selector.select_features({
        'clinical': clinical_data
    })
    
    # Analyze feature importance
    importance_scores = selector.get_feature_importance()
    
    # Visualize results
    visualizer = Visualizer({})
    visualizer.plot_feature_importance(importance_scores['clinical'])
    visualizer.plot_risk_stratification(risk_scores, selected_features['clinical'])

def genomic_biomarker_discovery():
    """
    Example of using feature selection for genomic biomarker discovery
    - Shows how to handle high-dimensional genomic data
    - Demonstrates group-wise feature selection
    - Includes pathway analysis visualization
    """
    # Generate synthetic genomic data
    n_samples = 500
    n_genes = 1000
    genomic_data = torch.randn(n_samples, n_genes)
    
    # Add some correlated gene groups
    for i in range(0, n_genes, 50):
        genomic_data[:, i:i+10] = genomic_data[:, i:i+10] * 0.7 + torch.randn(n_samples, 10) * 0.3
    
    # Configure feature selector
    selector_config = {
        'n_features': n_genes,
        'selection_method': 'group_lasso',
        'group_size': 50,
        'modalities': ['genomic']
    }
    
    # Initialize selector
    selector = AdaptiveFeatureSelector(selector_config)
    
    # Select features
    selected_features = selector.select_features({
        'genomic': genomic_data
    })
    
    # Analyze group structure
    groups = selector.analyze_feature_groups()
    
    # Visualize results
    visualizer = Visualizer({})
    visualizer.plot_gene_groups(groups['genomic'])
    visualizer.plot_pathway_enrichment(selected_features['genomic'])

def multimodal_disease_progression():
    """
    Example of using feature selection for disease progression modeling
    - Demonstrates multimodal feature selection
    - Shows temporal feature analysis
    - Includes progression visualization
    """
    # Generate synthetic longitudinal data
    n_samples = 200
    n_timepoints = 5
    
    # Clinical features
    clinical_data = torch.randn(n_samples, n_timepoints, 20)
    # Imaging features
    imaging_data = torch.randn(n_samples, n_timepoints, 100)
    # Genomic features
    genomic_data = torch.randn(n_samples, n_timepoints, 500)
    
    # Configure feature selector
    selector_config = {
        'n_features': {
            'clinical': 20,
            'imaging': 100,
            'genomic': 500
        },
        'selection_method': 'temporal_group_lasso',
        'temporal_consistency': True,
        'modalities': ['clinical', 'imaging', 'genomic']
    }
    
    # Initialize selector
    selector = AdaptiveFeatureSelector(selector_config)
    
    # Select features
    selected_features = selector.select_features({
        'clinical': clinical_data,
        'imaging': imaging_data,
        'genomic': genomic_data
    })
    
    # Analyze temporal patterns
    temporal_patterns = selector.analyze_temporal_patterns()
    
    # Visualize results
    visualizer = Visualizer({})
    visualizer.plot_temporal_patterns(temporal_patterns)
    visualizer.plot_disease_progression(selected_features)

def treatment_response_prediction():
    """
    Example of using feature selection for treatment response prediction
    - Shows how to handle treatment-specific features
    - Demonstrates interaction analysis
    - Includes response prediction visualization
    """
    # Generate synthetic treatment response data
    n_samples = 300
    n_features = {
        'clinical': 30,
        'genomic': 200,
        'treatment': 10
    }
    
    # Generate data
    clinical_data = torch.randn(n_samples, n_features['clinical'])
    genomic_data = torch.randn(n_samples, n_features['genomic'])
    treatment_data = torch.randn(n_samples, n_features['treatment'])
    
    # Add treatment-specific effects
    response = torch.sigmoid(
        clinical_data[:, 0] * 0.3 +
        genomic_data[:, 0] * 0.4 +
        treatment_data[:, 0] * 0.5 +
        clinical_data[:, 0] * treatment_data[:, 0] * 0.2  # Interaction effect
    )
    
    # Configure feature selector
    selector_config = {
        'n_features': n_features,
        'selection_method': 'interaction_lasso',
        'interaction_analysis': True,
        'modalities': ['clinical', 'genomic', 'treatment']
    }
    
    # Initialize selector
    selector = AdaptiveFeatureSelector(selector_config)
    
    # Select features
    selected_features = selector.select_features({
        'clinical': clinical_data,
        'genomic': genomic_data,
        'treatment': treatment_data
    })
    
    # Analyze interactions
    interactions = selector.analyze_feature_interactions()
    
    # Visualize results
    visualizer = Visualizer({})
    visualizer.plot_treatment_interactions(interactions)
    visualizer.plot_response_prediction(response, selected_features)

def main():
    """Run all examples"""
    print("Running clinical risk stratification example...")
    clinical_risk_stratification_example()
    
    print("\nRunning genomic biomarker discovery example...")
    genomic_biomarker_discovery()
    
    print("\nRunning multimodal disease progression example...")
    multimodal_disease_progression()
    
    print("\nRunning treatment response prediction example...")
    treatment_response_prediction()

if __name__ == "__main__":
    main() 