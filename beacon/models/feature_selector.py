import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.cluster import SpectralClustering
from ..core.base import BeaconBase

class AdaptiveFeatureSelector(BeaconBase):
    """Advanced adaptive feature selection for multimodal data"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature selector
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.feature_masks = {}
        self.importance_history = {}
        self.performance_history = {
            'original': [],
            'selected': []
        }
        self.thresholds = {}
        self.feature_interactions = {}
        self._initialize_metrics()
        self.selection_threshold = self.config.get('selection_threshold', 0.5)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        base_config = super()._get_default_config()
        base_config.update({
            'selection_method': 'ensemble',  # ensemble, attention, gradient, mutual_info, lasso, elastic_net
            'update_frequency': 5,
            'selection_threshold': 0.5,
            'min_features': 0.2,  # Minimum fraction of features to keep
            'smoothing_factor': 0.9,  # For exponential moving average
            'use_uncertainty': True,
            'modality_specific': True,
            'interaction_threshold': 0.3,
            'adaptive_threshold': True,
            'threshold_patience': 5,
            'threshold_delta': 0.05,
            'ensemble_weights': {
                'attention': 0.3,
                'gradient': 0.3,
                'mutual_info': 0.4
            },
            'lasso_alpha': 0.01,
            'elastic_net_alpha': 0.01,
            'elastic_net_l1_ratio': 0.5,
            'group_selection': {
                'enabled': False,
                'n_groups': 10,
                'method': 'spectral'  # spectral, hierarchical
            },
            'cross_modal': {
                'enabled': False,
                'interaction_threshold': 0.3,
                'fusion_method': 'weighted_sum'  # weighted_sum, attention
            }
        })
        return base_config
    
    def _initialize_metrics(self):
        """Initialize tracking metrics"""
        self.stability_scores = {}
        self.reduction_ratios = {}
        self.validation_scores = {}
    
    def initialize_masks(self, feature_dims: Dict[str, int]):
        """
        Initialize feature masks for each modality
        Args:
            feature_dims: Dictionary of feature dimensions for each modality
        """
        for modality, dim in feature_dims.items():
            self.feature_masks[modality] = torch.ones(dim)
            self.importance_history[modality] = []
            self.thresholds[modality] = self.config['selection_threshold']
            self.stability_scores[modality] = []
            self.reduction_ratios[modality] = []
            self.feature_interactions[modality] = {}
    
    def _compute_lasso_importance(self, 
                              features: torch.Tensor,
                              labels: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance using LASSO
        Args:
            features: Input features
            labels: Target labels
        Returns:
            Feature importance scores
        """
        # Convert to numpy for sklearn
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Fit LASSO
        lasso = Lasso(alpha=self.config['lasso_alpha'])
        lasso.fit(features_np, labels_np)
        
        # Get feature importance from coefficients
        importance = np.abs(lasso.coef_)
        return torch.from_numpy(importance).float()

    def _compute_elastic_net_importance(self,
                                    features: torch.Tensor,
                                    labels: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance using Elastic Net
        Args:
            features: Input features
            labels: Target labels
        Returns:
            Feature importance scores
        """
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        elastic = ElasticNet(
            alpha=self.config['elastic_net_alpha'],
            l1_ratio=self.config['elastic_net_l1_ratio']
        )
        elastic.fit(features_np, labels_np)
        
        importance = np.abs(elastic.coef_)
        return torch.from_numpy(importance).float()

    def _compute_importance_scores(self, 
                               batch: Dict[str, torch.Tensor],
                               model: nn.Module,
                               labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute feature importance scores using multiple methods
        Args:
            batch: Input batch
            model: Model instance
            labels: Optional labels for supervised methods
        Returns:
            Dictionary of importance scores per modality
        """
        if self.config['selection_method'] == 'lasso' and labels is not None:
            scores = {}
            for modality, features in batch.items():
                scores[modality] = self._compute_lasso_importance(features, labels)
        
        elif self.config['selection_method'] == 'elastic_net' and labels is not None:
            scores = {}
            for modality, features in batch.items():
                scores[modality] = self._compute_elastic_net_importance(features, labels)
        
        elif self.config['selection_method'] == 'ensemble':
            scores = {}
            for modality in batch.keys():
                ensemble_scores = torch.zeros_like(batch[modality][0])
                
                # Get attention-based scores
                if 'attention' in self.config['ensemble_weights']:
                    attention_scores = model.get_attention_weights(batch)[modality]
                    ensemble_scores += (self.config['ensemble_weights']['attention'] * 
                                     self._normalize_scores(attention_scores))
                
                # Get gradient-based scores
                if 'gradient' in self.config['ensemble_weights'] and labels is not None:
                    gradient_scores = self._compute_gradient_importance(batch[modality], 
                                                                     model, labels)
                    ensemble_scores += (self.config['ensemble_weights']['gradient'] * 
                                     self._normalize_scores(gradient_scores))
                
                # Get mutual information scores
                if 'mutual_info' in self.config['ensemble_weights'] and labels is not None:
                    mi_scores = self._compute_mutual_information(batch[modality], labels)
                    ensemble_scores += (self.config['ensemble_weights']['mutual_info'] * 
                                     self._normalize_scores(torch.from_numpy(mi_scores)))
                
                # Get LASSO scores
                if 'lasso' in self.config['ensemble_weights'] and labels is not None:
                    lasso_scores = self._compute_lasso_importance(batch[modality], labels)
                    ensemble_scores += (self.config['ensemble_weights']['lasso'] * 
                                     self._normalize_scores(lasso_scores))
                
                # Get Elastic Net scores
                if 'elastic_net' in self.config['ensemble_weights'] and labels is not None:
                    elastic_scores = self._compute_elastic_net_importance(batch[modality], labels)
                    ensemble_scores += (self.config['ensemble_weights']['elastic_net'] * 
                                     self._normalize_scores(elastic_scores))
                
                scores[modality] = ensemble_scores
        else:
            scores = self._compute_single_method_scores(batch, model, labels)
        
        # Apply cross-modal importance adjustment
        scores = self._compute_cross_modal_importance(batch, scores)
        
        return scores
    
    def _compute_gradient_importance(self, 
                                 features: torch.Tensor,
                                 model: nn.Module,
                                 labels: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient-based feature importance
        Args:
            features: Input features
            model: Model instance
            labels: Ground truth labels
        Returns:
            Gradient-based importance scores
        """
        features.requires_grad_(True)
        outputs = model(features)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        gradients = torch.autograd.grad(loss, features)[0]
        importance = torch.abs(gradients).mean(0)
        features.requires_grad_(False)
        return importance
    
    def _compute_mutual_information(self, 
                                features: torch.Tensor,
                                labels: torch.Tensor) -> np.ndarray:
        """
        Compute mutual information between features and labels
        Args:
            features: Input features
            labels: Ground truth labels
        Returns:
            Mutual information scores
        """
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mi_scores = mutual_info_classif(features_np, labels_np)
        return mi_scores
    
    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Normalize importance scores to [0, 1] range
        Args:
            scores: Raw importance scores
        Returns:
            Normalized importance scores
        """
        if torch.all(scores == 0):
            return scores
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    def _compute_feature_redundancy(self, 
                                features: torch.Tensor,
                                importance_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute feature redundancy based on correlation
        Args:
            features: Input features
            importance_scores: Feature importance scores
        Returns:
            Redundancy scores
        """
        # Compute feature correlations
        features_np = features.detach().cpu().numpy()
        correlations = np.corrcoef(features_np.T)
        
        # Compute redundancy scores
        redundancy = torch.zeros_like(importance_scores)
        for i in range(len(importance_scores)):
            # Find highly correlated features
            corr_features = np.where(np.abs(correlations[i]) > 0.8)[0]
            corr_features = corr_features[corr_features != i]
            
            if len(corr_features) > 0:
                # Reduce importance of redundant features
                redundancy[i] = importance_scores[corr_features].mean()
        
        return redundancy
    
    def _compute_single_method_scores(self,
                                    batch: Dict[str, torch.Tensor],
                                    model: nn.Module,
                                    labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute importance scores using a single method"""
        importance_scores = {}
        
        if self.config['selection_method'] == 'attention':
            attention_weights = model.get_attention_weights(batch)
            for modality in batch.keys():
                if modality in attention_weights:
                    importance_scores[modality] = attention_weights[modality].mean(0)
        
        elif self.config['selection_method'] == 'gradient':
            for modality, features in batch.items():
                features.requires_grad_(True)
                output = model(batch)
                grad = torch.autograd.grad(output.sum(), features)[0]
                importance_scores[modality] = grad.abs().mean(0)
                features.requires_grad_(False)
        
        elif self.config['selection_method'] == 'mutual_info':
            for modality, features in batch.items():
                if labels is not None:
                    mi_scores = mutual_info_classif(
                        features.detach().cpu().numpy(),
                        labels.detach().cpu().numpy()
                    )
                    importance_scores[modality] = torch.from_numpy(mi_scores).float()
                else:
                    # Fallback to correlation-based importance if labels not available
                    corr_matrix = torch.corrcoef(features.T)
                    importance_scores[modality] = corr_matrix.abs().mean(0)
        
        return importance_scores
    
    def _update_thresholds(self, performance_delta: float):
        """
        Update selection thresholds based on performance
        Args:
            performance_delta: Change in performance
        """
        if not self.config['adaptive_threshold']:
            return
            
        for modality in self.thresholds:
            if performance_delta > self.config['threshold_delta']:
                # Increase threshold if performance improves
                self.thresholds[modality] = min(
                    self.thresholds[modality] * 1.1,
                    0.9
                )
            elif performance_delta < -self.config['threshold_delta']:
                # Decrease threshold if performance degrades
                self.thresholds[modality] = max(
                    self.thresholds[modality] * 0.9,
                    self.config['min_features']
                )
    
    def _compute_feature_interactions(self, 
                                    batch: Dict[str, torch.Tensor],
                                    model: nn.Module) -> Dict[str, torch.Tensor]:
        """Compute interactions between features and modalities"""
        interactions = {}
        
        # Compute cross-modality correlations
        for modality1 in batch:
            for modality2 in batch:
                if modality1 < modality2:  # Avoid duplicate computations
                    corr = self._compute_cross_modal_correlation(
                        batch[modality1],
                        batch[modality2]
                    )
                    interactions[f"{modality1}_{modality2}"] = corr
        
        # Update feature interactions history
        for key, value in interactions.items():
            if key not in self.feature_interactions:
                self.feature_interactions[key] = []
            self.feature_interactions[key].append(value)
        
        return interactions
    
    def _compute_cross_modal_correlation(self,
                                       features1: torch.Tensor,
                                       features2: torch.Tensor) -> torch.Tensor:
        """Compute correlation between features from different modalities"""
        # Normalize features
        f1_norm = (features1 - features1.mean(0)) / (features1.std(0) + 1e-8)
        f2_norm = (features2 - features2.mean(0)) / (features2.std(0) + 1e-8)
        
        # Compute correlation matrix
        corr_matrix = torch.mm(f1_norm.T, f2_norm) / (len(features1) - 1)
        
        return corr_matrix
    
    def _compute_stability_score(self, modality: str) -> float:
        """Compute stability score for feature selection"""
        if len(self.importance_history[modality]) < 2:
            return 1.0
            
        # Compare current and previous feature masks
        prev_mask = self.importance_history[modality][-2] > self.thresholds[modality]
        curr_mask = self.importance_history[modality][-1] > self.thresholds[modality]
        
        # Compute Jaccard similarity
        intersection = (prev_mask & curr_mask).sum()
        union = (prev_mask | curr_mask).sum()
        
        return (intersection / union).item() if union > 0 else 1.0
    
    def _compute_reduction_ratio(self, modality: str) -> float:
        """Compute feature reduction ratio"""
        total_features = len(self.feature_masks[modality])
        selected_features = self.feature_masks[modality].sum()
        return 1.0 - (selected_features / total_features).item()
    
    def _identify_feature_groups(self, 
                             features: torch.Tensor,
                             importance_scores: torch.Tensor) -> Tuple[List[List[int]], torch.Tensor]:
        """
        Identify groups of related features
        Args:
            features: Input features
            importance_scores: Feature importance scores
        Returns:
            List of feature groups and group importance scores
        """
        # Compute feature similarity matrix
        features_np = features.detach().cpu().numpy()
        similarity = np.abs(np.corrcoef(features_np.T))
        
        # Apply spectral clustering
        n_groups = self.config['group_selection']['n_groups']
        clustering = SpectralClustering(
            n_clusters=n_groups,
            affinity='precomputed'
        )
        group_labels = clustering.fit_predict(similarity)
        
        # Organize features into groups
        groups = [[] for _ in range(n_groups)]
        for i, label in enumerate(group_labels):
            groups[label].append(i)
        
        # Compute group importance scores
        group_scores = torch.zeros(n_groups)
        for i, group in enumerate(groups):
            group_scores[i] = importance_scores[group].mean()
        
        return groups, group_scores

    def _select_features_by_group(self,
                              features: torch.Tensor,
                              importance_scores: torch.Tensor) -> torch.Tensor:
        """
        Select features based on group importance
        Args:
            features: Input features
            importance_scores: Feature importance scores
        Returns:
            Feature mask
        """
        if not self.config['group_selection']['enabled']:
            return importance_scores >= self.selection_threshold
        
        groups, group_scores = self._identify_feature_groups(features, importance_scores)
        
        # Create feature mask based on group importance
        mask = torch.zeros_like(importance_scores)
        sorted_groups = sorted(enumerate(groups), 
                             key=lambda x: group_scores[x[0]], 
                             reverse=True)
        
        total_features = len(importance_scores)
        min_features = int(total_features * self.config['min_features'])
        selected_features = 0
        
        for group_idx, group in sorted_groups:
            if selected_features >= min_features:
                break
            mask[group] = 1
            selected_features += len(group)
        
        return mask

    def _compute_cross_modal_importance(self,
                                    batch: Dict[str, torch.Tensor],
                                    base_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute feature importance considering cross-modal interactions
        Args:
            batch: Input batch
            base_scores: Base importance scores for each modality
        Returns:
            Adjusted importance scores
        """
        if not self.config['cross_modal']['enabled']:
            return base_scores
        
        adjusted_scores = {}
        interactions = self._compute_feature_interactions(batch)
        
        for modality1 in batch.keys():
            # Initialize with base scores
            adjusted_scores[modality1] = base_scores[modality1].clone()
            
            # Adjust based on cross-modal interactions
            for modality2 in batch.keys():
                if modality1 != modality2:
                    interaction_key = f"{modality1}_{modality2}"
                    if interaction_key in interactions:
                        interaction_strength = interactions[interaction_key].mean(1)
                        if self.config['cross_modal']['fusion_method'] == 'weighted_sum':
                            contribution = (interaction_strength * 
                                         base_scores[modality2].mean())
                            adjusted_scores[modality1] += (contribution * 
                                self.config['cross_modal']['interaction_threshold'])
                        elif self.config['cross_modal']['fusion_method'] == 'attention':
                            attention = torch.softmax(interaction_strength, dim=0)
                            contribution = (attention * base_scores[modality2].mean())
                            adjusted_scores[modality1] += contribution
        
        return adjusted_scores

    def update_feature_selection(self, 
                             batch: Dict[str, torch.Tensor],
                             model: nn.Module,
                             step: int,
                             labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Update feature selection based on importance scores
        Args:
            batch: Input batch
            model: Model instance
            step: Current training step
            labels: Optional labels
        Returns:
            Updated feature masks
        """
        if step % self.config['update_frequency'] != 0:
            return self.feature_masks
        
        # Compute importance scores
        importance_scores = self._compute_importance_scores(batch, model, labels)
        
        # Update feature masks for each modality
        for modality, scores in importance_scores.items():
            # Compute feature redundancy
            redundancy = self._compute_feature_redundancy(batch[modality], scores)
            
            # Adjust scores based on redundancy
            adjusted_scores = scores * (1 - redundancy)
            
            # Update importance history
            if modality not in self.importance_history:
                self.importance_history[modality] = []
            self.importance_history[modality].append(adjusted_scores)
            
            # Apply smoothing
            if len(self.importance_history[modality]) > 1:
                smoothed_scores = (self.config['smoothing_factor'] * adjusted_scores +
                                 (1 - self.config['smoothing_factor']) * 
                                 self.importance_history[modality][-2])
            else:
                smoothed_scores = adjusted_scores
            
            # Select features by group if enabled
            if self.config['group_selection']['enabled']:
                mask = self._select_features_by_group(batch[modality], smoothed_scores)
            else:
                # Update feature mask
                threshold = self.thresholds.get(modality, self.config['selection_threshold'])
                mask = smoothed_scores >= threshold
                
                # Ensure minimum number of features
                if mask.sum() < len(mask) * self.config['min_features']:
                    top_k = int(len(mask) * self.config['min_features'])
                    _, top_indices = torch.topk(smoothed_scores, top_k)
                    mask = torch.zeros_like(mask)
                    mask[top_indices] = True
            
            self.feature_masks[modality] = mask
            
            # Update metrics
            self.stability_scores[modality].append(self._compute_stability_score(modality))
            self.reduction_ratios[modality].append(self._compute_reduction_ratio(modality))
        
        return self.feature_masks
    
    def validate_selection(self,
                         val_batch: Dict[str, torch.Tensor],
                         model: nn.Module,
                         labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Validate feature selection performance
        Args:
            val_batch: Input batch
            model: Model instance
            labels: Optional labels
        Returns:
            Validation metrics
        """
        # Get predictions with original features
        with torch.no_grad():
            original_output = model(val_batch)
            if labels is not None:
                original_loss = nn.CrossEntropyLoss()(original_output, labels)
            else:
                original_loss = original_output.var(dim=1).mean()
        
        # Apply feature masks
        masked_batch = self.apply_masks(val_batch)
        
        # Get predictions with selected features
        with torch.no_grad():
            masked_output = model(masked_batch)
            if labels is not None:
                masked_loss = nn.CrossEntropyLoss()(masked_output, labels)
            else:
                masked_loss = masked_output.var(dim=1).mean()
        
        # Compute performance metrics
        metrics = {
            'original_performance': original_loss.item(),
            'selected_performance': masked_loss.item()
        }
        
        # Update performance history
        self.performance_history['original'].append(metrics['original_performance'])
        self.performance_history['selected'].append(metrics['selected_performance'])
        
        # Compute performance delta
        if len(self.performance_history['selected']) > 1:
            performance_delta = (self.performance_history['selected'][-1] -
                               self.performance_history['selected'][-2])
            self._update_thresholds(performance_delta)
        
        # Compute stability scores
        for modality in self.feature_masks:
            stability = self._compute_stability_score(modality)
            metrics[f'{modality}_stability'] = stability
        
        # Compute reduction ratios
        for modality in self.feature_masks:
            reduction = self._compute_reduction_ratio(modality)
            metrics[f'{modality}_reduction'] = reduction
        
        self.validation_scores = metrics
        return metrics
    
    def get_selected_features(self) -> Dict[str, torch.Tensor]:
        """Get current feature masks"""
        return self.feature_masks
    
    def get_importance_history(self) -> Dict[str, List[torch.Tensor]]:
        """Get importance score history"""
        return self.importance_history
    
    def apply_masks(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply feature masks to input batch
        Args:
            batch: Input batch
        Returns:
            Masked input batch
        """
        masked_batch = {}
        for modality, features in batch.items():
            if modality in self.feature_masks:
                mask = self.feature_masks[modality].to(features.device)
                masked_batch[modality] = features * mask
            else:
                masked_batch[modality] = features
        return masked_batch 