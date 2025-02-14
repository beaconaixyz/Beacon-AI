import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from beacon.utils.metrics import Metrics

class AdvancedVisualizer:
    """Advanced visualization tools for feature selection analysis"""
    
    def __init__(self, config: Dict):
        """
        Initialize visualizer
        Args:
            config: Configuration dictionary
        """
        self.config = config
        plt.style.use('seaborn')
    
    def plot_feature_importance_evolution(self,
                                        importance_history: Dict[str, List[torch.Tensor]],
                                        save_path: Optional[str] = None):
        """
        Plot evolution of feature importance over time
        Args:
            importance_history: History of importance scores
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(12, 6))
        
        for modality, history in importance_history.items():
            history_array = torch.stack(history).numpy()
            mean_importance = history_array.mean(axis=1)
            std_importance = history_array.std(axis=1)
            
            plt.plot(mean_importance, label=modality)
            plt.fill_between(
                range(len(mean_importance)),
                mean_importance - std_importance,
                mean_importance + std_importance,
                alpha=0.3
            )
        
        plt.xlabel('Time Step')
        plt.ylabel('Feature Importance')
        plt.title('Evolution of Feature Importance')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_feature_correlation_network(self,
                                       correlation_matrix: torch.Tensor,
                                       feature_names: List[str],
                                       threshold: float = 0.5,
                                       save_path: Optional[str] = None):
        """
        Plot feature correlation network
        Args:
            correlation_matrix: Feature correlation matrix
            feature_names: List of feature names
            threshold: Correlation threshold for drawing edges
            save_path: Optional path to save plot
        """
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, name in enumerate(feature_names):
            G.add_node(i, name=name)
        
        # Add edges for correlations above threshold
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(correlation_matrix[i, j]) > threshold:
                    G.add_edge(i, j, weight=abs(correlation_matrix[i, j]))
        
        plt.figure(figsize=(12, 12))
        
        # Draw network
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(
            G, pos,
            width=[G[u][v]['weight'] * 2 for u, v in G.edges()]
        )
        nx.draw_networkx_labels(G, pos, {i: data['name'] for i, data in G.nodes(data=True)})
        
        plt.title('Feature Correlation Network')
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_feature_embedding(self,
                             features: torch.Tensor,
                             importance_scores: torch.Tensor,
                             method: str = 'tsne',
                             save_path: Optional[str] = None):
        """
        Plot feature embedding with importance scores
        Args:
            features: Feature matrix
            importance_scores: Feature importance scores
            method: Embedding method ('tsne' or 'pca')
            save_path: Optional path to save plot
        """
        # Convert to numpy
        features_np = features.numpy()
        scores_np = importance_scores.numpy()
        
        # Apply dimensionality reduction
        if method == 'tsne':
            embedding = TSNE(n_components=2).fit_transform(features_np.T)
        else:
            embedding = PCA(n_components=2).fit_transform(features_np.T)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=scores_np,
            cmap='viridis',
            s=100
        )
        plt.colorbar(scatter, label='Feature Importance')
        
        plt.title(f'Feature Embedding ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_temporal_feature_patterns(self,
                                     temporal_patterns: Dict[str, torch.Tensor],
                                     save_path: Optional[str] = None):
        """
        Plot temporal patterns in feature selection
        Args:
            temporal_patterns: Dictionary of temporal patterns
            save_path: Optional path to save plot
        """
        n_modalities = len(temporal_patterns)
        fig, axes = plt.subplots(n_modalities, 1, figsize=(12, 4 * n_modalities))
        
        if n_modalities == 1:
            axes = [axes]
        
        for ax, (modality, patterns) in zip(axes, temporal_patterns.items()):
            patterns_np = patterns.numpy()
            
            sns.heatmap(
                patterns_np,
                ax=ax,
                cmap='coolwarm',
                center=0,
                cbar_kws={'label': 'Pattern Strength'}
            )
            ax.set_title(f'{modality} Temporal Patterns')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Feature Index')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_feature_stability(self,
                             stability_scores: Dict[str, torch.Tensor],
                             save_path: Optional[str] = None):
        """
        Plot feature selection stability
        Args:
            stability_scores: Dictionary of stability scores
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        positions = range(len(stability_scores))
        plt.boxplot(
            [scores.numpy() for scores in stability_scores.values()],
            labels=list(stability_scores.keys())
        )
        
        plt.ylabel('Stability Score')
        plt.title('Feature Selection Stability Across Modalities')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_performance_impact(self,
                              performance_metrics: Dict[str, List[float]],
                              feature_counts: List[int],
                              save_path: Optional[str] = None):
        """
        Plot impact of feature selection on model performance
        Args:
            performance_metrics: Dictionary of performance metrics
            feature_counts: List of feature counts
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        for metric_name, values in performance_metrics.items():
            plt.plot(feature_counts, values, marker='o', label=metric_name)
        
        plt.xlabel('Number of Selected Features')
        plt.ylabel('Performance Score')
        plt.title('Performance Impact of Feature Selection')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_group_interactions(self,
                              interaction_matrix: torch.Tensor,
                              group_names: List[str],
                              save_path: Optional[str] = None):
        """
        Plot interactions between feature groups
        Args:
            interaction_matrix: Matrix of group interactions
            group_names: Names of feature groups
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            interaction_matrix.numpy(),
            xticklabels=group_names,
            yticklabels=group_names,
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'Interaction Strength'}
        )
        
        plt.title('Feature Group Interactions')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def create_interactive_dashboard(self,
                                   results: Dict[str, Dict],
                                   save_path: Optional[str] = None):
        """
        Create comprehensive interactive dashboard
        Args:
            results: Dictionary of analysis results
            save_path: Optional path to save dashboard
        """
        # Create multi-panel figure
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Feature importance evolution
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_importance_evolution_subplot(results['importance_history'], ax1)
        
        # Performance impact
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_performance_impact_subplot(
            results['performance_metrics'],
            results['feature_counts'],
            ax2
        )
        
        # Feature stability
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_stability_subplot(results['stability_scores'], ax3)
        
        # Group interactions
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_interactions_subplot(
            results['interaction_matrix'],
            results['group_names'],
            ax4
        )
        
        # Temporal patterns
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_temporal_patterns_subplot(results['temporal_patterns'], ax5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def _plot_importance_evolution_subplot(self,
                                         importance_history: Dict[str, List[torch.Tensor]],
                                         ax: plt.Axes):
        """Plot feature importance evolution subplot"""
        for modality, history in importance_history.items():
            history_array = torch.stack(history).numpy()
            mean_importance = history_array.mean(axis=1)
            ax.plot(mean_importance, label=modality)
        
        ax.set_title('Feature Importance Evolution')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Importance')
        ax.legend()
    
    def _plot_performance_impact_subplot(self,
                                       performance_metrics: Dict[str, List[float]],
                                       feature_counts: List[int],
                                       ax: plt.Axes):
        """Plot performance impact subplot"""
        for metric_name, values in performance_metrics.items():
            ax.plot(feature_counts, values, marker='o', label=metric_name)
        
        ax.set_title('Performance Impact')
        ax.set_xlabel('Feature Count')
        ax.set_ylabel('Score')
        ax.legend()
    
    def _plot_stability_subplot(self,
                              stability_scores: Dict[str, torch.Tensor],
                              ax: plt.Axes):
        """Plot stability subplot"""
        ax.boxplot(
            [scores.numpy() for scores in stability_scores.values()],
            labels=list(stability_scores.keys())
        )
        ax.set_title('Selection Stability')
        ax.set_ylabel('Stability Score')
    
    def _plot_interactions_subplot(self,
                                 interaction_matrix: torch.Tensor,
                                 group_names: List[str],
                                 ax: plt.Axes):
        """Plot interactions subplot"""
        sns.heatmap(
            interaction_matrix.numpy(),
            xticklabels=group_names,
            yticklabels=group_names,
            cmap='coolwarm',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Interaction'}
        )
        ax.set_title('Group Interactions')
    
    def _plot_temporal_patterns_subplot(self,
                                      temporal_patterns: Dict[str, torch.Tensor],
                                      ax: plt.Axes):
        """Plot temporal patterns subplot"""
        for modality, patterns in temporal_patterns.items():
            patterns_np = patterns.numpy()
            sns.heatmap(
                patterns_np,
                ax=ax,
                cmap='coolwarm',
                center=0,
                cbar_kws={'label': 'Pattern Strength'}
            )
            ax.set_title(f'{modality} Temporal Patterns') 