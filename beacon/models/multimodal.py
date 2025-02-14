import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from ..core.base import BeaconModel
from .image_classifier import MedicalImageCNN
from .gnn import MolecularGNN

class MultimodalFusion(BeaconModel):
    """Multimodal fusion model for medical data analysis"""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            # Image pathway
            'image': {
                'enabled': True,
                'in_channels': 1,
                'base_filters': 32,
                'n_blocks': 3
            },
            
            # Genomic pathway
            'genomic': {
                'enabled': True,
                'conv_type': 'gat',
                'input_dim': 119,
                'hidden_dims': [64, 128],
                'num_heads': 4
            },
            
            # Clinical pathway
            'clinical': {
                'enabled': True,
                'input_dim': 32,
                'hidden_dims': [64, 32]
            },
            
            # Fusion settings
            'fusion': {
                'method': 'attention',  # or 'concatenate'
                'hidden_dim': 256,
                'num_heads': 4,
                'dropout_rate': 0.3
            },
            
            # Output settings
            'output_dim': 2,
            'task': 'classification',  # or 'regression'
            'learning_rate': 0.001
        }
    
    def _build_model(self) -> nn.Module:
        """Build multimodal fusion model architecture"""
        
        class MultimodalArchitecture(nn.Module):
            def __init__(self, config: Dict[str, Any]):
                super().__init__()
                self.config = config
                
                # Initialize pathway models
                if config['image']['enabled']:
                    self.image_model = MedicalImageCNN({
                        'in_channels': config['image']['in_channels'],
                        'base_filters': config['image']['base_filters'],
                        'n_blocks': config['image']['n_blocks'],
                        'output_dim': config['fusion']['hidden_dim']
                    })
                
                if config['genomic']['enabled']:
                    self.genomic_model = MolecularGNN({
                        'conv_type': config['genomic']['conv_type'],
                        'input_dim': config['genomic']['input_dim'],
                        'hidden_dims': config['genomic']['hidden_dims'],
                        'output_dim': config['fusion']['hidden_dim'],
                        'num_heads': config['genomic']['num_heads']
                    })
                
                if config['clinical']['enabled']:
                    self.clinical_pathway = nn.Sequential(
                        nn.Linear(config['clinical']['input_dim'], 
                                config['clinical']['hidden_dims'][0]),
                        nn.ReLU(),
                        nn.Dropout(config['fusion']['dropout_rate']),
                        nn.Linear(config['clinical']['hidden_dims'][0],
                                config['fusion']['hidden_dim'])
                    )
                
                # Initialize fusion mechanism
                if config['fusion']['method'] == 'attention':
                    self.fusion = MultiheadAttentionFusion(
                        hidden_dim=config['fusion']['hidden_dim'],
                        num_heads=config['fusion']['num_heads'],
                        dropout_rate=config['fusion']['dropout_rate']
                    )
                else:  # concatenate
                    input_dim = config['fusion']['hidden_dim'] * sum([
                        config['image']['enabled'],
                        config['genomic']['enabled'],
                        config['clinical']['enabled']
                    ])
                    self.fusion = nn.Sequential(
                        nn.Linear(input_dim, config['fusion']['hidden_dim']),
                        nn.ReLU(),
                        nn.Dropout(config['fusion']['dropout_rate'])
                    )
                
                # Output layer
                self.output = nn.Linear(
                    config['fusion']['hidden_dim'],
                    config['output_dim']
                )
            
            def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
                features = []
                attention_mask = []
                
                # Process each modality
                if self.config['image']['enabled'] and 'image' in batch:
                    image_features = self.image_model(batch['image'])
                    features.append(image_features)
                    attention_mask.append(torch.ones(image_features.size(0), 1))
                
                if self.config['genomic']['enabled'] and 'genomic' in batch:
                    genomic_features = self.genomic_model(batch['genomic'])
                    features.append(genomic_features)
                    attention_mask.append(torch.ones(genomic_features.size(0), 1))
                
                if self.config['clinical']['enabled'] and 'clinical' in batch:
                    clinical_features = self.clinical_pathway(batch['clinical'])
                    features.append(clinical_features)
                    attention_mask.append(torch.ones(clinical_features.size(0), 1))
                
                # Stack features
                features = torch.stack(features, dim=1)  # [batch_size, n_modalities, hidden_dim]
                attention_mask = torch.cat(attention_mask, dim=1)  # [batch_size, n_modalities]
                
                # Apply fusion
                if self.config['fusion']['method'] == 'attention':
                    fused = self.fusion(features, attention_mask)
                else:  # concatenate
                    fused = self.fusion(features.view(features.size(0), -1))
                
                # Output layer
                return self.output(fused)
        
        class MultiheadAttentionFusion(nn.Module):
            def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True
                )
                self.layer_norm = nn.LayerNorm(hidden_dim)
                self.dropout = nn.Dropout(dropout_rate)
            
            def forward(self, features: torch.Tensor, 
                      attention_mask: torch.Tensor) -> torch.Tensor:
                # Self-attention
                attended, attention_weights = self.attention(
                    features, features, features,
                    key_padding_mask=~attention_mask.bool()
                )
                
                # Residual connection and normalization
                features = features + self.dropout(attended)
                features = self.layer_norm(features)
                
                # Pool across modalities
                return torch.mean(features, dim=1)
        
        # Create model
        model = MultimodalArchitecture(self.config).to(self.device)
        self.logger.info(f"Created multimodal fusion model with architecture:\n{model}")
        return model
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step
        Args:
            batch: Dictionary containing data from each modality
        Returns:
            Tuple of (loss, predictions)
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        predictions = self.model(batch)
        
        # Calculate loss
        if self.config['task'] == 'regression':
            criterion = nn.MSELoss()
            loss = criterion(predictions, batch['target'])
        else:  # classification
            criterion = nn.CrossEntropyLoss()
            loss = criterion(predictions, batch['target'])
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, predictions
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Make predictions
        Args:
            batch: Dictionary containing data from each modality
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(batch)
            
            if self.config['task'] == 'classification':
                predictions = F.softmax(predictions, dim=1)
            
            return predictions
    
    def get_attention_weights(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Get attention weights between modalities
        Args:
            batch: Dictionary containing data from each modality
        Returns:
            Attention weights if using attention fusion
        """
        if self.config['fusion']['method'] != 'attention':
            return None
        
        self.model.eval()
        attention_weights = []
        
        def hook_fn(module, input, output):
            # Get attention weights from output tuple
            attention_weights.append(output[1])
        
        # Register hook on attention layer
        handle = self.model.fusion.attention.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            _ = self.model(batch)
        
        # Remove hook
        handle.remove()
        
        return attention_weights[0]
    
    def get_modality_embeddings(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get embeddings from each modality before fusion
        Args:
            batch: Dictionary containing data from each modality
        Returns:
            Dictionary of embeddings for each modality
        """
        self.model.eval()
        embeddings = {}
        
        with torch.no_grad():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            if self.config['image']['enabled'] and 'image' in batch:
                embeddings['image'] = self.model.image_model(batch['image'])
            
            if self.config['genomic']['enabled'] and 'genomic' in batch:
                embeddings['genomic'] = self.model.genomic_model(batch['genomic'])
            
            if self.config['clinical']['enabled'] and 'clinical' in batch:
                embeddings['clinical'] = self.model.clinical_pathway(batch['clinical'])
        
        return embeddings 