import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from ..core.base import BeaconModel
from typing import Dict, Any, List, Tuple, Optional

class MolecularGNN(BeaconModel):
    """Graph Neural Network for molecular structure analysis"""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            'conv_type': 'gcn',  # or 'gat'
            'input_dim': 32,
            'hidden_dims': [64, 128],
            'output_dim': 2,
            'num_heads': 4,  # for GAT
            'dropout_rate': 0.3,
            'pooling': 'mean',  # or 'max'
            'residual': True,
            'batch_norm': True,
            'learning_rate': 0.001
        }
    
    def _build_model(self) -> nn.Module:
        """Build GNN model architecture"""
        
        class GNNArchitecture(nn.Module):
            def __init__(self, config: Dict[str, Any]):
                super().__init__()
                self.config = config
                
                # Initialize graph convolution layers
                self.conv_layers = nn.ModuleList()
                self.batch_norms = nn.ModuleList()
                
                # Input layer
                if config['conv_type'] == 'gcn':
                    self.conv_layers.append(
                        GCNConv(config['input_dim'], config['hidden_dims'][0])
                    )
                else:  # GAT
                    self.conv_layers.append(
                        GATConv(
                            config['input_dim'],
                            config['hidden_dims'][0] // config['num_heads'],
                            heads=config['num_heads']
                        )
                    )
                
                if config['batch_norm']:
                    self.batch_norms.append(
                        nn.BatchNorm1d(config['hidden_dims'][0])
                    )
                
                # Hidden layers
                for i in range(len(config['hidden_dims']) - 1):
                    in_dim = config['hidden_dims'][i]
                    out_dim = config['hidden_dims'][i + 1]
                    
                    if config['conv_type'] == 'gcn':
                        self.conv_layers.append(GCNConv(in_dim, out_dim))
                    else:  # GAT
                        self.conv_layers.append(
                            GATConv(
                                in_dim,
                                out_dim // config['num_heads'],
                                heads=config['num_heads']
                            )
                        )
                    
                    if config['batch_norm']:
                        self.batch_norms.append(nn.BatchNorm1d(out_dim))
                
                # Output layer
                self.output = nn.Linear(
                    config['hidden_dims'][-1],
                    config['output_dim']
                )
                
                self.dropout = nn.Dropout(config['dropout_rate'])
            
            def forward(self, data) -> torch.Tensor:
                x, edge_index, batch = data.x, data.edge_index, data.batch
                
                # Initial features
                hidden = x
                
                # Graph convolution layers
                for i, conv in enumerate(self.conv_layers):
                    # Residual connection
                    identity = hidden
                    
                    # Apply convolution
                    hidden = conv(hidden, edge_index)
                    
                    # Batch normalization
                    if self.config['batch_norm']:
                        hidden = self.batch_norms[i](hidden)
                    
                    # Activation and dropout
                    hidden = F.relu(hidden)
                    hidden = self.dropout(hidden)
                    
                    # Add residual connection if dimensions match
                    if self.config['residual'] and hidden.shape == identity.shape:
                        hidden = hidden + identity
                
                # Global pooling
                if self.config['pooling'] == 'mean':
                    pooled = global_mean_pool(hidden, batch)
                else:  # max pooling
                    pooled = global_max_pool(hidden, batch)
                
                # Output layer
                out = self.output(pooled)
                return out
        
        # Create model
        model = GNNArchitecture(self.config).to(self.device)
        self.logger.info(f"Created GNN model with architecture:\n{model}")
        return model
    
    def train_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step
        Args:
            batch: Input batch from geometric DataLoader
        Returns:
            Tuple of (loss, predictions)
        """
        self.model.train()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Forward pass
        predictions = self.model(batch)
        
        # Calculate loss
        if self.config.get('task') == 'regression':
            criterion = nn.MSELoss()
            loss = criterion(predictions, batch.y)
        else:  # classification
            criterion = nn.CrossEntropyLoss()
            loss = criterion(predictions, batch.y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, predictions
    
    def predict(self, data: Any) -> torch.Tensor:
        """
        Make predictions
        Args:
            data: Input data
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            predictions = self.model(data)
            
            if self.config.get('task') == 'classification':
                predictions = F.softmax(predictions, dim=1)
            
            return predictions
    
    def get_node_embeddings(self, data: Any) -> torch.Tensor:
        """
        Get node embeddings from the last layer
        Args:
            data: Input data
        Returns:
            Node embeddings
        """
        self.model.eval()
        embeddings = []
        
        def hook_fn(module, input, output):
            embeddings.append(output)
        
        # Register hook on last convolution layer
        last_conv = self.model.conv_layers[-1]
        handle = last_conv.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            data = data.to(self.device)
            _ = self.model(data)
        
        # Remove hook
        handle.remove()
        
        return embeddings[0]
    
    def get_attention_weights(self, data: Any) -> Optional[List[torch.Tensor]]:
        """
        Get attention weights from GAT layers
        Args:
            data: Input data
        Returns:
            List of attention weights for each GAT layer
        """
        if self.config['conv_type'] != 'gat':
            return None
        
        self.model.eval()
        attention_weights = []
        
        def hook_fn(module, input, output):
            # GAT attention weights are in the additional output
            if isinstance(output, tuple):
                attention_weights.append(output[1])
        
        # Register hooks on GAT layers
        handles = []
        for conv in self.model.conv_layers:
            if isinstance(conv, GATConv):
                handle = conv.register_forward_hook(hook_fn)
                handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            data = data.to(self.device)
            _ = self.model(data)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return attention_weights 