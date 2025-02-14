import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from beacon.models.gnn import MolecularGNN

@pytest.fixture
def sample_graph_data():
    """Create sample graph data for testing"""
    # Create random graph with 10 nodes
    num_nodes = 10
    num_edges = 20
    num_node_features = 32
    num_graphs = 4
    
    graphs = []
    for _ in range(num_graphs):
        # Node features
        x = torch.randn(num_nodes, num_node_features)
        
        # Random edges
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Random target (for binary classification)
        y = torch.randint(0, 2, (1,))
        
        # Create graph
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)
    
    # Create batch
    batch = Batch.from_data_list(graphs)
    return batch

@pytest.fixture
def model_config():
    """Create model configuration for testing"""
    return {
        'conv_type': 'gcn',
        'input_dim': 32,
        'hidden_dims': [64, 128],
        'output_dim': 2,
        'num_heads': 4,
        'dropout_rate': 0.3,
        'pooling': 'mean',
        'residual': True,
        'batch_norm': True,
        'task': 'classification'
    }

def test_model_initialization(model_config):
    """Test model initialization"""
    model = MolecularGNN(model_config)
    
    # Check model components
    assert hasattr(model.model, 'conv_layers')
    assert hasattr(model.model, 'output')
    assert hasattr(model.model, 'dropout')
    
    # Check number of layers
    assert len(model.model.conv_layers) == len(model_config['hidden_dims'])
    
    # Check layer dimensions
    first_layer = model.model.conv_layers[0]
    assert first_layer.in_channels == model_config['input_dim']
    assert first_layer.out_channels == model_config['hidden_dims'][0]

def test_forward_pass(model_config, sample_graph_data):
    """Test forward pass"""
    model = MolecularGNN(model_config)
    
    # Forward pass
    outputs = model.model(sample_graph_data)
    
    # Check output shape
    assert outputs.shape == (len(sample_graph_data.y), model_config['output_dim'])
    assert not torch.isnan(outputs).any()

def test_training_step(model_config, sample_graph_data):
    """Test training step"""
    model = MolecularGNN(model_config)
    
    # Training step
    loss, predictions = model.train_step(sample_graph_data)
    
    # Check outputs
    assert isinstance(loss, torch.Tensor)
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(sample_graph_data.y), model_config['output_dim'])
    assert not torch.isnan(loss)
    assert not torch.isnan(predictions).any()

def test_prediction(model_config, sample_graph_data):
    """Test prediction"""
    model = MolecularGNN(model_config)
    
    # Make predictions
    predictions = model.predict(sample_graph_data)
    
    # Check predictions
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(sample_graph_data.y), model_config['output_dim'])
    
    if model_config['task'] == 'classification':
        # Check if probabilities sum to 1
        assert torch.allclose(predictions.sum(dim=1), torch.ones(len(sample_graph_data.y)))
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)

def test_node_embeddings(model_config, sample_graph_data):
    """Test node embeddings extraction"""
    model = MolecularGNN(model_config)
    
    # Get node embeddings
    embeddings = model.get_node_embeddings(sample_graph_data)
    
    # Check embeddings
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == sample_graph_data.x.shape[0]  # num nodes
    assert embeddings.shape[1] == model_config['hidden_dims'][-1]  # embedding dim
    assert not torch.isnan(embeddings).any()

def test_gat_attention(sample_graph_data):
    """Test GAT attention weights"""
    # Configure GAT model
    gat_config = {
        'conv_type': 'gat',
        'input_dim': 32,
        'hidden_dims': [64, 128],
        'output_dim': 2,
        'num_heads': 4,
        'dropout_rate': 0.3,
        'pooling': 'mean',
        'residual': True,
        'batch_norm': True,
        'task': 'classification'
    }
    
    model = MolecularGNN(gat_config)
    
    # Get attention weights
    attention_weights = model.get_attention_weights(sample_graph_data)
    
    # Check attention weights
    assert attention_weights is not None
    assert len(attention_weights) == len(gat_config['hidden_dims'])
    
    for weights in attention_weights:
        assert isinstance(weights, torch.Tensor)
        assert weights.dim() == 3  # (num_edges, num_heads, 1)
        assert weights.shape[1] == gat_config['num_heads']
        assert not torch.isnan(weights).any()

def test_regression_task(model_config, sample_graph_data):
    """Test regression task"""
    # Modify config and data for regression
    model_config['task'] = 'regression'
    model_config['output_dim'] = 1
    sample_graph_data.y = torch.randn(len(sample_graph_data.y), 1)
    
    model = MolecularGNN(model_config)
    
    # Test training step
    loss, predictions = model.train_step(sample_graph_data)
    assert predictions.shape == (len(sample_graph_data.y), 1)
    
    # Test prediction
    predictions = model.predict(sample_graph_data)
    assert predictions.shape == (len(sample_graph_data.y), 1)

def test_different_pooling(model_config, sample_graph_data):
    """Test different pooling methods"""
    # Test max pooling
    model_config['pooling'] = 'max'
    model = MolecularGNN(model_config)
    
    outputs = model.model(sample_graph_data)
    assert outputs.shape == (len(sample_graph_data.y), model_config['output_dim'])

def test_without_batch_norm(model_config, sample_graph_data):
    """Test model without batch normalization"""
    model_config['batch_norm'] = False
    model = MolecularGNN(model_config)
    
    outputs = model.model(sample_graph_data)
    assert outputs.shape == (len(sample_graph_data.y), model_config['output_dim'])

def test_without_residual(model_config, sample_graph_data):
    """Test model without residual connections"""
    model_config['residual'] = False
    model = MolecularGNN(model_config)
    
    outputs = model.model(sample_graph_data)
    assert outputs.shape == (len(sample_graph_data.y), model_config['output_dim'])

def test_save_load(tmp_path, model_config, sample_graph_data):
    """Test model save and load functionality"""
    model = MolecularGNN(model_config)
    
    # Get predictions before saving
    original_predictions = model.predict(sample_graph_data)
    
    # Save model
    save_path = tmp_path / "gnn_model.pt"
    model.save(str(save_path))
    
    # Load model
    new_model = MolecularGNN(model_config)
    new_model.load(str(save_path))
    
    # Get predictions after loading
    loaded_predictions = new_model.predict(sample_graph_data)
    
    # Compare predictions
    assert torch.allclose(original_predictions, loaded_predictions) 