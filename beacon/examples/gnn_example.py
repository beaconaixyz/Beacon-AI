import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch, DataLoader
from beacon.models.gnn import MolecularGNN
from typing import List, Dict, Tuple
import networkx as nx

def mol_to_graph(mol: Chem.Mol) -> Data:
    """
    Convert RDKit molecule to PyTorch Geometric graph
    Args:
        mol: RDKit molecule
    Returns:
        PyTorch Geometric Data object
    """
    # Get node features (atomic numbers)
    atomic_numbers = []
    for atom in mol.GetAtoms():
        atomic_numbers.append(atom.GetAtomicNum())
    
    x = torch.tensor(atomic_numbers, dtype=torch.long)
    
    # One-hot encode node features
    num_atom_types = 119  # Maximum atomic number + 1
    x = torch.nn.functional.one_hot(x, num_atom_types).float()
    
    # Get edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
    
    return Data(x=x, edge_index=edge_index)

def generate_synthetic_molecules(n_samples: int) -> List[Chem.Mol]:
    """
    Generate synthetic molecules for testing
    Args:
        n_samples: Number of molecules to generate
    Returns:
        List of RDKit molecules
    """
    molecules = []
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(=O)CC(C1=CC=C(C=C1)[N+](=O)[O-])CC(=O)C',  # Random molecule
    ]
    
    for _ in range(n_samples):
        # Randomly select a SMILES string
        smiles = np.random.choice(smiles_list)
        mol = Chem.MolFromSmiles(smiles)
        
        # Generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        
        molecules.append(mol)
    
    return molecules

def visualize_molecule(mol: Chem.Mol,
                      node_weights: torch.Tensor = None,
                      save_path: str = None):
    """
    Visualize molecule with optional node weights
    Args:
        mol: RDKit molecule
        node_weights: Optional node importance weights
        save_path: Optional path to save the plot
    """
    # Convert to NetworkX graph
    g = nx.Graph()
    
    # Add nodes
    for i, atom in enumerate(mol.GetAtoms()):
        g.add_node(i, symbol=atom.GetSymbol())
    
    # Add edges
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        g.add_edge(i, j)
    
    # Set up plot
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(g)
    
    # Draw edges
    nx.draw_networkx_edges(g, pos)
    
    # Draw nodes
    if node_weights is not None:
        # Normalize weights for coloring
        weights = node_weights.numpy()
        vmin, vmax = weights.min(), weights.max()
        node_colors = plt.cm.viridis((weights - vmin) / (vmax - vmin))
    else:
        node_colors = 'lightblue'
    
    nx.draw_networkx_nodes(g, pos, node_color=node_colors)
    
    # Add labels
    labels = {i: data['symbol'] for i, data in g.nodes(data=True)}
    nx.draw_networkx_labels(g, pos, labels)
    
    plt.title('Molecular Graph')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_training_history(history: Dict[str, List[float]]):
    """
    Plot training history
    Args:
        history: Dictionary containing loss and accuracy values
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Example of using the GNN model for molecular property prediction"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic molecules
    print("Generating synthetic molecules...")
    n_samples = 100
    molecules = generate_synthetic_molecules(n_samples)
    
    # Convert molecules to graphs
    graphs = []
    for mol in molecules:
        graph = mol_to_graph(mol)
        # Add random binary property (e.g., active/inactive)
        graph.y = torch.randint(0, 2, (1,))
        graphs.append(graph)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    
    # Model configuration
    model_config = {
        'conv_type': 'gat',  # Use GAT for better interpretability
        'input_dim': 119,    # Number of atom types
        'hidden_dims': [64, 128],
        'output_dim': 2,     # Binary classification
        'num_heads': 4,
        'dropout_rate': 0.3,
        'pooling': 'mean',
        'residual': True,
        'batch_norm': True,
        'task': 'classification'
    }
    
    # Initialize model
    print("\nInitializing GNN model...")
    model = MolecularGNN(model_config)
    
    # Training loop
    print("\nTraining model...")
    n_epochs = 10
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(n_epochs):
        # Training
        model.model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            loss, predictions = model.train_step(batch)
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = predictions.max(1)
            train_total += batch.y.size(0)
            train_correct += predicted.eq(batch.y).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss, predictions = model.train_step(batch)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = predictions.max(1)
                val_total += batch.y.size(0)
                val_correct += predicted.eq(batch.y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}\n")
    
    # Plot training history
    plot_training_history(history)
    
    # Analyze a sample molecule
    print("\nAnalyzing sample molecule...")
    sample_mol = molecules[0]
    sample_graph = mol_to_graph(sample_mol)
    
    # Get node embeddings
    embeddings = model.get_node_embeddings(sample_graph)
    
    # Get attention weights (for GAT)
    attention_weights = model.get_attention_weights(sample_graph)
    
    # Calculate node importance as mean attention
    if attention_weights:
        node_importance = torch.mean(attention_weights[-1], dim=(1, 2))
        print("\nVisualizing molecular graph with attention weights...")
        visualize_molecule(sample_mol, node_importance, 'molecule_attention.png')
    
    # Save model
    print("\nSaving model...")
    model.save('gnn_model.pt')
    
    print("\nDone!")

if __name__ == "__main__":
    main() 