import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from beacon.models.multimodal import MultimodalFusion
from beacon.examples.gnn_example import mol_to_graph
from typing import List, Dict, Tuple
import seaborn as sns

def generate_synthetic_data(n_samples: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Generate synthetic data for all modalities
    Args:
        n_samples: Number of samples to generate
    Returns:
        Tuple of (train_data, val_data)
    """
    # Generate image data (simulated medical images)
    def generate_image_batch(size: int) -> torch.Tensor:
        images = []
        for _ in range(size):
            # Create a synthetic medical image with a circular pattern
            x = np.linspace(-10, 10, 64)
            y = np.linspace(-10, 10, 64)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            Z = np.exp(-0.1 * R) * np.sin(R)
            
            # Add noise
            Z += np.random.normal(0, 0.1, Z.shape)
            
            # Normalize to [0, 1]
            Z = (Z - Z.min()) / (Z.max() - Z.min())
            
            images.append(Z)
        
        return torch.FloatTensor(images).unsqueeze(1)  # [B, 1, H, W]
    
    # Generate genomic data (molecular graphs)
    def generate_genomic_batch(size: int) -> List[Data]:
        smiles_list = [
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(=O)CC(C1=CC=C(C=C1)[N+](=O)[O-])CC(=O)C',  # Random molecule
        ]
        
        graphs = []
        for _ in range(size):
            smiles = np.random.choice(smiles_list)
            mol = Chem.MolFromSmiles(smiles)
            graph = mol_to_graph(mol)
            graphs.append(graph)
        
        return graphs
    
    # Generate clinical data
    def generate_clinical_batch(size: int) -> torch.Tensor:
        # Simulate clinical features (age, blood pressure, lab results, etc.)
        features = np.zeros((size, 32))
        
        # Age (normalized)
        features[:, 0] = np.random.normal(0.5, 0.15, size)
        
        # Blood pressure (systolic, diastolic)
        features[:, 1] = np.random.normal(120, 20, size) / 200  # Systolic
        features[:, 2] = np.random.normal(80, 10, size) / 200   # Diastolic
        
        # Lab results (random values)
        features[:, 3:] = np.random.normal(0, 1, (size, 29))
        
        return torch.FloatTensor(features)
    
    # Generate data
    train_size = int(0.8 * n_samples)
    val_size = n_samples - train_size
    
    # Training data
    train_data = {
        'image': generate_image_batch(train_size),
        'genomic': generate_genomic_batch(train_size),
        'clinical': generate_clinical_batch(train_size),
        'target': torch.randint(0, 2, (train_size,))  # Binary classification
    }
    
    # Validation data
    val_data = {
        'image': generate_image_batch(val_size),
        'genomic': generate_genomic_batch(val_size),
        'clinical': generate_clinical_batch(val_size),
        'target': torch.randint(0, 2, (val_size,))
    }
    
    return train_data, val_data

class MultimodalDataset:
    """Dataset class for handling multimodal data"""
    
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data['target'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {}
        
        # Get image
        if 'image' in self.data:
            sample['image'] = self.data['image'][idx]
        
        # Get genomic data
        if 'genomic' in self.data:
            sample['genomic'] = self.data['genomic'][idx]
        
        # Get clinical data
        if 'clinical' in self.data:
            sample['clinical'] = self.data['clinical'][idx]
        
        # Get target
        sample['target'] = self.data['target'][idx]
        
        return sample

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    collated = {}
    
    # Collate images
    if 'image' in batch[0]:
        collated['image'] = torch.stack([b['image'] for b in batch])
    
    # Collate genomic data (using Batch from PyTorch Geometric)
    if 'genomic' in batch[0]:
        collated['genomic'] = Batch.from_data_list([b['genomic'] for b in batch])
    
    # Collate clinical data
    if 'clinical' in batch[0]:
        collated['clinical'] = torch.stack([b['clinical'] for b in batch])
    
    # Collate targets
    collated['target'] = torch.stack([b['target'] for b in batch])
    
    return collated

def plot_attention_heatmap(attention_weights: torch.Tensor, modalities: List[str]):
    """
    Plot attention weights between modalities
    Args:
        attention_weights: Attention weights tensor [batch_size, n_modalities, n_modalities]
        modalities: List of modality names
    """
    # Average attention weights across batch
    avg_weights = attention_weights.mean(0).cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_weights, 
                xticklabels=modalities,
                yticklabels=modalities,
                annot=True,
                fmt='.2f',
                cmap='viridis')
    plt.title('Cross-modal Attention Weights')
    plt.tight_layout()
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
    """Example of using the multimodal fusion model"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    n_samples = 100
    train_data, val_data = generate_synthetic_data(n_samples)
    
    # Create datasets and dataloaders
    train_dataset = MultimodalDataset(train_data)
    val_dataset = MultimodalDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Model configuration
    model_config = {
        'image': {
            'enabled': True,
            'in_channels': 1,
            'base_filters': 32,
            'n_blocks': 3
        },
        'genomic': {
            'enabled': True,
            'conv_type': 'gat',
            'input_dim': 119,
            'hidden_dims': [64, 128],
            'num_heads': 4
        },
        'clinical': {
            'enabled': True,
            'input_dim': 32,
            'hidden_dims': [64, 32]
        },
        'fusion': {
            'method': 'attention',
            'hidden_dim': 256,
            'num_heads': 4,
            'dropout_rate': 0.3
        },
        'output_dim': 2,
        'task': 'classification',
        'learning_rate': 0.001
    }
    
    # Initialize model
    print("\nInitializing multimodal fusion model...")
    model = MultimodalFusion(model_config)
    
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
            train_total += batch['target'].size(0)
            train_correct += predicted.eq(batch['target']).sum().item()
        
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
                val_total += batch['target'].size(0)
                val_correct += predicted.eq(batch['target']).sum().item()
        
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
    
    # Analyze a sample batch
    print("\nAnalyzing sample batch...")
    sample_batch = next(iter(val_loader))
    
    # Get modality embeddings
    embeddings = model.get_modality_embeddings(sample_batch)
    print("\nModality embedding shapes:")
    for modality, emb in embeddings.items():
        print(f"{modality}: {emb.shape}")
    
    # Get attention weights
    if model_config['fusion']['method'] == 'attention':
        attention_weights = model.get_attention_weights(sample_batch)
        print("\nVisualizing cross-modal attention weights...")
        plot_attention_heatmap(
            attention_weights,
            ['Image', 'Genomic', 'Clinical']
        )
    
    # Save model
    print("\nSaving model...")
    model.save('multimodal_model.pt')
    
    print("\nDone!")

if __name__ == "__main__":
    main() 