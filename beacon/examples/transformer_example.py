import torch
import numpy as np
import matplotlib.pyplot as plt
from beacon.models.transformer import TransformerModel
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple

class SequenceDataset(Dataset):
    """Dataset class for sequence data"""
    
    def __init__(self, sequences: torch.Tensor, 
                 lengths: torch.Tensor,
                 labels: torch.Tensor):
        self.sequences = sequences
        self.lengths = lengths
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'sequences': self.sequences[idx],
            'lengths': self.lengths[idx],
            'targets': self.labels[idx]
        }

def generate_synthetic_data(n_samples: int,
                          seq_length: int,
                          input_dim: int,
                          n_classes: int = 2) -> Tuple[Dataset, Dataset]:
    """
    Generate synthetic sequence data
    Args:
        n_samples: Number of samples
        seq_length: Maximum sequence length
        input_dim: Input dimension
        n_classes: Number of classes
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Generate sequences with varying lengths
    sequences = torch.randn(n_samples, seq_length, input_dim)
    lengths = torch.randint(seq_length//2, seq_length + 1, (n_samples,))
    
    # Generate labels based on sequence patterns
    labels = []
    for i in range(n_samples):
        # Add some pattern for classification
        if torch.mean(sequences[i, :lengths[i], 0]) > 0:
            labels.append(1)
        else:
            labels.append(0)
    labels = torch.tensor(labels)
    
    # Split into train and validation
    train_size = int(0.8 * n_samples)
    
    train_dataset = SequenceDataset(
        sequences[:train_size],
        lengths[:train_size],
        labels[:train_size]
    )
    
    val_dataset = SequenceDataset(
        sequences[train_size:],
        lengths[train_size:],
        labels[train_size:]
    )
    
    return train_dataset, val_dataset

def visualize_attention(attention_weights: torch.Tensor,
                       sequence_length: int,
                       save_path: str = None):
    """
    Visualize attention weights
    Args:
        attention_weights: Attention weights tensor
        sequence_length: Actual sequence length
        save_path: Optional path to save the plot
    """
    # Average attention weights across heads
    avg_weights = attention_weights.mean(dim=1)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_weights[:sequence_length, :sequence_length],
              cmap='viridis')
    plt.colorbar()
    plt.title('Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
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
    """Example of using the Transformer model"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    n_samples = 1000
    seq_length = 50
    input_dim = 64
    train_dataset, val_dataset = generate_synthetic_data(
        n_samples, seq_length, input_dim
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Model configuration
    model_config = {
        'input_dim': input_dim,
        'num_heads': 4,
        'num_layers': 2,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'max_seq_length': seq_length,
        'positional_encoding': 'sinusoidal',
        'task': 'classification',
        'output_dim': 2
    }
    
    # Initialize model
    print("\nInitializing Transformer model...")
    model = TransformerModel(model_config)
    
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
            train_total += batch['targets'].size(0)
            train_correct += predicted.eq(batch['targets']).sum().item()
        
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
                val_total += batch['targets'].size(0)
                val_correct += predicted.eq(batch['targets']).sum().item()
        
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
    
    # Visualize attention weights
    print("\nVisualizing attention weights...")
    sample_batch = next(iter(val_loader))
    attention_weights = model.get_attention_weights(sample_batch)
    
    # Visualize first layer's attention for first sequence
    visualize_attention(
        attention_weights[0][0],  # First layer, first sample
        sample_batch['lengths'][0],  # Actual sequence length
        'attention_weights.png'
    )
    
    # Save model
    print("\nSaving model...")
    model.save('transformer_model.pt')
    
    print("\nDone!")

if __name__ == "__main__":
    main() 