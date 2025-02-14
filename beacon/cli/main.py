import click
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from ..models.medical_cnn import MedicalCNN
from ..data.image_processor import ImageProcessor
from ..interpretability.image_interpreter import ImageInterpreter
from ..visualization.visualizer import Visualizer
from ..evaluation.metrics import Metrics

@click.group()
def cli():
    """BEACON: Medical Image Analysis Framework"""
    pass

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output path for trained model')
@click.option('--device', '-d', default='cuda', help='Device to use (cuda/cpu)')
def train(config_path: str, data_path: str, output: str, device: str):
    """Train a model using the specified configuration and data"""
    # Load configuration
    config = load_config(config_path)
    config['device'] = device
    
    # Initialize components
    image_processor = ImageProcessor(config.get('image_processor', {}))
    model = MedicalCNN(config.get('model', {}))
    
    # Load and process data
    click.echo("Loading and processing data...")
    train_loader = image_processor.create_dataloader(
        data_path + '/train',
        batch_size=config.get('batch_size', 32)
    )
    val_loader = image_processor.create_dataloader(
        data_path + '/val',
        batch_size=config.get('batch_size', 32)
    )
    
    # Train model
    click.echo("Training model...")
    history = model.fit(train_loader, val_loader)
    
    # Save model and history
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_model(str(output_path / 'model.pth'))
        save_config(history, str(output_path / 'history.json'))
    
    click.echo("Training completed!")

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output path for results')
@click.option('--device', '-d', default='cuda', help='Device to use (cuda/cpu)')
def evaluate(model_path: str, data_path: str, output: str, device: str):
    """Evaluate a trained model on test data"""
    # Load model
    model = MedicalCNN({'device': device})
    model.load_model(model_path)
    
    # Initialize components
    image_processor = ImageProcessor({})
    metrics = Metrics({})
    
    # Load and process data
    click.echo("Loading and processing data...")
    test_loader = image_processor.create_dataloader(
        data_path,
        batch_size=32
    )
    
    # Evaluate model
    click.echo("Evaluating model...")
    results = model.evaluate(test_loader)
    
    # Calculate metrics
    evaluation_metrics = metrics.calculate_classification_metrics(
        results['true_labels'],
        results['predicted_labels'],
        results['prediction_scores']
    )
    
    # Save results
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        save_config(evaluation_metrics, str(output_path / 'metrics.json'))
    
    click.echo("Evaluation completed!")
    click.echo(f"Metrics: {evaluation_metrics}")

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output path for visualizations')
@click.option('--method', '-m', default='integrated_gradients', help='Interpretation method')
@click.option('--device', '-d', default='cuda', help='Device to use (cuda/cpu)')
def interpret(model_path: str, image_path: str, output: str, method: str, device: str):
    """Generate and visualize model interpretations"""
    # Load model
    model = MedicalCNN({'device': device})
    model.load_model(model_path)
    
    # Initialize components
    image_processor = ImageProcessor({})
    interpreter = ImageInterpreter(model, {'method': method})
    visualizer = Visualizer({})
    
    # Load and process image
    click.echo("Processing image...")
    image = image_processor.process_image(image_path)
    
    # Generate interpretation
    click.echo("Generating interpretation...")
    interpretation = interpreter.interpret(image.unsqueeze(0))
    
    # Visualize results
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save attribution map visualization
        visualizer.visualize_attribution(
            image,
            interpretation['attribution'],
            str(output_path / 'attribution.png')
        )
        
        # Save interpretation statistics
        stats = interpreter.get_interpretation_stats(interpretation['attribution'])
        save_config(stats, str(output_path / 'stats.json'))
    
    click.echo("Interpretation completed!")

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output path for results')
@click.option('--device', '-d', default='cuda', help='Device to use (cuda/cpu)')
def analyze(config_path: str, model_path: str, data_path: str, output: str, device: str):
    """Perform comprehensive analysis on a dataset"""
    # Load configuration
    config = load_config(config_path)
    config['device'] = device
    
    # Initialize components
    model = MedicalCNN(config.get('model', {}))
    model.load_model(model_path)
    
    image_processor = ImageProcessor(config.get('image_processor', {}))
    interpreter = ImageInterpreter(model, config.get('interpreter', {}))
    visualizer = Visualizer(config.get('visualizer', {}))
    metrics = Metrics(config.get('metrics', {}))
    
    # Load and process data
    click.echo("Loading and processing data...")
    data_loader = image_processor.create_dataloader(
        data_path,
        batch_size=config.get('batch_size', 32)
    )
    
    # Perform analysis
    click.echo("Performing analysis...")
    results = {
        'predictions': [],
        'interpretations': [],
        'metrics': []
    }
    
    for batch in data_loader:
        # Get predictions
        predictions = model.predict(batch)
        results['predictions'].append(predictions)
        
        # Generate interpretations
        interpretations = interpreter.interpret(batch)
        results['interpretations'].append(interpretations)
        
        # Calculate metrics
        batch_metrics = metrics.calculate_classification_metrics(
            batch[1] if isinstance(batch, tuple) else None,
            predictions.argmax(dim=1)
        )
        results['metrics'].append(batch_metrics)
    
    # Aggregate results
    aggregated_metrics = metrics.aggregate_metrics(results['metrics'])
    
    # Save results
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        save_config(aggregated_metrics, str(output_path / 'metrics.json'))
        
        # Save visualizations
        for i, (pred, interp) in enumerate(zip(results['predictions'], results['interpretations'])):
            visualizer.visualize_attribution(
                batch[0][i],
                interp['attribution'],
                str(output_path / f'sample_{i}_attribution.png')
            )
    
    click.echo("Analysis completed!")
    click.echo(f"Aggregated metrics: {aggregated_metrics}")

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    path = Path(path)
    
    if path.suffix == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    elif path.suffix in ['.yml', '.yaml']:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def save_config(config: Dict[str, Any], path: str):
    """Save configuration to file"""
    path = Path(path)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    elif path.suffix in ['.yml', '.yaml']:
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

if __name__ == '__main__':
    cli() 