import click
import yaml
import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any

from beacon.examples.integrated_analysis import (
    generate_synthetic_patient_data,
    process_data,
    train_models,
    evaluate_models,
    visualize_results
)
from beacon.data.loader import DataLoader
from beacon.utils.cross_validation import CrossValidator
from beacon.models.cancer_classifier import CancerClassifier
from beacon.models.image_classifier import MedicalImageCNN
from beacon.models.genomic_model import GenomicModel
from beacon.models.survival_model import SurvivalModel

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('beacon')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_dirs(config: Dict[str, Any]) -> None:
    """Create output directories if they don't exist"""
    Path(config['output']['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['output']['results_dir']).mkdir(parents=True, exist_ok=True)

def load_real_data(data_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load real data from files
    Args:
        data_dir: Path to data directory
        config: Configuration dictionary
    Returns:
        Dictionary containing loaded data
    """
    # Update config with data directory
    data_config = config.get('data', {})
    data_config['data_dir'] = data_dir
    
    # Initialize data loader
    loader = DataLoader(data_config)
    
    # Load all data
    data = loader.load_all_data()
    
    # Validate loaded data
    is_valid, error_msg = loader.validate_data(data)
    if not is_valid:
        raise ValueError(f"Invalid data: {error_msg}")
    
    return data

def train_with_cross_validation(data: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Train models using cross validation
    Args:
        data: Input data
        config: Configuration dictionary
        logger: Logger instance
    Returns:
        Dictionary of trained ensemble models
    """
    cv_config = config['cross_validation']
    validator = CrossValidator(cv_config)
    
    # Define model builders
    model_builders = {
        'cancer_classifier': lambda cfg: CancerClassifier(cfg),
        'image_classifier': lambda cfg: MedicalImageCNN(cfg),
        'genomic_model': lambda cfg: GenomicModel(cfg),
        'survival_model': lambda cfg: SurvivalModel(cfg)
    }
    
    # Train and evaluate each model type
    ensemble_models = {}
    for model_name, builder in model_builders.items():
        logger.info(f"\nTraining {model_name} with cross validation...")
        model_config = config['models'][model_name]
        
        # Create output directory for this model
        model_dir = Path(config['output']['model_dir']) / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Perform cross validation
        mean_metrics, fold_models = validator.cross_validate(
            data,
            builder,
            model_config,
            str(model_dir)
        )
        
        # Store ensemble model
        ensemble_models[model_name] = fold_models
        
        # Save metrics
        metrics_file = Path(config['output']['results_dir']) / f"{model_name}_cv_metrics.json"
        with open(metrics_file, 'w') as f:
            yaml.dump(mean_metrics, f, indent=4)
    
    return ensemble_models

@click.group()
def cli():
    """BEACON: AI Framework for Cancer Analysis"""
    pass

@cli.command()
@click.option('--config', '-c', default='config/default_config.yaml',
              help='Path to configuration file')
@click.option('--data-dir', '-d', default=None,
              help='Path to data directory (if using real data)')
@click.option('--output-dir', '-o', default='output',
              help='Path to output directory')
@click.option('--log-level', '-l', default='INFO',
              help='Logging level (DEBUG, INFO, WARNING, ERROR)')
@click.option('--cross-validate/--no-cross-validate', default=None,
              help='Enable/disable cross validation')
def run(config: str, data_dir: str, output_dir: str, log_level: str, cross_validate: bool):
    """Run integrated cancer analysis"""
    # Setup logging
    logger = setup_logging(log_level)
    logger.info("Starting BEACON analysis...")
    
    # Load configuration
    logger.info(f"Loading configuration from {config}")
    config = load_config(config)
    
    # Update output directories
    config['output']['model_dir'] = os.path.join(output_dir, 'models')
    config['output']['results_dir'] = os.path.join(output_dir, 'results')
    setup_output_dirs(config)
    
    # Override cross validation setting if specified
    if cross_validate is not None:
        config['cross_validation']['enabled'] = cross_validate
    
    # Set random seeds
    torch.manual_seed(config['training']['seed'])
    
    try:
        # Generate or load data
        if data_dir is None:
            logger.info("Generating synthetic data...")
            data = generate_synthetic_patient_data(
                n_samples=config['data']['n_samples']
            )
        else:
            logger.info(f"Loading data from {data_dir}...")
            data = load_real_data(data_dir, config)
        
        # Process data
        logger.info("Processing data...")
        processed_data = process_data(data)
        
        if config['cross_validation']['enabled']:
            # Train with cross validation
            logger.info("Training models with cross validation...")
            ensemble_models = train_with_cross_validation(processed_data, config, logger)
            
            # Make ensemble predictions
            logger.info("Making ensemble predictions...")
            validator = CrossValidator(config['cross_validation'])
            predictions = {}
            
            for model_name, models in ensemble_models.items():
                if model_name == 'survival_model':
                    pred_data = processed_data['clinical_features']
                elif model_name == 'image_classifier':
                    pred_data = processed_data['images']
                else:
                    pred_data = processed_data[f"{model_name}_features"]
                
                predictions[model_name] = validator.ensemble_predict(
                    models,
                    pred_data
                ).numpy()
            
            # Save predictions
            if config['output']['save_predictions']:
                predictions_dir = os.path.join(config['output']['results_dir'], 'predictions')
                loader = DataLoader(config['data'])
                loader.save_predictions(predictions, predictions_dir)
        
        else:
            # Train single models
            logger.info("Training models...")
            models = train_models(processed_data)
            
            # Evaluate models
            logger.info("Evaluating models...")
            results = evaluate_models(models, processed_data)
            
            # Save results
            if config['output']['save_models']:
                logger.info("Saving models...")
                for name, model in models.items():
                    save_path = os.path.join(config['output']['model_dir'], f"{name}.pt")
                    model.save(save_path)
            
            # Save predictions
            if config['output']['save_predictions']:
                logger.info("Saving predictions...")
                predictions = {
                    'cancer_classifier': models['cancer_classifier'].predict(
                        processed_data['clinical_features']
                    ).numpy(),
                    'image_classifier': models['image_classifier'].predict(
                        processed_data['images']
                    ).numpy(),
                    'genomic_model': models['genomic_model'].predict(
                        processed_data['genomic_features']
                    ).numpy(),
                    'survival_model': models['survival_model'].predict_risk(
                        processed_data['clinical_features']
                    ).numpy()
                }
                
                predictions_dir = os.path.join(config['output']['results_dir'], 'predictions')
                loader = DataLoader(config['data'])
                loader.save_predictions(predictions, predictions_dir)
            
            # Visualize results
            if config['output']['plot_metrics']:
                logger.info("Visualizing results...")
                visualize_results(results)
        
        logger.info("Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

@cli.command()
@click.argument('model_path')
@click.argument('data_path')
@click.option('--config', '-c', default='config/default_config.yaml',
              help='Path to configuration file')
@click.option('--output-dir', '-o', default='predictions',
              help='Path to output directory')
def predict(model_path: str, data_path: str, config: str, output_dir: str):
    """Make predictions using trained models"""
    logger = setup_logging()
    logger.info("Loading model and making predictions...")
    
    try:
        # Load configuration
        config = load_config(config)
        
        # Load data
        data = load_real_data(data_path, config)
        
        # Process data
        processed_data = process_data(data)
        
        # Load model
        model_name = Path(model_path).stem
        if model_name not in ['cancer_classifier', 'image_classifier', 
                            'genomic_model', 'survival_model']:
            raise ValueError(f"Unknown model type: {model_name}")
        
        model_class = {
            'cancer_classifier': CancerClassifier,
            'image_classifier': MedicalImageCNN,
            'genomic_model': GenomicModel,
            'survival_model': SurvivalModel
        }[model_name]
        
        model = model_class(config['models'][model_name])
        model.load(model_path)
        
        # Make predictions
        logger.info("Making predictions...")
        if model_name == 'image_classifier':
            predictions = model.predict(processed_data['images'])
        elif model_name == 'survival_model':
            predictions = model.predict_risk(processed_data['clinical_features'])
        else:
            predictions = model.predict(processed_data[f"{model_name}_features"])
        
        # Save predictions
        loader = DataLoader(config['data'])
        loader.save_predictions(
            {model_name: predictions.numpy()},
            output_dir
        )
        
        logger.info("Predictions complete!")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

@cli.command()
def init():
    """Initialize a new BEACON project"""
    logger = setup_logging()
    logger.info("Initializing new BEACON project...")
    
    # Create project structure
    dirs = ['data', 'config', 'models', 'results']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    
    # Create data subdirectories
    data_dirs = ['clinical', 'images', 'genomic', 'survival']
    for d in data_dirs:
        Path('data', d).mkdir(exist_ok=True)
    
    # Copy default configuration
    default_config = Path(__file__).parent / 'config' / 'default_config.yaml'
    if default_config.exists():
        with open(default_config, 'r') as src, open('config/config.yaml', 'w') as dst:
            dst.write(src.read())
    
    logger.info("Project initialized successfully!")

if __name__ == '__main__':
    cli() 