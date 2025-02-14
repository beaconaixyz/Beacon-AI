#!/usr/bin/env python3

"""
Model Serving Script for BEACON

This script provides a REST API for model inference using FastAPI.
"""

import os
import yaml
import torch
import uvicorn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import io

from beacon.models import BeaconModel
from beacon.utils.logger import setup_logger
from beacon.utils.preprocessing import preprocess_input
from beacon.utils.postprocessing import postprocess_output

# Initialize FastAPI app
app = FastAPI(
    title="BEACON Model Serving API",
    description="REST API for serving BEACON model predictions",
    version="1.0.0"
)

class PredictionInput(BaseModel):
    """Input data model for predictions"""
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    """Response data model for predictions"""
    predictions: Dict[str, List[float]]
    metadata: Optional[Dict[str, Any]] = None

class ModelServer:
    """Handles model serving and inference"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the model server.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.model_path = Path(config['model_path'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logger(
            name="server",
            log_file=self.output_dir / "serving.log"
        )
        
        # Load model
        self.logger.info(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = BeaconModel(checkpoint['config']['model'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessing transformers
        self.transformers = self._load_transformers()
        
        # Initialize request counter
        self.request_counter = 0

    def _load_transformers(self) -> Dict[str, Any]:
        """Load preprocessing transformers.

        Returns:
            Dictionary of transformers
        """
        transformers = {}
        transformer_dir = self.model_path.parent / 'transformers'
        
        if transformer_dir.exists():
            import joblib
            
            # Load scalers
            scaler_dir = transformer_dir / 'scalers'
            if scaler_dir.exists():
                for scaler_file in scaler_dir.glob('*.joblib'):
                    feature_name = scaler_file.stem.replace('_scaler', '')
                    transformers[f'{feature_name}_scaler'] = joblib.load(scaler_file)
            
            # Load encoders
            encoder_dir = transformer_dir / 'encoders'
            if encoder_dir.exists():
                for encoder_file in encoder_dir.glob('*.joblib'):
                    feature_name = encoder_file.stem.replace('_encoder', '')
                    transformers[f'{feature_name}_encoder'] = joblib.load(encoder_file)
        
        return transformers

    async def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess input data.

        Args:
            data: Input data dictionary

        Returns:
            Dictionary of preprocessed tensors
        """
        try:
            preprocessed = preprocess_input(
                data,
                transformers=self.transformers,
                config=self.config['preprocessing']
            )
            
            # Convert to tensors
            tensor_data = {}
            for key, value in preprocessed.items():
                if isinstance(value, np.ndarray):
                    tensor_data[key] = torch.from_numpy(value).to(self.device)
                elif isinstance(value, torch.Tensor):
                    tensor_data[key] = value.to(self.device)
                else:
                    raise ValueError(f"Unsupported data type for {key}: {type(value)}")
            
            return tensor_data
        
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    async def postprocess_predictions(
        self,
        predictions: torch.Tensor,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, List[float]]:
        """Postprocess model predictions.

        Args:
            predictions: Raw model predictions
            confidence_threshold: Optional confidence threshold for binary predictions

        Returns:
            Dictionary containing processed predictions
        """
        try:
            # Convert to numpy
            predictions_np = predictions.cpu().numpy()
            
            # Apply postprocessing
            processed = postprocess_output(
                predictions_np,
                config=self.config['postprocessing']
            )
            
            # Apply confidence threshold if specified
            if confidence_threshold is not None:
                processed['predicted_class'] = (
                    processed['probabilities'] > confidence_threshold
                ).astype(int)
            
            # Convert numpy arrays to lists for JSON serialization
            return {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in processed.items()
            }
        
        except Exception as e:
            self.logger.error(f"Error in postprocessing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Postprocessing error: {str(e)}")

    async def predict(
        self,
        data: Dict[str, Any],
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, List[float]]:
        """Generate predictions for input data.

        Args:
            data: Input data dictionary
            confidence_threshold: Optional confidence threshold for binary predictions

        Returns:
            Dictionary containing predictions
        """
        try:
            # Increment request counter
            self.request_counter += 1
            
            # Preprocess data
            preprocessed = await self.preprocess_data(data)
            
            # Generate predictions
            with torch.no_grad():
                predictions = self.model(preprocessed)
            
            # Postprocess predictions
            results = await self.postprocess_predictions(
                predictions,
                confidence_threshold
            )
            
            # Log request
            self.logger.info(
                f"Processed request {self.request_counter} - "
                f"Input shape: {data['inputs'].shape if 'inputs' in data else len(data)}"
            )
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize model server
model_server = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model server on startup."""
    global model_server
    
    # Load configuration
    config_path = os.environ.get('BEACON_CONFIG_PATH', 'config/serving.yml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize server
    model_server = ModelServer(config)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BEACON Model Serving API",
        "status": "active",
        "endpoints": [
            "/predict",
            "/predict/image",
            "/health",
            "/metrics"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Generate predictions for input data.

    Args:
        input_data: Input data model containing data and optional metadata

    Returns:
        Predictions and optional metadata
    """
    predictions = await model_server.predict(
        input_data.data,
        confidence_threshold=model_server.config['inference'].get('confidence_threshold')
    )
    
    return PredictionResponse(
        predictions=predictions,
        metadata=input_data.metadata
    )

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Generate predictions for an input image.

    Args:
        file: Uploaded image file

    Returns:
        Predictions for the image
    """
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Generate predictions
        predictions = await model_server.predict(
            {'image': image_array},
            confidence_threshold=model_server.config['inference'].get('confidence_threshold')
        )
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_server is not None,
        "device": str(model_server.device)
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint."""
    return {
        "total_requests": model_server.request_counter,
        "model_device": str(model_server.device),
        "model_path": str(model_server.model_path)
    }

def main():
    """Run the model serving application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start BEACON model serving")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--config", required=True, help="Path to serving config file")
    args = parser.parse_args()
    
    # Set config path environment variable
    os.environ['BEACON_CONFIG_PATH'] = args.config
    
    # Run server
    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main() 