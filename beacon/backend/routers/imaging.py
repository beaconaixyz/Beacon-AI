"""
Imaging data router for BEACON API

This module handles medical imaging data-related routes.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from typing import List
from beacon.backend.core.auth import get_current_active_user, check_permissions
from beacon.backend.db.database import get_db
from beacon.backend.models.user import User
from beacon.backend.core.config import settings
from scripts.data_processing.preprocess import DataPreprocessor
from scripts.training.train import ModelTrainer
from scripts.inference.predict import ModelPredictor

router = APIRouter()

@router.post("/upload")
async def upload_imaging_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload medical imaging data.

    Args:
        file: NPY file containing imaging data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Upload status
    """
    # Check permissions
    check_permissions(["upload_imaging"])

    try:
        # Validate file extension
        if not file.filename.endswith('.npy'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Only .npy files are supported."
            )

        # Read and validate data
        content = await file.read()
        with open(settings.UPLOAD_DIR / "temp.npy", "wb") as buffer:
            buffer.write(content)
        
        # Load and validate array shape
        data = np.load(settings.UPLOAD_DIR / "temp.npy")
        if len(data.shape) != 4:  # (samples, channels, height, width)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data shape. Expected 4D array (samples, channels, height, width)."
            )

        # Save file
        file_path = settings.UPLOAD_DIR / f"imaging_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        return {"message": "File uploaded successfully", "filename": file.filename}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/upload-labels")
async def upload_imaging_labels(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload labels for medical imaging data.

    Args:
        file: NPY file containing image labels
        current_user: Current authenticated user
        db: Database session

    Returns:
        Upload status
    """
    # Check permissions
    check_permissions(["upload_imaging"])

    try:
        # Validate file extension
        if not file.filename.endswith('.npy'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Only .npy files are supported."
            )

        # Read and validate data
        content = await file.read()
        with open(settings.UPLOAD_DIR / "temp_labels.npy", "wb") as buffer:
            buffer.write(content)
        
        # Load and validate array shape
        labels = np.load(settings.UPLOAD_DIR / "temp_labels.npy")
        if len(labels.shape) != 1:  # (samples,)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid labels shape. Expected 1D array."
            )

        # Save file
        file_path = settings.UPLOAD_DIR / f"imaging_labels_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        return {"message": "Labels uploaded successfully", "filename": file.filename}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/process")
async def process_imaging_data(
    data_filename: str,
    labels_filename: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Process medical imaging data.

    Args:
        data_filename: Name of the uploaded data file
        labels_filename: Name of the uploaded labels file
        current_user: Current authenticated user
        db: Database session

    Returns:
        Processing status
    """
    # Check permissions
    check_permissions(["process_imaging"])

    try:
        # Initialize preprocessor
        config = {
            'input_dir': str(settings.UPLOAD_DIR),
            'output_dir': str(settings.MODEL_PATH / 'processed')
        }
        preprocessor = DataPreprocessor(config)

        # Load and process data
        data = np.load(settings.UPLOAD_DIR / data_filename)
        processed_data = preprocessor.preprocess_imaging_data(data)
        
        # Load labels
        labels = np.load(settings.UPLOAD_DIR / labels_filename)
        
        # Save processed data
        np.save(settings.MODEL_PATH / 'processed/imaging_processed.npy', processed_data)
        np.save(settings.MODEL_PATH / 'processed/imaging_labels.npy', labels)

        return {"message": "Data processed successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/train")
async def train_imaging_model(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Train imaging model.

    Args:
        current_user: Current authenticated user
        db: Database session

    Returns:
        Training status
    """
    # Check permissions
    check_permissions(["train_imaging"])

    try:
        # Initialize trainer
        config = {
            'data_dir': str(settings.MODEL_PATH / 'processed'),
            'output_dir': str(settings.MODEL_PATH),
            'batch_size': settings.BATCH_SIZE,
            'num_workers': settings.NUM_WORKERS
        }
        trainer = ModelTrainer(config)

        # Train model
        trainer.train('imaging')

        return {"message": "Model trained successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/predict")
async def predict_imaging(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Make predictions on medical imaging data.

    Args:
        file: NPY file containing imaging data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Model predictions
    """
    # Check permissions
    check_permissions(["predict_imaging"])

    try:
        # Read and validate input data
        content = await file.read()
        with open(settings.UPLOAD_DIR / "temp_predict.npy", "wb") as buffer:
            buffer.write(content)
        
        data = np.load(settings.UPLOAD_DIR / "temp_predict.npy")
        
        # Initialize predictor
        config = {
            'model_dir': str(settings.MODEL_PATH),
            'output_dir': str(settings.MODEL_PATH / 'predictions')
        }
        predictor = ModelPredictor(config)

        # Make predictions
        predictions = predictor.predict('imaging', data)
        
        # Format results
        results = pd.DataFrame({
            'abnormality_prob': predictions.flatten()
        })

        return results.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 