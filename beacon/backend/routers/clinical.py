"""
Clinical data router for BEACON API

This module handles clinical data-related routes.
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
async def upload_clinical_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload clinical data.

    Args:
        file: CSV file containing clinical data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Upload status
    """
    # Check permissions
    check_permissions(["upload_clinical"])

    try:
        # Read and validate CSV file
        df = pd.read_csv(file.file)
        required_columns = [
            'age', 'weight', 'height', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'heart_rate', 'temperature',
            'glucose', 'cholesterol', 'smoking'
        ]
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {missing_cols}"
            )

        # Save file
        file_path = settings.UPLOAD_DIR / f"clinical_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return {"message": "File uploaded successfully", "filename": file.filename}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/process")
async def process_clinical_data(
    filename: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Process clinical data.

    Args:
        filename: Name of the uploaded file
        current_user: Current authenticated user
        db: Database session

    Returns:
        Processing status
    """
    # Check permissions
    check_permissions(["process_clinical"])

    try:
        # Initialize preprocessor
        config = {
            'input_dir': str(settings.UPLOAD_DIR),
            'output_dir': str(settings.MODEL_PATH / 'processed')
        }
        preprocessor = DataPreprocessor(config)

        # Process data
        processed_data = preprocessor.preprocess_clinical_data(
            pd.read_csv(settings.UPLOAD_DIR / filename)
        )
        
        # Save processed data
        preprocessor.save_processed_data(processed_data, 'clinical')

        return {"message": "Data processed successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/train")
async def train_clinical_model(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Train clinical model.

    Args:
        current_user: Current authenticated user
        db: Database session

    Returns:
        Training status
    """
    # Check permissions
    check_permissions(["train_clinical"])

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
        trainer.train('clinical')

        return {"message": "Model trained successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/predict")
async def predict_clinical(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Make clinical predictions.

    Args:
        file: CSV file containing clinical data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Model predictions
    """
    # Check permissions
    check_permissions(["predict_clinical"])

    try:
        # Read and validate input data
        df = pd.read_csv(file.file)
        
        # Initialize predictor
        config = {
            'model_dir': str(settings.MODEL_PATH),
            'output_dir': str(settings.MODEL_PATH / 'predictions')
        }
        predictor = ModelPredictor(config)

        # Make predictions
        predictions = predictor.predict('clinical', df)
        
        # Format results
        results = pd.DataFrame({
            'diabetes_prob': predictions[:, 0],
            'hypertension_prob': predictions[:, 1]
        })

        return results.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 