"""
Survival data router for BEACON API

This module handles survival data-related routes.
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
async def upload_survival_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload survival data.

    Args:
        file: CSV file containing survival data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Upload status
    """
    # Check permissions
    check_permissions(["upload_survival"])

    try:
        # Read and validate CSV file
        df = pd.read_csv(file.file)
        required_columns = [
            'time', 'event', 'age', 'stage', 'grade'
        ]
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {missing_cols}"
            )

        # Validate data types
        if not pd.to_numeric(df['time'], errors='coerce').notna().all():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Column 'time' must contain numeric values"
            )
            
        if not df['event'].isin([0, 1]).all():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Column 'event' must contain binary values (0 or 1)"
            )
            
        if not pd.to_numeric(df['age'], errors='coerce').notna().all():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Column 'age' must contain numeric values"
            )
            
        if not df['stage'].isin(['I', 'II', 'III', 'IV']).all():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Column 'stage' must contain values: I, II, III, or IV"
            )
            
        if not df['grade'].isin(['Low', 'Medium', 'High']).all():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Column 'grade' must contain values: Low, Medium, or High"
            )

        # Save file
        file_path = settings.UPLOAD_DIR / f"survival_{file.filename}"
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
async def process_survival_data(
    filename: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Process survival data.

    Args:
        filename: Name of the uploaded file
        current_user: Current authenticated user
        db: Database session

    Returns:
        Processing status
    """
    # Check permissions
    check_permissions(["process_survival"])

    try:
        # Initialize preprocessor
        config = {
            'input_dir': str(settings.UPLOAD_DIR),
            'output_dir': str(settings.MODEL_PATH / 'processed')
        }
        preprocessor = DataPreprocessor(config)

        # Load and process data
        data = pd.read_csv(settings.UPLOAD_DIR / filename)
        processed_data = preprocessor.preprocess_survival_data(data)
        
        # Save processed data
        processed_data.to_csv(
            settings.MODEL_PATH / 'processed/survival_processed.csv',
            index=False
        )

        return {"message": "Data processed successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/train")
async def train_survival_model(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Train survival model.

    Args:
        current_user: Current authenticated user
        db: Database session

    Returns:
        Training status
    """
    # Check permissions
    check_permissions(["train_survival"])

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
        trainer.train('survival')

        return {"message": "Model trained successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/predict")
async def predict_survival(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Make survival predictions.

    Args:
        file: CSV file containing survival data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Model predictions
    """
    # Check permissions
    check_permissions(["predict_survival"])

    try:
        # Read and validate input data
        df = pd.read_csv(file.file)
        required_columns = ['age', 'stage', 'grade']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Initialize predictor
        config = {
            'model_dir': str(settings.MODEL_PATH),
            'output_dir': str(settings.MODEL_PATH / 'predictions')
        }
        predictor = ModelPredictor(config)

        # Make predictions
        predictions = predictor.predict('survival', df)
        
        # Format results
        results = pd.DataFrame({
            'predicted_time': predictions[:, 0],
            'event_prob': predictions[:, 1]
        })

        return results.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 