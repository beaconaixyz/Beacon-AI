"""
Genomic data router for BEACON API

This module handles genomic data-related routes.
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

@router.post("/upload-expression")
async def upload_expression_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload gene expression data.

    Args:
        file: NPZ file containing expression data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Upload status
    """
    # Check permissions
    check_permissions(["upload_genomic"])

    try:
        # Validate file extension
        if not file.filename.endswith('.npz'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Only .npz files are supported."
            )

        # Save file
        file_path = settings.UPLOAD_DIR / f"expression_{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Validate data
        data = np.load(file_path)
        if 'expression' not in data.files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Expression data not found in file."
            )

        return {"message": "Expression data uploaded successfully", "filename": file.filename}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/upload-mutations")
async def upload_mutation_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload mutation data.

    Args:
        file: NPZ file containing mutation data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Upload status
    """
    # Check permissions
    check_permissions(["upload_genomic"])

    try:
        # Validate file extension
        if not file.filename.endswith('.npz'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Only .npz files are supported."
            )

        # Save file
        file_path = settings.UPLOAD_DIR / f"mutations_{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Validate data
        data = np.load(file_path)
        if 'mutations' not in data.files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Mutation data not found in file."
            )

        return {"message": "Mutation data uploaded successfully", "filename": file.filename}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/upload-cnv")
async def upload_cnv_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload copy number variation data.

    Args:
        file: NPZ file containing CNV data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Upload status
    """
    # Check permissions
    check_permissions(["upload_genomic"])

    try:
        # Validate file extension
        if not file.filename.endswith('.npz'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Only .npz files are supported."
            )

        # Save file
        file_path = settings.UPLOAD_DIR / f"cnv_{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Validate data
        data = np.load(file_path)
        if 'cnv' not in data.files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CNV data not found in file."
            )

        return {"message": "CNV data uploaded successfully", "filename": file.filename}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/process")
async def process_genomic_data(
    expression_filename: str,
    mutations_filename: str,
    cnv_filename: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Process genomic data.

    Args:
        expression_filename: Name of the expression data file
        mutations_filename: Name of the mutation data file
        cnv_filename: Name of the CNV data file
        current_user: Current authenticated user
        db: Database session

    Returns:
        Processing status
    """
    # Check permissions
    check_permissions(["process_genomic"])

    try:
        # Initialize preprocessor
        config = {
            'input_dir': str(settings.UPLOAD_DIR),
            'output_dir': str(settings.MODEL_PATH / 'processed')
        }
        preprocessor = DataPreprocessor(config)

        # Load data
        expression_data = dict(np.load(settings.UPLOAD_DIR / expression_filename))['expression']
        mutations_data = dict(np.load(settings.UPLOAD_DIR / mutations_filename))['mutations']
        cnv_data = dict(np.load(settings.UPLOAD_DIR / cnv_filename))['cnv']

        # Combine data
        genomic_data = {
            'expression': expression_data,
            'mutations': mutations_data,
            'cnv': cnv_data
        }

        # Process data
        processed_data = preprocessor.preprocess_genomic_data(genomic_data)
        
        # Save processed data
        np.savez(
            settings.MODEL_PATH / 'processed/genomic_processed.npz',
            **processed_data
        )

        return {"message": "Data processed successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/train")
async def train_genomic_model(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Train genomic model.

    Args:
        current_user: Current authenticated user
        db: Database session

    Returns:
        Training status
    """
    # Check permissions
    check_permissions(["train_genomic"])

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
        trainer.train('genomic')

        return {"message": "Model trained successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/predict")
async def predict_genomic(
    expression_file: UploadFile = File(...),
    mutations_file: UploadFile = File(...),
    cnv_file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Make predictions on genomic data.

    Args:
        expression_file: NPZ file containing expression data
        mutations_file: NPZ file containing mutation data
        cnv_file: NPZ file containing CNV data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Model predictions
    """
    # Check permissions
    check_permissions(["predict_genomic"])

    try:
        # Save and load expression data
        expression_content = await expression_file.read()
        with open(settings.UPLOAD_DIR / "temp_expression.npz", "wb") as buffer:
            buffer.write(expression_content)
        expression_data = dict(np.load(settings.UPLOAD_DIR / "temp_expression.npz"))['expression']

        # Save and load mutations data
        mutations_content = await mutations_file.read()
        with open(settings.UPLOAD_DIR / "temp_mutations.npz", "wb") as buffer:
            buffer.write(mutations_content)
        mutations_data = dict(np.load(settings.UPLOAD_DIR / "temp_mutations.npz"))['mutations']

        # Save and load CNV data
        cnv_content = await cnv_file.read()
        with open(settings.UPLOAD_DIR / "temp_cnv.npz", "wb") as buffer:
            buffer.write(cnv_content)
        cnv_data = dict(np.load(settings.UPLOAD_DIR / "temp_cnv.npz"))['cnv']

        # Combine data
        input_data = {
            'expression': expression_data,
            'mutations': mutations_data,
            'cnv': cnv_data
        }
        
        # Initialize predictor
        config = {
            'model_dir': str(settings.MODEL_PATH),
            'output_dir': str(settings.MODEL_PATH / 'predictions')
        }
        predictor = ModelPredictor(config)

        # Make predictions
        predictions = predictor.predict('genomic', input_data)
        
        # Format results
        results = pd.DataFrame({
            'expression_high_prob': predictions.flatten()
        })

        return results.to_dict(orient='records')

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 