"""
Main FastAPI application for BEACON

This module initializes and configures the FastAPI application.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from beacon.backend.routers import clinical, imaging, genomic, survival
from beacon.backend.core.config import settings
from beacon.backend.core.logging import setup_logging
import logging

# Setup logging
logger = setup_logging()

# Create FastAPI app
app = FastAPI(
    title="BEACON API",
    description="API for BEACON (Biomedical Evidence Analysis and Classification ONtology)",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(clinical.router, prefix="/api/v1/clinical", tags=["clinical"])
app.include_router(imaging.router, prefix="/api/v1/imaging", tags=["imaging"])
app.include_router(genomic.router, prefix="/api/v1/genomic", tags=["genomic"])
app.include_router(survival.router, prefix="/api/v1/survival", tags=["survival"])

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unexpected error occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    ) 