"""
FastAPI Application for Customer Churn Prediction API.

This module provides REST API endpoints for churn prediction including:
- Single customer predictions
- Batch predictions
- Model information and health checks
- Prediction feedback logging
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import asyncio
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.prediction_service import (
    PredictionService, CustomerInput, PredictionResult, ModelRegistry
)
from utils.logging_setup import get_notebook_logger

# Initialize logger
logger = get_notebook_logger(__name__)

# Global prediction service instance
prediction_service: Optional[PredictionService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global prediction_service
    
    try:
        logger.info("Initializing prediction service...")
        
        # Initialize model registry and prediction service
        model_registry = ModelRegistry()
        prediction_service = PredictionService(
            model_registry=model_registry,
            default_model_name="churn_predictor",
            default_model_version="latest"
        )
        
        logger.info("Prediction service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {e}")
        # Don't raise here - let the health check endpoint report the issue
        prediction_service = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down prediction service...")

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Advanced customer churn prediction service with ML-powered risk assessment and recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class CustomerInputAPI(BaseModel):
    """API model for customer input data."""
    model_config = {"extra": "forbid"}

    customer_id: str = Field(..., description="Unique customer identifier")
    
    # Demographics
    gender: str = Field(..., description="Customer gender")
    senior_citizen: bool = Field(..., description="Whether customer is a senior citizen")
    partner: bool = Field(..., description="Whether customer has a partner")
    dependents: bool = Field(..., description="Whether customer has dependents")
    
    # Account information
    tenure: int = Field(..., ge=0, description="Customer tenure in months")
    contract: str = Field(..., description="Contract type")
    paperless_billing: bool = Field(..., description="Whether customer uses paperless billing")
    payment_method: str = Field(..., description="Payment method")
    
    # Service usage
    phone_service: bool = Field(..., description="Whether customer has phone service")
    multiple_lines: str = Field(..., description="Multiple lines status")
    internet_service: str = Field(..., description="Internet service type")
    online_security: str = Field(..., description="Online security service")
    online_backup: str = Field(..., description="Online backup service")
    device_protection: str = Field(..., description="Device protection service")
    tech_support: str = Field(..., description="Tech support service")
    streaming_tv: str = Field(..., description="Streaming TV service")
    streaming_movies: str = Field(..., description="Streaming movies service")
    
    # Financial data
    monthly_charges: float = Field(..., ge=0, description="Monthly charges")
    total_charges: float = Field(..., ge=0, description="Total charges")
    
    # Optional enhanced features
    usage_minutes_monthly: Optional[float] = Field(None, ge=0, description="Monthly usage minutes")
    data_usage_gb_monthly: Optional[float] = Field(None, ge=0, description="Monthly data usage in GB")
    support_interactions_count: Optional[int] = Field(None, ge=0, description="Number of support interactions")
    complaint_count: Optional[int] = Field(None, ge=0, description="Number of complaints")
    satisfaction_score: Optional[float] = Field(None, ge=0, le=10, description="Customer satisfaction score")
    payment_delay_frequency: Optional[float] = Field(None, ge=0, le=1, description="Payment delay frequency")
    service_change_frequency: Optional[float] = Field(None, ge=0, description="Service change frequency")
    
    @field_validator('contract')
    @classmethod
    def validate_contract(cls, v):
        valid_contracts = ["Month-to-month", "One year", "Two year"]
        if v not in valid_contracts:
            raise ValueError(f"contract must be one of {valid_contracts}")
        return v
    
    @field_validator('internet_service')
    @classmethod
    def validate_internet_service(cls, v):
        valid_services = ["DSL", "Fiber optic", "No"]
        if v not in valid_services:
            raise ValueError(f"internet_service must be one of {valid_services}")
        return v
    
    def to_customer_input(self) -> CustomerInput:
        """Convert to CustomerInput object."""
        return CustomerInput(**self.model_dump())


class PredictionResultAPI(BaseModel):
    """API model for prediction results."""
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: str
    confidence_score: float = Field(..., ge=0, le=1)
    key_risk_factors: List[str]
    recommendations: List[str]
    model_version: str
    prediction_timestamp: str
    risk_score: Optional[float] = None
    risk_percentile: Optional[float] = None
    feature_contributions: Optional[Dict[str, float]] = None
    estimated_clv: Optional[float] = None
    retention_cost_benefit: Optional[float] = None
    
    @classmethod
    def from_prediction_result(cls, result: PredictionResult) -> 'PredictionResultAPI':
        """Create from PredictionResult object."""
        return cls(**result.to_dict())


class BatchPredictionRequest(BaseModel):
    """API model for batch prediction requests."""
    customers: List[CustomerInputAPI] = Field(..., description="List of customers for prediction")
    
    @field_validator('customers')
    @classmethod
    def validate_customers_not_empty(cls, v):
        if not v:
            raise ValueError("customers list cannot be empty")
        if len(v) > 1000:  # Reasonable batch size limit
            raise ValueError("batch size cannot exceed 1000 customers")
        return v


class BatchPredictionResponse(BaseModel):
    """API model for batch prediction responses."""
    predictions: List[PredictionResultAPI]
    batch_id: str
    processed_count: int
    failed_count: int
    processing_time_seconds: float
    errors: List[Dict[str, str]] = []


class ModelInfoResponse(BaseModel):
    """API model for model information."""
    model_name: str
    model_version: str
    model_type: str
    feature_count: int
    feature_names: List[str]
    risk_thresholds: Dict[str, float]
    performance_metrics: Optional[Dict[str, Any]] = None
    last_updated: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """API model for health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    feature_count: int
    uptime_seconds: float
    issues: List[str] = []


class FeedbackRequest(BaseModel):
    """API model for prediction feedback."""
    customer_id: str = Field(..., description="Customer ID from original prediction")
    prediction_id: Optional[str] = Field(None, description="Prediction ID if available")
    actual_churn: bool = Field(..., description="Actual churn outcome")
    intervention_applied: bool = Field(False, description="Whether retention intervention was applied")
    intervention_type: Optional[str] = Field(None, description="Type of intervention applied")
    outcome_date: str = Field(..., description="Date when outcome was observed")
    notes: Optional[str] = Field(None, description="Additional notes")


class ErrorResponse(BaseModel):
    """API model for error responses."""
    error: str
    detail: str
    timestamp: str
    request_id: Optional[str] = None


# Dependency to get prediction service
def get_prediction_service() -> PredictionService:
    """Dependency to get the prediction service."""
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not available. Check service health."
        )
    return prediction_service


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/model/health"
    }


@app.post("/predict", response_model=PredictionResultAPI)
async def predict_single_customer(customer: CustomerInputAPI):
    """
    Predict churn probability for a single customer.

    Returns comprehensive risk assessment including:
    - Churn probability and risk level
    - Confidence score
    - Key risk factors
    - Actionable recommendations
    - Business metrics (CLV, retention cost-benefit)
    """
    # Resolved inside the handler so request validation runs first: a
    # malformed body must return 422 even when no model is loaded
    service = get_prediction_service()
    try:
        logger.info(f"Processing prediction request for customer: {customer.customer_id}")
        
        # Convert API model to service model
        customer_input = customer.to_customer_input()
        
        # Make prediction
        result = service.predict(customer_input)
        
        logger.info(f"Prediction completed for customer {customer.customer_id}: {result.risk_level} risk")
        
        # Convert to API response model
        return PredictionResultAPI.from_prediction_result(result)
        
    except ValueError as e:
        logger.warning(f"Invalid input for customer {customer.customer_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction failed for customer {customer.customer_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_customers(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Predict churn probability for multiple customers in batch.

    Processes up to 1000 customers in a single request.
    Returns aggregated results with error handling for individual failures.
    """
    service = get_prediction_service()
    start_time = datetime.now()
    batch_id = f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}_{len(request.customers)}"
    
    logger.info(f"Processing batch prediction request {batch_id} with {len(request.customers)} customers")
    
    predictions = []
    errors = []
    
    try:
        # Convert API models to service models
        customer_inputs = [customer.to_customer_input() for customer in request.customers]
        
        # Make batch predictions
        results = service.batch_predict(customer_inputs)
        
        # Convert results to API models
        predictions = [PredictionResultAPI.from_prediction_result(result) for result in results]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Batch prediction {batch_id} completed: {len(predictions)} successful, {len(errors)} failed")
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            processed_count=len(predictions),
            failed_count=len(errors),
            processing_time_seconds=processing_time,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Batch prediction {batch_id} failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(service: PredictionService = Depends(get_prediction_service)):
    """
    Get information about the currently loaded model.
    
    Returns model metadata including:
    - Model name and version
    - Feature information
    - Performance metrics
    - Risk thresholds
    """
    try:
        info = service.get_model_info()
        
        return ModelInfoResponse(
            model_name=info.get("model_name", "unknown"),
            model_version=info.get("model_version", "unknown"),
            model_type=info.get("model_type", "unknown"),
            feature_count=info.get("feature_count", 0),
            feature_names=info.get("feature_names", []),
            risk_thresholds=info.get("risk_thresholds", {}),
            performance_metrics=info.get("performance_metrics"),
            last_updated=info.get("last_updated")
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@app.get("/model/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Check the health status of the prediction service.
    
    Returns service status including:
    - Overall health status
    - Model loading status
    - Feature count
    - Any issues or warnings
    """
    try:
        if prediction_service is None:
            return HealthCheckResponse(
                status="unhealthy",
                timestamp=datetime.now().isoformat(),
                model_loaded=False,
                feature_count=0,
                uptime_seconds=0,
                issues=["Prediction service not initialized"]
            )
        
        health = prediction_service.health_check()
        
        return HealthCheckResponse(
            status=health.get("status", "unknown"),
            timestamp=health.get("timestamp", datetime.now().isoformat()),
            model_loaded=health.get("model_loaded", False),
            feature_count=health.get("feature_count", 0),
            uptime_seconds=health.get("uptime_seconds", 0),
            issues=health.get("issues", [])
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=False,
            feature_count=0,
            uptime_seconds=0,
            issues=[f"Health check error: {str(e)}"]
        )


@app.post("/model/feedback")
async def submit_prediction_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit feedback on prediction accuracy for model improvement.
    
    Collects actual churn outcomes and intervention results to:
    - Improve model accuracy through retraining
    - Measure intervention effectiveness
    - Adjust risk thresholds based on business outcomes
    """
    try:
        logger.info(f"Received feedback for customer {feedback.customer_id}")
        
        # Add background task to process feedback
        background_tasks.add_task(process_feedback, feedback)
        
        return {
            "message": "Feedback received successfully",
            "customer_id": feedback.customer_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to process feedback for customer {feedback.customer_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process feedback")


async def process_feedback(feedback: FeedbackRequest):
    """Background task to process prediction feedback."""
    try:
        # Log feedback for analysis
        feedback_data = {
            "customer_id": feedback.customer_id,
            "prediction_id": feedback.prediction_id,
            "actual_churn": feedback.actual_churn,
            "intervention_applied": feedback.intervention_applied,
            "intervention_type": feedback.intervention_type,
            "outcome_date": feedback.outcome_date,
            "notes": feedback.notes,
            "received_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Processing feedback: {feedback_data}")
        
        # Here you would typically:
        # 1. Store feedback in database
        # 2. Update model performance metrics
        # 3. Trigger retraining if needed
        # 4. Adjust risk thresholds based on outcomes
        
        # For now, just log the feedback
        logger.info(f"Feedback processed for customer {feedback.customer_id}")
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )