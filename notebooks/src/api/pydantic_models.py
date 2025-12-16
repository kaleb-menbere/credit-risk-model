"""
Pydantic models for FastAPI request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

# Single prediction request
class PredictionRequest(BaseModel):
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer identifier (optional)"
    )
    transaction_count: float = Field(
        ..., 
        ge=0,
        description="Total number of transactions"
    )
    total_amount: float = Field(
        ...,
        description="Sum of all transaction amounts (can be negative)"
    )
    avg_amount: float = Field(
        ..., 
        ge=0,
        description="Average transaction amount"
    )
    amount_std: float = Field(
        ..., 
        ge=0,
        description="Standard deviation of transaction amounts"
    )
    recency_days: float = Field(
        ..., 
        ge=0,
        description="Days since last transaction"
    )
    customer_tenure_days: float = Field(
        ..., 
        ge=0,
        description="Customer tenure in days"
    )
    frequency_per_day: float = Field(
        ..., 
        ge=0,
        description="Average transactions per day"
    )
    high_risk_channel_use: float = Field(
        ..., 
        ge=0,
        description="Count of high-risk channel usage"
    )
    high_fraud_hour_ratio: float = Field(
        ..., 
        ge=0,
        le=1,
        description="Ratio of transactions in high-fraud hours (0-1)"
    )
    amount_cv: float = Field(
        ..., 
        ge=0,
        description="Coefficient of variation of amounts"
    )
    
    @validator('high_fraud_hour_ratio')
    def validate_ratio(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Ratio must be between 0 and 1')
        return v

# Single prediction response
class PredictionResponse(BaseModel):
    customer_id: str
    prediction: int = Field(..., ge=0, le=1, description="0=low risk, 1=high risk")
    probability: float = Field(..., ge=0, le=1, description="Risk probability")
    risk_level: str = Field(..., description="low|medium|high")
    confidence: str = Field(..., description="low|medium|high")
    recommendation: str = Field(..., description="Action recommendation")
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None,
        description="Feature importance scores"
    )
    model_version: str = Field(..., description="Model version")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST12345",
                "prediction": 1,
                "probability": 0.85,
                "risk_level": "high",
                "confidence": "high",
                "recommendation": "Review application manually",
                "feature_importance": {"transaction_count": 0.15, "amount_cv": 0.25},
                "model_version": "2.0.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

# Batch prediction request
class BatchPredictionRequest(BaseModel):
    customers: List[PredictionRequest] = Field(
        ...,
        max_items=1000,
        description="List of customers to predict (max 1000)"
    )

# Batch prediction response item
class BatchPredictionItem(BaseModel):
    customer_id: str
    prediction: Optional[int] = Field(None, ge=0, le=1)
    probability: Optional[float] = Field(None, ge=0, le=1)
    risk_level: str
    status: str = Field(..., description="success|failed")

# Batch prediction response
class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]
    total_count: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    error_count: int = Field(..., ge=0)
    errors: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of errors if any"
    )
    timestamp: datetime = Field(..., description="Batch completion timestamp")

# Health check response
class HealthResponse(BaseModel):
    status: str = Field(..., description="healthy|degraded|unhealthy")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Model loaded status")
    scaler_loaded: bool = Field(..., description="Scaler loaded status")
    timestamp: datetime = Field(..., description="Check timestamp")

# Model information
class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model class/type")
    accuracy: float = Field(..., ge=0, le=1, description="Accuracy score")
    roc_auc: float = Field(..., ge=0, le=1, description="ROC-AUC score")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score")
    features: List[str] = Field(..., description="Feature names")
    training_date: str = Field(..., description="Model training date")
    description: str = Field(..., description="Model description")