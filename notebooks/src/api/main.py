"""
Minimal FastAPI app for credit risk prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Optional

# ============ Initialize FastAPI ============
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using RFM features",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Pydantic Models ============
class PredictionRequest(BaseModel):
    customer_id: Optional[str] = None
    transaction_count: float
    total_amount: float
    avg_amount: float
    amount_std: float
    recency_days: float
    customer_tenure_days: float
    frequency_per_day: float
    high_risk_channel_use: float
    high_fraud_hour_ratio: float
    amount_cv: float

# ============ Load Model ============
MODEL = None
SCALER = None

def load_model():
    """Try to load model and scaler"""
    global MODEL, SCALER
    
    try:
        # Try multiple possible paths
        model_paths = [
            "models/task5/best_model.pkl",
            "../models/task5/best_model.pkl",
            "../../models/task5/best_model.pkl"
        ]
        
        scaler_paths = [
            "models/task5/scaler.pkl",
            "../models/task5/scaler.pkl", 
            "../../models/task5/scaler.pkl"
        ]
        
        # Load model
        for path in model_paths:
            if os.path.exists(path):
                MODEL = joblib.load(path)
                print(f"✅ Model loaded from: {path}")
                break
        
        # Load scaler
        for path in scaler_paths:
            if os.path.exists(path):
                SCALER = joblib.load(path)
                print(f"✅ Scaler loaded from: {path}")
                break
                
    except Exception as e:
        print(f"⚠ Could not load model artifacts: {e}")

# Load model on startup
load_model()

# ============ API Endpoints ============
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "status": "healthy",
        "message": "Credit Risk Prediction API",
        "version": "2.0.0",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    """Make prediction"""
    try:
        if MODEL is None or SCALER is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please check /health endpoint."
            )
        
        # Prepare features
        features = np.array([
            request.transaction_count,
            request.total_amount,
            request.avg_amount,
            request.amount_std,
            request.recency_days,
            request.customer_tenure_days,
            request.frequency_per_day,
            request.high_risk_channel_use,
            request.high_fraud_hour_ratio,
            request.amount_cv
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = SCALER.transform(features)
        
        # Make prediction
        probability = MODEL.predict_proba(features_scaled)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Determine risk level
        risk_level = "high" if prediction == 1 else "low"
        confidence = "high" if (probability > 0.7 or probability < 0.3) else "medium"
        
        recommendation = "Reject application" if prediction == 1 else "Approve with standard terms"
        
        return {
            "customer_id": request.customer_id or "anonymous",
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "confidence": confidence,
            "recommendation": recommendation,
            "model_version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/features")
def get_features():
    """Get feature list"""
    features = [
        "transaction_count",
        "total_amount", 
        "avg_amount",
        "amount_std",
        "recency_days",
        "customer_tenure_days",
        "frequency_per_day",
        "high_risk_channel_use",
        "high_fraud_hour_ratio",
        "amount_cv"
    ]
    
    return {
        "features": features,
        "count": len(features),
        "description": "RFM-based features for credit risk prediction"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)