# main.py at project root
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import numpy as np
import joblib
import os

app = FastAPI(title="Credit Risk API")

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

@app.get("/")
def root():
    return {"message": "Credit Risk API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
def predict(request: PredictionRequest):
    # Mock response for testing
    return {
        "prediction": 0,
        "probability": 0.25,
        "risk_level": "low",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)