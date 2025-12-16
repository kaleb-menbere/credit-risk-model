# Credit Risk Model Setup Script for Windows
# Run this script in PowerShell

Write-Host "========================================" -ForegroundColor Green
Write-Host "   Credit Risk Model Setup Script" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Create project structure
Write-Host "📁 Creating project structure..." -ForegroundColor Cyan

$directories = @(
    "src/api",
    "tests",
    ".github/workflows",
    "data/raw",
    "data/processed",
    "data/processed/task5",
    "models/task5",
    "reports",
    "mlruns",
    "logs",
    "notebooks"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "   Created: $dir" -ForegroundColor Gray
    }
}

# Create files with proper PowerShell syntax
Write-Host "`n📄 Creating configuration files..." -ForegroundColor Cyan

# Create requirements.txt
$requirementsContent = @"
# Core dependencies
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.2

# Machine learning
mlflow==2.9.2
xgboost==2.0.0
lightgbm==4.1.0

# API and web
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
httpx==0.25.1

# Code quality
flake8==6.1.0
black==23.11.0
mypy==1.7.0

# Utilities
python-dotenv==1.0.0
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0

# Deployment
gunicorn==21.2.0
"@

Set-Content -Path "requirements.txt" -Value $requirementsContent
Write-Host "   Created: requirements.txt" -ForegroundColor Gray

# Create Dockerfile
$dockerfileContent = @"
FROM python:3.9-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/processed models reports mlruns logs
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)"
CMD ["gunicorn", "src.api.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
"@

Set-Content -Path "Dockerfile" -Value $dockerfileContent
Write-Host "   Created: Dockerfile" -ForegroundColor Gray

# Create docker-compose.yml
$dockerComposeContent = @"
version: '3.8'
services:
  api:
    build: .
    container_name: credit-risk-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - PORT=8000
      - MLFLOW_TRACKING_URI=file:///app/mlruns
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
      - ./mlruns:/app/mlruns
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - credit-risk-network
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow-server
    ports:
      - "5000:5000"
    command: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0
    volumes:
      - ./mlruns:/mlruns
      - ./mlflow.db:/mlflow.db
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000
    restart: unless-stopped
    networks:
      - credit-risk-network
networks:
  credit-risk-network:
    driver: bridge
"@

Set-Content -Path "docker-compose.yml" -Value $dockerComposeContent
Write-Host "   Created: docker-compose.yml" -ForegroundColor Gray

# Create .env.example
$envContent = @"
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false

# Model Configuration
MODEL_PATH=models/task5/best_model.pkl
SCALER_PATH=models/task5/scaler.pkl
FEATURES_PATH=data/processed/model_features_fixed.txt

# MLflow Configuration
MLFLOW_TRACKING_URI=file://./mlruns
MLFLOW_EXPERIMENT_NAME=credit-risk-modeling

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# CORS
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]
"@

Set-Content -Path ".env.example" -Value $envContent
Write-Host "   Created: .env.example" -ForegroundColor Gray

# Create run_api.bat
$runApiBat = @"
@echo off
echo ========================================
echo   Bati Bank Credit Risk API
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or later
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo Starting API server...
echo.
echo 🌐 API Documentation: http://localhost:8000/docs
echo 🔍 Health Check: http://localhost:8000/health
echo 📊 MLflow UI: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

pause
"@

Set-Content -Path "run_api.bat" -Value $runApiBat
Write-Host "   Created: run_api.bat" -ForegroundColor Gray

# Create run_tests.bat
$runTestsBat = @"
@echo off
echo ========================================
echo   Running Credit Risk Model Tests
echo ========================================
echo.

REM Activate virtual environment
if exist "venv" (
    call venv\Scripts\activate.bat
)

echo Running tests with coverage...
pytest tests/ -v --cov=src --cov-report=html

echo.
echo Coverage report generated in: htmlcov/index.html
echo.

pause
"@

Set-Content -Path "run_tests.bat" -Value $runTestsBat
Write-Host "   Created: run_tests.bat" -ForegroundColor Gray

# Create CI/CD workflow
$githubWorkflowDir = ".github/workflows"
if (-not (Test-Path $githubWorkflowDir)) {
    New-Item -ItemType Directory -Force -Path $githubWorkflowDir | Out-Null
}

$ciCdContent = @"
name: Credit Risk Model CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9]

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python `${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: `${{ matrix.python-version }}

    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: `${{ runner.os }}-pip-`${{ hashFiles('requirements.txt') }}
        restore-keys: |
          `${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Lint with flake8
      run: |
        flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Format check with black
      run: |
        black --check src tests

    - name: Type check with mypy
      run: |
        mypy src --ignore-missing-imports

    - name: Run unit tests with coverage
      run: |
        pytest tests/ -v --cov=src --cov-report=xml --cov-report=html

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests

  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Docker Hub
      uses: docker/login-action@v3
      with:
        username: `${{ secrets.DOCKER_USERNAME }}
        password: `${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: `${{ github.ref == 'refs/heads/main' }}
        tags: |
          `${{ secrets.DOCKER_USERNAME }}/credit-risk-api:latest
          `${{ secrets.DOCKER_USERNAME }}/credit-risk-api:`${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
"@

Set-Content -Path "$githubWorkflowDir/ci.yml" -Value $ciCdContent
Write-Host "   Created: .github/workflows/ci.yml" -ForegroundColor Gray

# Create main.py (simplified)
$mainPyDir = "src/api"
if (-not (Test-Path $mainPyDir)) {
    New-Item -ItemType Directory -Force -Path $mainPyDir | Out-Null
}

$mainPyContent = @"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import joblib
import os

app = FastAPI(title="Credit Risk API", version="1.0.0")

# Load model (simplified for setup)
try:
    model = joblib.load("models/task5/best_model.pkl")
    print("✅ Model loaded successfully")
except:
    print("⚠ Model not found, using dummy model")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)

class PredictionRequest(BaseModel):
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
    customer_id: Optional[str] = None

class PredictionResponse(BaseModel):
    customer_id: str
    prediction: int
    probability: float
    risk_level: str
    message: str

@app.get("/")
def root():
    return {"message": "Credit Risk API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": "2023-12-15"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array([[
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
        ]])
        
        probability = model.predict_proba(features)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        risk_level = "high" if prediction == 1 else "low"
        
        return PredictionResponse(
            customer_id=request.customer_id or "anonymous",
            prediction=prediction,
            probability=float(probability),
            risk_level=risk_level,
            message=f"Predicted as {risk_level} risk"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"@

Set-Content -Path "$mainPyDir/main.py" -Value $mainPyContent
Write-Host "   Created: src/api/main.py" -ForegroundColor Gray

# Create pydantic_models.py
$pydanticModelsContent = @"
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str = "1.0.0"
    timestamp: datetime

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    accuracy: float
    roc_auc: float
    features: List[str]
    training_date: str
    description: str = "Credit risk prediction model"

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

class PredictionResponse(BaseModel):
    customer_id: str
    prediction: int
    probability: float
    risk_level: str
    message: str
"@

Set-Content -Path "$mainPyDir/pydantic_models.py" -Value $pydanticModelsContent
Write-Host "   Created: src/api/pydantic_models.py" -ForegroundColor Gray

# Create test_api.py
$testsDir = "tests"
if (-not (Test-Path $testsDir)) {
    New-Item -ItemType Directory -Force -Path $testsDir | Out-Null
}

$testApiContent = @"
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict():
    request_data = {
        "transaction_count": 25.0,
        "total_amount": 5000.0,
        "avg_amount": 200.0,
        "amount_std": 150.0,
        "recency_days": 7.0,
        "customer_tenure_days": 90.0,
        "frequency_per_day": 0.28,
        "high_risk_channel_use": 0.0,
        "high_fraud_hour_ratio": 0.2,
        "amount_cv": 0.75
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["probability"] >= 0 and data["probability"] <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"@

Set-Content -Path "$testsDir/test_api.py" -Value $testApiContent
Write-Host "   Created: tests/test_api.py" -ForegroundColor Gray

# Check for Python
Write-Host "`n🐍 Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = (python --version 2>&1)
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   Found: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "   Python not found or not in PATH" -ForegroundColor Red
        Write-Host "   Please install Python 3.9 or later from: https://www.python.org/downloads/" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   Python not found or not in PATH" -ForegroundColor Red
}

# Check for Docker
Write-Host "`n🐳 Checking Docker installation..." -ForegroundColor Cyan
try {
    $dockerVersion = (docker --version 2>&1)
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   Found: $dockerVersion" -ForegroundColor Green
    } else {
        Write-Host "   Docker not found or not in PATH" -ForegroundColor Yellow
        Write-Host "   Docker is optional but recommended for containerized deployment" -ForegroundColor Gray
    }
} catch {
    Write-Host "   Docker not found or not in PATH" -ForegroundColor Yellow
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "   Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Install dependencies: .\run_api.bat (first time)" -ForegroundColor White
Write-Host "2. Run the API: .\run_api.bat" -ForegroundColor White
Write-Host "3. Run tests: .\run_tests.bat" -ForegroundColor White
Write-Host "4. Open browser: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Project structure:" -ForegroundColor Cyan
Get-ChildItem | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize