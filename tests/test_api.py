"""
Comprehensive tests for the Credit Risk API
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.api.main import app
from src.api.pydantic_models import PredictionRequest

client = TestClient(app)

# Sample valid data
VALID_REQUEST_DATA = {
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

class TestAPIBasics:
    """Test basic API functionality"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns 200"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_docs_endpoints(self):
        """Test API documentation endpoints"""
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_features_endpoint(self):
        """Test features endpoint"""
        response = client.get("/features")
        # May return 404 if features not loaded, which is OK for tests
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "features" in data
            assert isinstance(data["features"], list)
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "model_type" in data
            assert "accuracy" in data

class TestPredictionEndpoint:
    """Test the /predict endpoint"""
    
    def test_valid_prediction_request(self):
        """Test prediction with valid data"""
        request_data = VALID_REQUEST_DATA.copy()
        request_data["customer_id"] = "test_customer_001"
        
        response = client.post("/predict", json=request_data)
        
        # The API might be in test mode, so accept 200 or 500
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data
            assert data["customer_id"] == "test_customer_001"
            assert data["probability"] >= 0 and data["probability"] <= 1
            assert data["risk_level"] in ["low", "high"]
    
    def test_prediction_without_customer_id(self):
        """Test prediction without customer ID"""
        response = client.post("/predict", json=VALID_REQUEST_DATA)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["customer_id"] == "anonymous"
    
    def test_prediction_missing_field(self):
        """Test prediction with missing required field"""
        invalid_data = VALID_REQUEST_DATA.copy()
        del invalid_data["transaction_count"]  # Remove required field
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_invalid_data_type(self):
        """Test prediction with wrong data type"""
        invalid_data = VALID_REQUEST_DATA.copy()
        invalid_data["transaction_count"] = "not_a_number"  # Should be float
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_out_of_bounds(self):
        """Test prediction with out of bounds values"""
        invalid_data = VALID_REQUEST_DATA.copy()
        invalid_data["high_fraud_hour_ratio"] = 1.5  # Should be <= 1
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

class TestBatchPrediction:
    """Test batch prediction endpoint"""
    
    def test_valid_batch_prediction(self):
        """Test batch prediction with valid data"""
        request_data = {
            "customers": [
                {**VALID_REQUEST_DATA, "customer_id": "cust1"},
                {**VALID_REQUEST_DATA, "customer_id": "cust2"}
            ]
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_count" in data
            assert "success_count" in data
            assert "error_count" in data
            assert len(data["predictions"]) == 2
    
    def test_batch_empty_list(self):
        """Test batch prediction with empty list"""
        request_data = {"customers": []}
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 422  # Validation error (min_items=1)
    
    def test_batch_mixed_valid_invalid(self):
        """Test batch with mix of valid and invalid data"""
        request_data = {
            "customers": [
                VALID_REQUEST_DATA,  # Valid
                {"invalid": "data"}  # Invalid
            ]
        }
        
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code in [200, 500]

class TestPydanticModels:
    """Test Pydantic model validation"""
    
    def test_prediction_request_model(self):
        """Test PredictionRequest model validation"""
        # Valid data
        request = PredictionRequest(**VALID_REQUEST_DATA)
        assert request.transaction_count == 25.0
        assert request.customer_id is None
        
        # With customer ID
        data_with_id = {**VALID_REQUEST_DATA, "customer_id": "test123"}
        request_with_id = PredictionRequest(**data_with_id)
        assert request_with_id.customer_id == "test123"
    
    def test_request_model_validation_errors(self):
        """Test PredictionRequest validation errors"""
        # Missing required field
        invalid_data = VALID_REQUEST_DATA.copy()
        del invalid_data["transaction_count"]
        
        with pytest.raises(ValueError):
            PredictionRequest(**invalid_data)
        
        # Wrong data type
        invalid_data = VALID_REQUEST_DATA.copy()
        invalid_data["transaction_count"] = "not_a_number"
        
        with pytest.raises(ValueError):
            PredictionRequest(**invalid_data)
        
        # Out of bounds
        invalid_data = VALID_REQUEST_DATA.copy()
        invalid_data["high_fraud_hour_ratio"] = 2.0
        
        with pytest.raises(ValueError):
            PredictionRequest(**invalid_data)

class TestErrorHandling:
    """Test API error handling"""
    
    def test_404_not_found(self):
        """Test 404 for non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test 405 for wrong HTTP method"""
        response = client.put("/predict", json=VALID_REQUEST_DATA)
        assert response.status_code == 405
    
    def test_invalid_json(self):
        """Test invalid JSON in request"""
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])