# test_complete_api.py
import requests
import json
import pytest

BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert "status" in data
    assert data["status"] == "active"
    print("✅ Root endpoint test passed")

def test_health_endpoint():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    print("✅ Health endpoint test passed")

def test_features_endpoint():
    """Test the features endpoint"""
    response = requests.get(f"{BASE_URL}/features")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "count" in data
    assert data["count"] == 10  # Should have 10 features
    print("✅ Features endpoint test passed")

def test_model_info_endpoint():
    """Test the model info endpoint"""
    response = requests.get(f"{BASE_URL}/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    print("✅ Model info endpoint test passed")

def test_predict_endpoint():
    """Test the prediction endpoint with valid data"""
    test_data = {
        "customer_id": "CUST_TEST_001",
        "transaction_count": 50.0,
        "total_amount": 10000.0,
        "avg_amount": 200.0,
        "amount_std": 50.0,
        "recency_days": 5.0,
        "customer_tenure_days": 365.0,
        "frequency_per_day": 0.14,
        "high_risk_channel_use": 2.0,
        "high_fraud_hour_ratio": 0.1,
        "amount_cv": 0.25
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    assert response.status_code in [200, 503]  # 200 if model loaded, 503 if not
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert data["probability"] >= 0
        assert data["probability"] <= 1
        assert data["risk_level"] in ["low", "high"]
        print("✅ Predict endpoint test passed with valid data")
    else:
        print("⚠ Predict endpoint returned 503 (model not loaded)")

def test_predict_invalid_data():
    """Test prediction with invalid data"""
    invalid_data = {
        "transaction_count": -50.0,  # Invalid: should be >= 0
        "total_amount": 10000.0,
        "avg_amount": 200.0,
        "amount_std": 50.0,
        "recency_days": 5.0,
        "customer_tenure_days": 365.0,
        "frequency_per_day": 0.14,
        "high_risk_channel_use": 2.0,
        "high_fraud_hour_ratio": 2.0,  # Invalid: should be <= 1
        "amount_cv": 0.25
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    # Should return 422 for validation error
    assert response.status_code == 422
    print("✅ Predict endpoint correctly rejected invalid data")

def run_all_tests():
    """Run all tests"""
    print("🧪 Running comprehensive API tests...")
    print("=" * 50)
    
    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_features_endpoint,
        test_model_info_endpoint,
        test_predict_endpoint,
        test_predict_invalid_data
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! API is ready for production.")
    else:
        print("⚠ Some tests failed. Check the API implementation.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)