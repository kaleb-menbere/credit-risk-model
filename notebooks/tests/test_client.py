"""
Test client for Credit Risk API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_all_endpoints():
    """Test all API endpoints"""
    print("🧪 Testing Credit Risk API...")
    print("=" * 60)
    
    # 1. Test root endpoint
    print("1. Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print()
    
    # 2. Test health endpoint
    print("2. Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print()
    
    # 3. Test model info
    print("3. Testing model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        info = response.json()
        print(f"   Model: {info.get('model_name')}")
        print(f"   Accuracy: {info.get('accuracy'):.3f}")
        print(f"   Features: {len(info.get('features', []))}")
    print()
    
    # 4. Test features endpoint
    print("4. Testing features endpoint...")
    response = requests.get(f"{BASE_URL}/features")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        features = response.json()
        print(f"   Feature count: {features.get('count')}")
    print()
    
    # 5. Test single prediction
    print("5. Testing single prediction...")
    test_data = {
        "customer_id": "test_customer_001",
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
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        prediction = response.json()
        print(f"   Risk Level: {prediction.get('risk_level')}")
        print(f"   Probability: {prediction.get('probability'):.3f}")
        print(f"   Recommendation: {prediction.get('recommendation')}")
    else:
        print(f"   Error: {response.text}")
    print()
    
    # 6. Test batch prediction
    print("6. Testing batch prediction...")
    batch_data = {
        "customers": [
            {
                "customer_id": "batch_customer_1",
                "transaction_count": 10.0,
                "total_amount": 1000.0,
                "avg_amount": 100.0,
                "amount_std": 50.0,
                "recency_days": 5.0,
                "customer_tenure_days": 30.0,
                "frequency_per_day": 0.33,
                "high_risk_channel_use": 0.0,
                "high_fraud_hour_ratio": 0.1,
                "amount_cv": 0.5
            },
            {
                "customer_id": "batch_customer_2",
                "transaction_count": 100.0,
                "total_amount": 50000.0,
                "avg_amount": 500.0,
                "amount_std": 1000.0,
                "recency_days": 1.0,
                "customer_tenure_days": 365.0,
                "frequency_per_day": 0.27,
                "high_risk_channel_use": 5.0,
                "high_fraud_hour_ratio": 0.5,
                "amount_cv": 2.0
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        batch_result = response.json()
        print(f"   Total: {batch_result.get('total_count')}")
        print(f"   Success: {batch_result.get('success_count')}")
        print(f"   Errors: {batch_result.get('error_count')}")
    print()
    
    # 7. Test stats endpoint
    print("7. Testing stats endpoint...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        stats = response.json()
        print(f"   Version: {stats.get('api_version')}")
    print()
    
    print("=" * 60)
    print("✅ All tests completed!")
    
    return True

if __name__ == "__main__":
    # Wait a moment for API to start
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    try:
        test_all_endpoints()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        print("   Start it with: uvicorn src.api.main:app --reload")