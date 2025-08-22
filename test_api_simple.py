"""
Simple API test to verify the FastAPI application works correctly.
"""

import requests
import json
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing Customer Churn Prediction API...")
    
    try:
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Message: {data.get('message')}")
            print(f"Version: {data.get('version')}")
        
        # Test health check
        print("\n2. Testing health check...")
        response = requests.get(f"{base_url}/model/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            health = response.json()
            print(f"Health Status: {health.get('status')}")
            print(f"Model Loaded: {health.get('model_loaded')}")
            print(f"Feature Count: {health.get('feature_count')}")
            if health.get('issues'):
                print(f"Issues: {health.get('issues')}")
        
        # Test model info
        print("\n3. Testing model info...")
        response = requests.get(f"{base_url}/model/info")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            info = response.json()
            print(f"Model Name: {info.get('model_name')}")
            print(f"Model Version: {info.get('model_version')}")
            print(f"Model Type: {info.get('model_type')}")
        elif response.status_code == 503:
            print("Service unavailable - model not loaded")
        
        # Test single prediction
        print("\n4. Testing single prediction...")
        customer_data = {
            "customer_id": "TEST_API_001",
            "gender": "Female",
            "senior_citizen": False,
            "partner": True,
            "dependents": False,
            "tenure": 24,
            "contract": "One year",
            "paperless_billing": True,
            "payment_method": "Credit card (automatic)",
            "phone_service": True,
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "Yes",
            "online_backup": "No",
            "device_protection": "Yes",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
            "monthly_charges": 75.50,
            "total_charges": 1815.00
        }
        
        response = requests.post(f"{base_url}/predict", json=customer_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            prediction = response.json()
            print(f"Customer ID: {prediction.get('customer_id')}")
            print(f"Churn Probability: {prediction.get('churn_probability'):.1%}")
            print(f"Risk Level: {prediction.get('risk_level')}")
            print(f"Confidence: {prediction.get('confidence_score'):.1%}")
        elif response.status_code == 503:
            print("Service unavailable - model not loaded")
        else:
            print(f"Error: {response.text}")
        
        print("\n✅ API test completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("Please start the API server first:")
        print("  python api/run_server.py")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_api()