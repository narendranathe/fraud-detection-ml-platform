"""
Test FastAPI fraud detection service
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# Sample transaction
sample_transaction = {
    "transaction_id": "TXN_TEST_001",
    "customer_id": "CUST_123456",
    "merchant_id": "MERCH_78901",
    "merchant_category": "online_shopping",
    "amount": 1250.50,
    "device_type": "web",
    "distance_from_home": 450.0,
    "merchant_risk_score": 0.65,
    "customer_age": 35,
    "account_age_days": 730,
    "hour": 14,
    "day_of_week": 2,
    "is_weekend": 0
}

# Test health endpoint
print("Testing /health endpoint...")
response = requests.get(f"{BASE_URL}/health")
print(json.dumps(response.json(), indent=2))

# Test prediction endpoint
print("\nTesting /predict endpoint...")
response = requests.post(f"{BASE_URL}/predict", json=sample_transaction)
print(json.dumps(response.json(), indent=2))

print("\n✅ API is working!")