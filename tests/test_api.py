"""
Test FastAPI fraud detection service using TestClient.

This runs the API in-process so it does not depend on a server already
running on localhost:8000. The model artifact is not required; the test
validates the contract in demo/rule-based fallback mode.
"""

import json
import os
import sys

# Allow importing src.api.main when running this script directly from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

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

# A second sample for batch testing
sample_transaction_2 = {
    "transaction_id": "TXN_TEST_002",
    "customer_id": "CUST_123457",
    "merchant_id": "MERCH_78902",
    "merchant_category": "gas_station",
    "amount": 45.00,
    "device_type": "mobile",
    "distance_from_home": 5.0,
    "merchant_risk_score": 0.20,
    "customer_age": 28,
    "account_age_days": 365,
    "hour": 9,
    "day_of_week": 1,
    "is_weekend": 0
}

# Test health endpoint
print("Testing /health endpoint...")
response = client.get("/health")
print(json.dumps(response.json(), indent=2))
assert response.status_code == 200
assert "model_loaded" in response.json()

# Test prediction endpoint
print("\nTesting /predict endpoint...")
response = client.post("/predict", json=sample_transaction)
print(json.dumps(response.json(), indent=2))
assert response.status_code == 200
pred = response.json()
assert pred["transaction_id"] == sample_transaction["transaction_id"]
assert "model_version" in pred
assert "latency_ms" in pred

# Test batch prediction endpoint
print("\nTesting /predict/batch endpoint...")
response = client.post("/predict/batch", json=[sample_transaction, sample_transaction_2])
batch_result = response.json()
print(json.dumps(batch_result, indent=2))
assert response.status_code == 200
assert batch_result.get("batch_size") == 2, "Batch size should be 2"
assert len(batch_result.get("predictions", [])) == 2, "Should return 2 predictions"
assert "model_version" in batch_result, "Batch response should include model_version"
for pred in batch_result["predictions"]:
    assert "model_version" in pred, "Each prediction should include model_version"
    assert "latency_ms" in pred, "Each prediction should include latency_ms"

print("\n✅ API is working!")
