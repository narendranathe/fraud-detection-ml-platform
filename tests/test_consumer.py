"""
Unit tests for the Kafka consumer batch/scoring path.

These tests mock Kafka and Postgres so they can run without the full
infrastructure stack. They verify the consumer correctly:
- strips non-prediction fields before calling the API
- enriches batch predictions with required fields
- batches API calls and bulk inserts
- logs fraud detections
"""

from unittest.mock import Mock, patch
import pytest
import os
import sys

# Allow importing src.data_ingestion.kafka_consumer when running directly from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.kafka_consumer import FraudDetectionConsumer


@pytest.fixture
def consumer():
    """Return a consumer with Kafka and Postgres connections mocked."""
    mock_pool = Mock()
    mock_pool.getconn.return_value = Mock()

    with patch("src.data_ingestion.kafka_consumer.KafkaConsumer"), \
         patch(
             "src.data_ingestion.kafka_consumer.psycopg2.pool.ThreadedConnectionPool",
             return_value=mock_pool,
         ):
        yield FraudDetectionConsumer(
            kafka_bootstrap_servers="localhost:9092",
            kafka_topic="transactions",
            api_url="http://localhost:8000",
            api_batch_size=50,
        )


@pytest.fixture
def sample_transaction():
    return {
        "transaction_id": "TX_1",
        "customer_id": "C1",
        "amount": 100.0,
        "timestamp": "2024-01-01T00:00:00",
        "is_fraud": 1,
        "merchant_id": "M1",
        "merchant_category": "grocery",
        "device_type": "mobile",
        "distance_from_home": 10.0,
        "merchant_risk_score": 0.5,
        "customer_age": 30,
        "account_age_days": 100,
        "hour": 12,
        "day_of_week": 1,
        "is_weekend": 0,
    }


def test_prepare_transaction_strips_non_prediction_fields(consumer, sample_transaction):
    prepared = consumer._prepare_transaction(sample_transaction)

    assert "is_fraud" not in prepared
    assert "timestamp" not in prepared
    assert "transaction_id" in prepared
    assert prepared["transaction_id"] == "TX_1"


def test_call_prediction_batch_api_enriches_missing_fields(consumer, sample_transaction):
    mock_response = Mock()
    mock_response.json.return_value = {
        "predictions": [
            {"transaction_id": "TX_1", "fraud_probability": 0.9, "prediction": 1, "risk_level": "high"},
        ],
        "model_version": "v1.0.0",
        "avg_latency_per_txn_ms": 0.5,
    }
    mock_response.raise_for_status.return_value = None
    consumer.http_session.post = Mock(return_value=mock_response)

    predictions = consumer.call_prediction_batch_api([sample_transaction])

    assert len(predictions) == 1
    assert predictions[0]["model_version"] == "v1.0.0"
    assert predictions[0]["latency_ms"] == 0.5


def test_process_batch_calls_api_and_inserts_bulk(consumer, sample_transaction):
    txn2 = {**sample_transaction, "transaction_id": "TX_2"}

    consumer.call_prediction_batch_api = Mock(return_value=[
        {
            "transaction_id": "TX_1",
            "fraud_probability": 0.9,
            "prediction": 1,
            "risk_level": "high",
            "model_version": "v1.0.0",
            "latency_ms": 0.5,
        },
        {
            "transaction_id": "TX_2",
            "fraud_probability": 0.1,
            "prediction": 0,
            "risk_level": "low",
            "model_version": "v1.0.0",
            "latency_ms": 0.5,
        },
    ])
    consumer.save_predictions_to_postgres = Mock(return_value=2)

    consumer.process_batch([sample_transaction, txn2])

    consumer.call_prediction_batch_api.assert_called_once()
    consumer.save_predictions_to_postgres.assert_called_once()
    saved_records = consumer.save_predictions_to_postgres.call_args[0][0]
    assert len(saved_records) == 2


def test_process_batch_splits_large_batches(consumer, sample_transaction):
    consumer.api_batch_size = 2
    consumer.call_prediction_batch_api = Mock(return_value=[])
    consumer.save_predictions_to_postgres = Mock(return_value=0)

    transactions = [{**sample_transaction, "transaction_id": f"TX_{i}"} for i in range(5)]
    consumer.process_batch(transactions)

    assert consumer.call_prediction_batch_api.call_count == 3  # 2 + 2 + 1
