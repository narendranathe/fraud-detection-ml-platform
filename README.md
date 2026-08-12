# Real-Time Fraud Detection ML Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.5-red)](https://kafka.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)](https://fastapi.tiangolo.com/)

Kafka consumers score a transaction stream with a LightGBM model served by FastAPI. Measured P99 is 1.12ms per prediction. Every prediction, probability, and latency goes to PostgreSQL and Prometheus, so the numbers below come out of the database, not from memory. The model is trained on 100,000 synthetic transactions with a 2.03% fraud rate, tracked in MLflow.

The data is made up and the stack runs on one host. Take the numbers as a check that the parts connect, not as proof of scale.

---

## What surprised me

I assumed the model would be the slow part, so I measured it first. It wasn't. The 1.12ms P99 is almost all overhead: HTTP parsing, Pydantic validation, and building a one-row pandas DataFrame per request in `main.py`. LightGBM scores a single row in microseconds. I spent a day profiling the wrong thing.

The consumer opens a new Postgres connection for every prediction (`kafka_consumer.py`). At 100 TPS that is 100 connections a second, and Postgres runs out of patience long before Kafka does. There is a `/predict/batch` endpoint in `main.py` that I wrote and then never wired into the consumer. If I pick this project back up, that is the first fix.

An earlier draft of this README claimed exactly-once processing. It isn't true, and it doesn't need to be. Offsets auto-commit, and the insert uses `ON CONFLICT (transaction_id) DO NOTHING`, so a replayed message is a no-op. Idempotent writes get you the same guarantee without the ceremony.

The fraud threshold is 0.5 because that is the library default. With 2,034 fraud rows out of 100,000, a model that never fires is already 97.97% accurate, so the cutoff has to come from the precision-recall curve and from what a missed fraud costs versus a false alarm. I haven't done that tuning. It is the weakest part of the project right now.

If the model file is missing, the API quietly switches to a rule: amount over 1000 scores 0.8. `/health` reports `model_loaded: false`, but nothing alerts on it and the 200s keep coming. In production that means serving rule-based guesses for days while every dashboard says healthy.

---

## How it works

```
Kafka producer ──▶ Kafka topic ──▶ Python consumer ──▶ FastAPI /predict ──▶ PostgreSQL
(100 TPS)                        (50 msg batches)      (LightGBM, P99 1.12ms)      │
                                                                                    ▼
                                                                              Prometheus ──▶ Grafana
                                                                              (latency percentiles,
                                                                               fraud rate, throughput)
```

1. `data/generate_synthetic_data.py` writes 100,000 labeled transactions (2,034 fraud, 2.03%) to `data/raw/`.
2. The producer streams them at 100 TPS into a partitioned topic.
3. The consumer reads in 50-message batches and posts each transaction to `/predict`.
4. FastAPI scores with LightGBM (P50 0.45ms, P95 0.89ms, P99 1.12ms) and the consumer persists ID, amount, probability, and latency to PostgreSQL.
5. Prometheus scrapes `/metrics`; Grafana renders throughput, latency percentiles, and detection rate.

---

## Dashboards

Grafana, during a 100 TPS run:

![Grafana dashboard](<grafana dashboard picture.jpg>)

Prediction latency sum over 5m, from Prometheus:

![prometheus latency_seconds_sum 5m dashboard](https://github.com/user-attachments/assets/77b12934-0cf4-4a02-8e08-c3cfee9577ad)

The API ships with OpenAPI docs at `/docs`:

![FastAPI Swagger UI](image.png)

---

## Measured results

| Metric | Value | How it was measured |
|--------|-------|---------------------|
| Scoring latency P50 | 0.45ms | Prometheus histogram on `/predict`, single sample, no batching |
| Scoring latency P95 | 0.89ms | Same |
| Scoring latency P99 | 1.12ms | Same |
| Producer throughput | 100 TPS | Rate-limited loop in `kafka_producer.py` |
| Training data | 100,000 rows, 2.03% fraud | Generator output |
| Fraud detected in demo run | ~2% of 180+ scored transactions | Row count in the `predictions` table |

Training runs and parameters are in MLflow (`http://localhost:5000` when the stack is up, or `python view_mlflow_results.py`).

---

## Stack

| Layer | Choice |
|-------|--------|
| Streaming | Apache Kafka 3.5, partitioned topics |
| Model | LightGBM, scikit-learn pipeline, MLflow tracking |
| Serving | FastAPI + Uvicorn, Pydantic validation |
| Storage | PostgreSQL 16, Redis 7 feature cache |
| Monitoring | Prometheus + Grafana, config in `monitoring/` |
| Batch | Airflow DAGs in `airflow/` |
| Infra | Docker Compose, 8GB RAM minimum |

---

## Run it

```bash
# 1. Infrastructure (FastAPI :8000, Prometheus :9090, Grafana :3000, MLflow :5000, Airflow :8080)
cd docker && docker compose up -d

# 2. Python environment
conda create -n fraud-detection python=3.11 -y
conda activate fraud-detection
pip install -r requirements.txt

# 3. Training data (100k rows, ~2% fraud)
python data/generate_synthetic_data.py

# 4. Pipeline, three terminals
python src/api/main.py                          # scoring API
python src/data_ingestion/kafka_producer.py     # 100 TPS stream
python src/data_ingestion/kafka_consumer.py     # scoring consumer
```

Verify:

```bash
docker exec -it fraud-postgres psql -U fraud_user -d fraud_detection \
  -c "SELECT COUNT(*) FROM predictions;"

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TEST_001", "customer_id": "CUST_123", "merchant_id": "MERCH_456",
       "merchant_category": "online_shopping", "amount": 1500.00, "device_type": "web",
       "distance_from_home": 450.0, "merchant_risk_score": 0.65, "customer_age": 35,
       "account_age_days": 730, "hour": 14, "day_of_week": 2, "is_weekend": 0}'
```

API docs at `http://localhost:8000/docs`. Grafana login is `admin/admin`.

---

## Structure

```
├── data/generate_synthetic_data.py   # labeled transaction generator
├── src/
│   ├── data_ingestion/               # kafka_producer.py, kafka_consumer.py
│   ├── api/main.py                   # FastAPI scoring service
│   └── utils/
├── docker/                           # docker-compose.yml, init-db.sql
├── monitoring/                       # prometheus.yml, grafana/
├── airflow/                          # batch DAGs
├── artifacts/                        # trained model artifacts
└── tests/                            # test_api.py
```

---

## License

MIT
