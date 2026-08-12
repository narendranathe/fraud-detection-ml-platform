# Real-Time Fraud Detection ML Platform

Kafka consumers score a transaction stream with a LightGBM classifier behind FastAPI, at a measured P99 of 1.12ms per prediction. Each prediction, its probability, and its latency land in PostgreSQL and Prometheus, so detection rate and scoring speed are things you query, not things I claim. Trained on 100,000 synthetic transactions with a 2.03% fraud rate; runs tracked in MLflow.

The data is made up and the stack runs on one host. What this proves: the parts connect, and each claim has a number you can check. What it does not prove: fraud at bank scale.

---

## What this project taught me

**The model was never the slow part.** The 1.12ms P99 includes HTTP parsing, Pydantic validation, and a one-row DataFrame built per request in `main.py`. LightGBM itself scores in microseconds. Tuning the model would have optimized the cheapest line on the bill.

**The throughput ceiling is connection churn, not Kafka.** The consumer opens a fresh Postgres connection per prediction (`kafka_consumer.py`), so 100 TPS means 100 new connections a second. A `/predict/batch` endpoint exists and the consumer never calls it. Pooling and batching would buy more throughput than any broker tuning.

**Idempotence beat exactly-once.** Offsets auto-commit, and the sink writes with `ON CONFLICT (transaction_id) DO NOTHING`. A replayed message does no harm. That is the property exactly-once delivery pretends to give you, at a fraction of the complexity.

**The default threshold is wrong at a 2% base rate.** With 2,034 fraud rows in 100,000, predicting "no" every time scores 97.97% accuracy. The 0.5 cutoff in `main.py` comes from the library default, not from the cost of a missed fraud versus a false alarm. Picking the cutoff off the precision-recall curve is the actual work, and this repo does not do it yet.

**A silent fallback is an outage with extra steps.** If the model file is missing, the API switches to a rule (amount over 1000 gets 0.8) and keeps returning 200s. `/health` exposes `model_loaded`, but nothing alerts on it. In a real system I would make that fallback loud, page on it, and treat it as degraded service.

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
