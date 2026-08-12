# Real-Time Fraud Detection ML Platform

Kafka consumers score streaming transactions with a LightGBM classifier behind FastAPI, at a measured P99 of 1.12ms per prediction. Every prediction, its probability, and its latency land in PostgreSQL and Prometheus, so detection rate and scoring speed are queryable rather than asserted. Trained on 100,000 synthetic transactions with a 2.03% fraud rate; experiments tracked in MLflow.

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
2. The producer streams them at a sustained 100 TPS into a partitioned topic.
3. The consumer reads in 50-message batches and calls the FastAPI scoring endpoint.
4. FastAPI runs the LightGBM model (P50 0.45ms, P95 0.89ms, P99 1.12ms, single-sample scoring) and persists transaction ID, amount, probability, and prediction to PostgreSQL.
5. Prometheus scrapes the API; Grafana renders throughput, latency percentiles, and fraud detection rate.

Scope note: the dataset is synthetic and the scale is a single Docker Compose host. The point is the wiring, instrumentation, and measured latency, not benchmark claims against production payment traffic.

---

## Measured results

| Metric | Value | How it was measured |
|--------|-------|---------------------|
| Scoring latency P50 | 0.45ms | Prometheus histogram on `/predict`, single sample, no batching |
| Scoring latency P95 | 0.89ms | Same |
| Scoring latency P99 | 1.12ms | Same |
| Producer throughput | 100 TPS sustained | `src/data_ingestion/kafka_producer.py` rate-limited loop |
| Consumer batch size | 50 messages | `kafka_consumer.py` poll configuration |
| Training data | 100,000 rows, 2.03% fraud | `data/generate_synthetic_data.py` output |
| Fraud detected in demo run | ~2% of 180+ scored transactions | PostgreSQL `predictions` table count |

Model and training runs are in MLflow (`http://localhost:5000` when the stack is up; `view_mlflow_results.py` prints them without the UI).

---

## Stack

| Layer | Choice |
|-------|--------|
| Streaming | Apache Kafka 3.5, partitioned topics |
| Model | LightGBM classifier, scikit-learn feature pipeline, MLflow tracking |
| Serving | FastAPI + Uvicorn, Pydantic validation |
| Storage | PostgreSQL 16 (predictions), Redis 7 (feature cache) |
| Monitoring | Prometheus scrape config in `monitoring/`, Grafana dashboards |
| Batch workflows | Airflow DAGs in `airflow/` |
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
# predictions landing in Postgres
docker exec -it fraud-postgres psql -U fraud_user -d fraud_detection \
  -c "SELECT COUNT(*) FROM predictions;"

# sample prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TEST_001", "customer_id": "CUST_123", "merchant_id": "MERCH_456",
       "merchant_category": "online_shopping", "amount": 1500.00, "device_type": "web",
       "distance_from_home": 450.0, "merchant_risk_score": 0.65, "customer_age": 35,
       "account_age_days": 730, "hour": 14, "day_of_week": 2, "is_weekend": 0}'
```

API docs at `http://localhost:8000/docs`; Grafana login is `admin/admin`.

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
├── notebooks/                        # training and EDA
├── artifacts/                        # trained model artifacts
└── tests/
```

---

## License

MIT
