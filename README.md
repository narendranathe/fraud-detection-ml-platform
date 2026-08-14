# Real-Time Fraud Detection ML Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.5-red)](https://kafka.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)](https://fastapi.tiangolo.com/)

I built this to learn Kafka past the quickstart level: a producer at 100 TPS, a consumer that scores each transaction with LightGBM behind FastAPI, predictions and latencies in Postgres, metrics in Prometheus. Scoring P99 came out at 1.12ms. The data is synthetic (100k rows, 2,034 fraud) and it all runs on one laptop, so the numbers mean the pipeline holds together. They don't mean it survives real payment traffic.

---

## What I'd fix

Things I know are wrong or lazy, in rough order of how much they'd hurt in production:

- ✅ **Consumer throughput** — fixed: the consumer now pools Postgres connections, reuses an HTTP session, calls `/predict/batch`, and bulk inserts with `execute_values`.
- ✅ **Fraud threshold** — fixed: `src/models/train.py` derives a cost-weighted threshold from the validation PR curve (default 10:1 missed-fraud vs false-alarm cost) and persists it to `artifacts/models/threshold.json`. The API loads and uses it instead of the hard-coded 0.5.
- If the model pickle is missing, the API silently scores with a rule (`amount > 1000` gets 0.8) and keeps returning 200s. `/health` reports `model_loaded: false` but nothing alerts on it. The Swagger screenshot below is that fallback running. It should fail the deploy or page someone, not improvise.
- Most of the 1.12ms is FastAPI overhead: request parsing, Pydantic, a one-row DataFrame built per call. LightGBM itself takes microseconds. If latency ever actually mattered, the DataFrame goes first, not the model.
- Offsets auto-commit and the insert is `ON CONFLICT (transaction_id) DO NOTHING`. Not exactly-once, but replays are no-ops, which is the property I actually wanted.

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
| Scoring throughput | ~0.5 predictions/s | Grafana panel, full run below |
| Training data | 100,000 rows, 2.03% fraud | Generator output |
| Fraud flagged in longest run | 21 of 2,082 (0.94%) | Grafana counters, see proof of work |

Training runs and parameters are in MLflow at `http://localhost:5000`, the compose server that training logs to by default. Set `MLFLOW_TRACKING_URI=file:./mlruns` to keep a local file store instead, and run `python view_mlflow_results.py` for a terminal dump of the same store.

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

## Proof of work

### Grafana, end of a full pipeline run

![Grafana dashboard](<grafana dashboard picture.jpg>)

Producer, consumer, API, and Postgres ran together for about 75 minutes. Total Predictions reads 2,082, Fraud Detected reads 21, and the Fraud Rate gauge sits at 0.937%. The panel that matters is Predictions Per Second: it holds ~0.5 ops/s from 16:00 to 17:15 while the producer was firing at 100 TPS. That gap is the consumer's serial loop (one HTTP call and one fresh Postgres connection per transaction) falling behind, which is why connection pooling and the unused batch endpoint are the first item in What I'd fix. The 0.94% flag rate against a 2.03% training base rate is the threshold problem, same list.

### Prometheus, where the latency numbers come from

![prometheus latency_seconds_sum 5m dashboard](https://github.com/user-attachments/assets/77b12934-0cf4-4a02-8e08-c3cfee9577ad)

The query is `rate(prediction_latency_seconds_sum[5m])`, total scoring seconds accumulated per second over a 5-minute window. At roughly one prediction per second that rate equals mean latency per score: baseline ~0.5ms, with a spike to ~1.7ms around 22:52 when the load increased. The P50 / P95 / P99 figures in the results table come from the `prediction_latency_seconds` histogram this counter feeds.

### Swagger, the API contract (and the fallback, visible)

![FastAPI Swagger UI](image.png)

`/predict` try-it-out with the sample transaction, response 200: probability 0.8, prediction 1, risk level high, latency 5ms. Look at the probability: exactly 0.8 on a $1,500 transaction is the demo rule (`amount > 1000`) firing, which means this run happened before the model artifact was mounted. The screenshot does double duty. It documents the request and response shape, and it shows the silent fallback from What I'd fix serving real 200s with nobody paged.

---

## Structure

```
├── data/generate_synthetic_data.py   # labeled transaction generator
├── src/
│   ├── data_ingestion/               # kafka_producer.py, kafka_consumer.py
│   ├── api/main.py                   # FastAPI scoring service
│   ├── models/                       # train.py, evaluate.py
│   ├── utils/                        # threshold.py (cost-weighted threshold selection)
│   └── ...
├── docker/                           # docker-compose.yml, init-db.sql
├── monitoring/                       # prometheus.yml, grafana/
├── airflow/                          # batch DAGs
├── artifacts/models/                 # trained model + threshold.json
└── tests/                            # test_api.py, test_consumer.py, test_threshold.py
```

---

## License

MIT
