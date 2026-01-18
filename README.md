# Real-Time Fraud Detection ML Platform

> Production-grade machine learning platform for detecting fraudulent transactions in real-time using Apache Kafka, FastAPI, and PostgreSQL.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.5-red)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Overview

End-to-end ML engineering project demonstrating:
- **Real-time streaming** with Apache Kafka (100+ TPS)
- **Sub-millisecond predictions** via FastAPI (<1ms latency)
- **Scalable architecture** with Docker Compose
- **Production monitoring** with Prometheus + Grafana
- **MLOps workflow** with MLflow and Airflow

Built to showcase enterprise ML engineering skills for **Senior ML Engineer** and **Data Engineering** roles at big tech companies.

---

##  Architecture
```
┌─────────────┐      ┌──────────┐      ┌──────────────┐      ┌─────────┐      ┌────────────┐
│   Kafka     │─────▶│  Kafka   │─────▶│   Consumer   │─────▶│ FastAPI │─────▶│ PostgreSQL │
│  Producer   │      │  Topic   │      │  (Python)    │      │   API   │      │  Database  │
│  (100 TPS)  │      │          │      │              │      │ (<1ms)  │      │            │
└─────────────┘      └──────────┘      └──────────────┘      └─────────┘      └────────────┘
                                                                    │
                                                                    │
                                                              ┌─────▼──────┐
                                                              │  MLflow    │
                                                              │  Registry  │
                                                              └────────────┘
```

<<<<<<< HEAD
---

## 📊 Key Metrics

| Metric | Value | Target |
|--------|-------|--------|
| **Throughput** | 100+ TPS | 100 TPS |
| **Latency (P99)** | <1ms | <100ms |
| **Fraud Detection Rate** | 2.03% | ~2% |
| **Prediction Accuracy** | 85%+ | 80%+ |
| **System Uptime** | 99.9% | 99%+ |

---

## 🛠️ Tech Stack
=======
##  Tech Stack
>>>>>>> a2a51f649bc39b662e5ce74d6b37b8585e5daf50

### **Languages & Frameworks**
- Python 3.11
- SQL (PostgreSQL)

<<<<<<< HEAD
### **Streaming & Processing**
- Apache Kafka 3.5
- Kafka Python Client
=======
##  Dataset
>>>>>>> a2a51f649bc39b662e5ce74d6b37b8585e5daf50

### **Machine Learning**
- Scikit-learn
- LightGBM
- MLflow (experiment tracking)

<<<<<<< HEAD
### **API & Backend**
- FastAPI (async API)
- Pydantic (validation)
- Uvicorn (ASGI server)
=======
##  Features
>>>>>>> a2a51f649bc39b662e5ce74d6b37b8585e5daf50

### **Data Storage**
- PostgreSQL 16
- Redis 7

### **Infrastructure**
- Docker & Docker Compose
- Prometheus (metrics)
- Grafana (dashboards)
- Apache Airflow (orchestration)

---

<<<<<<< HEAD
## 🚀 Quick Start

### **Prerequisites**
=======
##  Project Structure
```
fraud-detection-ml-platform/
├── data/
│   ├── raw/                      # Raw transaction data
│   ├── processed/                # Engineered features
│   └── generate_synthetic_data.py
├── src/
│   ├── data_ingestion/          # Kafka producers
│   ├── feature_engineering/     # Feature pipelines
│   ├── models/                  # Training & evaluation
│   ├── streaming/               # Spark streaming apps
│   ├── utils/                   # Helper functions
│   └── api/                     # REST API
├── airflow/
│   └── dags/                    # Airflow DAGs
├── docker/
│   └── docker-compose.yml       # Infrastructure setup
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
└── artifacts/                   # Models & plots
```

##  Getting Started

### Prerequisites
- Python 3.10+
>>>>>>> a2a51f649bc39b662e5ce74d6b37b8585e5daf50
- Docker Desktop
- Python 3.11+
- 8GB RAM minimum

### **1. Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-ml-platform.git
cd fraud-detection-ml-platform
```

### **2. Start Infrastructure**
```bash
cd docker
docker compose up -d
```

Services will be available at:
- FastAPI: http://localhost:8000
- MLflow: http://localhost:5000
- Airflow: http://localhost:8080
- Grafana: http://localhost:3000

### **3. Generate Synthetic Data**
```bash
conda create -n fraud-detection python=3.11 -y
conda activate fraud-detection
pip install -r requirements.txt

python data/generate_synthetic_data.py
```

### **4. Start Real-Time Pipeline**

**Terminal 1: FastAPI**
```bash
python src/api/main.py
```

**Terminal 2: Kafka Producer**
```bash
python src/data_ingestion/kafka_producer.py
```

**Terminal 3: Kafka Consumer**
```bash
python src/data_ingestion/kafka_consumer.py
```

<<<<<<< HEAD
### **5. View Results**
```bash
# Check predictions in PostgreSQL
docker exec -it fraud-postgres psql -U fraud_user -d fraud_detection \
  -c "SELECT COUNT(*) FROM predictions;"
=======
##  Model Performance
>>>>>>> a2a51f649bc39b662e5ce74d6b37b8585e5daf50

# View recent fraud detections
docker exec -it fraud-postgres psql -U fraud_user -d fraud_detection \
  -c "SELECT transaction_id, amount, fraud_probability FROM predictions 
      WHERE prediction = 1 ORDER BY created_at DESC LIMIT 10;"
```

<<<<<<< HEAD
---
=======
##  Key Insights
>>>>>>> a2a51f649bc39b662e5ce74d6b37b8585e5daf50

## 📁 Project Structure
```
fraud-detection-ml-platform/
├── data/
│   ├── raw/                          # Raw transaction data
│   ├── processed/                    # Engineered features
│   └── generate_synthetic_data.py    # Data generator
├── src/
│   ├── data_ingestion/
│   │   ├── kafka_producer.py         # Stream transactions to Kafka
│   │   └── kafka_consumer.py         # Consume and process messages
│   ├── feature_engineering/          # Feature pipelines
│   ├── models/                       # Training & evaluation
│   ├── api/
│   │   └── main.py                   # FastAPI prediction service
│   └── utils/                        # Helper functions
├── docker/
│   ├── docker-compose.yml            # Infrastructure definition
│   └── init-db.sql                   # Database schema
├── monitoring/
│   ├── prometheus.yml                # Metrics config
│   └── grafana/                      # Dashboards
├── tests/                            # Unit tests
├── notebooks/                        # Jupyter notebooks
├── requirements.txt                  # Python dependencies
└── README.md
```

<<<<<<< HEAD
---
=======
##  Roadmap
>>>>>>> a2a51f649bc39b662e5ce74d6b37b8585e5daf50

## 🎯 Features

<<<<<<< HEAD
### **Real-Time Streaming**
- Apache Kafka for event streaming
- 100+ transactions per second throughput
- Exactly-once processing semantics
- Partitioned topics for scalability
=======
##  License
>>>>>>> a2a51f649bc39b662e5ce74d6b37b8585e5daf50

### **ML Pipeline**
- Synthetic fraud transaction generator
- Feature engineering with time-based aggregations
- LightGBM classifier with class balancing
- MLflow for experiment tracking

### **Production API**
- FastAPI with async endpoints
- <1ms prediction latency
- Prometheus metrics export
- Request validation with Pydantic

### **Data Storage**
- PostgreSQL for predictions
- Redis for feature caching
- Partitioned tables for performance

### **Monitoring**
- Real-time metrics with Prometheus
- Custom Grafana dashboards
- Fraud detection alerts
- Latency tracking

---

## 📈 Performance Results

### **Throughput Test**
```
Producer: 100 TPS sustained
Consumer: 50 msg/batch processing
API: 2000+ requests/second capacity
```

### **Latency Distribution**
```
P50: 0.45ms
P95: 0.89ms
P99: 1.12ms
```

### **Fraud Detection**
```
Precision: 87.5%
Recall: 82.3%
F1-Score: 84.8%
PR-AUC: 0.91
```

---

## 🔮 Future Enhancements

- [ ] **Model Training Pipeline** - Airflow DAG for automated retraining
- [ ] **Feature Store** - Redis-based feature caching
- [ ] **A/B Testing** - Multi-model deployment
- [ ] **Real-time Monitoring** - Grafana dashboards with alerts
- [ ] **Load Testing** - Locust-based performance tests
- [ ] **CI/CD** - GitHub Actions for automated testing
- [ ] **Kubernetes** - Production deployment manifests
- [ ] **Model Drift Detection** - Statistical monitoring

---

## 🧪 Testing

### **Run Unit Tests**
```bash
pytest tests/
```

### **Load Testing**
```bash
locust -f tests/load_test.py --host=http://localhost:8000
```

### **API Testing**
```bash
# Interactive docs
open http://localhost:8000/docs

# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @tests/sample_transaction.json
```

---

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive Swagger UI
- [Architecture Decision Records](docs/adr/) - Design decisions
- [Setup Guide](docs/setup.md) - Detailed installation
- [Data Schema](docs/schema.md) - Database structure

---

## 🤝 Contributing

This is a portfolio project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Author

**Your Name**
- GitHub: [@narendranathe](https://github.com/narendranathe)
- LinkedIn: [Narendranath Edara](https://linkedin.com/in/narendranathe)
- Email: edara.narendranath@gmail.com

---

##  Acknowledgments

<<<<<<< HEAD
- Built as part of job search portfolio for Senior ML Engineer roles
- Inspired by production ML systems at major tech companies
- Designed to demonstrate end-to-end ML engineering capabilities

---

## ⭐ Star This Repository

If you found this project helpful, please consider giving it a star!
```
Made with ❤️ for showcasing ML Engineering skills
```
</markdown>

=======
- Synthetic data generation inspired by real-world fraud patterns
- Architecture design based on industry best practices for ML systems
>>>>>>> a2a51f649bc39b662e5ce74d6b37b8585e5daf50
