# Real-Time Fraud Detection ML Platform

End-to-end ML platform for detecting fraudulent transactions using Apache Kafka, Spark Streaming, MLflow, and Airflow.

## 🎯 Project Overview

This project demonstrates production-grade ML engineering skills including:
- Real-time fraud detection with machine learning
- Streaming data pipeline with Kafka and Spark
- MLOps with MLflow for experiment tracking and model registry
- Orchestration with Apache Airflow
- Feature engineering with time-based aggregations
- Model monitoring and drift detection

## 🏗️ Architecture
```
Data Sources → Kafka → Spark Streaming → Real-Time Predictions → Monitoring
                ↓                ↓
            Delta Lake    Feature Store
                ↓                ↓
            Airflow → MLflow → Model Registry → Databricks
```

## 🛠️ Tech Stack

- **Languages**: Python, SQL
- **ML/Data**: Scikit-learn, LightGBM, XGBoost, Pandas, NumPy
- **Streaming**: Apache Kafka, Apache Spark Structured Streaming
- **MLOps**: MLflow, Airflow
- **Storage**: Delta Lake
- **Infrastructure**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana

## 📊 Dataset

- 100,000 synthetic transactions
- 10,000 users, 5,000 merchants
- ~2% fraud rate with realistic patterns
- 30+ engineered features

## 🚀 Features

### Data Pipeline
- Synthetic transaction data generator with realistic fraud patterns
- Feature engineering with point-in-time correctness
- Time-based aggregations (1h, 24h, 7d windows)
- User and merchant behavior features
- Velocity and anomaly detection features

### ML Model
- LightGBM classifier for fraud detection
- Handles class imbalance with balanced weights
- PR-AUC optimization for fraud detection
- Feature importance analysis
- Model explainability with SHAP

### MLOps
- Experiment tracking with MLflow
- Model versioning and registry
- Automated retraining pipeline
- Performance monitoring
- A/B testing framework

## 📁 Project Structure
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

## 🏃 Getting Started

### Prerequisites
- Python 3.10+
- Docker Desktop
- Git

### Installation

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-ml-platform.git
cd fraud-detection-ml-platform
```

2. Create virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/Mac
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Generate synthetic data
```bash
python data/generate_synthetic_data.py
```

5. Run feature engineering
```bash
python src/feature_engineering/batch_features.py
```

6. Train model
```bash
python src/models/train.py
```

7. Evaluate model
```bash
python src/models/evaluate.py
```

## 📈 Model Performance

- **PR-AUC**: 0.XX
- **ROC-AUC**: 0.XX
- **F1-Score**: 0.XX
- **Precision**: 0.XX
- **Recall**: 0.XX

## 🔍 Key Insights

- Top fraud indicators: velocity features, amount deviation, distance from home
- Model catches XX% of fraud amount with <X% false positive rate
- Real-time inference latency: <100ms

## 🚧 Roadmap

- [x] Data generation pipeline
- [x] Feature engineering
- [x] Model training with MLflow
- [x] Model evaluation
- [ ] Kafka streaming pipeline
- [ ] Spark real-time inference
- [ ] Airflow orchestration
- [ ] REST API for predictions
- [ ] Model monitoring dashboard
- [ ] Online learning implementation

## 📝 License

This project is for portfolio and educational purposes.

## 👤 Author

**Narendranath**
- Data Engineer with 4+ years of experience
- Specializing in Big Data, ETL/ELT, and ML pipelines
- [LinkedIn](your-linkedin-url)
- [Email](your-email)

## 🙏 Acknowledgments

- Synthetic data generation inspired by real-world fraud patterns
- Architecture design based on industry best practices for ML systems