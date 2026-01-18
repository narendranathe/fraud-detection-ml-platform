"""
FastAPI Fraud Detection Service
Real-time fraud prediction API with <100ms latency
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from contextlib import asynccontextmanager
import time
import joblib
import pandas as pd
from loguru import logger
from pathlib import Path
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

logger.add("logs/api.log", rotation="10 MB")

# Prometheus metrics
PREDICTIONS_COUNTER = Counter('fraud_predictions_total', 'Total predictions made')
FRAUD_DETECTED_COUNTER = Counter('fraud_detected_total', 'Total fraud cases detected')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

# Global model cache
MODEL = None
MODEL_VERSION = "v1.0.0"


def load_model():
    """Load latest trained model"""
    global MODEL, MODEL_VERSION
    
    try:
        model_path = Path("artifacts/models/fraud_detector_model.pkl")
        if model_path.exists():
            MODEL = joblib.load(model_path)
            logger.info(f"✅ Loaded model from {model_path}")
        else:
            logger.warning(f"⚠️ Model not found at {model_path} - Using demo mode")
            MODEL = None
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Fraud Detection API...")
    load_model()
    yield
    # Shutdown
    logger.info("👋 Shutting down Fraud Detection API...")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with ML",
    version="1.0.0",
    lifespan=lifespan
)


class Transaction(BaseModel):
    """Transaction input schema"""
    transaction_id: str
    customer_id: str
    merchant_id: str
    merchant_category: str
    amount: float = Field(..., gt=0)
    device_type: str
    distance_from_home: float = Field(..., ge=0)
    merchant_risk_score: float = Field(..., ge=0, le=1)
    customer_age: int = Field(..., ge=18, le=100)
    account_age_days: int = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    """Prediction output schema"""
    transaction_id: str
    fraud_probability: float
    prediction: int
    risk_level: str
    latency_ms: float
    model_version: str


@app.get("/")
async def root():
    """Health check"""
    return {
        "service": "Fraud Detection API",
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION,
        "timestamp": time.time()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """Predict fraud probability for a transaction"""
    
    with PREDICTION_LATENCY.time():
        start_time = time.time()
        
        if MODEL is None:
            # Demo mode: use rule-based prediction
            fraud_prob = 0.8 if transaction.amount > 1000 else 0.2
        else:
            # Convert to DataFrame
            data = pd.DataFrame([transaction.dict()])
            
            # Make prediction
            fraud_prob = MODEL.predict_proba(data)[0][1]
        
        # Binary prediction (threshold = 0.5)
        prediction = 1 if fraud_prob > 0.5 else 0
        
        # Update metrics
        PREDICTIONS_COUNTER.inc()
        if prediction == 1:
            FRAUD_DETECTED_COUNTER.inc()
        
        # Risk level
        if fraud_prob < 0.3:
            risk_level = "low"
        elif fraud_prob < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Prediction: {transaction.transaction_id} | "
            f"Prob: {fraud_prob:.4f} | Latency: {latency_ms:.2f}ms"
        )
        
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(fraud_prob, 4),
            prediction=prediction,
            risk_level=risk_level,
            latency_ms=round(latency_ms, 2),
            model_version=MODEL_VERSION
        )


@app.post("/predict/batch")
async def predict_batch(transactions: List[Transaction]):
    """Batch prediction endpoint"""
    start_time = time.time()
    
    results = []
    
    if MODEL is None:
        # Demo mode
        for txn in transactions:
            fraud_prob = 0.8 if txn.amount > 1000 else 0.2
            prediction = 1 if fraud_prob > 0.5 else 0
            
            PREDICTIONS_COUNTER.inc()
            if prediction == 1:
                FRAUD_DETECTED_COUNTER.inc()
            
            results.append({
                "transaction_id": txn.transaction_id,
                "fraud_probability": round(fraud_prob, 4),
                "prediction": prediction,
                "risk_level": "high" if fraud_prob > 0.7 else "medium" if fraud_prob > 0.3 else "low"
            })
    else:
        # Real model
        data = pd.DataFrame([t.dict() for t in transactions])
        fraud_probs = MODEL.predict_proba(data)[:, 1]
        
        for txn, prob in zip(transactions, fraud_probs):
            prediction = 1 if prob > 0.5 else 0
            
            PREDICTIONS_COUNTER.inc()
            if prediction == 1:
                FRAUD_DETECTED_COUNTER.inc()
            
            results.append({
                "transaction_id": txn.transaction_id,
                "fraud_probability": round(float(prob), 4),
                "prediction": prediction,
                "risk_level": "high" if prob > 0.7 else "medium" if prob > 0.3 else "low"
            })
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "predictions": results,
        "batch_size": len(transactions),
        "latency_ms": round(latency_ms, 2),
        "avg_latency_per_txn_ms": round(latency_ms / len(transactions), 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")