-- Initialize database schema for fraud detection platform

-- Feature store tables
CREATE TABLE IF NOT EXISTS feature_store (
    customer_id VARCHAR(50) PRIMARY KEY,
    txn_count_1h INT DEFAULT 0,
    txn_count_24h INT DEFAULT 0,
    txn_count_7d INT DEFAULT 0,
    total_amount_1h DECIMAL(10,2) DEFAULT 0,
    total_amount_24h DECIMAL(10,2) DEFAULT 0,
    total_amount_7d DECIMAL(10,2) DEFAULT 0,
    avg_amount_7d DECIMAL(10,2) DEFAULT 0,
    distinct_merchants_24h INT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feature_store_customer ON feature_store(customer_id);

-- Predictions log
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    fraud_probability DECIMAL(5,4) NOT NULL,
    prediction INT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    latency_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_customer ON predictions(customer_id);

-- Model metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_metrics_version ON model_metrics(model_version);

-- Model drift tracking
CREATE TABLE IF NOT EXISTS drift_metrics (
    drift_id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL,
    drift_score DECIMAL(10,6) NOT NULL,
    threshold DECIMAL(10,6) NOT NULL,
    is_drifted BOOLEAN NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_drift_timestamp ON drift_metrics(timestamp);

-- Alert log
CREATE TABLE IF NOT EXISTS alerts (
    alert_id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE feature_store IS 'Real-time feature cache for low-latency predictions';
COMMENT ON TABLE predictions IS 'Log of all fraud predictions for monitoring and audit';
COMMENT ON TABLE model_metrics IS 'Model performance metrics over time';
COMMENT ON TABLE drift_metrics IS 'Feature drift detection results';
COMMENT ON TABLE alerts IS 'System alerts and anomalies';