"""
Kafka Consumer for Real-Time Fraud Detection
Consumes transactions from Kafka, calls batch API, saves predictions to Postgres
"""

import json
import time
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import requests
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
from loguru import logger
from pathlib import Path
from typing import Dict, Any, List, Tuple

logger.add("logs/kafka_consumer.log", rotation="10 MB")


class FraudDetectionConsumer:
    """Consume transactions and detect fraud in real-time"""
    
    def __init__(
        self,
        kafka_bootstrap_servers: str = 'localhost:9092',
        kafka_topic: str = 'transactions',
        api_url: str = 'http://localhost:8000',
        postgres_config: Dict[str, str] = None,
        api_batch_size: int = 50,
        db_pool_min: int = 1,
        db_pool_max: int = 10
    ):
        self.kafka_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.api_url = api_url
        self.api_batch_size = api_batch_size
        
        # Postgres config
        if postgres_config is None:
            self.postgres_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'fraud_detection',
                'user': 'fraud_user',
                'password': 'fraud_password'
            }
        else:
            self.postgres_config = postgres_config
        
        # Reusable HTTP session for keep-alive
        self.http_session = requests.Session()
        
        # Postgres connection pool
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=db_pool_min,
            maxconn=db_pool_max,
            **self.postgres_config
        )
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='fraud-detection-consumer-group',
            max_poll_records=100
        )
        
        # Test Postgres connection
        self._test_postgres_connection()
        
        logger.info(f"✅ Kafka Consumer initialized")
        logger.info(f"   Topic: {kafka_topic}")
        logger.info(f"   API: {api_url}")
        logger.info(f"   Postgres: {self.postgres_config['host']}:{self.postgres_config['port']}")
        logger.info(f"   API batch size: {api_batch_size}")
    
    def _test_postgres_connection(self):
        """Test Postgres connection from the pool"""
        conn = None
        try:
            conn = self.db_pool.getconn()
            logger.info("✅ Postgres connection successful")
        except Exception as e:
            logger.error(f"❌ Postgres connection failed: {e}")
            raise
        finally:
            if conn is not None:
                self.db_pool.putconn(conn)
    
    def _prepare_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-prediction fields before sending to API"""
        return {
            k: v
            for k, v in transaction.items()
            if k not in ('is_fraud', 'timestamp')
        }
    
    def call_prediction_batch_api(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Call FastAPI batch prediction endpoint"""
        try:
            txns = [self._prepare_transaction(t) for t in transactions]
            
            response = self.http_session.post(
                f"{self.api_url}/predict/batch",
                json=txns,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            predictions = data.get('predictions', [])
            
            # Ensure each prediction has the fields the consumer needs
            default_model_version = data.get('model_version', 'unknown')
            default_latency = data.get('avg_latency_per_txn_ms', 0)
            for pred in predictions:
                pred.setdefault('model_version', default_model_version)
                pred.setdefault('latency_ms', default_latency)
            
            return predictions
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch API call failed for {len(transactions)} transactions: {e}")
            return []
    
    def save_predictions_to_postgres(self, records: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> int:
        """Bulk save predictions to Postgres using execute_values"""
        if not records:
            return 0
        
        conn = None
        try:
            conn = self.db_pool.getconn()
            cursor = conn.cursor()
            
            values = [
                (
                    prediction['transaction_id'],
                    transaction['customer_id'],
                    transaction['timestamp'],
                    transaction['amount'],
                    prediction['fraud_probability'],
                    prediction['prediction'],
                    prediction['model_version'],
                    prediction['latency_ms']
                )
                for transaction, prediction in records
            ]
            
            execute_values(
                cursor,
                """
                    INSERT INTO predictions (
                        transaction_id, customer_id, timestamp, amount,
                        fraud_probability, prediction, model_version, latency_ms
                    )
                    VALUES %s
                    ON CONFLICT (transaction_id) DO NOTHING
                """,
                values
            )
            
            conn.commit()
            cursor.close()
            return len(values)
            
        except Exception as e:
            logger.error(f"Failed to save predictions to Postgres: {e}")
            if conn is not None:
                conn.rollback()
            return 0
        finally:
            if conn is not None:
                self.db_pool.putconn(conn)
    
    def process_batch(self, messages: List[Dict[str, Any]]):
        """Process a batch of transactions in API-sized chunks"""
        start_time = time.time()
        predictions_saved = 0
        
        for i in range(0, len(messages), self.api_batch_size):
            chunk = messages[i:i + self.api_batch_size]
            chunk_by_id = {m['transaction_id']: m for m in chunk}
            
            predictions = self.call_prediction_batch_api(chunk)
            
            if not predictions:
                continue
            
            records = []
            for prediction in predictions:
                transaction = chunk_by_id.get(prediction['transaction_id'])
                if transaction is not None:
                    records.append((transaction, prediction))
            
            saved = self.save_predictions_to_postgres(records)
            predictions_saved += saved
            
            for prediction in predictions:
                if prediction['prediction'] == 1:
                    transaction = chunk_by_id.get(prediction['transaction_id'])
                    if transaction is not None:
                        logger.warning(
                            f"🚨 FRAUD DETECTED: {transaction['transaction_id']} | "
                            f"Amount: ${transaction['amount']:.2f} | "
                            f"Probability: {prediction['fraud_probability']:.4f}"
                        )
        
        batch_time = time.time() - start_time
        
        logger.info(
            f"Processed batch: {len(messages)} messages | "
            f"Saved: {predictions_saved} predictions | "
            f"Time: {batch_time:.2f}s"
        )
    
    def consume(self, batch_size: int = 100):
        """Start consuming messages from Kafka"""
        logger.info(f"🚀 Starting to consume from topic: {self.kafka_topic}")
        
        try:
            batch = []
            
            for message in self.consumer:
                transaction = message.value
                batch.append(transaction)
                
                # Process in batches matching Kafka poll size
                if len(batch) >= batch_size:
                    self.process_batch(batch)
                    batch = []
                    
        except KeyboardInterrupt:
            logger.warning("⚠️ Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.close()
            logger.info("✅ Kafka Consumer closed")
    
    def close(self):
        """Close all resources"""
        try:
            self.consumer.close()
        except Exception as e:
            logger.warning(f"Error closing Kafka consumer: {e}")
        
        try:
            self.http_session.close()
        except Exception as e:
            logger.warning(f"Error closing HTTP session: {e}")
        
        try:
            self.db_pool.closeall()
        except Exception as e:
            logger.warning(f"Error closing Postgres pool: {e}")


if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize consumer
    consumer = FraudDetectionConsumer(
        kafka_bootstrap_servers='localhost:9092',
        kafka_topic='transactions',
        api_url='http://localhost:8000',
        api_batch_size=50
    )
    
    # Start consuming
    consumer.consume(batch_size=100)
