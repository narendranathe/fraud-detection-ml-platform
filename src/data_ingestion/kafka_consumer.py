"""
Kafka Consumer for Real-Time Fraud Detection
Consumes transactions from Kafka, calls API, saves predictions to Postgres
"""

import json
import time
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import requests
import psycopg2
from psycopg2.extras import execute_values
from loguru import logger
from pathlib import Path
from typing import Dict, Any, List

logger.add("logs/kafka_consumer.log", rotation="10 MB")


class FraudDetectionConsumer:
    """Consume transactions and detect fraud in real-time"""
    
    def __init__(
        self,
        kafka_bootstrap_servers: str = 'localhost:9092',
        kafka_topic: str = 'transactions',
        api_url: str = 'http://localhost:8000',
        postgres_config: Dict[str, str] = None
    ):
        self.kafka_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.api_url = api_url
        
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
    
    def _test_postgres_connection(self):
        """Test Postgres connection"""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            conn.close()
            logger.info("✅ Postgres connection successful")
        except Exception as e:
            logger.error(f"❌ Postgres connection failed: {e}")
            raise
    
    def call_prediction_api(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Call FastAPI to get fraud prediction"""
        try:
            # Remove is_fraud field (ground truth, not used for prediction)
            txn_copy = transaction.copy()
            txn_copy.pop('is_fraud', None)
            
            response = requests.post(
                f"{self.api_url}/predict",
                json=txn_copy,
                timeout=5
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed for {transaction['transaction_id']}: {e}")
            return None
    
    def save_prediction_to_postgres(self, transaction: Dict[str, Any], prediction: Dict[str, Any]):
        """Save prediction to Postgres"""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cursor = conn.cursor()
            
            insert_query = """
                INSERT INTO predictions (
                    transaction_id, customer_id, timestamp, amount,
                    fraud_probability, prediction, model_version, latency_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
            """
            
            cursor.execute(insert_query, (
                prediction['transaction_id'],
                transaction['customer_id'],
                transaction['timestamp'],
                transaction['amount'],
                prediction['fraud_probability'],
                prediction['prediction'],
                prediction['model_version'],
                prediction['latency_ms']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save prediction to Postgres: {e}")
    
    def process_batch(self, messages: List[Dict[str, Any]]):
        """Process a batch of transactions"""
        start_time = time.time()
        
        predictions_saved = 0
        
        for message in messages:
            # Call API
            prediction = self.call_prediction_api(message)
            
            if prediction:
                # Save to Postgres
                self.save_prediction_to_postgres(message, prediction)
                predictions_saved += 1
                
                # Log fraud detections
                if prediction['prediction'] == 1:
                    logger.warning(
                        f"🚨 FRAUD DETECTED: {message['transaction_id']} | "
                        f"Amount: ${message['amount']:.2f} | "
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
                
                # Process in batches
                if len(batch) >= batch_size:
                    self.process_batch(batch)
                    batch = []
                    
        except KeyboardInterrupt:
            logger.warning("⚠️ Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.consumer.close()
            logger.info("✅ Kafka Consumer closed")


if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize consumer
    consumer = FraudDetectionConsumer(
        kafka_bootstrap_servers='localhost:9092',
        kafka_topic='transactions',
        api_url='http://localhost:8000'
    )
    
    # Start consuming
    consumer.consume(batch_size=50)