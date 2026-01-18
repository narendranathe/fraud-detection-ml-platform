"""
Kafka Producer for Real-Time Transaction Streaming
Simulates live transaction feed from payment processor
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger
import pandas as pd

logger.add("logs/kafka_producer.log", rotation="10 MB")


class TransactionProducer:
    """Stream transactions to Kafka topic"""
    
    def __init__(
        self, 
        bootstrap_servers: str = 'localhost:9092',
        topic: str = 'transactions',
        transactions_per_second: int = 100
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.tps = transactions_per_second
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1,
            compression_type='gzip'
        )
        
        logger.info(f"✅ Kafka Producer initialized: {bootstrap_servers}")
        logger.info(f"   Topic: {topic} | Target TPS: {transactions_per_second}")
    
    def load_streaming_data(self, filepath: str = 'data/raw/streaming_transactions.jsonl') -> pd.DataFrame:
        """Load pre-generated streaming data"""
        transactions = []
        with open(filepath, 'r') as f:
            for line in f:
                transactions.append(json.loads(line))
        
        df = pd.DataFrame(transactions)
        logger.info(f"📂 Loaded {len(df):,} transactions from {filepath}")
        return df
    
    def format_transaction(self, row: pd.Series) -> Dict[str, Any]:
        """Format transaction for Kafka message"""
        return {
            'transaction_id': row['transaction_id'],
            'timestamp': row['timestamp'] if isinstance(row['timestamp'], str) else row['timestamp'].isoformat(),
            'customer_id': row['customer_id'],
            'merchant_id': row['merchant_id'],
            'merchant_category': row['merchant_category'],
            'amount': float(row['amount']),
            'device_type': row['device_type'],
            'distance_from_home': float(row['distance_from_home']),
            'merchant_risk_score': float(row['merchant_risk_score']),
            'customer_age': int(row['customer_age']),
            'account_age_days': int(row['account_age_days']),
            'hour': int(row['hour']),
            'day_of_week': int(row['day_of_week']),
            'is_weekend': int(row['is_weekend']),
            'is_fraud': int(row['is_fraud'])
        }
    
    def send_transaction(self, transaction: Dict[str, Any]) -> None:
        """Send single transaction to Kafka"""
        try:
            future = self.producer.send(
                self.topic,
                key=transaction['customer_id'],
                value=transaction
            )
            
            record_metadata = future.get(timeout=10)
            
            logger.debug(
                f"✓ Sent {transaction['transaction_id']} to "
                f"partition {record_metadata.partition} offset {record_metadata.offset}"
            )
            
        except KafkaError as e:
            logger.error(f"Failed to send transaction: {e}")
            raise
    
    def stream_transactions(
        self, 
        df: pd.DataFrame, 
        loop: bool = False,
        delay_between_batches: int = 10
    ) -> None:
        """Stream transactions at specified TPS"""
        delay = 1.0 / self.tps
        
        logger.info(f"🚀 Starting transaction stream (TPS: {self.tps})")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                logger.info(f"📡 Streaming iteration {iteration} ({len(df):,} transactions)")
                
                for idx, row in df.iterrows():
                    transaction = self.format_transaction(row)
                    self.send_transaction(transaction)
                    time.sleep(delay)
                    
                    if (idx + 1) % 1000 == 0:
                        logger.info(f"   Sent {idx + 1:,}/{len(df):,} transactions...")
                
                logger.info(f"✅ Completed iteration {iteration}")
                
                if not loop:
                    break
                
                logger.info(f"⏳ Waiting {delay_between_batches}s before next iteration...")
                time.sleep(delay_between_batches)
                
        except KeyboardInterrupt:
            logger.warning("⚠️ Stream interrupted by user")
        finally:
            self.close()
    
    def close(self):
        """Flush and close producer"""
        logger.info("Flushing remaining messages...")
        self.producer.flush()
        self.producer.close()
        logger.info("✅ Kafka Producer closed")


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    
    producer = TransactionProducer(
        bootstrap_servers='localhost:9092',
        topic='transactions',
        transactions_per_second=100
    )
    
    df = producer.load_streaming_data('data/raw/streaming_transactions.jsonl')
    producer.stream_transactions(df, loop=True, delay_between_batches=30)