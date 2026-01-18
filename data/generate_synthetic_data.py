"""
Synthetic Fraud Transaction Data Generator
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

np.random.seed(42)

def generate_transactions(n=100000):
    """Generate synthetic transaction data"""
    
    print(f"Generating {n:,} transactions...")
    
    # Categories
    categories = ['grocery', 'gas_station', 'restaurant', 'retail', 
                  'online_shopping', 'entertainment', 'travel', 'utilities']
    devices = ['mobile', 'web', 'pos_terminal', 'atm']
    
    # Generate data
    data = {
        'transaction_id': [f'TXN_{i:010d}' for i in range(n)],
        'timestamp': [datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365), 
                      hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60)) 
                      for _ in range(n)],
        'customer_id': [f'CUST_{np.random.randint(100000, 999999)}' for _ in range(n)],
        'merchant_id': [f'MERCH_{np.random.randint(10000, 99999)}' for _ in range(n)],
        'merchant_category': np.random.choice(categories, n),
        'amount': np.random.lognormal(4, 1.2, n),
        'device_type': np.random.choice(devices, n),
        'distance_from_home': np.random.exponential(50, n),
        'merchant_risk_score': np.random.beta(2, 5, n),
        'customer_age': np.random.randint(18, 80, n),
        'account_age_days': np.random.randint(30, 3650, n),
    }
    
    df = pd.DataFrame(data)
    
    # Add time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Add fraud label (2% fraud rate)
    df['is_fraud'] = (np.random.random(n) < 0.02).astype(int)
    
    # Make fraud cases more extreme
    fraud_mask = df['is_fraud'] == 1
    df.loc[fraud_mask, 'amount'] *= np.random.uniform(2, 5, fraud_mask.sum())
    df.loc[fraud_mask, 'distance_from_home'] *= np.random.uniform(3, 10, fraud_mask.sum())
    
    print(f"✅ Generated {len(df):,} transactions")
    print(f"   Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    
    return df


def save_datasets(df, output_dir='data'):
    """Save train/test/streaming datasets"""
    
    Path(output_dir).mkdir(exist_ok=True)
    Path(f"{output_dir}/raw").mkdir(exist_ok=True)
    
    # Split data
    train_size = int(0.70 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    # Save CSV files
    train_df.to_csv(f'{output_dir}/raw/train_data.csv', index=False)
    val_df.to_csv(f'{output_dir}/raw/validation_data.csv', index=False)
    test_df.to_csv(f'{output_dir}/raw/test_data.csv', index=False)
    
    # Save streaming data (JSONL format for Kafka)
    streaming_df = test_df.sample(frac=0.3, random_state=42)
    
    with open(f'{output_dir}/raw/streaming_transactions.jsonl', 'w') as f:
        for _, row in streaming_df.iterrows():
            json.dump(row.to_dict(), f, default=str)
            f.write('\n')
    
    print(f"\n💾 Saved datasets:")
    print(f"   Train: {len(train_df):,} rows → {output_dir}/raw/train_data.csv")
    print(f"   Validation: {len(val_df):,} rows → {output_dir}/raw/validation_data.csv")
    print(f"   Test: {len(test_df):,} rows → {output_dir}/raw/test_data.csv")
    print(f"   Streaming: {len(streaming_df):,} rows → {output_dir}/raw/streaming_transactions.jsonl")


if __name__ == "__main__":
    # Generate data
    df = generate_transactions(n=100000)
    
    # Save datasets
    save_datasets(df)
    
    print("\n✅ Data generation complete!")