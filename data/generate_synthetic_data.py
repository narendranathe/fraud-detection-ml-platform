"""
Generate realistic synthetic transaction data with fraud patterns
"""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)

class TransactionGenerator:
    def __init__(self, n_users=10000, n_merchants=5000):
        self.n_users = n_users
        self.n_merchants = n_merchants
        
        print(f"Initializing transaction generator...")
        print(f"  Users: {n_users:,}")
        print(f"  Merchants: {n_merchants:,}")
        
        # Generate user profiles
        self.users = self._generate_users()
        self.merchants = self._generate_merchants()
        
    def _generate_users(self):
        """Generate user profiles with home locations and spending patterns"""
        print("  Generating user profiles...")
        users = []
        for i in range(self.n_users):
            user = {
                'user_id': f'USER_{i:06d}',
                'home_lat': float(fake.latitude()),
                'home_lon': float(fake.longitude()),
                'avg_transaction_amount': float(np.random.lognormal(3, 1)),
                'transaction_frequency': int(np.random.poisson(5)),
                'preferred_merchants': list(np.random.choice(self.n_merchants, size=5, replace=False)),
                'is_fraudster': bool(np.random.random() < 0.02)
            }
            users.append(user)
        return pd.DataFrame(users)
    
    def _generate_merchants(self):
        """Generate merchant profiles"""
        print("  Generating merchant profiles...")
        merchants = []
        categories = ['grocery', 'restaurant', 'gas_station', 'retail', 
                     'electronics', 'pharmacy', 'entertainment', 'travel']
        
        for i in range(self.n_merchants):
            merchant = {
                'merchant_id': f'MERCH_{i:06d}',
                'merchant_name': fake.company(),
                'merchant_category': np.random.choice(categories),
                'merchant_lat': float(fake.latitude()),
                'merchant_lon': float(fake.longitude()),
                'fraud_risk_score': float(np.random.random())
            }
            merchants.append(merchant)
        return pd.DataFrame(merchants)
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in km"""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def generate_transaction(self, timestamp, user_idx=None):
        """Generate a single transaction"""
        
        # Select user
        if user_idx is None:
            user_idx = np.random.randint(0, len(self.users))
        user = self.users.iloc[user_idx]
        
        # Determine if this will be fraud
        is_fraud = False
        
        if user['is_fraudster'] and np.random.random() < 0.3:
            # Fraudster executing fraud
            is_fraud = True
            amount = float(np.random.uniform(500, 5000))
            merchant_idx = np.random.randint(0, len(self.merchants))
            merchant = self.merchants.iloc[merchant_idx]
        else:
            # Normal transaction
            amount = float(np.random.lognormal(np.log(user['avg_transaction_amount']), 0.5))
            amount = max(1.0, min(amount, 1000.0))
            
            # Use preferred merchant 80% of time
            if np.random.random() < 0.8 and len(user['preferred_merchants']) > 0:
                merchant_idx = int(user['preferred_merchants'][np.random.randint(0, len(user['preferred_merchants']))])
            else:
                merchant_idx = np.random.randint(0, len(self.merchants))
            
            merchant = self.merchants.iloc[merchant_idx]
            
            # Small chance of normal transaction being fraud
            if np.random.random() < 0.001:
                is_fraud = True
        
        # Calculate distance
        distance = self._haversine_distance(
            user['home_lat'], user['home_lon'],
            merchant['merchant_lat'], merchant['merchant_lon']
        )
        
        transaction = {
            'transaction_id': f'TXN_{timestamp.strftime("%Y%m%d%H%M%S")}_{user_idx}_{np.random.randint(1000)}',
            'timestamp': timestamp.isoformat(),
            'user_id': user['user_id'],
            'merchant_id': merchant['merchant_id'],
            'merchant_name': merchant['merchant_name'],
            'merchant_category': merchant['merchant_category'],
            'amount': round(float(amount), 2),
            'merchant_lat': float(merchant['merchant_lat']),
            'merchant_lon': float(merchant['merchant_lon']),
            'home_lat': float(user['home_lat']),
            'home_lon': float(user['home_lon']),
            'distance_from_home': round(float(distance), 2),
            'is_fraud': int(is_fraud)
        }
        
        return transaction
    
    def generate_batch(self, n_transactions=100000, start_date=None):
        """Generate a batch of historical transactions"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        
        print(f"\n  Generating {n_transactions:,} transactions...")
        transactions = []
        
        for i in range(n_transactions):
            # Random timestamp over 90 days
            random_seconds = np.random.randint(0, 90 * 24 * 3600)
            timestamp = start_date + timedelta(seconds=random_seconds)
            
            txn = self.generate_transaction(timestamp)
            transactions.append(txn)
            
            if (i + 1) % 10000 == 0:
                print(f"    Progress: {i + 1:,}/{n_transactions:,} transactions...")
        
        df = pd.DataFrame(transactions)
        
        # Add day of week and hour features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour_of_day'] = df['timestamp'].dt.hour
        
        return df


def main():
    print("="*60)
    print("FRAUD DETECTION - SYNTHETIC DATA GENERATOR")
    print("="*60)
    
    # Create generator
    generator = TransactionGenerator(n_users=10000, n_merchants=5000)
    
    # Generate 100k historical transactions
    df = generator.generate_batch(n_transactions=100000)
    
    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/raw/transactions.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📁 Output: {output_path}")
    print(f"📊 Total Transactions: {len(df):,}")
    print(f"\n💰 Fraud Statistics:")
    print(f"   Fraud Count: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"   Fraud Amount: ${df[df['is_fraud']==1]['amount'].sum():,.2f}")
    print(f"   Normal Amount: ${df[df['is_fraud']==0]['amount'].sum():,.2f}")
    print(f"\n📈 Amount Statistics:")
    print(df['amount'].describe())
    print(f"\n🔍 Sample Transactions:")
    print(df.head(10))
    
    # Save user and merchant profiles
    generator.users.to_csv('data/raw/users.csv', index=False)
    generator.merchants.to_csv('data/raw/merchants.csv', index=False)
    
    print(f"\n✅ Saved user profiles: data/raw/users.csv")
    print(f"✅ Saved merchant profiles: data/raw/merchants.csv")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()