"""
Batch feature engineering for model training
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import os

class FeatureEngineering:
    
    def __init__(self, transactions_df):
        self.df = transactions_df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Loaded {len(self.df):,} transactions")
    
    def create_time_features(self):
        """Extract time-based features"""
        print("Creating time features...")
        
        self.df['hour_of_day'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['day_of_month'] = self.df['timestamp'].dt.day
        self.df['month'] = self.df['timestamp'].dt.month
        
        # Is weekend
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # Is night (10pm - 6am)
        self.df['is_night'] = ((self.df['hour_of_day'] >= 22) | 
                                (self.df['hour_of_day'] <= 6)).astype(int)
        
        # Is business hours (9am - 5pm weekday)
        self.df['is_business_hours'] = (
            (self.df['hour_of_day'] >= 9) & 
            (self.df['hour_of_day'] <= 17) & 
            (self.df['day_of_week'] < 5)
        ).astype(int)
        
        return self
    
    def create_amount_features(self):
        """Create amount-based features"""
        print("Creating amount features...")
        
        # Log transform
        self.df['amount_log'] = np.log1p(self.df['amount'])
        
        # Round amount (fraud often uses round numbers)
        self.df['is_round_amount'] = (self.df['amount'] % 10 == 0).astype(int)
        self.df['is_very_round'] = (self.df['amount'] % 100 == 0).astype(int)
        
        return self
    
    def create_user_aggregates(self):
        """Create user-level aggregate features"""
        print("Creating user aggregate features...")
        
        user_features_list = []
        
        for user_id in self.df['user_id'].unique():
            user_df = self.df[self.df['user_id'] == user_id].copy()
            
            for idx in range(len(user_df)):
                row = user_df.iloc[idx]
                
                # Get history before this transaction
                history = user_df.iloc[:idx]
                
                if len(history) == 0:
                    # First transaction for this user
                    features = {
                        'user_txn_count_all_time': 0,
                        'user_avg_amount': 0,
                        'user_std_amount': 0,
                        'user_total_amount': 0,
                        'user_max_amount': 0,
                        'user_distinct_merchants': 0,
                        'amount_deviation_from_avg': 0,
                        'is_amount_unusual': 0
                    }
                else:
                    avg_amt = history['amount'].mean()
                    std_amt = history['amount'].std() if len(history) > 1 else 0
                    
                    features = {
                        'user_txn_count_all_time': len(history),
                        'user_avg_amount': avg_amt,
                        'user_std_amount': std_amt,
                        'user_total_amount': history['amount'].sum(),
                        'user_max_amount': history['amount'].max(),
                        'user_distinct_merchants': history['merchant_id'].nunique(),
                        'amount_deviation_from_avg': (row['amount'] - avg_amt) / avg_amt if avg_amt > 0 else 0,
                        'is_amount_unusual': int(row['amount'] > avg_amt + 2 * std_amt) if std_amt > 0 else 0
                    }
                
                user_features_list.append(features)
        
        # Merge back
        user_features_df = pd.DataFrame(user_features_list)
        self.df = pd.concat([self.df.reset_index(drop=True), user_features_df.reset_index(drop=True)], axis=1)
        
        return self
    
    def create_velocity_features(self):
        """Create velocity features (transactions per time window)"""
        print("Creating velocity features...")
        
        velocity_features_list = []
        
        for user_id in self.df['user_id'].unique():
            user_df = self.df[self.df['user_id'] == user_id].copy()
            
            for idx in range(len(user_df)):
                row = user_df.iloc[idx]
                current_time = row['timestamp']
                history = user_df.iloc[:idx]
                
                if len(history) == 0:
                    features = {
                        'txn_count_1h': 0,
                        'txn_count_24h': 0,
                        'txn_count_7d': 0,
                        'amount_sum_1h': 0,
                        'amount_sum_24h': 0,
                        'amount_sum_7d': 0,
                        'high_velocity_1h': 0,
                        'high_velocity_24h': 0
                    }
                else:
                    # Count transactions in windows
                    last_1h = history[history['timestamp'] >= current_time - timedelta(hours=1)]
                    last_24h = history[history['timestamp'] >= current_time - timedelta(hours=24)]
                    last_7d = history[history['timestamp'] >= current_time - timedelta(days=7)]
                    
                    features = {
                        'txn_count_1h': len(last_1h),
                        'txn_count_24h': len(last_24h),
                        'txn_count_7d': len(last_7d),
                        'amount_sum_1h': last_1h['amount'].sum() if len(last_1h) > 0 else 0,
                        'amount_sum_24h': last_24h['amount'].sum() if len(last_24h) > 0 else 0,
                        'amount_sum_7d': last_7d['amount'].sum() if len(last_7d) > 0 else 0,
                        'high_velocity_1h': int(len(last_1h) >= 5),
                        'high_velocity_24h': int(len(last_24h) >= 20)
                    }
                
                velocity_features_list.append(features)
        
        velocity_df = pd.DataFrame(velocity_features_list)
        self.df = pd.concat([self.df.reset_index(drop=True), velocity_df.reset_index(drop=True)], axis=1)
        
        return self
    
    def create_merchant_features(self):
        """Create merchant-level features"""
        print("Creating merchant features...")
        
        merchant_features_list = []
        
        for merchant_id in self.df['merchant_id'].unique():
            merchant_df = self.df[self.df['merchant_id'] == merchant_id].copy()
            
            for idx in range(len(merchant_df)):
                row = merchant_df.iloc[idx]
                history = merchant_df.iloc[:idx]
                
                if len(history) == 0:
                    features = {
                        'merchant_txn_count': 0,
                        'merchant_fraud_rate': 0,
                        'merchant_avg_amount': 0
                    }
                else:
                    features = {
                        'merchant_txn_count': len(history),
                        'merchant_fraud_rate': history['is_fraud'].mean() if 'is_fraud' in history.columns else 0,
                        'merchant_avg_amount': history['amount'].mean()
                    }
                
                merchant_features_list.append(features)
        
        merchant_features_df = pd.DataFrame(merchant_features_list)
        self.df = pd.concat([self.df.reset_index(drop=True), merchant_features_df.reset_index(drop=True)], axis=1)
        
        return self
    
    def create_all_features(self):
        """Create all features"""
        print("\n🔧 Starting feature engineering...")
        print("="*60)
        
        self.create_time_features()
        self.create_amount_features()
        self.create_user_aggregates()
        self.create_velocity_features()
        self.create_merchant_features()
        
        print("="*60)
        print("✅ Feature engineering complete!")
        print(f"Total features: {len(self.df.columns)}")
        
        return self.df


def main():
    print("Loading transaction data...")
    
    # Load data
    df = pd.read_csv('data/raw/transactions.csv')
    
    print(f"Loaded {len(df):,} transactions")
    
    # Create features
    fe = FeatureEngineering(df)
    df_features = fe.create_all_features()
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Save
    output_path = 'data/processed/transactions_with_features.csv'
    df_features.to_csv(output_path, index=False)
    
    print(f"\n📁 Saved to: {output_path}")
    print(f"\n📊 Feature Columns ({len(df_features.columns)}):")
    print(df_features.columns.tolist())
    
    print(f"\n📈 Sample Data:")
    print(df_features.head())


if __name__ == "__main__":
    main()