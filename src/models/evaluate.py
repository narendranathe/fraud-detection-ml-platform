"""
Model evaluation and analysis
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_model_and_data():
    """Load trained model and test data"""
    print("="*60)
    print("LOADING MODEL AND DATA")
    print("="*60)
    
    # Load model
    print("\n📦 Loading trained model...")
    model = joblib.load('artifacts/fraud_model.pkl')
    print("   ✅ Model loaded")
    
    # Load data
    print("\n📊 Loading test data...")
    df = pd.read_csv('data/processed/transactions_with_features.csv')
    print(f"   ✅ Loaded {len(df):,} transactions")
    
    # Same features as training
    FEATURE_COLS = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours',
        'amount', 'amount_log', 'is_round_amount', 'is_very_round',
        'user_txn_count_all_time', 'user_avg_amount', 'user_std_amount',
        'user_max_amount', 'user_distinct_merchants',
        'amount_deviation_from_avg', 'is_amount_unusual',
        'txn_count_1h', 'txn_count_24h', 'txn_count_7d',
        'amount_sum_1h', 'amount_sum_24h', 'amount_sum_7d',
        'high_velocity_1h', 'high_velocity_24h',
        'merchant_txn_count', 'merchant_fraud_rate', 'merchant_avg_amount',
        'distance_from_home'
    ]
    
    # Use last 20% as test set (same as training)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\n📐 Test set size: {len(test_df):,} transactions")
    
    X_test = test_df[FEATURE_COLS].fillna(0)
    y_test = test_df['is_fraud']
    
    return model, X_test, y_test, test_df


def analyze_predictions(model, X_test, y_test, test_df):
    """Analyze model predictions"""
    print("\n" + "="*60)
    print("MODEL EVALUATION & ANALYSIS")
    print("="*60)
    
    # Get predictions
    print("\n🔮 Generating predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("   ✅ Predictions generated")
    
    # Add predictions to dataframe
    test_df['fraud_score'] = y_pred_proba
    test_df['predicted_fraud'] = y_pred
    
    # Confusion matrix breakdown
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 CONFUSION MATRIX BREAKDOWN:")
    print(f"   True Negatives:  {tn:,} (Correctly identified normal)")
    print(f"   False Positives: {fp:,} (Normal flagged as fraud)")
    print(f"   False Negatives: {fn:,} (Fraud missed)")
    print(f"   True Positives:  {tp:,} (Correctly caught fraud)")
    
    # Calculate rates
    print(f"\n📈 DETECTION RATES:")
    fpr = fp/(fp+tn)*100 if (fp+tn) > 0 else 0
    fnr = fn/(fn+tp)*100 if (fn+tp) > 0 else 0
    tpr = tp/(tp+fn)*100 if (tp+fn) > 0 else 0
    
    print(f"   False Positive Rate: {fpr:.2f}%")
    print(f"   False Negative Rate: {fnr:.2f}%")
    print(f"   True Positive Rate:  {tpr:.2f}% (Recall)")
    
    # Business impact
    fraud_amount = test_df[test_df['is_fraud']==1]['amount'].sum()
    caught_amount = test_df[(test_df['is_fraud']==1) & (test_df['predicted_fraud']==1)]['amount'].sum()
    missed_amount = test_df[(test_df['is_fraud']==1) & (test_df['predicted_fraud']==0)]['amount'].sum()
    
    print(f"\n💰 BUSINESS IMPACT:")
    print(f"   Total Fraud Amount:   ${fraud_amount:,.2f}")
    print(f"   Caught Fraud Amount:  ${caught_amount:,.2f} ({caught_amount/fraud_amount*100:.1f}%)")
    print(f"   Missed Fraud Amount:  ${missed_amount:,.2f} ({missed_amount/fraud_amount*100:.1f}%)")
    
    # Score distribution
    print(f"\n🎯 FRAUD SCORE DISTRIBUTION:")
    normal_mean_score = test_df[test_df['is_fraud']==0]['fraud_score'].mean()
    fraud_mean_score = test_df[test_df['is_fraud']==1]['fraud_score'].mean()
    print(f"   Normal transactions (mean score): {normal_mean_score:.4f}")
    print(f"   Fraud transactions (mean score):  {fraud_mean_score:.4f}")
    print(f"   Separation: {fraud_mean_score - normal_mean_score:.4f}")
    
    # High-risk transactions
    high_risk = test_df[test_df['fraud_score'] > 0.8]
    print(f"\n⚠️  HIGH RISK TRANSACTIONS (score > 0.8):")
    print(f"   Count: {len(high_risk):,}")
    if len(high_risk) > 0:
        print(f"   Actual fraud: {high_risk['is_fraud'].sum():,} ({high_risk['is_fraud'].mean()*100:.1f}%)")
    
    # Sample of caught fraud
    print(f"\n🎯 SAMPLE CAUGHT FRAUD (Top 5):")
    caught_fraud = test_df[(test_df['is_fraud']==1) & (test_df['predicted_fraud']==1)].nlargest(5, 'fraud_score')
    if len(caught_fraud) > 0:
        for idx, row in caught_fraud.iterrows():
            print(f"   TXN: {row['transaction_id']} | Amount: ${row['amount']:.2f} | Score: {row['fraud_score']:.4f}")
    else:
        print("   None found")
    
    # Sample of missed fraud
    print(f"\n❌ SAMPLE MISSED FRAUD (Lowest scores):")
    missed_fraud = test_df[(test_df['is_fraud']==1) & (test_df['predicted_fraud']==0)].nsmallest(5, 'fraud_score')
    if len(missed_fraud) > 0:
        for idx, row in missed_fraud.iterrows():
            print(f"   TXN: {row['transaction_id']} | Amount: ${row['amount']:.2f} | Score: {row['fraud_score']:.4f}")
    else:
        print("   None - perfect detection!")
    
    # Score distribution by decile
    print(f"\n📊 FRAUD RATE BY SCORE DECILE:")
    test_df['score_decile'] = pd.qcut(test_df['fraud_score'], q=10, labels=False, duplicates='drop')
    for decile in sorted(test_df['score_decile'].unique()):
        decile_df = test_df[test_df['score_decile'] == decile]
        fraud_rate = decile_df['is_fraud'].mean() * 100
        score_range = f"{decile_df['fraud_score'].min():.3f} - {decile_df['fraud_score'].max():.3f}"
        print(f"   Decile {int(decile)+1}: Score {score_range} | Fraud Rate: {fraud_rate:.2f}%")
    
    # Save analysis
    os.makedirs('artifacts', exist_ok=True)
    output_path = 'artifacts/predictions_analysis.csv'
    test_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved detailed predictions to: {output_path}")
    
    # Summary statistics
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], digits=4))
    
    print("="*60 + "\n")


def main():
    try:
        # Load model and data
        model, X_test, y_test, test_df = load_model_and_data()
        
        # Analyze
        analyze_predictions(model, X_test, y_test, test_df)
        
        print("✅ Evaluation complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Make sure you've run the training script first:")
        print("   python src\\models\\train.py")
        print("\nThis will create:")
        print("   - artifacts/fraud_model.pkl")
        print("   - data/processed/transactions_with_features.csv")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()