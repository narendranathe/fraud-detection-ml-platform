"""
Model training with MLflow tracking
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, auc, classification_report,
    confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
)
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Define feature columns
FEATURE_COLS = [
    # Time features
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours',
    
    # Amount features
    'amount', 'amount_log', 'is_round_amount', 'is_very_round',
    
    # User aggregates
    'user_txn_count_all_time', 'user_avg_amount', 'user_std_amount',
    'user_max_amount', 'user_distinct_merchants',
    'amount_deviation_from_avg', 'is_amount_unusual',
    
    # Velocity features
    'txn_count_1h', 'txn_count_24h', 'txn_count_7d',
    'amount_sum_1h', 'amount_sum_24h', 'amount_sum_7d',
    'high_velocity_1h', 'high_velocity_24h',
    
    # Merchant features
    'merchant_txn_count', 'merchant_fraud_rate', 'merchant_avg_amount',
    
    # Distance
    'distance_from_home'
]


def load_and_prepare_data():
    """Load and prepare data for training"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    df = pd.read_csv('data/processed/transactions_with_features.csv')
    print(f"Loaded {len(df):,} transactions")
    
    # Remove any NaN values
    df = df.fillna(0)
    
    # Check if all features exist
    missing_features = [f for f in FEATURE_COLS if f not in df.columns]
    if missing_features:
        print(f"\n⚠️  Warning: Missing features: {missing_features}")
        FEATURE_COLS_FILTERED = [f for f in FEATURE_COLS if f in df.columns]
    else:
        FEATURE_COLS_FILTERED = FEATURE_COLS
    
    # Separate features and target
    X = df[FEATURE_COLS_FILTERED]
    y = df['is_fraud']
    
    # Time-based split (first 80% for training, last 20% for testing)
    split_idx = int(len(df) * 0.8)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {len(X_train):,} samples ({y_train.sum():,} fraud, {y_train.mean()*100:.2f}%)")
    print(f"   Testing:  {len(X_test):,} samples ({y_test.sum():,} fraud, {y_test.mean()*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test, FEATURE_COLS_FILTERED


def train_model(X_train, X_test, y_train, y_test, feature_cols, params):
    """Train LightGBM model with MLflow tracking"""
    
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Set MLflow tracking (local directory)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("fraud-detection")
    
    with mlflow.start_run(run_name="lgbm_baseline"):
        
        print("\n🚀 Training LightGBM model...")
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Train model
        model = LGBMClassifier(**params, random_state=42, verbose=-1)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision_score_val = precision_score(y_test, y_pred)
        recall_score_val = recall_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision_score_val)
        mlflow.log_metric("recall", recall_score_val)
        
        print(f"\n📈 MODEL PERFORMANCE:")
        print(f"   PR-AUC:    {pr_auc:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Precision: {precision_score_val:.4f}")
        print(f"   Recall:    {recall_score_val:.4f}")
        
        # Classification report
        print(f"\n📊 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Create artifacts directory
        os.makedirs('artifacts', exist_ok=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        confusion_matrix_path = 'artifacts/confusion_matrix.png'
        plt.savefig(confusion_matrix_path, dpi=100)
        mlflow.log_artifact(confusion_matrix_path)
        plt.close()
        print(f"   ✅ Saved: {confusion_matrix_path}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        feature_importance_path = 'artifacts/feature_importance.png'
        plt.savefig(feature_importance_path, dpi=100)
        mlflow.log_artifact(feature_importance_path)
        plt.close()
        print(f"   ✅ Saved: {feature_importance_path}")
        
        print(f"\n🎯 TOP 10 FEATURES:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:30s} {row['importance']:.4f}")
        
        # PR curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, color='blue')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_curve_path = 'artifacts/pr_curve.png'
        plt.savefig(pr_curve_path, dpi=100)
        mlflow.log_artifact(pr_curve_path)
        plt.close()
        print(f"   ✅ Saved: {pr_curve_path}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        import joblib
        model_path = 'artifacts/fraud_model.pkl'
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        print(f"   ✅ Saved: {model_path}")
        
        print(f"\n✅ Model logged to MLflow")
        print(f"📂 MLflow tracking: ./mlruns")
        
        return model, pr_auc, roc_auc, f1


def main():
    print("\n" + "="*60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data()
    
    # Model parameters
    params = {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'class_weight': 'balanced'
    }
    
    print(f"\n⚙️  Model Parameters:")
    for k, v in params.items():
        print(f"   {k}: {v}")
    
    # Train model
    model, pr_auc, roc_auc, f1 = train_model(X_train, X_test, y_train, y_test, feature_cols, params)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📊 FINAL METRICS:")
    print(f"   PR-AUC:   {pr_auc:.4f}")
    print(f"   ROC-AUC:  {roc_auc:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"\n📁 Artifacts saved to: ./artifacts/")
    print(f"📂 MLflow runs saved to: ./mlruns/")
    print(f"\n💡 To view MLflow UI, run:")
    print(f"   mlflow ui")
    print(f"   Then open: http://localhost:5000")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()