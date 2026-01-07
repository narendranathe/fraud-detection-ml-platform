"""
View MLflow experiment results without UI
"""
import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Get client
client = MlflowClient()

# Get experiment
try:
    experiment = client.get_experiment_by_name("fraud-detection")
    
    if experiment:
        print("="*60)
        print("MLFLOW EXPERIMENT RESULTS")
        print("="*60)
        print(f"\nExperiment: {experiment.name}")
        print(f"Experiment ID: {experiment.experiment_id}")
        
        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        print(f"\nTotal Runs: {len(runs)}")
        print("\n" + "="*60)
        
        for i, run in enumerate(runs, 1):
            print(f"\nRUN #{i}")
            print(f"Run ID: {run.info.run_id}")
            print(f"Status: {run.info.status}")
            
            print("\nParameters:")
            for key, value in run.data.params.items():
                print(f"  {key}: {value}")
            
            print("\nMetrics:")
            for key, value in run.data.metrics.items():
                print(f"  {key}: {value:.4f}")
            
            print("\nArtifacts:")
            artifacts = client.list_artifacts(run.info.run_id)
            for artifact in artifacts:
                print(f"  {artifact.path}")
            
            print("-"*60)
    else:
        print("❌ No experiment found named 'fraud-detection'")
        print("\nAvailable experiments:")
        all_experiments = client.search_experiments()
        for exp in all_experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
            
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure you've run the training script first:")
    print("  python src\\models\\train.py")