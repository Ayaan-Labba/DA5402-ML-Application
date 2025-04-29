import os
import sys
import argparse
import mlflow

# Set path to root
sys.path.append(os.getcwd())

from utils.logger import setup_logger

# Setup logging
logger = setup_logger("Hyperparameter Tuning")

def get_latest_run_id(experiment_name, mlflow_tracking_uri):
    """
    Get the latest successful run ID from MLflow
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experiment '{experiment_name}' not found")
        return None
    
    # Get runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="")
    
    if runs.empty:
        logger.error(f"No runs found for experiment '{experiment_name}'")
        return None
    
    # Sort by start time (descending) and get the latest run
    runs = runs.sort_values("start_time", ascending=False)
    latest_run_id = runs.iloc[0]["run_id"]
    
    return latest_run_id

def main():
    parser = argparse.ArgumentParser(description='Save latest MLflow run ID')
    parser.add_argument('--experiment_name', type=str, default='forex_prediction',
                        help='MLflow experiment name')
    parser.add_argument('--mlflow_tracking_uri', type=str, default='./mlruns', 
                        help='MLflow tracking URI')
    parser.add_argument('--output_file', type=str, default='models/latest_run_id.txt',
                        help='Output file to save the run ID')
    
    args = parser.parse_args()
    
    # Get latest run ID
    run_id = get_latest_run_id(args.experiment_name, args.mlflow_tracking_uri)
    
    if run_id:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        # Save run ID to file
        with open(args.output_file, 'w') as f:
            f.write(run_id)
        
        logger.info(f"Latest run ID '{run_id}' saved to {args.output_file}")
    else:
        logger.error("Failed to get latest run ID")

if __name__ == "__main__":
    main()