import os
import sys
import argparse
import json
from datetime import datetime

# Set path to root
sys.path.append(os.getcwd())

from utils.logger import setup_logger

# Setup logging
logger = setup_logger("Hyperparameter Tuning")

def check_drift_status(evaluation_results_dir, threshold, trigger_rebuild):
    """
    Check drift status from evaluation results and decide if model rebuild is needed
    """
    # Load evaluation results
    results_file = os.path.join(evaluation_results_dir, "evaluation_results.json")
    
    if not os.path.exists(results_file):
        logger.error(f"Evaluation results file not found: {results_file}")
        return False, {}
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Check drift status
    drift_detected = results.get("drift_detected", False)
    drift_results = results.get("drift_results", {})
    
    # Calculate drift severity (percentage of features with drift)
    total_features = len(drift_results)
    if total_features == 0:
        drift_severity = 0
    else:
        drifted_features = sum(1 for feature in drift_results.values() if feature.get("drift_detected", False))
        drift_severity = drifted_features / total_features
    
    # Determine if rebuild is needed
    rebuild_needed = drift_detected and drift_severity >= threshold and trigger_rebuild
    
    # Create drift status report
    status = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "drift_detected": drift_detected,
        "drift_severity": drift_severity,
        "rebuild_needed": rebuild_needed,
        "threshold": threshold,
        "model_run_id": results.get("model_run_id", "unknown"),
        "metrics": results.get("metrics", {}),
        "drifted_features_count": sum(1 for feature in drift_results.values() if feature.get("drift_detected", False)),
        "total_features": total_features
    }
    
    return rebuild_needed, status

def main():
    parser = argparse.ArgumentParser(description='Check drift status and trigger model rebuild if needed')
    parser.add_argument('--evaluation_results', type=str, required=True, 
                        help='Directory containing evaluation results')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Drift severity threshold to trigger rebuild (0-1)')
    parser.add_argument('--trigger_rebuild', type=int, default=1,
                        help='Flag to enable/disable model rebuild on drift (0/1)')
    
    args = parser.parse_args()
    
    # Convert trigger_rebuild to boolean
    trigger_rebuild = bool(args.trigger_rebuild)
    
    # Check drift status
    rebuild_needed, status = check_drift_status(
        args.evaluation_results, 
        args.threshold,
        trigger_rebuild
    )
    
    # Save drift status
    os.makedirs("model_monitoring", exist_ok=True)
    with open("model_monitoring/drift_status.json", 'w') as f:
        json.dump(status, f, indent=4)
    
    # Log drift status
    if status["drift_detected"]:
        logger.warning(f"Data drift detected! Severity: {status['drift_severity']:.2f}")
        if rebuild_needed:
            logger.warning("Model rebuild will be triggered!")
        else:
            logger.info("Model rebuild not triggered - below threshold or disabled.")
    else:
        logger.info("No significant data drift detected.")
    
    # Create trigger file for DVC if rebuild is needed
    if rebuild_needed:
        with open("model_monitoring/rebuild_trigger.txt", 'w') as f:
            f.write("1")
    
    return rebuild_needed

if __name__ == "__main__":
    main()