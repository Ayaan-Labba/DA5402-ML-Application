import argparse
import os
import sys
import json
import mlflow
import torch
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Save MLflow model for deployment")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID of the model to save")
    parser.add_argument("--mlflow_tracking_uri", type=str, required=True, help="MLflow tracking URI")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model artifacts")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the model artifacts from MLflow
        print(f"Downloading model artifacts for run ID: {args.run_id}")
        model_path = os.path.join(output_dir, "model")
        
        # Download the PyTorch model
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(args.run_id)
        artifact_uri = run.info.artifact_uri
        
        # Download the model file
        mlflow.artifacts.download_artifacts(
            artifact_uri=os.path.join(artifact_uri, "model"),
            dst_path=output_dir
        )
        
        # Save model metadata
        model_metadata = {
            "run_id": args.run_id,
            "model_path": "model/data/model.pth",
            "created_at": run.info.start_time,
            "metrics": run.data.metrics
        }
        
        with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
            json.dump(model_metadata, f, indent=4)
        
        print(f"Model successfully saved to {output_dir}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()