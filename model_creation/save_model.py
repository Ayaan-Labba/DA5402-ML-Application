import argparse
import os
import sys
import json
import mlflow
import torch
from pathlib import Path

# Set path to root
sys.path.append(os.getcwd())

from utils.logger import setup_logger

# Setup logging
logger = setup_logger("Save model weights")

def parse_args():
    parser = argparse.ArgumentParser(description="Save MLflow model for deployment")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID of the model to save")
    parser.add_argument("--mlflow_tracking_uri", type=str, required=True, help="MLflow tracking URI")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the model artifacts")
    return parser.parse_args()

def main():
    args = parse_args()
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading model artifacts for run ID: {args.run_id}")
        model_dir = os.path.join(output_dir, "model")
        
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(args.run_id)
        artifact_uri = run.info.artifact_uri

        # Download the model artifact directory
        mlflow.artifacts.download_artifacts(
            artifact_uri=os.path.join(artifact_uri, "model"),
            dst_path=output_dir
        )

        # Load the original model.pth file
        raw_model_path = os.path.join(model_dir, "data", "model.pth")
        logger.info(f"Loading model from {raw_model_path}")
        checkpoint = torch.load(raw_model_path, map_location=torch.device('cpu'))

        # Re-save the checkpoint
        cleaned_model_path = os.path.join(output_dir, "model.pth")
        logger.info(f"Saving cleaned model to {cleaned_model_path}")
        torch.save({
            "model_state_dict": checkpoint["model_state_dict"],
            "input_size": checkpoint["input_size"],
            "hidden_size": checkpoint["hidden_size"],
            "num_layers": checkpoint["num_layers"],
            "dropout": checkpoint["dropout"],
            "output_size": checkpoint["output_size"],
            "scaler_params": checkpoint.get("scaler_params", {})
        }, cleaned_model_path)

        # Save model metadata
        model_metadata = {
            "run_id": args.run_id,
            "model_path": "model.pth",
            "created_at": run.info.start_time,
            "metrics": run.data.metrics
        }

        metadata_path = os.path.join(output_dir, "model_metadata.json")
        logger.info(f"Saving model metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=4)

        logger.info(f"Model successfully saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()