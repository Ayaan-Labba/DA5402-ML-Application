import argparse
import os
import sys
import json
import mlflow
import torch
from pathlib import Path

# Set path to root
sys.path.append(os.getcwd())

# Dummy model class (update as per your model)
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

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

        print(f"Downloading model artifacts for run ID: {args.run_id}")
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
        checkpoint = torch.load(raw_model_path, map_location=torch.device('cpu'), weights_only=True)

        # Extract hyperparameters and state_dict
        model = LSTMModel(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            output_size=checkpoint["output_size"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Re-save clean model.pth (only state_dict + metadata)
        cleaned_model_path = os.path.join(output_dir, "model.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
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

        with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
            json.dump(model_metadata, f, indent=4)

        print(f"Model successfully saved to {output_dir}")

    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()