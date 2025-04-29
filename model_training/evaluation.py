import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.pytorch
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

# Set path to root
sys.path.append(os.getcwd())

from utils.logger import setup_logger

# Setup logging
logger = setup_logger("Hyperparameter Tuning")

def prepare_inference_data(data_path, sequence_length):
    """
    Prepare data for model inference and drift detection
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Remove date column if it exists
    if 'date' in df.columns:
        df = df.drop('date', axis=1)
    
    # Extract features and target
    features = df.drop('target', axis=1).values
    target = df['target'].values
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i + sequence_length])
        y.append(target[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor, df, scaler

def load_model(run_id, mlflow_tracking_uri):
    """
    Load model from MLflow model registry
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Load model
    logger.info(f"Loading model from run: {run_id}")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    
    return model

def evaluate_model(model, X, y):
    """
    Evaluate model on test data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_device = X.to(device)
        y_device = y.to(device)
        
        # Make predictions
        predictions = model(X_device)
        
        # Calculate metrics
        mse = torch.mean((predictions - y_device) ** 2).item()
        mae = torch.mean(torch.abs(predictions - y_device)).item()
        
        # Calculate directional accuracy
        actual_direction = (y_device[1:] > y_device[:-1]).float()
        pred_direction = (predictions[1:] > predictions[:-1]).float()
        directional_accuracy = torch.mean((actual_direction == pred_direction).float()).item()
        
        # Move tensors back to CPU for further processing
        predictions_np = predictions.cpu().numpy()
        y_np = y_device.cpu().numpy()
        
    metrics = {
        "mse": mse,
        "mae": mae,
        "directional_accuracy": directional_accuracy
    }
    
    return predictions_np, y_np, metrics

def detect_drift(reference_data, current_data, threshold=0.05):
    """
    Detect data drift using Kolmogorov-Smirnov test
    """
    # Flatten the sequence data
    reference_flat = reference_data.reshape(-1, reference_data.shape[-1])
    current_flat = current_data.reshape(-1, current_data.shape[-1])
    
    # Check drift for each feature
    drift_results = {}
    drift_detected = False
    
    for feature_idx in range(reference_flat.shape[1]):
        ref_feature = reference_flat[:, feature_idx]
        curr_feature = current_flat[:, feature_idx]
        
        # Run KS test
        ks_stat, p_value = ks_2samp(ref_feature, curr_feature)
        
        drift_results[f"feature_{feature_idx}"] = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drift_detected": p_value < threshold
        }
        
        if p_value < threshold:
            drift_detected = True
    
    return drift_detected, drift_results

def visualize_drift(reference_data, current_data, drift_results, output_dir):
    """
    Create visualizations for data drift
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten the sequence data
    reference_flat = reference_data.reshape(-1, reference_data.shape[-1])
    current_flat = current_data.reshape(-1, current_data.shape[-1])
    
    # Create distribution plots for each feature
    for feature_idx in range(reference_flat.shape[1]):
        ref_feature = reference_flat[:, feature_idx]
        curr_feature = current_flat[:, feature_idx]
        
        drift_info = drift_results[f"feature_{feature_idx}"]
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(ref_feature, label="Reference Data", fill=True, alpha=0.3)
        sns.kdeplot(curr_feature, label="Current Data", fill=True, alpha=0.3)
        
        title = f"Feature {feature_idx} Distribution"
        if drift_info["drift_detected"]:
            title += f" - DRIFT DETECTED (p={drift_info['p_value']:.4f})"
        else:
            title += f" - No Drift (p={drift_info['p_value']:.4f})"
            
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"drift_feature_{feature_idx}.png"))
        plt.close()
    
    # Create comparison plot of actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.title("Forecast vs Actual")
    # Plotting code will be added here during evaluation
    plt.savefig(os.path.join(output_dir, "forecast_vs_actual.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate forex prediction model')
    parser.add_argument('--model_run_id', type=str, required=True, help='MLflow run ID of the model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length for LSTM')
    parser.add_argument('--reference_data_path', type=str, required=True, 
                        help='Path to reference data for drift detection')
    parser.add_argument('--mlflow_tracking_uri', type=str, default='./mlruns', 
                        help='MLflow tracking URI')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Load reference data for drift detection
    X_ref, _, _, _ = prepare_inference_data(args.reference_data_path, args.sequence_length)
    
    # Load evaluation data
    X_eval, y_eval, df_eval, scaler = prepare_inference_data(args.data_path, args.sequence_length)
    
    # Load model
    model = load_model(args.model_run_id, args.mlflow_tracking_uri)
    
    # Evaluate model
    predictions, actuals, metrics = evaluate_model(model, X_eval, y_eval)
    
    # Detect drift
    drift_detected, drift_results = detect_drift(X_ref.numpy(), X_eval.numpy())
    
    # Create visualizations
    visualize_drift(X_ref.numpy(), X_eval.numpy(), drift_results, args.output_dir)
    
    # Add forecast vs actual to the visualization
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Forecast', alpha=0.7)
    plt.title("Forecast vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "forecast_vs_actual.png"))
    plt.close()
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        "model_run_id": args.model_run_id,
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "drift_detected": drift_detected,
        "drift_results": {k: {"ks_statistic": float(v["ks_statistic"]), 
                             "p_value": float(v["p_value"]), 
                             "drift_detected": v["drift_detected"]} 
                         for k, v in drift_results.items()}
    }
    
    with open(os.path.join(args.output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")
    logger.info(f"Model performance: MAE={metrics['mae']:.4f}, Dir Acc={metrics['directional_accuracy']:.4f}")
    logger.info(f"Drift detected: {drift_detected}")
    
    # If drift detected, record this for the feedback loop
    if drift_detected:
        with open(os.path.join(args.output_dir, "drift_detected.txt"), 'w') as f:
            f.write("1")
        logger.info("Data drift detected! Model may need retraining.")
    else:
        logger.info("No significant data drift detected.")

if __name__ == "__main__":
    main()