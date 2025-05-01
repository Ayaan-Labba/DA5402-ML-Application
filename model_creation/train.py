import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import json
from datetime import datetime
from tqdm import tqdm

# Set path to root
sys.path.append(os.getcwd())

from utils.logger import setup_logger

# Set up logging
logger = setup_logger("model_training")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(data_path, train_sequence):
    """
    Load and prepare data for LSTM model
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Remove date column if it exists
    if 'date' in df.columns:
        df = df.drop('date', axis=1)
    
    # Extract features and target
    features = df.drop('target', axis=1).values
    target = df['target'].values
    
    # Create sequences
    X, y = [], []
    for i in range(len(features) - train_sequence):
        X.append(features[i:i + train_sequence])
        y.append(target[i + train_sequence])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Split into training and testing (before scaling!)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Flatten X for scaling (scale each timestep feature independently)
    num_features = X.shape[2]
    scaler = StandardScaler()
    
    X_train_flat = X_train_raw.reshape(-1, num_features)
    X_test_flat = X_test_raw.reshape(-1, num_features)
    
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train_raw.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test_raw.shape)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test, scaler


def train_and_evaluate(X_train, y_train, X_val, y_val, model_params, training_params):
    """
    Train and evaluate LSTM model
    """
    # Unpack parameters
    input_size = X_train.shape[2]  # Number of features
    hidden_size = model_params['hidden_size']
    num_layers = model_params['num_layers']
    output_size = 1  # Regression task
    dropout = model_params['dropout']
    
    batch_size = training_params['batch_size']
    learning_rate = training_params['learning_rate']
    num_epochs = training_params['num_epochs']
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss function and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Start MLflow run
    run_name = f"hs{hidden_size}_nl{num_layers}_dr{dropout}_bs{batch_size}_lr{learning_rate}"
    with mlflow.start_run(run_name=run_name):
        # Log model parameters
        mlflow.log_params({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "input_size": input_size,
            "train_sequence": X_train.shape[1]
        })
        
        # Training loop
        logger.info("Starting training...")

        # best_val_loss = float('inf')
        # patience = training_params.get('patience', 10)
        # counter = 0
        # best_model_state = None

        pbar = tqdm(range(num_epochs), desc="Training Progress")
        for epoch in pbar:
            model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_loader)
            
            # Evaluate on val set
            model.eval()
            with torch.no_grad():
                val_X, val_y = X_val.to(device), y_val.to(device)
                predictions = model(val_X)
                val_loss = criterion(predictions, val_y).item()
                
                # Calculate metrics
                mae = torch.mean(torch.abs(predictions - val_y)).item()
                
                # For directional accuracy
                actual_direction = (val_y[1:] > val_y[:-1]).float()
                pred_direction = (predictions[1:] > predictions[:-1]).float()
                directional_accuracy = torch.mean((actual_direction == pred_direction).float()).item()
            
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_model_state = model.state_dict()
            #     counter = 0
            # else:
            #     counter += 1
            #     if counter >= patience:
            #         logger.info(f"Early stopping triggered at epoch {epoch+1}")
            #         break

            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "mae": mae,
                "directional_accuracy": directional_accuracy
            }, step=epoch)
            
            # Update tqdm progress bar in-place
            pbar.set_postfix({
                'Train Loss': f"{avg_loss:.4f}",
                'Val Loss': f"{val_loss:.4f}",
                'MAE': f"{mae:.4f}",
                'Dir Acc': f"{directional_accuracy:.4f}"
            })
        
        # Save model
        mlflow.pytorch.log_model(model, "model")
        
        # Get run ID for reference
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Return model and metrics for further use
        final_metrics = {
            "val_loss": val_loss,
            "mae": mae,
            "directional_accuracy": directional_accuracy,
            "run_id": run_id
        }
        
        return model, final_metrics

def save_model_info(metrics, model_dir, run_id):
    """Save model info and metrics to file"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Add timestamp to metrics
    metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics['run_id'] = run_id
    
    # Save metrics to file
    with open(os.path.join(model_dir, f"model_info_{run_id}.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Model info saved to {model_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model for forex prediction')
    parser.add_argument('--data_path', type=str, required=True, help='Path to prepared data')
    parser.add_argument('--train_sequence', type=int, default=20, help='Sequence length for LSTM')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--mlflow_tracking_uri', type=str, default='http://localhost:5000', 
                        help='MLflow tracking URI')
    parser.add_argument('--experiment_name', type=str, default='forex_prediction',
                        help='MLflow experiment name')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save model info')
    
    args = parser.parse_args()
    
    # Set up MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    # Prepare data
    X_train, y_train, X_val, y_val, _ = prepare_data(args.data_path, args.train_sequence)
    
    # Model parameters
    model_params = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    # Training parameters
    training_params = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs
    }
    
    # Train and evaluate model
    _, metrics = train_and_evaluate(X_train, y_train, X_val, y_val, 
                                      model_params, training_params)
    
    # Save model info
    save_model_info(metrics, args.model_dir, metrics['run_id'])
    
    # Log completion
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()