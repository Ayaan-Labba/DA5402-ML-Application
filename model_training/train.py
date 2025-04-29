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

def prepare_data(data_path, sequence_length):
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return X_train, y_train, X_test, y_test, scaler

def train_and_evaluate(X_train, y_train, X_test, y_test, model_params, training_params):
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
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "input_size": input_size,
            "sequence_length": X_train.shape[1]
        })
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(num_epochs):
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
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_X, test_y = X_test.to(device), y_test.to(device)
                predictions = model(test_X)
                test_loss = criterion(predictions, test_y).item()
                
                # Calculate metrics
                mae = torch.mean(torch.abs(predictions - test_y)).item()
                
                # For directional accuracy
                actual_direction = (test_y[1:] > test_y[:-1]).float()
                pred_direction = (predictions[1:] > predictions[:-1]).float()
                directional_accuracy = torch.mean((actual_direction == pred_direction).float()).item()
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "test_loss": test_loss,
                "mae": mae,
                "directional_accuracy": directional_accuracy
            }, step=epoch)
            
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"MAE: {mae:.4f}, "
                  f"Dir Acc: {directional_accuracy:.4f}")
        
        # Save model
        mlflow.pytorch.log_model(model, "model")
        
        # Get run ID for reference
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Return model and metrics for further use
        final_metrics = {
            "test_loss": test_loss,
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
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length for LSTM')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--mlflow_tracking_uri', type=str, default='./mlruns', 
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
    X_train, y_train, X_test, y_test, scaler = prepare_data(args.data_path, args.sequence_length)
    
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
    model, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, 
                                      model_params, training_params)
    
    # Save model info
    save_model_info(metrics, args.model_dir, metrics['run_id'])
    
    # Log completion
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()