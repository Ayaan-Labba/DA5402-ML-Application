import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import mlflow
import json
from itertools import product

# Set path to root
sys.path.append(os.getcwd())

from model_training.train import prepare_data, train_and_evaluate, LSTMModel
from utils.logger import setup_logger

# Setup logging
logger = setup_logger("Hyperparameter Tuning")

def run_hyperparameter_tuning(data_path, sequence_length, param_grid, mlflow_tracking_uri, experiment_name):
    """
    Run hyperparameter tuning with MLflow tracking
    """
    # Set up MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler = prepare_data(data_path, sequence_length)
    
    # Generate all combinations of hyperparameters
    param_combinations = list(product(
        param_grid['hidden_size'],
        param_grid['num_layers'],
        param_grid['dropout'],
        param_grid['batch_size'],
        param_grid['learning_rate'],
        param_grid['num_epochs']
    ))
    
    logger.info(f"Running hyperparameter tuning with {len(param_combinations)} combinations")
    
    # Store results
    results = []
    
    # Run training for each combination
    for hidden_size, num_layers, dropout, batch_size, learning_rate, num_epochs in param_combinations:
        logger.info(f"Training with: hidden_size={hidden_size}, num_layers={num_layers}, "
              f"dropout={dropout}, batch_size={batch_size}, "
              f"learning_rate={learning_rate}, num_epochs={num_epochs}")
        
        model_params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        training_params = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs
        }
        
        # Train and evaluate model
        model, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, 
                                          model_params, training_params)
        
        # Store results
        result = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'test_loss': metrics['test_loss'],
            'mae': metrics['mae'],
            'directional_accuracy': metrics['directional_accuracy'],
            'run_id': metrics['run_id']
        }
        
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Find best model based on test loss
    best_idx = results_df['test_loss'].idxmin()
    best_model = results_df.iloc[best_idx]
    
    logger.info(f"Best model found with test_loss: {best_model['test_loss']:.4f}")
    logger.info(f"Best hyperparameters: {best_model.to_dict()}")
    
    # Save results
    os.makedirs('tuning_results', exist_ok=True)
    results_df.to_csv('tuning_results/hyperparameter_results.csv', index=False)
    
    with open('tuning_results/best_params.json', 'w') as f:
        json.dump(best_model.to_dict(), f, indent=4)
    
    return best_model.to_dict()

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for forex prediction model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to prepared data')
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length for LSTM')
    parser.add_argument('--mlflow_tracking_uri', type=str, default='./mlruns', 
                        help='MLflow tracking URI')
    parser.add_argument('--experiment_name', type=str, default='forex_prediction_tuning',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Define hyperparameter grid
    param_grid = {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2],
        'dropout': [0.1, 0.2, 0.3],
        'batch_size': [16, 32, 64],
        'learning_rate': [0.01, 0.001],
        'num_epochs': [30, 50]
    }
    
    # Run hyperparameter tuning
    best_params = run_hyperparameter_tuning(
        args.data_path,
        args.sequence_length,
        param_grid,
        args.mlflow_tracking_uri,
        args.experiment_name
    )
    
    logger.info("Hyperparameter tuning completed successfully!")

if __name__ == "__main__":
    main()