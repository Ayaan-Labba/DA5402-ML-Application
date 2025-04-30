from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
import json
import os
import logging
import mlflow
from datetime import datetime, timedelta
import time
from prometheus_client import REGISTRY, Counter, Histogram, start_http_server
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("forex_prediction_api")

def get_or_create_counter(name, description, labelnames):
    try:
        return REGISTRY._names_to_collectors[name]
    except KeyError:
        return Counter(name, description, labelnames)

def get_or_create_histogram(name, description, labelnames):
    try:
        return REGISTRY._names_to_collectors[name]
    except KeyError:
        return Histogram(name, description, labelnames)

PREDICTION_COUNTER = get_or_create_counter(
    "forex_predictions_total", 
    "Total number of forex predictions made",
    ["currency_pair"]
)

PREDICTION_LATENCY = get_or_create_histogram(
    "forex_prediction_latency_seconds", 
    "Time taken for prediction",
    ["currency_pair"]
)

app = FastAPI(
    title="Forex Rate Prediction API",
    description="API for predicting forex exchange rates using LSTM model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model path
MODEL_DIR = os.environ.get("MODEL_DIR", "model_deployment/model")
METADATA_PATH = os.environ.get("MODEL_METADATA", "model_deployment/model_metadata.json")

# Data schemas
class PredictionInput(BaseModel):
    sequence: List[float]
    currency_pair: str = "USD_INR"
    horizon: int = 1

class PredictionOutput(BaseModel):
    predicted_value: float
    prediction_timestamp: str
    confidence_interval: Optional[Dict[str, float]] = None
    model_version: str

# Load model and metadata
model = None
model_metadata = None
scaler = None

def load_model():
    global model, model_metadata, scaler
    
    try:
        # Load model metadata
        logger.info(f"Loading model metadata from {METADATA_PATH}")
        with open(METADATA_PATH, "r") as f:
            model_metadata = json.load(f)
        
        # Load PyTorch model
        model_path = os.path.join(MODEL_DIR, "data/model.pth")
        logger.info(f"Loading model from {model_path}")
        
        # Load the model state dict
        model_state = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Reconstruct the model
        # Extract model parameters from metadata or state dict
        hidden_size = model_state.get("hidden_size", 512)
        num_layers = model_state.get("num_layers", 2)
        dropout = model_state.get("dropout", 0.4)
        input_size = model_state.get("input_size", 1)
        output_size = model_state.get("output_size", 1)
        
        # Create LSTM model
        class LSTMModel(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                                          batch_first=True, dropout=dropout)
                self.fc = torch.nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out
        
        # Instantiate model
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
        
        # Load the state dict into the model
        model.load_state_dict(model_state["model_state_dict"])
        model.eval()
        
        # Load scaler parameters if available
        scaler_params = model_state.get("scaler_params", None)
        if scaler_params:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaler.min_, scaler.scale_ = scaler_params["min"], scaler_params["scale"]
        
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Load the model
    if not load_model():
        logger.error("Failed to load model at startup")
    else:
        logger.info("Model loaded successfully at startup")

@app.get("/")
async def root():
    return {"message": "Forex Rate Prediction API is running"}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": model_metadata.get("run_id", "unknown")}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        if not load_model():
            raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        start_time = time.time()
        
        # Prepare input sequence
        sequence = np.array(input_data.sequence).reshape(-1, 1)
        
        # Scale input if scaler is available
        if scaler:
            sequence = scaler.transform(sequence)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            prediction = model(sequence_tensor)
            
        # Convert prediction back to original scale if needed
        predicted_value = prediction.item()
        if scaler:
            predicted_value = scaler.inverse_transform(np.array([[predicted_value]]))[0, 0]
        
        # Calculate simple confidence interval (example)
        std_dev = 0.02 * abs(predicted_value)  # Simplified estimate
        confidence_interval = {
            "lower_bound": predicted_value - 1.96 * std_dev,
            "upper_bound": predicted_value + 1.96 * std_dev
        }
        
        # Record metrics
        prediction_time = time.time() - start_time
        PREDICTION_COUNTER.labels(currency_pair=input_data.currency_pair).inc()
        PREDICTION_LATENCY.labels(currency_pair=input_data.currency_pair).observe(prediction_time)
        
        logger.info(f"Prediction made for {input_data.currency_pair}: {predicted_value}")
        
        return {
            "predicted_value": float(predicted_value),
            "prediction_timestamp": datetime.now().isoformat(),
            "confidence_interval": confidence_interval,
            "model_version": model_metadata.get("run_id", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    if model_metadata is None:
        raise HTTPException(status_code=404, detail="Model metadata not available")
    
    return {
        "model_version": model_metadata.get("run_id", "unknown"),
        "created_at": model_metadata.get("created_at", "unknown"),
        "metrics": model_metadata.get("metrics", {})
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)