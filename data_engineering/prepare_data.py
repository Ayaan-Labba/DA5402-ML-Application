import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set path to root
sys.path.append(os.getcwd())

from configs.config import FOREX_PAIRS
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('data_transformation')

def transform_data(df, sequence_length=60, prediction_horizon=1):
    """
    Transform forex data for LSTM model
    
    Args:
        df (pandas.DataFrame): DataFrame with forex data
        sequence_length (int): Number of days in input sequence
        prediction_horizon (int): Number of days to predict ahead
        
    Returns:
        pandas.DataFrame: Transformed DataFrame ready for model training
    """
    try:
        df_new = df.tail(sequence_length)

        # Convert date to datetime if it's not already
        if df_new['date'].dtype != 'datetime64[ns]':
            df_new['date'] = pd.to_datetime(df_new['date'])
        
        # Sort by date
        df_new = df_new.sort_values('date')
        
        # Create a clean dataframe with only the needed columns
        clean_df = df_new[['date', 'open', 'high', 'low', 'close']].copy()
        
        # Feature engineering
        # Add technical indicators
        
        # 1. Moving averages
        clean_df['MA5'] = clean_df['close'].rolling(window=5).mean()    # weekly
        clean_df['MA10'] = clean_df['close'].rolling(window=10).mean()  # biweekly
        clean_df['MA22'] = clean_df['close'].rolling(window=22).mean()  # monthly
        
        # 2. Exponential Moving Average
        df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()     # weekly
        df['EMA_22'] = df['close'].ewm(span=22, adjust=False).mean()    # monthly

        # 3. Relative Strength Index (RSI)
        delta = clean_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=10).mean()
        avg_loss = loss.rolling(window=10).mean()
        
        rs = avg_gain / avg_loss
        clean_df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. Bollinger Bands
        BB_middle = clean_df['MA22']
        clean_df['BB_std'] = clean_df['close'].rolling(window=22).std()
        clean_df['BB_upper'] = BB_middle + 2 * clean_df['BB_std']
        clean_df['BB_lower'] = BB_middle - 2 * clean_df['BB_std']
        
        # 5. MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26

        # 6. Average True Range (ATR)
        tr = np.maximum((df['high'] - df['low']), 
                              np.maximum(abs(df['high'] - df['low'].shift()), abs(df['high'] - df['low'].shift()))
                              )
        df['ATR_14'] = tr.rolling(window=10).mean()

        # 7. Price rate of change
        clean_df['price_roc'] = clean_df['close'].pct_change(periods=5) * 100
        
        # 8. Daily returns
        clean_df['daily_return'] = clean_df['close'].pct_change() * 100
        
        # Drop rows with NaN values (due to rolling windows)
        clean_df = clean_df.dropna()
        
        # Create target variable - next day's closing price
        clean_df['target'] = clean_df['close'].shift(-prediction_horizon)
        
        # Drop the last row which doesn't have a target
        clean_df = clean_df[:-prediction_horizon]
        
        logger.info(f"Transformed data shape: {clean_df.shape}")
        return clean_df
    
    except Exception as e:
        logger.error(f"Error transforming data: {str(e)}")
        return pd.DataFrame()

def transform_all_pairs(input_dir="data/raw_data", output_dir="data/prepared_data", sequence_length=60, prediction_horizon=1):
    """
    Transform data for all forex pairs and save to CSV files
    
    Args:
        input_dir (str): Directory with raw data files
        output_dir (str): Directory to save processed data files
        sequence_length (int): Number of days in input sequence
        prediction_horizon (int): Number of days to predict ahead
        
    Returns:
        dict: Dictionary of DataFrames with forex pair as key
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    transformed_data = {}
    
    for pair in FOREX_PAIRS:
        # Construct input file path
        in_filename = pair.replace('/', '_') + '.csv'
        input_path = os.path.join(input_dir, in_filename)
        
        if os.path.exists(input_path):
            # Read data
            df = pd.read_csv(input_path)
            
            # Transform data
            transformed_df = transform_data(df, sequence_length, prediction_horizon)
            
            if len(transformed_df) > 0:
                # Save to CSV
                out_filename = pair.replace('/', '_') + '_transformed.csv'
                output_path = os.path.join(output_dir, out_filename)
                transformed_df.to_csv(output_path, index=False)
                logger.info(f"Saved transformed data for {pair} to {output_path}")
                
                # Add to dictionary
                transformed_data[pair] = transformed_df
            else:
                logger.warning(f"Transformation resulted in empty DataFrame for {pair}")
        else:
            logger.warning(f"Raw data file not found for {pair}: {input_path}")
    
    return transformed_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform forex data for LSTM model')
    parser.add_argument('--input_dir', type=str, default='../data/raw', 
                        help='Directory with raw data files')
    parser.add_argument('--output_dir', type=str, default='../data/processed', 
                        help='Directory to save processed data files')
    parser.add_argument('--sequence_length', type=int, default=60, 
                        help='Number of days in input sequence')
    parser.add_argument('--prediction_horizon', type=int, default=1, 
                        help='Number of days to predict ahead')
    
    args = parser.parse_args()
    
    # Transform all pairs
    transformed_data = transform_all_pairs(args.input_dir, args.output_dir, 
                                         args.sequence_length, args.prediction_horizon)
    
    logger.info(f"Transformed data for {len(transformed_data)} forex pairs")