import requests
import pandas as pd
import time
import json
import sys

# Set path to root
sys.path.append("../")

from configs.config import ALPHA_VANTAGE_API_KEY, BASE_URL, FOREX_PAIRS, DEFAULT_OUTPUT_SIZE
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('data_collect')

def fetch_data(base_curr, to_curr, output_size=DEFAULT_OUTPUT_SIZE):
    """
    Fetch forex data from Alpha Vantage API
    
    Args:
        base_curr (str): From currency symbol (e.g., 'USD')
        to_curr (str): To currency symbol (e.g., 'INR')
        output_size (str): 'compact' for latest 100 data points, 'full' for all available data
        
    Returns:
        pandas.DataFrame: DataFrame containing forex data
    """

    logger.info(f"Fetching forex data for {base_curr}/{to_curr}")
    
    try:
        params = {
            'function': 'FX_DAILY',
            'from_symbol': base_curr,
            'to_symbol': to_curr,
            'outputsize': output_size,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Check for error message in the response
        if 'Error Message' in data:
            logger.error(f"API Error: {data['Error Message']}")
            return None
            
        # Check for time series data
        if 'Time Series FX (Daily)' not in data: # Use current key used by API
            logger.error(f"Unexpected API response structure: {json.dumps(data)[:200]}...")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data['Time Series FX (Daily)'], orient='index')
        
        # Rename columns to more readable format
        df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close'
        }, inplace=True)
        
        # Convert values to float
        for col in df.columns:
            df[col] = df[col].astype(float)
        
        # Add date column and forex pair info
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        df['from_currency'] = base_curr
        df['to_currency'] = to_curr
        df['pair'] = f"{base_curr}/{to_curr}"
        
        logger.info(f"Successfully fetched {len(df)} records for {base_curr}/{to_curr}")
        return df
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return None
    except ValueError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching forex data: {str(e)}")
        return None

def fetch_all_pairs(forex_pairs=FOREX_PAIRS, delay=15):
    """
    Fetch data for all specified forex pairs
    
    Args:
        forex_pairs (list): List of forex pairs in 'ABC/XYZ' format
        delay (int): Delay between API calls in seconds (to avoid rate limits)
        
    Returns:
        dict: Dictionary of DataFrames with forex pair as key
    """
    all_data = {}
    
    for pair in forex_pairs:
        logger.info(f"Processing forex pair: {pair}")
        
        try:
            from_symbol, to_symbol = pair.split('/')
            
            df = fetch_data(from_symbol, to_symbol)
            
            if df is not None and not df.empty:
                all_data[pair] = df
            
            # Respect API rate limits with a delay
            if pair != forex_pairs[-1]:  # No need to wait after the last request
                logger.info(f"Waiting {delay} seconds before next API call")
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error processing {pair}: {str(e)}")
    
    return all_data

if __name__ == "__main__":
    # For testing purposes
    data = fetch_all_pairs()
    for pair, df in data.items():
        print(f"{pair}: {len(df)} records")
        print(df.head())