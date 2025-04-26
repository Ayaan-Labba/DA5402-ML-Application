import sys
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import os

# Set path to root
sys.path.append(os.getcwd())

from configs.config import FOREX_PAIRS, DB_CONNECTION_STRING
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('data_loading')

def create_DBengine():
    """
    Create database engine connection
    
    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine
    """
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        logger.info("Database engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Error creating database engine: {str(e)}")
        raise

def load_pair_data(pair, start_date=None, end_date=None):
    """
    Extract forex data for a specific pair from the database
    
    Args:
        pair (str): Forex pair in 'XXX/YYY' format
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        
    Returns:
        pandas.DataFrame: DataFrame with forex data
    """
    try:
        # Create database engine
        engine = create_DBengine()
        
        # Build query
        query = f"SELECT * FROM forex_raw WHERE pair = '{pair}'"
        
        if start_date:
            query += f" AND date >= '{start_date}'"
        
        if end_date:
            query += f" AND date <= '{end_date}'"
        
        query += " ORDER BY date ASC"
        
        # Execute query with connection
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        
        logger.info(f"Extracted {len(df)} records for {pair}")
        return df
    
    except Exception as e:
        logger.error(f"Error extracting data for {pair}: {str(e)}")
        return pd.DataFrame()

def load_all_pairs(output_dir="data/raw_data", start_date=None, end_date=None):
    """
    Extract data for all forex pairs and save to CSV files
    
    Args:
        output_dir (str): Directory to save raw data files
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary of DataFrames with forex pair as key
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    forex_data = {}
    
    for pair in FOREX_PAIRS:
        # Extract data
        df = load_pair_data(pair, start_date, end_date)
        
        if len(df) > 0:
            # Save to CSV
            pair_filename = pair.replace('/', '_') + '.csv'
            output_path = os.path.join(output_dir, pair_filename)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved raw data for {pair} to {output_path}")
            
            # Add to dictionary
            forex_data[pair] = df
        else:
            logger.warning(f"No data found for {pair}")
    
    return forex_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract forex data from database')
    parser.add_argument('--output_dir', type=str, default='../data/raw', 
                        help='Directory to save raw data files')
    parser.add_argument('--start_date', type=str, default=None, 
                        help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), 
                        help='End date in YYYY-MM-DD format')
    
    args = parser.parse_args()
    
    # Extract all pairs
    forex_data = load_all_pairs(args.output_dir, args.start_date, args.end_date)
    
    logger.info(f"Extracted data for {len(forex_data)} forex pairs")