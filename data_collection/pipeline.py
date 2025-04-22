import argparse
import sys

# Set path to root
sys.path.append("../")

from data_collection.data_collect import fetch_all_pairs
from data_collection.data_store import create_DBengine, create_tables, store_raw_data, check_availability
from configs.config import FOREX_PAIRS
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('main_pipeline')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Forex Data Pipeline Runner')
    parser.add_argument('--collect', action='store_true', help='Collect forex data from Alpha Vantage')
    parser.add_argument('--check', action='store_true', help='Check data availability')
    parser.add_argument('--pairs', nargs='+', help='Forex pairs to process (e.g., EUR/USD GBP/USD)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back for data quality check')
    
    return parser.parse_args()

def run_pipeline(collect=False, check=False, pairs=None, days=30):
    """Run the forex data pipeline with specified options"""
    
    logger.info("Starting forex data pipeline")
    
    if not pairs:
        pairs = FOREX_PAIRS
    
    logger.info(f"Processing forex pairs: {', '.join(pairs)}")
    
    # Initialize database
    engine = create_DBengine()
    create_tables(engine)
    
    if check:
        logger.info("Checking data availability")
        for pair in FOREX_PAIRS:
            check_availability(engine, pair)
    
    if collect:
        logger.info("Collecting forex data")
        forex_data = fetch_all_pairs(pairs)
        
        if forex_data:
            records_stored = store_raw_data(engine, forex_data)
            logger.info(f"Stored {records_stored} raw forex records")
        else:
            logger.error("No forex data collected")
    
    logger.info("Forex data pipeline completed")

if __name__ == "__main__":
    args = parse_arguments()
    
    # If no specific operation is selected, run the complete pipeline
    if not (args.collect or args.check):
        run_pipeline(collect=True, check=True, pairs=args.pairs, days=args.days)
    else:
        run_pipeline(collect=args.collect, check=args.check, pairs=args.pairs, days=args.days)