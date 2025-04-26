import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
BASE_URL = 'https://www.alphavantage.co/query'

# List of forex pairs to track
FOREX_PAIRS = [
    'USD/OMR',  # US Dollar / Omani Rial
    'USD/INR',  # US Dollar / Indian Rupee
    'USD/NZD',  # US Dollar / NZ Dollar
    'USD/EUR',  # US Dollar / Euro
]

# Data collection parameters
DEFAULT_OUTPUT_SIZE = 'full'  # 'compact' for latest 100 datapoints, 'full' for all available data

# Database configuration
DB_USERNAME = os.getenv('DB_USERNAME', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5433')
DB_NAME = os.getenv('DB_NAME', 'forex_data')

# Database connection string
DB_CONNECTION_STRING = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"