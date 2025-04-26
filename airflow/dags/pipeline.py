from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from sqlalchemy import text
import sys
import pandas as pd
import pendulum


# Set path to root
sys.path.append("/home/ayaan-ubuntu/DA5402/DA5402-ML-Application")

# Import project modules
from data_collection.data_collect import fetch_all_pairs
from data_collection.data_store import create_DBengine, create_tables, store_raw_data, check_availability
from configs.config import FOREX_PAIRS
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('forex_dag')

# Define default arguments for the DAG
default_args = {
    'owner': 'Ayaan Labba',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
dag = DAG(
    'forex_data_pipeline',
    default_args=default_args,
    description='Forex data collection pipeline',
    schedule='0 1 * * *',  # Run daily at 1:00 AM
    start_date=pendulum.now().subtract(days=1),
    catchup=False,
    tags=['forex', 'data_pipeline'],
)

# Task to initialize database tables
def init_database():
    logger.info("Initializing database tables")
    engine = create_DBengine()
    create_tables(engine)
    logger.info("Database initialization completed")
    return "Database initialized"

# Task to fetch forex data from Alpha Vantage
def fetch_forex_data():
    logger.info("Starting forex data collection")
    forex_data = fetch_all_pairs()
    if not forex_data:
        logger.error("No forex data fetched")
        return False
    
    logger.info(f"Fetched data for {len(forex_data)} forex pairs")

    # Store real dataframes inside XCom temporarily serialized
    forex_data_dict = {}
    for pair, df in forex_data.items():
        df_serializable = df.copy()
        for col in df_serializable.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            df_serializable[col] = df_serializable[col].astype(str)
        forex_data_dict[pair] = df_serializable.to_dict()
    
    return forex_data_dict

# Task to store raw forex data in database
def store_raw_data_db(**kwargs):
    logger.info("Starting raw data storage")
    ti = kwargs['ti']
    forex_data_dict = ti.xcom_pull(task_ids='fetch_forex_data')
    
    if not forex_data_dict:
        logger.warning("No forex data to store")
        return False
    
    forex_data = {}
    for pair, data_dict in forex_data_dict.items():
        df = pd.DataFrame.from_dict(data_dict)
        
        # Convert datetime columns back to datetime
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass  # not a datetime column
        
        forex_data[pair] = df
    
    engine = create_DBengine()
    records_stored = store_raw_data(engine, forex_data)
    logger.info(f"Stored {records_stored} raw forex records")
    
    # Check data availability for each pair
    for pair in FOREX_PAIRS:
        earliest, latest, count = check_availability(engine, pair)
        logger.info(f"Pair {pair}: {count} records from {earliest} to {latest}")
    
    return True

# Task to check data quality
def check_data_availability():
    logger.info("Checking data availability")
    engine = create_DBengine()
    
    try:
        logger.info("Checking data availability")
        for pair in FOREX_PAIRS:
            check_availability(engine, pair)
                
    except Exception as e:
        logger.error(f"Error in availability check: {str(e)}")
        return f"Data availability check failed: {str(e)}"

# Define task dependencies
start = EmptyOperator(
    task_id='start_pipeline',
    dag=dag,
)

init_db = PythonOperator(
    task_id='init_database',
    python_callable=init_database,
    dag=dag,
)

fetch_forex = PythonOperator(
    task_id='fetch_forex_data',
    python_callable=fetch_forex_data,
    dag=dag,
)

store_raw = PythonOperator(
    task_id='store_raw_data',
    python_callable=store_raw_data_db,
    op_kwargs={},
    dag=dag,
)

quality_check = PythonOperator(
    task_id='check_data_availability',
    python_callable=check_data_availability,
    dag=dag,
)

end = EmptyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Set up dependencies
start >> init_db >> fetch_forex >> store_raw >> quality_check >> end

if __name__ == "__main__":
    # For testing purposes
    init_database()
    forex_data_dict = fetch_forex_data()
    if forex_data_dict:
        context = {'ti': lambda: None}
        context['ti'].xcom_pull = lambda task_ids: forex_data_dict
        store_raw_data(**context)
        quality_check_result = check_data_availability()
        print(quality_check_result)