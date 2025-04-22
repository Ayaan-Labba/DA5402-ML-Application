import sys
from sqlalchemy import create_engine, Table, Column, String, Float, Date, MetaData, text
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime

# Set path to root
sys.path.append("../")

from configs.config import DB_CONNECTION_STRING
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('data_storage')

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

def create_tables(engine):
    """
    Create necessary tables in the database if they don't exist
    
    Args:
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine
        
    Returns:
        sqlalchemy.MetaData: Metadata object with table definitions
    """
    try:
        metadata = MetaData()
        
        # Raw forex data table
        forex_raw = Table(
            'forex_raw', 
            metadata,
            Column('date', Date, primary_key=True),
            Column('pair', String, primary_key=True),
            Column('from_currency', String),
            Column('to_currency', String),
            Column('open', Float),
            Column('high', Float),
            Column('low', Float),
            Column('close', Float),
            Column('inserted_at', Date),
            Column('updated_at', Date)
        )
        
        # Create tables in the database
        metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        return metadata
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

def store_raw_data(engine, forex_data):
    """
    Store forex data in the database using upsert (insert or update)
    
    Args:
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine
        forex_data (dict): Dictionary of DataFrames with forex pair as key
        
    Returns:
        int: Total number of records stored
    """
    total_records = 0
    now = datetime.now().date()
    
    try:
        metadata = MetaData()
        forex_raw = Table('forex_raw', metadata, autoload_with=engine)
        
        with engine.begin() as conn:
            for pair, df in forex_data.items():
                logger.info(f"Storing data for {pair}, {len(df)} records")
                
                # Add timestamp columns
                df['inserted_at'] = now
                df['updated_at'] = now
                
                # Convert DataFrame to list of dictionaries
                records = df.to_dict('records')
                total_records += len(records)
                
                # Use PostgreSQL's upsert feature
                stmt = insert(forex_raw).values(records)
                # On conflict, update all fields except primary keys
                stmt = stmt.on_conflict_do_update(
                    constraint=forex_raw.primary_key,
                    set_={
                        'open': stmt.excluded.open,
                        'high': stmt.excluded.high,
                        'low': stmt.excluded.low,
                        'close': stmt.excluded.close,
                        'updated_at': stmt.excluded.updated_at
                    }
                )
                
                conn.execute(stmt)
                
                logger.info(f"Stored {len(records)} records for {pair}")
        
        logger.info(f"Total records stored: {total_records}")
        return total_records
    
    except Exception as e:
        logger.error(f"Error storing raw forex data: {str(e)}")
        raise

def check_availability(engine, pair):
    """
    Check the availability of data for a specific forex pair
    
    Args:
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine
        pair (str): Forex pair in 'XXX/YYY' format
        
    Returns:
        tuple: (earliest_date, latest_date, count)
    """
    try:
        with engine.connect() as conn:
            # Get earliest date
            query = text(f"SELECT MIN(date) FROM forex_raw WHERE pair = '{pair}'")
            earliest_date = conn.execute(query).scalar()
            
            # Get latest date
            query = text(f"SELECT MAX(date) FROM forex_raw WHERE pair = '{pair}'")
            latest_date = conn.execute(query).scalar()
            
            # Get count
            query = text(f"SELECT COUNT(*) FROM forex_raw WHERE pair = '{pair}'")
            count = conn.execute(query).scalar()
            
            logger.info(f"Data for {pair}: {count} records from {earliest_date} to {latest_date}")
            
            return earliest_date, latest_date, count
    
    except Exception as e:
        logger.error(f"Error checking data availability: {str(e)}")
        return None, None, 0

if __name__ == "__main__":
    # For testing purposes
    engine = create_DBengine()
    create_tables(engine)
    
    # Test data availability
    from configs.config import FOREX_PAIRS
    for pair in FOREX_PAIRS:
        check_availability(engine, pair)