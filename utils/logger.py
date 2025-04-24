import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from pythonjsonlogger import jsonlogger

# Get logs in json format
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.now(timezone.utc).isoformat()
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

def setup_logger(name, log_level=logging.INFO):
    """Set up a logger with file and console handlers"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler - display logs in console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(funcName)s:%(lineno)d | %(message)s') # not in json format
    console_handler.setFormatter(console_formatter)
    
    # File handler - print logs to logs/<log_file>
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(module)s %(function)s:%(line)s %(message)s') # in json format
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger