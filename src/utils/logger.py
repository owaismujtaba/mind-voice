import logging
import os
from pathlib import Path

def create_logger(name, level=logging.INFO):
    """
    Create a logger that logs to both console and a file.
    
    Args:
        name (str): Logger name.
        log_file (str): Path to save the log file.
        level (int): Logging level.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    cur_dir = os.getcwd()
    log_dir = Path(cur_dir, 'logs')
    name = f'{name}.log'
    log_file =  Path(log_dir, name)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

