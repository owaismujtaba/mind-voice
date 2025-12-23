import logging
import os
import yaml
from pathlib import Path
import numpy as np
import random
import tensorflow as tf 


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

def load_config(config_path):
    """
    Load a YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def load_all_filepaths_from_directory(directory, extensions=None, startswith=None, sub=None):
    """
    Load all file paths from a directory with specified extensions.
    
    Args:
        directory (str): Directory to search for files.
        extensions (list, optional): List of file extensions to include. If None, include all files.
    
    Returns:
        list: List of file paths.
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if extensions is None or any(file.endswith(ext) for ext in extensions):
                if startswith is None or file.startswith(startswith): 
                    if sub == None:  
                        file_paths.append(os.path.join(root, file))
                    else:
                        if sub in file:
                            file_paths.append(os.path.join(root, file))
    return sorted(file_paths)

    

def set_all_seeds(seed=12345678):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    

def log_info(logger, text):
    """
    Log an info message using the provided logger.
    
    Args:
        logger (logging.Logger): Logger instance.
        text (str): Message to log.
    """
    logger.info('='*50)
    logger.info(text)
    logger.info('='*50)