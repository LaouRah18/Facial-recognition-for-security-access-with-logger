"""
utils.py
Small utilities: logger setup and project root detection.
"""
import logging
import os


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))




def setup_logger(name: str, log_file: str, level=logging.INFO):
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)


    console = logging.StreamHandler()
    console.setFormatter(formatter)


    logger = logging.getLogger(name)
    logger.setLevel(level)


    # Avoid adding multiple handlers if called more than once
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console)


    return logger