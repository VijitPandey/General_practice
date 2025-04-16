import os 
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import dill

def save_object(file_path, obj):
    """
    Save the object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)