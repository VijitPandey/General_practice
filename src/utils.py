import os 
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import dill

from sklearn.metrics import r2_score

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
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the models and return the best model based on R2 score.
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_score = model.score(X_test, y_test)
            report[model_name] = r2_score
        
        return report

    except Exception as e:
        raise CustomException(e, sys)