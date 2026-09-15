import os
import pickle
import sys

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.ml_house_price_prediction.exception import CustomException
from src.ml_house_price_prediction.logger import logging

load_dotenv()


def read_sql_data():
    """Read data from a MySQL table when environment variables are configured."""
    logging.info("Reading SQL database started.")

    try:
        import pymysql

        host = os.getenv("host")
        user = os.getenv("user")
        password = os.getenv("password")
        database = os.getenv("db")

        if not all([host, user, password, database]):
            raise ValueError("Database environment variables are incomplete.")

        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
        )
        logging.info("Database connection established.")

        df = pd.read_sql_query("SELECT * FROM students", connection)
        logging.info("SQL data loaded successfully.")
        return df

    except Exception as exc:
        raise CustomException(exc, sys) from exc


def save_object(file_path, obj):
    """Save any Python object to disk using pickle."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as exc:
        raise CustomException(exc, sys) from exc


def evaluate_models(X_train, X_test, y_train, y_test, models, param):
    """Evaluate multiple models with basic grid-search tuning."""
    try:
        report = {}

        for model_name, model in models.items():
            model_params = param.get(model_name, {})
            grid_search = GridSearchCV(model, model_params, cv=3)
            grid_search.fit(X_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info("Model: %s | Train R2: %.4f | Test R2: %.4f", model_name, train_score, test_score)
            report[model_name] = test_score

        return report

    except Exception as exc:
        raise CustomException(exc, sys) from exc


def load_object(file_path):
    """Load a saved object from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as exc:
        raise CustomException(exc, sys) from exc

