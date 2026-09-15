import sys

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.ml_house_price_prediction.config.configuration import ConfigManager
from src.ml_house_price_prediction.entity.artifact_entity import ModelTrainerArtifact
from src.ml_house_price_prediction.exception import CustomException
from src.ml_house_price_prediction.logger import logging
from src.ml_house_price_prediction.utils import load_object, save_object


class ModelTrainer:
    """Train a simple regression model and save it."""

    def __init__(self, config: ConfigManager):
        self.config = config

    def initiate_model_training(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            for data in (train_df, test_df):
                if "date" in data.columns:
                    data["date"] = pd.to_datetime(data["date"]).dt.year

            preprocessor_path = self.config.get_data_transformation_config().preprocessor_object_path
            preprocessor = load_object(preprocessor_path)

            X_train = preprocessor.transform(train_df.drop(columns=["price"]))
            y_train = train_df["price"]

            X_test = preprocessor.transform(test_df.drop(columns=["price"]))
            y_test = test_df["price"]

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            model_path = self.config.get_model_trainer_config().trained_model_path
            save_object(model_path, model)

            artifact = ModelTrainerArtifact(
                trained_model_path=model_path,
                train_r2_score=train_r2,
                test_r2_score=test_r2,
            )

            logging.info("Model training completed. Train R2=%s, Test R2=%s", train_r2, test_r2)
            return artifact

        except Exception as exc:
            raise CustomException(exc, sys) from exc
