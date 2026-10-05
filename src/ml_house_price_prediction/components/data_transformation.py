import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_house_price_prediction.config.configuration import ConfigManager
from ml_house_price_prediction.exception import CustomException
from ml_house_price_prediction.logger import logging
from ml_house_price_prediction.utils import save_object


class DataTransformation:
    """Prepare the features before model training."""

    def __init__(self, config: ConfigManager):
        self.config = config

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"]).dt.year

        return data

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "sqft_lot",
                "floors",
                "waterfront",
                "view",
                "condition",
                "sqft_above",
                "sqft_basement",
                "yr_built",
                "yr_renovated",
                "date",
            ]

            categorical_columns = [
                "street",
                "city",
                "statezip",
                "country",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as exc:
            raise CustomException(exc, sys) from exc

    def initiate_data_transformation(self, train_path):
        try:
            train_df = self._prepare_features(pd.read_csv(train_path))

            target_column = "price"
            X_train = train_df.drop(columns=[target_column])

            preprocessor = self.get_data_transformer_object()

            preprocessor.fit(X_train)

            save_object(self.config.get_data_transformation_config().preprocessor_object_path, preprocessor)

            logging.info("Data transformation completed successfully.")
            return self.config.get_data_transformation_config().preprocessor_object_path

        except Exception as exc:
            raise CustomException(exc, sys) from exc
