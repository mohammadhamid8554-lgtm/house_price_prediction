import os
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml_house_price_prediction.config.configuration import ConfigManager
from src.ml_house_price_prediction.entity.artifact_entity import DataTransformationArtifact
from src.ml_house_price_prediction.exception import CustomException
from src.ml_house_price_prediction.logger import logging
from src.ml_house_price_prediction.utils import save_object


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

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = self._prepare_features(pd.read_csv(train_path))
            test_df = self._prepare_features(pd.read_csv(test_path))

            target_column = "price"
            X_train = train_df.drop(columns=[target_column])
            X_test = test_df.drop(columns=[target_column])

            preprocessor = self.get_data_transformer_object()

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            save_object(self.config.get_data_transformation_config().preprocessor_object_path, preprocessor)

            dense_train = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
            dense_test = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

            pd.DataFrame(dense_train).to_csv(os.path.join("artifacts", "train_transformed.csv"), index=False)
            pd.DataFrame(dense_test).to_csv(os.path.join("artifacts", "test_transformed.csv"), index=False)

            artifact = DataTransformationArtifact(
                transformed_train_path=os.path.join("artifacts", "train_transformed.csv"),
                transformed_test_path=os.path.join("artifacts", "test_transformed.csv"),
                preprocessor_path=self.config.get_data_transformation_config().preprocessor_object_path,
            )

            logging.info("Data transformation completed successfully.")
            return artifact

        except Exception as exc:
            raise CustomException(exc, sys) from exc
