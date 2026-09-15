import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.ml_house_price_prediction.exception import CustomException
from src.ml_house_price_prediction.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """Load the raw dataset, split it, and save train/test files."""

    def __init__(self, config: DataIngestionConfig = None):
        self.ingestion_config = config or DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            source_path = os.path.join("src", "notebook", "data", "raw.csv")
            df = pd.read_csv(source_path)

            logging.info("Dataset loaded successfully. Shape: %s", df.shape)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train/Test split completed successfully.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as exc:
            raise CustomException(exc, sys) from exc

