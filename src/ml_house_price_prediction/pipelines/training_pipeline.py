import sys

from src.ml_house_price_prediction.components.data_ingestion import DataIngestion
from src.ml_house_price_prediction.components.data_transformation import DataTransformation
from src.ml_house_price_prediction.components.model_trainer import ModelTrainer
from src.ml_house_price_prediction.config.configuration import ConfigManager
from src.ml_house_price_prediction.exception import CustomException
from src.ml_house_price_prediction.logger import logging


class TrainPipeline:
    """Run the full training process from raw data to trained model."""

    def __init__(self):
        self.config = ConfigManager()

    def run_pipeline(self):
        try:
            data_ingestion = DataIngestion(self.config.get_data_ingestion_config())
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation(self.config)
            data_transformation.initiate_data_transformation(train_path, test_path)

            model_trainer = ModelTrainer(self.config)
            model_trainer.initiate_model_training(train_path, test_path)

            logging.info("Training pipeline completed successfully.")
            return train_path, test_path

        except Exception as exc:
            raise CustomException(exc, sys) from exc
