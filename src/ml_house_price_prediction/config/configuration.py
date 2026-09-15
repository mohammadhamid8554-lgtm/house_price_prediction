import os
from dataclasses import dataclass

from src.ml_house_price_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)


@dataclass
class ConfigManager:
    def __init__(self):
        self.artifacts_dir = os.path.join("artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            train_data_path=os.path.join(self.artifacts_dir, "train.csv"),
            test_data_path=os.path.join(self.artifacts_dir, "test.csv"),
            raw_data_path=os.path.join(self.artifacts_dir, "raw.csv"),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            preprocessor_object_path=os.path.join(self.artifacts_dir, "preprocessor.pkl"),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig(
            trained_model_path=os.path.join(self.artifacts_dir, "model.pkl"),
        )
