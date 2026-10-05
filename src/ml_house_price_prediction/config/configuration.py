from dataclasses import dataclass
from pathlib import Path

from ml_house_price_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)


@dataclass
class ConfigManager:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[3]
        self.artifacts_dir = project_root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            train_data_path=str(self.artifacts_dir / "train.csv"),
            test_data_path=str(self.artifacts_dir / "test.csv"),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            preprocessor_object_path=str(self.artifacts_dir / "preprocessor.pkl"),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig(
            trained_model_path=str(self.artifacts_dir / "model.pkl"),
        )
