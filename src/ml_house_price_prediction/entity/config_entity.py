from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str
    raw_data_path: str


@dataclass(frozen=True)
class DataTransformationConfig:
    preprocessor_object_path: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_path: str
