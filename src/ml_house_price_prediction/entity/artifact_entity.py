from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_data_path: str
    test_data_path: str
    raw_data_path: str


@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str
    preprocessor_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    train_r2_score: float
    test_r2_score: float
