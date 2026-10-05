from dataclasses import dataclass


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    train_r2_score: float
    test_r2_score: float
