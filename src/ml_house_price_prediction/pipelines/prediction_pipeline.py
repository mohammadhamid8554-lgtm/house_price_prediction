import sys
from pathlib import Path

import pandas as pd

from ml_house_price_prediction.exception import CustomException
from ml_house_price_prediction.utils import load_object


class PredictPipeline:
    """Load the saved model and preprocessor once, then serve predictions."""

    def __init__(self, model_path: str | Path | None = None, preprocessor_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[3]
        self.model_path = Path(model_path) if model_path else project_root / "artifacts" / "model.pkl"
        self.preprocessor_path = (
            Path(preprocessor_path) if preprocessor_path else project_root / "artifacts" / "preprocessor.pkl"
        )
        self.model = load_object(str(self.model_path))
        self.preprocessor = load_object(str(self.preprocessor_path))

    def predict(self, features: pd.DataFrame):
        try:
            prepared_features = features.copy()
            if "date" in prepared_features.columns:
                prepared_features["date"] = pd.to_datetime(prepared_features["date"], errors="raise").dt.year

            transformed_features = self.preprocessor.transform(prepared_features)
            return self.model.predict(transformed_features)

        except Exception as exc:
            raise CustomException(exc, sys) from exc
