import numpy as np
import pandas as pd

from ml_house_price_prediction.pipelines import prediction_pipeline
from ml_house_price_prediction.pipelines.prediction_pipeline import PredictPipeline


def test_prediction_prepares_sale_date_and_returns_model_output(monkeypatch, tmp_path):
    transformed_features = {}

    class FakePreprocessor:
        def transform(self, features):
            transformed_features["frame"] = features
            return features

    class FakeModel:
        def predict(self, features):
            return np.array([425000.0])

    objects = {"model.pkl": FakeModel(), "preprocessor.pkl": FakePreprocessor()}

    def fake_load_object(file_path):
        return objects[file_path.rsplit("\\", maxsplit=1)[-1]]

    monkeypatch.setattr(prediction_pipeline, "load_object", fake_load_object)
    pipeline = PredictPipeline(tmp_path / "model.pkl", tmp_path / "preprocessor.pkl")
    features = pd.DataFrame(
        [
            {
                "date": "2014-05-02",
                "bedrooms": 3.0,
                "city": "Shoreline",
            }
        ]
    )

    result = pipeline.predict(features)

    assert result.tolist() == [425000.0]
    assert transformed_features["frame"].loc[0, "date"] == 2014
