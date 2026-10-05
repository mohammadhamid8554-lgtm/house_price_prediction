from fastapi.testclient import TestClient

from ml_house_price_prediction.api import main as api_main


class FakePredictionPipeline:
    def predict(self, features):
        return [425000.0]


def test_predict_endpoint_returns_estimated_price(monkeypatch, tmp_path):
    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    model_path.touch()
    preprocessor_path.touch()
    monkeypatch.setattr(api_main, "MODEL_PATH", model_path)
    monkeypatch.setattr(api_main, "PREPROCESSOR_PATH", preprocessor_path)
    monkeypatch.setattr(api_main, "get_prediction_pipeline", lambda: FakePredictionPipeline())
    payload = {
        "date": "2014-05-02",
        "bedrooms": 3,
        "bathrooms": 1.5,
        "sqft_living": 1340,
        "sqft_lot": 7912,
        "floors": 1.5,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "sqft_above": 1340,
        "sqft_basement": 0,
        "yr_built": 1955,
        "yr_renovated": 2005,
        "street": "18810 Densmore Ave N",
        "city": "Shoreline",
        "statezip": "WA 98133",
        "country": "USA",
    }

    response = TestClient(api_main.app).post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"predicted_price": 425000.0}


def test_predict_endpoint_rejects_missing_features():
    response = TestClient(api_main.app).post("/predict", json={"city": "Shoreline"})

    assert response.status_code == 422
