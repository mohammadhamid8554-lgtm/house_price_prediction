import pickle
from pathlib import Path

from src.ml_house_price_prediction.exception import CustomException
from src.ml_house_price_prediction.logger import logging
from src.ml_house_price_prediction.utils import save_object, load_object


def test_custom_exception_has_message():
    try:
        raise ValueError("sample failure")
    except ValueError as exc:
        err = CustomException(exc, __import__("sys"))
        assert "sample failure" in str(err)


def test_object_round_trip(tmp_path):
    file_path = tmp_path / "model.pkl"
    payload = {"model": "linear_regression", "score": 0.95}

    save_object(str(file_path), payload)
    loaded = load_object(str(file_path))

    assert loaded == payload
    assert logging is not None
