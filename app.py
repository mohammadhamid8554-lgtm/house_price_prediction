"""ASGI entry point: run with ``uvicorn app:app``."""

from ml_house_price_prediction.api.main import app

__all__ = ["app"]