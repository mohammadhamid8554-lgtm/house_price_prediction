import sys

from src.ml_house_price_prediction.exception import CustomException
from src.ml_house_price_prediction.logger import logging
from src.ml_house_price_prediction.pipelines.training_pipeline import TrainPipeline


if __name__ == "__main__":
    logging.info("Application execution started.")

    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logging.info("Training pipeline completed successfully.")

    except Exception as exc:
        raise CustomException(exc, sys) from exc