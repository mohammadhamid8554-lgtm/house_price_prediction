import sys

from src.ml_house_price_prediction.exception import CustomException
from src.ml_house_price_prediction.logger import logging
from src.ml_house_price_prediction.pipelines.training_pipeline import TrainPipeline


if __name__ == "__main__":
    logging.info("Project execution started.")

    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        logging.info("Project completed successfully.")

    except Exception as exc:
        raise CustomException(exc, sys) from exc
