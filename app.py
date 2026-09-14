import sys
from src.ml_house_price_prediction.logger import logging
from src.ml_house_price_prediction.exception import CustomException

if __name__ == "__main__":
    logging.info("The execution has started!!")

    try:
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)