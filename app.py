import sys
from src.ml_house_price_prediction.logger import logging
from src.ml_house_price_prediction.exception import CustomException

if __name__ == "__main__":
    logging.info("The execution has started!!")

    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Division by zero error occurred.")
        raise CustomException(e, sys)