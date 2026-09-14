from src.ml_house_price_prediction.logger import logging
from src.ml_house_price_prediction.exception import CustomException
import sys
if __name__ == "__main":
    logging.info("The execution has started!!")

try:
    a = 1 / 0
except Exception as e:
    raise CustomException(e, sys) # type: ignore