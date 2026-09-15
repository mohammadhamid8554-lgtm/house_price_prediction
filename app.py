import sys
from src.ml_house_price_prediction.logger import logging
from src.ml_house_price_prediction.exception import CustomException
from src.ml_house_price_prediction.components.data_ingestion import DataIngestion




if __name__ == "__main__":
    logging.info("The execution has started!!")

    try:
        # Data Ingestion

        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        
    except Exception as e:
        raise CustomException(e, sys)