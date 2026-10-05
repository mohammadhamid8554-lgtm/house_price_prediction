import pandas as pd

from ml_house_price_prediction.components.data_ingestion import (
    DataIngestion,
    DataIngestionConfig,
)


def test_data_ingestion_reads_standard_data_directory(tmp_path):
    config = DataIngestionConfig(
        train_data_path=str(tmp_path / "train.csv"),
        test_data_path=str(tmp_path / "test.csv"),
    )

    train_path, test_path = DataIngestion(config).initiate_data_ingestion()

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    assert len(train) + len(test) == 4600
    assert set(train.columns) == set(test.columns)
    assert not (tmp_path / "raw.csv").exists()
