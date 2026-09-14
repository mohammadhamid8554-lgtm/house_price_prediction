import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

projec_name = "ml_house_price_prediction"

list_of_files = [

    f"src/{projec_name}/__init__.py",
    f"src/{projec_name}/components/__init__.py",
    f"src/{projec_name}/components/data_ingestion.py",
    f"src/{projec_name}/components/data_transformation.py",
    f"src/{projec_name}/components/model_trainer.py",
    f"src/{projec_name}/components/model_monitoring.py",
    f"src/{projec_name}/pipelines/__init__.py",
    f"src/{projec_name}/pipelines/__init__.py",
    f"src/{projec_name}/pipelines/training_pipeline.py",
    f"src/{projec_name}/pipelines/prediction_pipeline.py",
    f"src/{projec_name}/exception_handling.py",
    f"src/{projec_name}/logger.py",
    f"src/{projec_name}/utils.py",
    "app.py",
    "docker_file",
    "main.py",
    "setup.py",
    "requirements.txt"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Create Directory: {filedir}")

    if (not os.path.exists(filename) or os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists!!")