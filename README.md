# House Price Prediction

This project trains a house-price regression model and serves predictions through a
FastAPI service with a Streamlit form.

## Project layout

```text
data/                         Source dataset
notebooks/                    EDA and model exploration
artifacts/                    Saved model and preprocessor
src/ml_house_price_prediction/ Training, inference, and API code
tests/                        Automated tests
app.py                        FastAPI entry point
streamlit_app.py              Streamlit user interface
main.py                       Model-training entry point
```

## Setup

Run these commands from the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements install the project in editable mode and include the API and UI
dependencies.

## Run the project

Start the prediction API in one terminal:

```powershell
python -m uvicorn app:app --reload
```

Start the Streamlit interface in a second terminal:

```powershell
streamlit run streamlit_app.py
```

Open the local Streamlit URL shown in the terminal. The API documentation is
available at `http://127.0.0.1:8000/docs`; `GET /health` reports whether both
saved model artifacts are available.

If the model artifacts are missing or need to be refreshed, run the training
pipeline from the project root:

```powershell
python main.py
```

Training reads `data/raw.csv` and writes the model and preprocessor to
`artifacts/`. Temporary train/test splits are also written there while training.
The API uses the model artifacts, applies the same sale-date year conversion
used during training, and returns an estimated price.

## Tests

Run the test suite from the project root:

```powershell
python -m pytest
```
