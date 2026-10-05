import os
from datetime import date

import requests
import streamlit as st

st.set_page_config(page_title="House Price Estimator", page_icon="🏠")
st.title("House Price Estimator")
st.write("Enter the house details to get an estimated price.")

default_api_url = os.getenv("PREDICTION_API_URL", "http://127.0.0.1:8000").strip()
api_url = st.sidebar.text_input("Prediction API URL", default_api_url).strip().rstrip("/")
if api_url and "://" not in api_url:
    api_url = f"https://{api_url}"

with st.form("house_features"):
    st.subheader("House details")
    left, right = st.columns(2)

    with left:
        sale_date = st.date_input("Sale date", value=date(2014, 5, 2))
        bedrooms = st.number_input("Bedrooms", min_value=0.0, value=3.0, step=1.0)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, value=1.5, step=0.5)
        sqft_living = st.number_input("Living area (sq ft)", min_value=0.0, value=1340.0, step=50.0)
        sqft_lot = st.number_input("Lot area (sq ft)", min_value=0.0, value=7912.0, step=100.0)
        floors = st.number_input("Floors", min_value=0.0, value=1.5, step=0.5)
        waterfront = st.number_input("Waterfront (0 or 1)", min_value=0.0, max_value=1.0, value=0.0, step=1.0)
        view = st.number_input("View score", min_value=0.0, value=0.0, step=1.0)
        condition = st.number_input("Condition score", min_value=1.0, max_value=5.0, value=3.0, step=1.0)

    with right:
        sqft_above = st.number_input("Above-ground area (sq ft)", min_value=0.0, value=1340.0, step=50.0)
        sqft_basement = st.number_input("Basement area (sq ft)", min_value=0.0, value=0.0, step=50.0)
        yr_built = st.number_input("Year built", min_value=1800.0, max_value=2100.0, value=1955.0, step=1.0)
        yr_renovated = st.number_input("Year renovated (0 if never)", min_value=0.0, max_value=2100.0, value=0.0, step=1.0)
        street = st.text_input("Street", value="18810 Densmore Ave N")
        city = st.text_input("City", value="Shoreline")
        statezip = st.text_input("State and ZIP", value="WA 98133")
        country = st.text_input("Country", value="USA")

    submitted = st.form_submit_button("Estimate price")

if submitted:
    payload = {
        "date": sale_date.isoformat(),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "street": street,
        "city": city,
        "statezip": statezip,
        "country": country,
    }

    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
    except requests.HTTPError as exc:
        response = exc.response
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"Prediction API returned {response.status_code}: {detail}")
    except requests.RequestException as exc:
        st.error(f"Could not connect to the prediction API at {api_url}: {exc}")
    else:
        st.success(f"Estimated price: ${result['predicted_price']:,.2f}")
