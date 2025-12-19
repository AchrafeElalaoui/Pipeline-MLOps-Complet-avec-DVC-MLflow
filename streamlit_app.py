import os
import time

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = int(os.getenv("FASTAPI_TIMEOUT", "120"))
DEFAULT_RETRIES = int(os.getenv("FASTAPI_RETRIES", "1"))

FEATURE_COLUMNS = [
    "Store",
    "Dept",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
    "Type",
    "Size",
    "Year",
    "Month",
    "Week",
]


def call_api(api_url: str, payload: dict, timeout: int, retries: int) -> tuple[int, dict, float]:
    start = time.perf_counter()
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{api_url}/predict", json=payload, timeout=timeout
            )
            elapsed = (time.perf_counter() - start) * 1000
            try:
                data = response.json()
            except ValueError:
                data = {"error": response.text}
            return response.status_code, data, elapsed
        except requests.Timeout as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1)
                continue
            elapsed = (time.perf_counter() - start) * 1000
            return 0, {"error": "Request timed out. Increase timeout or retry."}, elapsed
        except requests.RequestException as exc:
            last_error = exc
            elapsed = (time.perf_counter() - start) * 1000
            return 0, {"error": str(exc)}, elapsed

    elapsed = (time.perf_counter() - start) * 1000
    return 0, {"error": str(last_error)}, elapsed


def fetch_models(api_url: str, timeout: int) -> list[str]:
    try:
        response = requests.get(f"{api_url}/models", timeout=timeout)
        data = response.json()
        return data.get("models", [])
    except requests.RequestException:
        return []


def build_record(form_values: dict) -> dict:
    record = {}
    for col in FEATURE_COLUMNS:
        record[col] = form_values.get(col)
    return record


st.set_page_config(page_title="Retail Demand Forecast", layout="wide")
st.title("Retail Demand Forecast")
st.write("Streamlit UI connected to the FastAPI model service.")

api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL)
with st.expander("Request settings"):
    timeout = st.number_input(
        "Request timeout (seconds)", min_value=1, value=DEFAULT_TIMEOUT, step=5
    )
    retries = st.number_input(
        "Retries on timeout", min_value=0, value=DEFAULT_RETRIES, step=1
    )

models = fetch_models(api_url, timeout)
if not models:
    models = ["model", "rf", "gbr"]
selected_model = st.selectbox("Model", options=models, index=0)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Single prediction")
    with st.form("single_prediction"):
        form_values = {
            "Store": st.number_input("Store", min_value=1, value=1, step=1),
            "Dept": st.number_input("Dept", min_value=1, value=1, step=1),
            "IsHoliday": st.checkbox("IsHoliday"),
            "Temperature": st.number_input("Temperature", value=42.31),
            "Fuel_Price": st.number_input("Fuel_Price", value=2.572),
            "MarkDown1": st.number_input("MarkDown1", value=0.0),
            "MarkDown2": st.number_input("MarkDown2", value=0.0),
            "MarkDown3": st.number_input("MarkDown3", value=0.0),
            "MarkDown4": st.number_input("MarkDown4", value=0.0),
            "MarkDown5": st.number_input("MarkDown5", value=0.0),
            "CPI": st.number_input("CPI", value=211.0963582),
            "Unemployment": st.number_input("Unemployment", value=8.106),
            "Type": st.selectbox("Type", options=["A", "B", "C"], index=0),
            "Size": st.number_input("Size", value=151315),
            "Year": st.number_input("Year", min_value=2000, value=2010, step=1),
            "Month": st.number_input("Month", min_value=1, max_value=12, value=2, step=1),
            "Week": st.number_input("Week", min_value=1, max_value=53, value=5, step=1),
        }
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "records": [build_record(form_values)],
            "model_name": selected_model,
        }
        status, data, latency_ms = call_api(api_url, payload, timeout, retries)
        if status == 200:
            st.success(
                f"Prediction ({data.get('model_name', selected_model)}): {data['predictions'][0]:.2f}"
            )
            st.caption(f"Latency: {latency_ms:.2f} ms")
        else:
            st.error(f"Error {status}: {data}")

with col_right:
    st.subheader("Batch prediction")
    st.write("Upload a CSV with the required feature columns.")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            st.write(df.head())
            if st.button("Predict batch"):
                records = df[FEATURE_COLUMNS].to_dict(orient="records")
                payload = {
                    "records": records,
                    "model_name": selected_model,
                }
                status, data, latency_ms = call_api(api_url, payload, timeout, retries)
                if status == 200:
                    preds = data["predictions"]
                    df_out = df.copy()
                    df_out["prediction"] = preds
                    st.success(f"Generated {len(preds)} predictions.")
                    st.caption(f"Latency: {latency_ms:.2f} ms")
                    st.dataframe(df_out.head())
                    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions", csv_bytes, "predictions.csv")
                else:
                    st.error(f"Error {status}: {data}")

st.divider()
if st.button("Check API health"):
    try:
        health = requests.get(f"{api_url}/health", timeout=10).json()
        st.write(health)
    except requests.RequestException as exc:
        st.error(str(exc))
