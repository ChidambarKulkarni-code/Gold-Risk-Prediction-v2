import streamlit as st
import requests

BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/data_outputs/"

def load_json(file):
    return requests.get(BASE_URL + file).json()

st.set_page_config(page_title="Gold Price Dashboard", layout="wide")

st.title("Gold Price Prediction Dashboard")

data = load_json("latest_prediction.json")
meta = load_json("metadata.json")

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"₹{data['current_price']}")
col2.metric("Predicted Price", f"₹{data['predicted_price']}")
col3.metric("Change %", f"{data['predicted_return_pct']:.2f}%")

st.write("Forecast Date:", data["forecast_date"])
st.write("Model Used:", data["model_used"])
st.write("Last Updated:", meta["last_updated"])

st.warning("For educational purposes only.")
