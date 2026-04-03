import streamlit as st
import requests

# IMPORTANT: Replace with your real GitHub raw URL
BASE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/data_outputs/"

st.set_page_config(page_title="Gold Price Dashboard", layout="wide")

def load_json(file_name: str):
    url = BASE_URL + file_name
    try:
        response = requests.get(url, timeout=20)
    except requests.RequestException as e:
        st.error(f"Network error while loading {file_name}")
        st.write("URL:", url)
        st.write("Error:", str(e))
        st.stop()

    if response.status_code != 200:
        st.error(f"Could not load {file_name}")
        st.write("URL:", url)
        st.write("Status code:", response.status_code)
        st.write("Response preview:")
        st.code(response.text[:500] if response.text else "No response text")
        st.stop()

    try:
        return response.json()
    except ValueError:
        st.error(f"{file_name} is not valid JSON")
        st.write("URL:", url)
        st.write("Response preview:")
        st.code(response.text[:500] if response.text else "Empty response")
        st.stop()

st.title("Gold Price Prediction Dashboard")

data = load_json("latest_prediction.json")
meta = load_json("metadata.json")

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"₹{data.get('current_price', 'N/A')}")
col2.metric("Predicted Price", f"₹{data.get('predicted_price', 'N/A')}")

pred_ret = data.get("predicted_return_pct", 0)
try:
    pred_ret_display = f"{float(pred_ret):.2f}%"
except Exception:
    pred_ret_display = str(pred_ret)

col3.metric("Change %", pred_ret_display)

st.write("Forecast Date:", data.get("forecast_date", "N/A"))
st.write("Model Used:", data.get("model_used", "N/A"))
st.write("Last Updated:", meta.get("last_updated", "N/A"))

with st.expander("Debug info"):
    st.write("Base URL:", BASE_URL)
    st.write("latest_prediction.json URL:", BASE_URL + "latest_prediction.json")
    st.write("metadata.json URL:", BASE_URL + "metadata.json")

st.warning("For educational and analytical use only. Not financial advice.")
