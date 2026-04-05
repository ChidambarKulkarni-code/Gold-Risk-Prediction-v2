import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Gold Price Prediction Dashboard", layout="wide")

summary_path = "data_outputs/summary.json"
predictions_path = "data_outputs/final_predictions.csv"
metrics_path = "data_outputs/test_metrics.csv"
weights_path = "data_outputs/ensemble_weights.csv"

with open(summary_path, "r") as f:
    summary = json.load(f)

final_predictions = pd.read_csv(predictions_path)
test_metrics = pd.read_csv(metrics_path)
ensemble_weights = pd.read_csv(weights_path)

st.title("🟡 Gold Price Prediction Dashboard")
st.caption("Display-only dashboard. Results are exported from the Colab notebook.")
st.info(summary["data_note"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Derived Gold Price (INR/10g)", f"₹{summary['latest_price']:,.2f}")
col2.metric("Ensemble Forecast", f"₹{summary['ensemble_forecast']:,.2f}")
col3.metric("Expected Change", f"{summary['expected_change_pct']:.2f}%")
col4.metric("Forecast Date", summary["forecast_date"])

st.markdown("---")

st.write(
    f"**90% forecast interval:** ₹{summary['interval_lower']:,.2f} to ₹{summary['interval_upper']:,.2f}"
)

change = summary["expected_change_pct"]

if change > 0.2:
    takeaway = f"The notebook suggests a modest upward move of about {change:.2f}% for the next trading day."
elif change < -0.2:
    takeaway = f"The notebook suggests a modest downward move of about {abs(change):.2f}% for the next trading day."
else:
    takeaway = f"The notebook suggests the next trading day may remain broadly stable, with an expected move of {change:.2f}%."

st.success(takeaway)

st.markdown("---")

st.subheader("📈 Test Forecast vs Actual")

chart_df = final_predictions.copy()
chart_df["date"] = pd.to_datetime(chart_df["date"])
chart_df = chart_df.set_index("date")

st.line_chart(chart_df[["ActualPrice", "PredictedPrice"]])

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Test Metrics")
    st.dataframe(test_metrics, use_container_width=True)

with col_b:
    st.subheader("⚖️ Ensemble Weights")
    st.dataframe(ensemble_weights, use_container_width=True)

st.markdown("---")

with st.expander("Show prediction details"):
    st.dataframe(final_predictions, use_container_width=True)
