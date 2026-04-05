# ============================================
# Gold Price Prediction Dashboard (Streamlit)
# ============================================

# This Streamlit app is VIEW ONLY.
# It only reads already-saved files and displays them.
# It does NOT train any model.
# It does NOT make predictions.
# It does NOT recalculate metrics.

import json
import os

import pandas as pd
import streamlit as st


# --------------------------------------------
# Page title
# --------------------------------------------
st.title("Gold Price Prediction Dashboard")


# --------------------------------------------
# Helper function to check whether a file exists
# --------------------------------------------
def check_file(file_name: str) -> bool:
    """
    Check if a required file exists in the same folder as app.py.
    If the file is missing, show a Streamlit error message.
    """
    if not os.path.exists(file_name):
        st.error(f"Missing file: {file_name}")
        return False
    return True


# --------------------------------------------
# Helper function to clean feature names
# --------------------------------------------
def clean_feature_name(feature_name: str) -> str:
    """
    Convert technical feature names into cleaner display names.
    Example: gold_return_1d -> Gold Return 1D
    """
    text = str(feature_name).replace("_", " ").strip()
    return text.title()


# --------------------------------------------
# File names
# --------------------------------------------
summary_file = "final_summary.json"
feature_file = "feature_importance.csv"
actual_pred_file = "actual_vs_predicted.csv"
strategy_file = "strategy_performance.csv"


# --------------------------------------------
# Check that all required files are present
# --------------------------------------------
all_files_present = all(
    [
        check_file(summary_file),
        check_file(feature_file),
        check_file(actual_pred_file),
        check_file(strategy_file),
    ]
)

# Stop the app if any file is missing
if not all_files_present:
    st.stop()


# --------------------------------------------
# Load data files
# --------------------------------------------

# Load JSON summary file
with open(summary_file, "r", encoding="utf-8") as file:
    summary = json.load(file)

# Load CSV files
feature_df = pd.read_csv(feature_file)
actual_pred_df = pd.read_csv(actual_pred_file)
strategy_df = pd.read_csv(strategy_file)


# --------------------------------------------
# Main Prediction Section
# --------------------------------------------
st.subheader("Main Prediction")

predicted_price = summary.get("predicted_price", "N/A")
direction = summary.get("direction", "N/A")
probability = summary.get("probability", "N/A")
predicted_date = summary.get("predicted_date", "N/A")
lower_bound = summary.get("lower_bound", "N/A")
upper_bound = summary.get("upper_bound", "N/A")

col1, col2, col3 = st.columns(3)

if isinstance(predicted_price, (int, float)):
    col1.metric("Predicted Price", f"₹{predicted_price:,.2f}")
else:
    col1.metric("Predicted Price", str(predicted_price))

col2.metric("Direction", str(direction))

if isinstance(probability, (int, float)):
    col3.metric("Probability", f"{probability:.2f}%")
else:
    col3.metric("Probability", str(probability))

st.write(f"**Predicted Date:** {predicted_date}")

if isinstance(lower_bound, (int, float)) and isinstance(upper_bound, (int, float)):
    st.write(f"**Prediction Range:** ₹{lower_bound:,.2f} to ₹{upper_bound:,.2f}")
else:
    st.write(f"**Prediction Range:** {lower_bound} to {upper_bound}")

st.markdown("---")


# --------------------------------------------
# Strategy Section
# --------------------------------------------
st.subheader("Strategy")

signal = summary.get("signal", "N/A")
signal_explanation = summary.get("signal_explanation", "")

# If explanation is missing or too generic, show a better display message
default_explanations = {
    "BUY": "BUY signal is generated because the forecast indicates upward movement with strong confidence.",
    "SELL": "SELL signal is generated because the forecast indicates downward movement with strong confidence.",
    "HOLD": "HOLD signal is generated because the expected movement is limited or unclear.",
}

if not signal_explanation or "based on the project forecast" in str(signal_explanation).lower():
    signal_explanation = default_explanations.get(str(signal).upper(), "Signal explanation not available.")

st.write(f"**Signal:** {signal}")
st.write(signal_explanation)

st.markdown("---")


# --------------------------------------------
# Risk Section
# --------------------------------------------
st.subheader("Risk")

risk_level = summary.get("risk_level", "N/A")
volatility = summary.get("volatility", "N/A")

st.write(f"**Risk Level:** {risk_level}")

# Fix NaN display issue
if pd.isna(volatility):
    st.write("**Volatility:** Not available")
elif isinstance(volatility, (int, float)):
    st.write(f"**Volatility:** {volatility:.6f}")
else:
    st.write(f"**Volatility:** {volatility}")

st.markdown("---")


# --------------------------------------------
# Interpretation Section
# --------------------------------------------
st.subheader("Interpretation")

interpretation_text = summary.get("interpretation_text", "No interpretation available.")
st.write(interpretation_text)

st.markdown("---")


# --------------------------------------------
# Feature Importance Section
# --------------------------------------------
st.subheader("Feature Importance")

required_feature_columns = {"Feature", "Importance"}

if required_feature_columns.issubset(feature_df.columns):
    # Keep only needed columns
    feature_df = feature_df[["Feature", "Importance"]].copy()

    # Clean feature names for better display
    feature_df["Feature"] = feature_df["Feature"].apply(clean_feature_name)

    # Sort features from highest importance to lowest
    top_features_df = feature_df.sort_values(by="Importance", ascending=False)

    st.write("Top features from the notebook output:")

    # Show only top 10 in the table for cleaner display
    st.write(top_features_df.head(10))

    # Show top 10 in chart
    chart_df = top_features_df.head(10).set_index("Feature")
    st.bar_chart(chart_df["Importance"])
else:
    st.error("feature_importance.csv must contain these columns: Feature, Importance")

st.markdown("---")


# --------------------------------------------
# Charts Section - Actual vs Predicted
# --------------------------------------------
st.subheader("Actual vs Predicted")

required_actual_pred_columns = {"date", "actual_price", "predicted_price"}

if required_actual_pred_columns.issubset(actual_pred_df.columns):
    actual_pred_df = actual_pred_df.copy()

    # Convert date column safely
    actual_pred_df["date"] = pd.to_datetime(actual_pred_df["date"], errors="coerce")
    actual_pred_df = actual_pred_df.dropna(subset=["date"])

    # Convert values to numeric safely
    actual_pred_df["actual_price"] = pd.to_numeric(actual_pred_df["actual_price"], errors="coerce")
    actual_pred_df["predicted_price"] = pd.to_numeric(actual_pred_df["predicted_price"], errors="coerce")
    actual_pred_df = actual_pred_df.dropna(subset=["actual_price", "predicted_price"])

    # Simple logic to label chart correctly
    # If values are very small, they are likely returns rather than prices
    max_abs_value = actual_pred_df[["actual_price", "predicted_price"]].abs().max().max()

    if max_abs_value < 1000:
        st.write("Showing actual vs predicted returns from saved notebook output.")
    else:
        st.write("Showing actual vs predicted prices from saved notebook output.")

    actual_pred_chart = actual_pred_df.set_index("date")[["actual_price", "predicted_price"]]
    st.line_chart(actual_pred_chart)
else:
    st.error(
        "actual_vs_predicted.csv must contain these columns: date, actual_price, predicted_price"
    )

st.markdown("---")


# --------------------------------------------
# Charts Section - Strategy Performance
# --------------------------------------------
st.subheader("Strategy Performance")

required_strategy_columns = {"date", "strategy_cumulative", "buy_hold_cumulative"}

if required_strategy_columns.issubset(strategy_df.columns):
    strategy_df = strategy_df.copy()

    # Convert date column safely
    strategy_df["date"] = pd.to_datetime(strategy_df["date"], errors="coerce")
    strategy_df = strategy_df.dropna(subset=["date"])

    # Convert values to numeric safely
    strategy_df["strategy_cumulative"] = pd.to_numeric(strategy_df["strategy_cumulative"], errors="coerce")
    strategy_df["buy_hold_cumulative"] = pd.to_numeric(strategy_df["buy_hold_cumulative"], errors="coerce")
    strategy_df = strategy_df.dropna(subset=["strategy_cumulative", "buy_hold_cumulative"])

    st.write("Cumulative performance comparison of strategy vs buy-and-hold.")
    strategy_chart = strategy_df.set_index("date")[["strategy_cumulative", "buy_hold_cumulative"]]
    st.line_chart(strategy_chart)
else:
    st.error(
        "strategy_performance.csv must contain these columns: date, strategy_cumulative, buy_hold_cumulative"
    )
