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

# Show price
if isinstance(predicted_price, (int, float)):
    col1.metric("Predicted Price", f"₹{predicted_price:,.2f}")
else:
    col1.metric("Predicted Price", str(predicted_price))

# Show direction
col2.metric("Direction", str(direction))

# Show probability
if isinstance(probability, (int, float)):
    col3.metric("Probability", f"{probability:.2f}%")
else:
    col3.metric("Probability", str(probability))

st.write(f"**Predicted Date:** {predicted_date}")

if isinstance(lower_bound, (int, float)) and isinstance(upper_bound, (int, float)):
    st.write(f"**Prediction Range:** ₹{lower_bound:,.2f} to ₹{upper_bound:,.2f}")
else:
    st.write(f"**Prediction Range:** {lower_bound} to {upper_bound}")


# --------------------------------------------
# Strategy Section
# --------------------------------------------
st.subheader("Strategy")

signal = summary.get("signal", "N/A")
signal_explanation = summary.get("signal_explanation", "N/A")

st.write(f"**Signal:** {signal}")
st.write(signal_explanation)


# --------------------------------------------
# Risk Section
# --------------------------------------------
st.subheader("Risk")

risk_level = summary.get("risk_level", "N/A")
volatility = summary.get("volatility", "N/A")

st.write(f"**Risk Level:** {risk_level}")
st.write(f"**Volatility:** {volatility}")


# --------------------------------------------
# Interpretation Section
# --------------------------------------------
st.subheader("Interpretation")

interpretation_text = summary.get("interpretation_text", "No interpretation available.")
st.write(interpretation_text)


# --------------------------------------------
# Feature Importance Section
# --------------------------------------------
st.subheader("Feature Importance")

# Make sure required columns exist
required_feature_columns = {"Feature", "Importance"}

if required_feature_columns.issubset(feature_df.columns):
    # Sort features from highest importance to lowest
    top_features_df = feature_df.sort_values(by="Importance", ascending=False)

    # Show the full table
    st.write("Top features from the notebook output:")
    st.write(top_features_df)

    # Show a simple bar chart for the top 10 features
    chart_df = top_features_df.head(10).set_index("Feature")
    st.bar_chart(chart_df["Importance"])
else:
    st.error("feature_importance.csv must contain these columns: Feature, Importance")


# --------------------------------------------
# Charts Section - Actual vs Predicted
# --------------------------------------------
st.subheader("Actual vs Predicted Prices")

required_actual_pred_columns = {"date", "actual_price", "predicted_price"}

if required_actual_pred_columns.issubset(actual_pred_df.columns):
    # Convert date column to datetime
    actual_pred_df["date"] = pd.to_datetime(actual_pred_df["date"], errors="coerce")

    # Remove rows where date could not be read
    actual_pred_df = actual_pred_df.dropna(subset=["date"])

    # Set date as index for line chart
    actual_pred_chart = actual_pred_df.set_index("date")[["actual_price", "predicted_price"]]

    st.line_chart(actual_pred_chart)
else:
    st.error(
        "actual_vs_predicted.csv must contain these columns: date, actual_price, predicted_price"
    )


# --------------------------------------------
# Charts Section - Strategy Performance
# --------------------------------------------
st.subheader("Strategy Performance")

required_strategy_columns = {"date", "strategy_cumulative", "buy_hold_cumulative"}

if required_strategy_columns.issubset(strategy_df.columns):
    # Convert date column to datetime
    strategy_df["date"] = pd.to_datetime(strategy_df["date"], errors="coerce")

    # Remove rows where date could not be read
    strategy_df = strategy_df.dropna(subset=["date"])

    # Set date as index for line chart
    strategy_chart = strategy_df.set_index("date")[
        ["strategy_cumulative", "buy_hold_cumulative"]
    ]

    st.line_chart(strategy_chart)
else:
    st.error(
        "strategy_performance.csv must contain these columns: date, strategy_cumulative, buy_hold_cumulative"
    )
