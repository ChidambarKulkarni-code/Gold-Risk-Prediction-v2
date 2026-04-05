# ============================================
# Gold Price Prediction Dashboard (10/10 Version)
# ============================================

# IMPORTANT:
# This Streamlit app is VIEW ONLY.
# It only reads saved output files from the notebook.
# It does NOT train any model.
# It does NOT make any prediction.
# It does NOT recalculate model metrics.
# It only displays saved results in a cleaner way.

import json
import os

import pandas as pd
import streamlit as st


# --------------------------------------------
# Basic page setup
# --------------------------------------------
st.title("Gold Price Prediction Dashboard")
st.write("This dashboard is view-only and displays saved notebook outputs.")


# --------------------------------------------
# File names
# --------------------------------------------
summary_file = "final_summary.json"
feature_file = "feature_importance.csv"
actual_pred_file = "actual_vs_predicted.csv"
strategy_file = "strategy_performance.csv"


# --------------------------------------------
# Helper function: check if file exists
# --------------------------------------------
def check_file(file_name: str) -> bool:
    """
    Check whether a required file is present.
    If missing, show a clear error in Streamlit.
    """
    if not os.path.exists(file_name):
        st.error(f"Missing file: {file_name}")
        return False
    return True


# --------------------------------------------
# Helper function: safely load JSON
# --------------------------------------------
def load_json_file(file_name: str):
    """
    Load a JSON file safely.
    If there is an issue, show an error message and stop the app.
    """
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        st.error(f"Could not read {file_name}: {error}")
        st.stop()


# --------------------------------------------
# Helper function: safely load CSV
# --------------------------------------------
def load_csv_file(file_name: str):
    """
    Load a CSV file safely.
    If there is an issue, show an error message and stop the app.
    """
    try:
        return pd.read_csv(file_name)
    except Exception as error:
        st.error(f"Could not read {file_name}: {error}")
        st.stop()


# --------------------------------------------
# Helper function: clean feature names
# --------------------------------------------
def clean_feature_name(feature_name: str) -> str:
    """
    Convert technical feature names into cleaner display names.
    Example:
    gold_return_1d -> Gold Return 1D
    sp500_ret_1d_lag10 -> S&P 500 Return 1D Lag 10
    """
    text = str(feature_name).strip().lower()

    # Replace common abbreviations first
    text = text.replace("sp500", "s&p 500")
    text = text.replace("usd_inr", "usd/inr")
    text = text.replace("gold_usd", "gold usd")

    # Replace words
    text = text.replace("ret", "return")
    text = text.replace("lag", "lag ")

    # Replace underscores with spaces
    text = text.replace("_", " ")

    # Fix spacing issues
    text = " ".join(text.split())

    # Capitalize each word
    text = text.title()

    # Clean final formatting for common finance terms
    text = text.replace("Usd/Inr", "USD/INR")
    text = text.replace("Gold Usd", "Gold USD")
    text = text.replace("S&P 500", "S&P 500")
    text = text.replace("1D", "1D")
    text = text.replace("10D", "10D")

    return text


# --------------------------------------------
# Helper function: format money
# --------------------------------------------
def format_money(value):
    """
    Format number as Indian rupee if numeric.
    Otherwise return the value as text.
    """
    if isinstance(value, (int, float)) and not pd.isna(value):
        return f"₹{value:,.2f}"
    return str(value)


# --------------------------------------------
# Helper function: format percentage
# --------------------------------------------
def format_percent(value):
    """
    Format a number as a percentage.
    Otherwise return the value as text.
    """
    if isinstance(value, (int, float)) and not pd.isna(value):
        return f"{value:.2f}%"
    return str(value)


# --------------------------------------------
# Check required files before running
# --------------------------------------------
all_files_present = all(
    [
        check_file(summary_file),
        check_file(feature_file),
        check_file(actual_pred_file),
        check_file(strategy_file),
    ]
)

if not all_files_present:
    st.stop()


# --------------------------------------------
# Load saved notebook outputs
# --------------------------------------------
summary = load_json_file(summary_file)
feature_df = load_csv_file(feature_file)
actual_pred_df = load_csv_file(actual_pred_file)
strategy_df = load_csv_file(strategy_file)


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
col1.metric("Predicted Price", format_money(predicted_price))
col2.metric("Direction", str(direction))
col3.metric("Probability", format_percent(probability))

st.write(f"**Predicted Date:** {predicted_date}")

if (
    isinstance(lower_bound, (int, float))
    and not pd.isna(lower_bound)
    and isinstance(upper_bound, (int, float))
    and not pd.isna(upper_bound)
):
    st.write(f"**Prediction Range:** {format_money(lower_bound)} to {format_money(upper_bound)}")
else:
    st.write("**Prediction Range:** Not available")

st.markdown("---")


# --------------------------------------------
# Strategy Section
# --------------------------------------------
st.subheader("Strategy")

signal = summary.get("signal", "N/A")
signal_explanation = summary.get("signal_explanation", "")
direction_text = str(direction).upper()

# If explanation is weak or missing, use a cleaner default display line
if not signal_explanation or "based on the project forecast" in str(signal_explanation).lower():
    if str(signal).upper() == "BUY":
        signal_explanation = "BUY signal is generated because the saved forecast indicates upward movement with strong confidence."
    elif str(signal).upper() == "SELL":
        signal_explanation = "SELL signal is generated because the saved forecast indicates downward movement with strong confidence."
    elif str(signal).upper() == "HOLD":
        signal_explanation = "HOLD signal is generated because the expected movement is limited or not strong enough."
    else:
        signal_explanation = "Signal explanation is not available in the saved output."

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

if pd.isna(volatility):
    st.write("**Volatility:** Not available in saved output")
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
    feature_df = feature_df[["Feature", "Importance"]].copy()

    # Convert importance to numeric safely
    feature_df["Importance"] = pd.to_numeric(feature_df["Importance"], errors="coerce")
    feature_df = feature_df.dropna(subset=["Importance"])

    # Clean display names
    feature_df["Feature"] = feature_df["Feature"].apply(clean_feature_name)

    # Sort by importance
    feature_df = feature_df.sort_values(by="Importance", ascending=False)

    st.write("Top features from the notebook output:")

    # Show only the top 10 rows for a cleaner dashboard
    display_feature_df = feature_df.head(10).reset_index(drop=True)
    st.write(display_feature_df)

    # Bar chart
    feature_chart_df = display_feature_df.set_index("Feature")
    st.bar_chart(feature_chart_df["Importance"])
else:
    st.error("feature_importance.csv must contain these columns: Feature, Importance")

st.markdown("---")


# --------------------------------------------
# Actual vs Predicted Chart Section
# --------------------------------------------
st.subheader("Actual vs Predicted")

required_actual_pred_columns = {"date", "actual_price", "predicted_price"}

if required_actual_pred_columns.issubset(actual_pred_df.columns):
    actual_pred_df = actual_pred_df.copy()

    # Convert and clean data
    actual_pred_df["date"] = pd.to_datetime(actual_pred_df["date"], errors="coerce")
    actual_pred_df["actual_price"] = pd.to_numeric(actual_pred_df["actual_price"], errors="coerce")
    actual_pred_df["predicted_price"] = pd.to_numeric(actual_pred_df["predicted_price"], errors="coerce")

    actual_pred_df = actual_pred_df.dropna(subset=["date", "actual_price", "predicted_price"])
    actual_pred_df = actual_pred_df.sort_values("date")

    # Detect whether saved values look like prices or returns
    max_abs_value = actual_pred_df[["actual_price", "predicted_price"]].abs().max().max()

    if max_abs_value < 1000:
        st.write("Showing actual vs predicted returns from saved notebook output.")
        chart_df = actual_pred_df.set_index("date")[["actual_price", "predicted_price"]].copy()
        chart_df.columns = ["Actual Return", "Predicted Return"]
    else:
        st.write("Showing actual vs predicted prices from saved notebook output.")
        chart_df = actual_pred_df.set_index("date")[["actual_price", "predicted_price"]].copy()
        chart_df.columns = ["Actual Price", "Predicted Price"]

    st.line_chart(chart_df)
else:
    st.error("actual_vs_predicted.csv must contain these columns: date, actual_price, predicted_price")

st.markdown("---")


# --------------------------------------------
# Strategy Performance Chart Section
# --------------------------------------------
st.subheader("Strategy Performance")

required_strategy_columns = {"date", "strategy_cumulative", "buy_hold_cumulative"}

if required_strategy_columns.issubset(strategy_df.columns):
    strategy_df = strategy_df.copy()

    # Convert and clean data
    strategy_df["date"] = pd.to_datetime(strategy_df["date"], errors="coerce")
    strategy_df["strategy_cumulative"] = pd.to_numeric(strategy_df["strategy_cumulative"], errors="coerce")
    strategy_df["buy_hold_cumulative"] = pd.to_numeric(strategy_df["buy_hold_cumulative"], errors="coerce")

    strategy_df = strategy_df.dropna(subset=["date", "strategy_cumulative", "buy_hold_cumulative"])
    strategy_df = strategy_df.sort_values("date")

    st.write("Cumulative performance comparison of strategy vs buy-and-hold.")

    strategy_chart = strategy_df.set_index("date")[["strategy_cumulative", "buy_hold_cumulative"]].copy()
    strategy_chart.columns = ["Strategy Return", "Buy & Hold Return"]

    st.line_chart(strategy_chart)
else:
    st.error(
        "strategy_performance.csv must contain these columns: date, strategy_cumulative, buy_hold_cumulative"
    )
