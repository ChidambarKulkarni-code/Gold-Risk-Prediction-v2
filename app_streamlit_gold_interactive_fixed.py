
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Gold Price Prediction Dashboard", layout="wide")

st.title("🟡 Gold Price Prediction Dashboard")
st.caption("Interactive gold price prediction app with safer data handling for Streamlit Cloud.")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("⚙️ Controls")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
n_estimators = st.sidebar.slider("Model Complexity (Trees)", 50, 300, 100, step=10)

# -----------------------
# Data loader
# -----------------------
@st.cache_data(show_spinner=False)
def load_data(start):
    # Safer Yahoo download for Streamlit Cloud
    df = yf.download(
        "GC=F",
        start=str(start),
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Handle possible MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if "Close" not in df.columns:
        return pd.DataFrame()

    out = df[["Close"]].copy()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna()

    # Feature engineering
    out["Return"] = out["Close"].pct_change()
    out["Lag1"] = out["Close"].shift(1)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    return out

df = load_data(start_date)

# -----------------------
# Validation
# -----------------------
if df.empty or len(df) < 30:
    st.error("Not enough clean Yahoo Finance data was downloaded to train the model. Try an earlier start date such as 2010-01-01.")
    st.stop()

required_cols = ["Lag1", "Close"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Required columns missing after preprocessing: {missing}")
    st.stop()

# Keep only numeric clean rows
model_df = df[["Lag1", "Close"]].copy()
model_df["Lag1"] = pd.to_numeric(model_df["Lag1"], errors="coerce")
model_df["Close"] = pd.to_numeric(model_df["Close"], errors="coerce")
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

if len(model_df) < 30:
    st.error("Too few valid rows remain after cleaning. Please choose an earlier start date.")
    st.stop()

# -----------------------
# Train / test split
# -----------------------
split = int(len(model_df) * 0.8)

if split <= 5 or len(model_df) - split <= 5:
    st.error("Train/test split created too few rows. Use an earlier start date.")
    st.stop()

train = model_df.iloc[:split].copy()
test = model_df.iloc[split:].copy()

X_train = train[["Lag1"]]
y_train = train["Close"]
X_test = test[["Lag1"]]
y_test = test["Close"]

# Final safety checks before fit
if X_train.empty or y_train.empty:
    st.error("Training data is empty after preprocessing.")
    st.stop()

if X_test.empty or y_test.empty:
    st.error("Test data is empty after preprocessing.")
    st.stop()

if X_train.isna().any().any() or y_train.isna().any():
    st.error("NaN values found in training data.")
    st.stop()

if X_test.isna().any().any() or y_test.isna().any():
    st.error("NaN values found in test data.")
    st.stop()

# -----------------------
# Model
# -----------------------
model = RandomForestRegressor(
    n_estimators=n_estimators,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
next_pred = model.predict(pd.DataFrame({"Lag1": [model_df["Close"].iloc[-1]]}))[0]

# -----------------------
# Top KPIs
# -----------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Data Points", f"{len(model_df):,}")
col2.metric("🧪 Train Rows", f"{len(train):,}")
col3.metric("📉 MAE", f"{mae:,.2f}")
col4.metric("📈 RMSE", f"{rmse:,.2f}")

st.markdown("---")

# -----------------------
# Interactivity
# -----------------------
view = st.selectbox(
    "Choose what to view",
    ["Price Trend", "Prediction vs Actual", "Prediction Table"]
)

if view == "Price Trend":
    st.subheader("📈 Gold Price Trend")
    st.line_chart(model_df["Close"])
    st.caption("This chart shows the historical closing price series used for training and testing.")

elif view == "Prediction vs Actual":
    st.subheader("🤖 Prediction vs Actual")
    chart_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Predicted": preds
        },
        index=test.index
    )
    st.line_chart(chart_df)
    st.caption("The closer the two lines are, the better the model is tracking the actual gold price.")

else:
    st.subheader("📋 Prediction Table")
    pred_table = pd.DataFrame(
        {
            "Date": test.index,
            "Actual": y_test.values,
            "Predicted": preds,
            "Error": y_test.values - preds
        }
    ).reset_index(drop=True)
    st.dataframe(pred_table, use_container_width=True)
    st.download_button(
        "⬇️ Download Predictions CSV",
        pred_table.to_csv(index=False).encode("utf-8"),
        file_name="gold_test_predictions.csv",
        mime="text/csv"
    )

st.markdown("---")
st.subheader("🔮 Next Day Prediction")
st.success(f"Predicted Gold Price: ₹{next_pred:,.2f}")
st.caption("Prediction is based on the previous day's closing price using a Random Forest model.")

with st.expander("ℹ️ Model Details"):
    st.write("Model used: Random Forest Regressor")
    st.write(f"Feature used: previous day's closing price (Lag1)")
    st.write(f"Training period rows: {len(train):,}")
    st.write(f"Testing period rows: {len(test):,}")

st.markdown("---")
