
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Gold Price Prediction Dashboard", layout="wide")

st.title("🟡 Gold Price Prediction Dashboard")
st.caption("Flexible version: works even with short date ranges by adapting features automatically.")

st.sidebar.header("⚙️ Controls")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
test_ratio = st.sidebar.slider("Test Size", 0.10, 0.35, 0.20, 0.05)
run_btn = st.sidebar.button("Run Model")

@st.cache_data(show_spinner=False)
def load_gold_data(start):
    df = yf.download(
        "GC=F",
        start=str(start),
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if "Close" not in df.columns:
        return pd.DataFrame()

    df = df[["Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    return df

def build_features(df):
    data = df.copy()
    data["Lag1"] = data["Close"].shift(1)
    data["Lag2"] = data["Close"].shift(2)
    data["Lag3"] = data["Close"].shift(3)
    data["MA3"] = data["Close"].rolling(3).mean()
    data["MA7"] = data["Close"].rolling(7).mean()
    data["MA14"] = data["Close"].rolling(14).mean()
    data["MA30"] = data["Close"].rolling(30).mean()
    data["Return1"] = data["Close"].pct_change()
    data["Return3"] = data["Close"].pct_change(3)
    data["Volatility3"] = data["Return1"].rolling(3).std()
    data["Volatility7"] = data["Return1"].rolling(7).std()
    data["TrendIndex"] = np.arange(len(data))
    data = data.replace([np.inf, -np.inf], np.nan)
    return data

def choose_feature_set(feature_df):
    candidate_sets = [
        ["Lag1", "Lag2", "Lag3", "MA3", "MA7", "MA14", "MA30", "Return1", "Return3", "Volatility3", "Volatility7", "TrendIndex"],
        ["Lag1", "Lag2", "Lag3", "MA3", "MA7", "MA14", "Return1", "Return3", "Volatility3", "Volatility7", "TrendIndex"],
        ["Lag1", "Lag2", "Lag3", "MA3", "MA7", "Return1", "Return3", "Volatility3", "TrendIndex"],
        ["Lag1", "Lag2", "MA3", "MA7", "Return1", "Volatility3", "TrendIndex"],
        ["Lag1", "Lag2", "MA3", "Return1", "TrendIndex"],
        ["Lag1", "Lag2", "TrendIndex"],
        ["Lag1", "TrendIndex"],
        ["Lag1"]
    ]

    for cols in candidate_sets:
        temp = feature_df[["Close"] + cols].copy()
        temp = temp.replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp) >= 12:
            return temp, cols

    temp = feature_df[["Close", "Lag1"]].copy().replace([np.inf, -np.inf], np.nan).dropna()
    return temp, ["Lag1"]

def train_models(feature_df, feature_cols, test_ratio_value):
    if len(feature_df) < 12:
        raise ValueError("The selected date range is too short. Please choose at least a few weeks of data.")

    split_idx = int(len(feature_df) * (1 - test_ratio_value))
    split_idx = max(8, split_idx)
    split_idx = min(split_idx, len(feature_df) - 4)

    train = feature_df.iloc[:split_idx].copy()
    test = feature_df.iloc[split_idx:].copy()

    if len(train) < 8 or len(test) < 4:
        raise ValueError("Too few train/test rows after split. Please use a slightly earlier start date.")

    X_train = train[feature_cols]
    y_train = train["Close"]
    X_test = test[feature_cols]
    y_test = test["Close"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    rows = []
    preds_dict = {}

    active_models = models if len(train) >= 25 else {"Linear Regression": LinearRegression()}

    for name, model in active_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))

        rows.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae
        })
        preds_dict[name] = {
            "model": model,
            "preds": preds
        }

    metrics_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Model"]
    best_model = preds_dict[best_name]["model"]
    best_preds = preds_dict[best_name]["preds"]

    next_features = feature_df[feature_cols].iloc[[-1]]
    next_pred = float(best_model.predict(next_features)[0])

    return {
        "metrics_df": metrics_df,
        "best_name": best_name,
        "best_model": best_model,
        "best_preds": best_preds,
        "X_test": X_test,
        "y_test": y_test,
        "test": test,
        "next_pred": next_pred,
        "feature_cols": feature_cols,
        "train_rows": len(train),
        "test_rows": len(test),
    }

def run_pipeline(start_date, test_ratio):
    raw_df = load_gold_data(start_date)
    if raw_df.empty:
        raise ValueError("Could not download gold price data from Yahoo Finance for that date range.")

    feature_df = build_features(raw_df)
    usable_df, feature_cols = choose_feature_set(feature_df)

    if len(usable_df) < 12:
        raise ValueError("The selected date range is too short. Please choose at least a few weeks of data.")

    result = train_models(usable_df, feature_cols, test_ratio)
    return raw_df, usable_df, feature_cols, result

if run_btn or "run_once" not in st.session_state:
    st.session_state["run_once"] = True
    try:
        raw_df, df, feature_cols, result = run_pipeline(start_date, test_ratio)
        st.session_state["raw_df"] = raw_df
        st.session_state["df"] = df
        st.session_state["feature_cols"] = feature_cols
        st.session_state["result"] = result
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    if not all(k in st.session_state for k in ["raw_df", "df", "feature_cols", "result"]):
        st.info("Click 'Run Model' in the sidebar.")
        st.stop()

raw_df = st.session_state["raw_df"]
df = st.session_state["df"]
feature_cols = st.session_state["feature_cols"]
result = st.session_state["result"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Downloaded Rows", f"{len(raw_df):,}")
c2.metric("Usable Rows", f"{len(df):,}")
c3.metric("Best Model", result["best_name"])
c4.metric("Next-Day Prediction", f"₹{result['next_pred']:,.2f}")

st.markdown("---")

st.subheader("📊 Model Comparison")
st.dataframe(result["metrics_df"], use_container_width=True)
st.success(f"Best model selected: {result['best_name']}")
st.caption(f"Adaptive features used for this date range: {', '.join(feature_cols)}")

view = st.selectbox(
    "Choose what to view",
    ["Prediction vs Actual", "Historical Price Trend", "Prediction Table"]
)

if view == "Prediction vs Actual":
    chart_df = pd.DataFrame(
        {
            "Actual": result["y_test"].values,
            "Predicted": result["best_preds"]
        },
        index=result["test"].index
    )
    st.subheader(f"📈 {result['best_name']}: Prediction vs Actual")
    st.line_chart(chart_df)
    st.caption("The app automatically reduces feature complexity for short date ranges so the model can still run.")

elif view == "Historical Price Trend":
    st.subheader("📉 Historical Gold Price Trend")
    st.line_chart(raw_df["Close"])
    st.caption("Historical closing prices downloaded from Yahoo Finance (GC=F).")

else:
    pred_table = pd.DataFrame(
        {
            "Date": result["test"].index,
            "Actual": result["y_test"].values,
            "Predicted": result["best_preds"],
            "Error": result["y_test"].values - result["best_preds"]
        }
    ).reset_index(drop=True)
    st.subheader("📋 Test Prediction Table")
    st.dataframe(pred_table, use_container_width=True)
    st.download_button(
        "⬇️ Download Predictions CSV",
        pred_table.to_csv(index=False).encode("utf-8"),
        file_name="gold_test_predictions.csv",
        mime="text/csv"
    )

st.markdown("---")
st.subheader("🔮 Next-Day Prediction")
st.info(f"Predicted Gold Price: ₹{result['next_pred']:,.2f}")

with st.expander("ℹ️ Model Details"):
    st.write(f"Training rows: {result['train_rows']}")
    st.write(f"Testing rows: {result['test_rows']}")
    st.write(f"Features used: {feature_cols}")
    st.write("For short date ranges, the app automatically switches to a simpler feature set so it can still train.")
