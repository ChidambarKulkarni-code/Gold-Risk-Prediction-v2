
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

st.set_page_config(page_title="Gold Price Prediction Dashboard", layout="wide")

SEED = 42
YAHOO_TICKERS = {
    "gold_usd_oz": "GC=F",
    "usd_inr": "INR=X",
    "crude_oil": "CL=F",
    "silver_usd_oz": "SI=F",
    "nifty50": "^NSEI",
    "sp500": "^GSPC",
}

st.title("🟡 Gold Price Prediction Dashboard")
st.caption("Notebook-aligned pipeline: Yahoo data → preprocessing → leakage-safe features → strict time split → multi-model comparison → ensemble → next-day price forecast.")

st.sidebar.header("⚙️ Controls")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2004-01-01"))
train_ratio = st.sidebar.slider("Train Ratio", 0.50, 0.85, 0.70, 0.05)
val_ratio = st.sidebar.slider("Validation Ratio", 0.05, 0.30, 0.15, 0.05)
run_btn = st.sidebar.button("Run Pipeline")

if train_ratio + val_ratio >= 0.95:
    st.sidebar.error("Train Ratio + Validation Ratio must be less than 0.95")
    st.stop()

def _clean_download(ticker: str, series_name: str, start_date: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date,
        progress=False,
        auto_adjust=True,
        threads=True
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["date", series_name])

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            s = df.iloc[:, 0]
    else:
        s = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

    out = s.rename(series_name).to_frame().reset_index()
    out.columns = ["date", series_name]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values("date").reset_index(drop=True)
    out[series_name] = pd.to_numeric(out[series_name], errors="coerce")
    return out

@st.cache_data(show_spinner=False)
def fetch_market_data(start_date_str: str) -> pd.DataFrame:
    market_frames = {}
    for name, ticker in YAHOO_TICKERS.items():
        market_frames[name] = _clean_download(ticker, name, start_date_str)

    if market_frames["gold_usd_oz"].empty or market_frames["usd_inr"].empty:
        raise ValueError("Gold USD or USD/INR data could not be downloaded from Yahoo Finance.")

    master_df = market_frames["gold_usd_oz"][["date"]].copy().sort_values("date").reset_index(drop=True)
    for name, frame in market_frames.items():
        master_df = master_df.merge(frame, on="date", how="left")

    raw_df = master_df.copy().sort_values("date").reset_index(drop=True)
    return raw_df

def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    market_cols = ["gold_usd_oz", "usd_inr", "crude_oil", "silver_usd_oz", "nifty50", "sp500"]
    for col in market_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in market_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    df["gold_inr_10g"] = (df["gold_usd_oz"] * df["usd_inr"] / 31.1034768) * 10.0

    required_cols = ["gold_usd_oz", "usd_inr", "crude_oil", "silver_usd_oz", "sp500", "gold_inr_10g"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    if "nifty50" in df.columns:
        df["nifty50"] = df["nifty50"].ffill().bfill()

    return df

def build_features(clean_df: pd.DataFrame):
    feat = clean_df.copy().sort_values("date").reset_index(drop=True)

    feat["target_return"] = feat["gold_inr_10g"].pct_change(1).shift(-1)
    feat["target_price"] = feat["gold_inr_10g"].shift(-1)

    feat["gold_return_1d"] = feat["gold_inr_10g"].pct_change(1)
    feat["gold_usd_ret_1d"] = feat["gold_usd_oz"].pct_change(1)
    feat["usd_inr_ret_1d"] = feat["usd_inr"].pct_change(1)
    feat["crude_ret_1d"] = feat["crude_oil"].pct_change(1)
    feat["silver_ret_1d"] = feat["silver_usd_oz"].pct_change(1)
    feat["nifty_ret_1d"] = feat["nifty50"].pct_change(1)
    feat["sp500_ret_1d"] = feat["sp500"].pct_change(1)

    return_cols = [
        "gold_return_1d",
        "gold_usd_ret_1d",
        "usd_inr_ret_1d",
        "crude_ret_1d",
        "silver_ret_1d",
        "nifty_ret_1d",
        "sp500_ret_1d",
    ]

    for col in return_cols:
        for lag in [1, 2, 3, 5, 10]:
            feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

    past_gold_ret = feat["gold_return_1d"].shift(1)
    past_usd_ret = feat["usd_inr_ret_1d"].shift(1)
    past_crude_ret = feat["crude_ret_1d"].shift(1)
    past_silver_ret = feat["silver_ret_1d"].shift(1)
    past_nifty_ret = feat["nifty_ret_1d"].shift(1)
    past_sp500_ret = feat["sp500_ret_1d"].shift(1)

    for window in [5, 10, 21, 63]:
        feat[f"gold_ret_mean_{window}d"] = past_gold_ret.rolling(window).mean()
        feat[f"gold_ret_std_{window}d"] = past_gold_ret.rolling(window).std()
        feat[f"usd_ret_mean_{window}d"] = past_usd_ret.rolling(window).mean()
        feat[f"crude_ret_mean_{window}d"] = past_crude_ret.rolling(window).mean()
        feat[f"silver_ret_mean_{window}d"] = past_silver_ret.rolling(window).mean()
        feat[f"nifty_ret_mean_{window}d"] = past_nifty_ret.rolling(window).mean()
        feat[f"sp500_ret_mean_{window}d"] = past_sp500_ret.rolling(window).mean()

    # past price trend features using only historical information
    feat["gold_ma_5"] = feat["gold_inr_10g"].shift(1).rolling(5).mean()
    feat["gold_ma_10"] = feat["gold_inr_10g"].shift(1).rolling(10).mean()
    feat["gold_ma_21"] = feat["gold_inr_10g"].shift(1).rolling(21).mean()
    feat["gold_vol_21"] = feat["gold_inr_10g"].shift(1).rolling(21).std()

    feature_cols = [c for c in feat.columns if c not in [
        "date", "target_return", "target_price",
        "gold_usd_oz", "usd_inr", "crude_oil", "silver_usd_oz", "nifty50", "sp500"
    ]]

    model_df = feat.dropna(subset=feature_cols + ["target_return", "target_price", "gold_inr_10g"]).reset_index(drop=True)
    return model_df, feature_cols

def ret_to_price(current_price, predicted_return):
    return np.array(current_price) * (1.0 + np.array(predicted_return))

def price_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

@st.cache_data(show_spinner=False)
def run_pipeline(start_date_str: str, train_ratio_value: float, val_ratio_value: float):
    raw_df = fetch_market_data(start_date_str)
    clean_df = preprocess_data(raw_df)
    model_df, feature_cols = build_features(clean_df)

    if len(model_df) < 250:
        raise ValueError("Too few usable rows after feature engineering. Use an earlier start date to give the notebook-style pipeline enough history.")

    X_all = model_df[feature_cols].copy()
    y_ret = model_df["target_return"].copy()
    y_price = model_df["target_price"].copy()
    cp_all = model_df["gold_inr_10g"].copy()
    dates_all = model_df["date"].copy()

    n = len(model_df)
    train_end = int(n * train_ratio_value)
    val_end = int(n * (train_ratio_value + val_ratio_value))

    X_train_raw = X_all.iloc[:train_end].copy()
    X_val_raw = X_all.iloc[train_end:val_end].copy()
    X_test_raw = X_all.iloc[val_end:].copy()

    y_train_ret = y_ret.iloc[:train_end].copy()
    y_val_ret = y_ret.iloc[train_end:val_end].copy()
    y_test_ret = y_ret.iloc[val_end:].copy()

    y_train_price = y_price.iloc[:train_end].copy()
    y_val_price = y_price.iloc[train_end:val_end].copy()
    y_test_price = y_price.iloc[val_end:].copy()

    cp_train = cp_all.iloc[:train_end].copy()
    cp_val = cp_all.iloc[train_end:val_end].copy()
    cp_test = cp_all.iloc[val_end:].copy()

    dates_train = dates_all.iloc[:train_end].copy()
    dates_val = dates_all.iloc[train_end:val_end].copy()
    dates_test = dates_all.iloc[val_end:].copy()

    if min(len(X_train_raw), len(X_val_raw), len(X_test_raw)) < 30:
        raise ValueError("One of the train/validation/test segments is too small. Use an earlier start date or adjust the ratios.")

    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train_raw)
    X_val_std = std_scaler.transform(X_val_raw)
    X_test_std = std_scaler.transform(X_test_raw)

    trained_models = {}
    val_pred_price = {}
    test_pred_price = {}
    val_rows = []
    test_rows = []

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_std, y_train_ret)
    lr_val_price = ret_to_price(cp_val.values, lr.predict(X_val_std))
    lr_test_price = ret_to_price(cp_test.values, lr.predict(X_test_std))
    trained_models["LinearRegression"] = lr
    val_pred_price["LinearRegression"] = lr_val_price
    test_pred_price["LinearRegression"] = lr_test_price

    m = price_metrics(y_val_price.values, lr_val_price); m["Model"] = "LinearRegression"; val_rows.append(m)
    m = price_metrics(y_test_price.values, lr_test_price); m["Model"] = "LinearRegression"; test_rows.append(m)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=5,
        min_samples_leaf=20, min_samples_split=40,
        max_features="sqrt", random_state=SEED, n_jobs=-1
    )
    rf.fit(X_train_raw, y_train_ret)
    rf_val_price = ret_to_price(cp_val.values, rf.predict(X_val_raw))
    rf_test_price = ret_to_price(cp_test.values, rf.predict(X_test_raw))
    trained_models["RandomForest"] = rf
    val_pred_price["RandomForest"] = rf_val_price
    test_pred_price["RandomForest"] = rf_test_price

    m = price_metrics(y_val_price.values, rf_val_price); m["Model"] = "RandomForest"; val_rows.append(m)
    m = price_metrics(y_test_price.values, rf_test_price); m["Model"] = "RandomForest"; test_rows.append(m)

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        min_child_weight=10, subsample=0.80, colsample_bytree=0.80,
        gamma=1.0, reg_alpha=0.5, reg_lambda=5.0,
        objective="reg:squarederror", random_state=SEED, n_jobs=2
    )
    xgb.fit(X_train_raw, y_train_ret)
    xgb_val_price = ret_to_price(cp_val.values, xgb.predict(X_val_raw))
    xgb_test_price = ret_to_price(cp_test.values, xgb.predict(X_test_raw))
    trained_models["XGBoost"] = xgb
    val_pred_price["XGBoost"] = xgb_val_price
    test_pred_price["XGBoost"] = xgb_test_price

    m = price_metrics(y_val_price.values, xgb_val_price); m["Model"] = "XGBoost"; val_rows.append(m)
    m = price_metrics(y_test_price.values, xgb_test_price); m["Model"] = "XGBoost"; test_rows.append(m)

    # LightGBM
    lgbm = LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15,
        max_depth=3, min_child_samples=40, subsample=0.80,
        colsample_bytree=0.80, reg_alpha=0.5, reg_lambda=5.0,
        random_state=SEED, verbose=-1
    )
    lgbm.fit(X_train_raw, y_train_ret)
    lgbm_val_price = ret_to_price(cp_val.values, lgbm.predict(X_val_raw))
    lgbm_test_price = ret_to_price(cp_test.values, lgbm.predict(X_test_raw))
    trained_models["LightGBM"] = lgbm
    val_pred_price["LightGBM"] = lgbm_val_price
    test_pred_price["LightGBM"] = lgbm_test_price

    m = price_metrics(y_val_price.values, lgbm_val_price); m["Model"] = "LightGBM"; val_rows.append(m)
    m = price_metrics(y_test_price.values, lgbm_test_price); m["Model"] = "LightGBM"; test_rows.append(m)

    val_results_df = pd.DataFrame(val_rows)[["Model", "RMSE", "MAE", "MAPE", "R2"]].sort_values("RMSE").reset_index(drop=True)
    test_results_df = pd.DataFrame(test_rows)[["Model", "RMSE", "MAE", "MAPE", "R2"]].sort_values("RMSE").reset_index(drop=True)

    tabular_model_names = ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"]
    eligible = val_results_df[(val_results_df["Model"].isin(tabular_model_names)) & (val_results_df["R2"] > 0)].copy()
    if len(eligible) == 0:
        eligible = val_results_df[val_results_df["Model"].isin(tabular_model_names)].copy()

    best_rmse = eligible["RMSE"].min()
    eligible = eligible[eligible["RMSE"] <= best_rmse * 1.5].copy()
    eligible = eligible.sort_values("RMSE").head(3).copy()
    eligible["inv_rmse"] = 1.0 / eligible["RMSE"]
    eligible["Weight"] = eligible["inv_rmse"] / eligible["inv_rmse"].sum()

    ensemble_weights_df = eligible[["Model", "RMSE", "Weight"]].sort_values("Weight", ascending=False).reset_index(drop=True)

    ens_val_price = np.zeros(len(y_val_price))
    ens_test_price = np.zeros(len(y_test_price))
    for _, row in ensemble_weights_df.iterrows():
        ens_val_price += row["Weight"] * np.array(val_pred_price[row["Model"]])
        ens_test_price += row["Weight"] * np.array(test_pred_price[row["Model"]])

    ensemble_val_metrics = price_metrics(y_val_price.values, ens_val_price)
    ensemble_test_metrics = price_metrics(y_test_price.values, ens_test_price)

    latest_row = model_df[feature_cols].dropna().tail(1).copy()
    latest_date = model_df.loc[latest_row.index[0], "date"]
    latest_price = float(model_df.loc[latest_row.index[0], "gold_inr_10g"])
    next_day = pd.bdate_range(start=latest_date, periods=2)[-1]

    next_predictions = {}
    for model_name in tabular_model_names:
        model = trained_models[model_name]
        if model_name == "LinearRegression":
            X_input = std_scaler.transform(latest_row)
        else:
            X_input = latest_row.values
        pred_ret = float(model.predict(X_input)[0])
        pred_price = float(latest_price * (1.0 + pred_ret))
        next_predictions[model_name] = pred_price

    ensemble_next_pred = 0.0
    for _, row in ensemble_weights_df.iterrows():
        ensemble_next_pred += row["Weight"] * next_predictions[row["Model"]]

    interval_model = ensemble_weights_df.iloc[0]["Model"]
    val_residuals = np.array(y_val_price.values) - np.array(val_pred_price[interval_model]).reshape(-1)
    lower_resid = float(np.percentile(val_residuals, 5))
    upper_resid = float(np.percentile(val_residuals, 95))
    ensemble_lower = ensemble_next_pred + lower_resid
    ensemble_upper = ensemble_next_pred + upper_resid

    test_eval_df = pd.DataFrame({
        "date": pd.to_datetime(dates_test.values),
        "ActualPrice": np.array(y_test_price.values),
        "PredictedPrice": np.array(ens_test_price),
    })
    test_eval_df["Lower90"] = test_eval_df["PredictedPrice"] + lower_resid
    test_eval_df["Upper90"] = test_eval_df["PredictedPrice"] + upper_resid

    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "model_df": model_df,
        "feature_cols": feature_cols,
        "val_results_df": val_results_df,
        "test_results_df": test_results_df,
        "ensemble_weights_df": ensemble_weights_df,
        "ensemble_val_metrics": ensemble_val_metrics,
        "ensemble_test_metrics": ensemble_test_metrics,
        "latest_price": latest_price,
        "latest_date": latest_date,
        "next_day": next_day,
        "next_predictions": next_predictions,
        "ensemble_next_pred": ensemble_next_pred,
        "ensemble_lower": ensemble_lower,
        "ensemble_upper": ensemble_upper,
        "test_eval_df": test_eval_df,
        "train_rows": len(X_train_raw),
        "val_rows": len(X_val_raw),
        "test_rows": len(X_test_raw),
        "train_start": dates_train.min(),
        "train_end_date": dates_train.max(),
        "val_start": dates_val.min(),
        "val_end_date": dates_val.max(),
        "test_start": dates_test.min(),
        "test_end_date": dates_test.max(),
    }

if run_btn or "notebook_app_result" not in st.session_state:
    try:
        st.session_state["notebook_app_result"] = run_pipeline(str(start_date), float(train_ratio), float(val_ratio))
    except Exception as e:
        st.error(str(e))
        st.stop()

res = st.session_state["notebook_app_result"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Price (INR/10g)", f"₹{res['latest_price']:,.2f}")
c2.metric("Ensemble Forecast", f"₹{res['ensemble_next_pred']:,.2f}")
c3.metric("Expected Change", f"{((res['ensemble_next_pred']/res['latest_price'])-1)*100:,.2f}%")
c4.metric("Forecast Date", str(pd.to_datetime(res["next_day"]).date()))

st.markdown("---")

col_a, col_b = st.columns([1.1, 1])
with col_a:
    st.subheader("📊 Test Set Performance")
    test_df = res["test_results_df"].copy()
    ens_row = pd.DataFrame([{
        "Model": "Ensemble",
        "RMSE": res["ensemble_test_metrics"]["RMSE"],
        "MAE": res["ensemble_test_metrics"]["MAE"],
        "MAPE": res["ensemble_test_metrics"]["MAPE"],
        "R2": res["ensemble_test_metrics"]["R2"],
    }])
    st.dataframe(pd.concat([test_df, ens_row], ignore_index=True).sort_values("RMSE"), use_container_width=True)

with col_b:
    st.subheader("⚖️ Ensemble Weights")
    st.dataframe(res["ensemble_weights_df"], use_container_width=True)

st.subheader("🔮 Next-Day Predictions by Model")
next_pred_df = pd.DataFrame(
    [{"Model": k, "PredictedPrice": v} for k, v in res["next_predictions"].items()]
)
next_pred_df = pd.concat([
    next_pred_df,
    pd.DataFrame([{"Model": "Ensemble", "PredictedPrice": res["ensemble_next_pred"]}])
], ignore_index=True)
st.dataframe(next_pred_df, use_container_width=True)

st.info(
    f"90% interval around ensemble forecast: ₹{res['ensemble_lower']:,.2f} to ₹{res['ensemble_upper']:,.2f}"
)

view = st.selectbox(
    "Choose view",
    ["Ensemble vs Actual (Test)", "Historical Gold INR Trend", "Pipeline Summary", "Feature List"]
)

if view == "Ensemble vs Actual (Test)":
    chart_df = res["test_eval_df"].set_index("date")[["ActualPrice", "PredictedPrice", "Lower90", "Upper90"]]
    st.line_chart(chart_df)
    st.caption("Notebook-aligned test view using ensemble predictions and a 90% interval built from validation residuals.")

elif view == "Historical Gold INR Trend":
    hist = res["clean_df"].set_index("date")[["gold_inr_10g"]]
    st.line_chart(hist)
    st.caption("Derived India gold price in INR per 10 grams from Yahoo Gold Futures and USD/INR.")

elif view == "Pipeline Summary":
    summary_df = pd.DataFrame({
        "Segment": ["Train", "Validation", "Test"],
        "Rows": [res["train_rows"], res["val_rows"], res["test_rows"]],
        "Start": [res["train_start"], res["val_start"], res["test_start"]],
        "End": [res["train_end_date"], res["val_end_date"], res["test_end_date"]],
    })
    st.dataframe(summary_df, use_container_width=True)
    st.write(f"Usable feature rows: {len(res['model_df']):,}")
    st.write(f"Feature count: {len(res['feature_cols']):,}")

else:
    st.write(res["feature_cols"])

with st.expander("What matches the notebook source of truth"):
    st.write(
        "- Same Yahoo tickers and INR/10g conversion\n"
        "- Same target design: predict next-day return, evaluate next-day price\n"
        "- Same leakage-safe lag and rolling features\n"
        "- Same strict time-based train/validation/test split\n"
        "- Same model family for the deployable tabular models: Linear Regression, Random Forest, XGBoost, LightGBM\n"
        "- Same validation-weighted ensemble idea\n"
        "- Same next-day forecast flow using latest complete feature row"
    )
