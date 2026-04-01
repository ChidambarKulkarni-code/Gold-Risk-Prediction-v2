import warnings
warnings.filterwarnings('ignore')

from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional models
HAS_XGB = True
HAS_LGBM = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
except Exception:
    HAS_LGBM = False

st.set_page_config(page_title='Gold Price Prediction Dashboard', layout='wide')

TICKERS = {
    'gold_usd_oz': 'GC=F',
    'usd_inr': 'INR=X',
    'crude_oil': 'CL=F',
    'silver_usd_oz': 'SI=F',
    'nifty50': '^NSEI',
    'sp500': '^GSPC',
}


def download_yahoo_series(ticker: str, series_name: str, start_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=['date', series_name])

    if isinstance(df.columns, pd.MultiIndex):
        if ('Adj Close', ticker) in df.columns:
            s = df[('Adj Close', ticker)]
        elif ('Close', ticker) in df.columns:
            s = df[('Close', ticker)]
        else:
            s = df.iloc[:, 0]
    else:
        s = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']

    out = s.rename(series_name).to_frame().reset_index()
    out.columns = ['date', series_name]
    out['date'] = pd.to_datetime(out['date']).dt.tz_localize(None)
    return out.sort_values('date').reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fetch_market_data(start_date: str) -> pd.DataFrame:
    market_frames = {name: download_yahoo_series(ticker, name, start_date) for name, ticker in TICKERS.items()}
    if market_frames['gold_usd_oz'].empty or market_frames['usd_inr'].empty:
        raise ValueError('Could not download required Yahoo Finance series.')

    master_df = market_frames['gold_usd_oz'][['date']].copy().sort_values('date').reset_index(drop=True)
    for name, frame in market_frames.items():
        master_df = master_df.merge(frame, on='date', how='left')

    df = master_df.copy().sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
    market_cols = list(TICKERS.keys())
    for col in market_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].ffill()

    df['gold_inr_10g'] = (df['gold_usd_oz'] * df['usd_inr'] / 31.1034768) * 10.0
    df = df.dropna(subset=['gold_usd_oz', 'usd_inr', 'crude_oil', 'silver_usd_oz', 'sp500', 'gold_inr_10g']).reset_index(drop=True)
    if 'nifty50' in df.columns:
        df['nifty50'] = df['nifty50'].ffill().bfill()
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


@st.cache_data(show_spinner=False)
def build_features(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feat = clean_df.copy().sort_values('date').reset_index(drop=True)
    feat['target_return'] = feat['gold_inr_10g'].pct_change(1).shift(-1)
    feat['target_price'] = feat['gold_inr_10g'].shift(-1)

    feat['gold_return_1d'] = feat['gold_inr_10g'].pct_change(1)
    feat['gold_usd_ret_1d'] = feat['gold_usd_oz'].pct_change(1)
    feat['usd_inr_ret_1d'] = feat['usd_inr'].pct_change(1)
    feat['crude_ret_1d'] = feat['crude_oil'].pct_change(1)
    feat['silver_ret_1d'] = feat['silver_usd_oz'].pct_change(1)
    feat['nifty_ret_1d'] = feat['nifty50'].pct_change(1)
    feat['sp500_ret_1d'] = feat['sp500'].pct_change(1)

    return_cols = [
        'gold_return_1d', 'gold_usd_ret_1d', 'usd_inr_ret_1d', 'crude_ret_1d',
        'silver_ret_1d', 'nifty_ret_1d', 'sp500_ret_1d'
    ]

    for col in return_cols:
        for lag in [1, 2, 3, 5, 10]:
            feat[f'{col}_lag{lag}'] = feat[col].shift(lag)

    past_gold_ret = feat['gold_return_1d'].shift(1)
    past_usd_ret = feat['usd_inr_ret_1d'].shift(1)
    past_crude_ret = feat['crude_ret_1d'].shift(1)
    past_silver_ret = feat['silver_ret_1d'].shift(1)
    past_nifty_ret = feat['nifty_ret_1d'].shift(1)

    for window in [5, 10, 21, 63]:
        feat[f'gold_ret_mean_{window}d'] = past_gold_ret.rolling(window).mean()
        feat[f'gold_ret_std_{window}d'] = past_gold_ret.rolling(window).std()
        feat[f'usd_ret_mean_{window}d'] = past_usd_ret.rolling(window).mean()
        feat[f'crude_ret_mean_{window}d'] = past_crude_ret.rolling(window).mean()
        feat[f'silver_ret_mean_{window}d'] = past_silver_ret.rolling(window).mean()
        feat[f'nifty_ret_mean_{window}d'] = past_nifty_ret.rolling(window).mean()

    past_gold_price = feat['gold_inr_10g'].shift(1)
    feat['rsi_14'] = compute_rsi(past_gold_price, 14)
    feat['rsi_7'] = compute_rsi(past_gold_price, 7)

    ema12 = past_gold_price.ewm(span=12, adjust=False).mean()
    ema26 = past_gold_price.ewm(span=26, adjust=False).mean()
    feat['macd'] = ema12 - ema26
    feat['macd_signal'] = feat['macd'].ewm(span=9, adjust=False).mean()
    feat['macd_hist'] = feat['macd'] - feat['macd_signal']

    bb_mid = past_gold_price.rolling(20).mean()
    bb_std = past_gold_price.rolling(20).std()
    feat['bb_upper'] = bb_mid + 2 * bb_std
    feat['bb_lower'] = bb_mid - 2 * bb_std
    feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / bb_mid
    feat['bb_pct'] = (past_gold_price - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'])

    feat['day_of_week'] = feat['date'].dt.dayofweek
    feat['month'] = feat['date'].dt.month
    feat['quarter'] = feat['date'].dt.quarter
    feat['is_month_end'] = feat['date'].dt.is_month_end.astype(int)

    exclude_cols = {'date', 'target_return', 'target_price'}
    feature_cols = [c for c in feat.columns if c not in exclude_cols]
    model_df = feat.dropna(subset=feature_cols + ['target_return', 'target_price']).reset_index(drop=True)
    return model_df, feature_cols


def ret_to_price(current_price, predicted_return):
    return np.array(current_price) * (1.0 + np.array(predicted_return))


def price_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


@st.cache_resource(show_spinner=False)
def train_pipeline(start_date: str, train_ratio: float, val_ratio: float):
    clean_df = fetch_market_data(start_date)
    model_df, feature_cols = build_features(clean_df)

    X_all = model_df[feature_cols].copy()
    y_ret = model_df['target_return'].copy()
    y_price = model_df['target_price'].copy()
    cp_all = model_df['gold_inr_10g'].copy()
    dates_all = model_df['date'].copy()

    n = len(model_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train_raw = X_all.iloc[:train_end].copy()
    X_val_raw = X_all.iloc[train_end:val_end].copy()
    X_test_raw = X_all.iloc[val_end:].copy()

    y_train_ret = y_ret.iloc[:train_end].copy()
    y_val_ret = y_ret.iloc[train_end:val_end].copy()
    y_test_ret = y_ret.iloc[val_end:].copy()

    y_val_price = y_price.iloc[train_end:val_end].copy()
    y_test_price = y_price.iloc[val_end:].copy()
    cp_val = cp_all.iloc[train_end:val_end].copy()
    cp_test = cp_all.iloc[val_end:].copy()
    dates_val = dates_all.iloc[train_end:val_end].copy()
    dates_test = dates_all.iloc[val_end:].copy()

    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train_raw)
    X_val_std = std_scaler.transform(X_val_raw)
    X_test_std = std_scaler.transform(X_test_raw)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            random_state=42,
        )
    if HAS_LGBM:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )

    results = []
    val_pred_price = {}
    test_pred_price = {}
    fitted_models = {}

    for name, model in models.items():
        if name == 'LinearRegression':
            model.fit(X_train_std, y_train_ret)
            val_pred = ret_to_price(cp_val.values, model.predict(X_val_std))
            test_pred = ret_to_price(cp_test.values, model.predict(X_test_std))
        else:
            model.fit(X_train_raw, y_train_ret)
            val_pred = ret_to_price(cp_val.values, model.predict(X_val_raw))
            test_pred = ret_to_price(cp_test.values, model.predict(X_test_raw))

        fitted_models[name] = model
        val_pred_price[name] = val_pred
        test_pred_price[name] = test_pred

        metric = price_metrics(y_test_price.values, test_pred)
        metric['Model'] = name
        results.append(metric)

    metrics_df = pd.DataFrame(results).sort_values(['RMSE', 'MAE']).reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]['Model']

    latest_row_raw = X_all.tail(1).copy()
    current_price = float(cp_all.iloc[-1])
    next_actual_date = pd.to_datetime(dates_all.iloc[-1]) + pd.Timedelta(days=1)

    forecasts = []
    for name, model in fitted_models.items():
        if name == 'LinearRegression':
            pred_ret = float(model.predict(std_scaler.transform(latest_row_raw))[0])
        else:
            pred_ret = float(model.predict(latest_row_raw)[0])
        pred_price = float(ret_to_price([current_price], [pred_ret])[0])
        forecasts.append({
            'Model': name,
            'CurrentPrice': current_price,
            'PredictedReturnPct': pred_ret * 100,
            'PredictedNextPrice': pred_price,
            'ForecastDate': next_actual_date.date().isoformat(),
        })
    forecast_df = pd.DataFrame(forecasts).sort_values('PredictedNextPrice', ascending=False).reset_index(drop=True)

    best_test_df = pd.DataFrame({
        'date': pd.to_datetime(dates_test.values),
        'ActualPrice': np.array(y_test_price.values),
        'PredictedPrice': np.array(test_pred_price[best_model_name]),
    })

    residuals = np.array(y_val_price.values) - np.array(val_pred_price[best_model_name])
    lower_resid = np.percentile(residuals, 5)
    upper_resid = np.percentile(residuals, 95)
    best_test_df['Lower90'] = best_test_df['PredictedPrice'] + lower_resid
    best_test_df['Upper90'] = best_test_df['PredictedPrice'] + upper_resid

    return {
        'clean_df': clean_df,
        'model_df': model_df,
        'feature_cols': feature_cols,
        'metrics_df': metrics_df,
        'forecast_df': forecast_df,
        'best_model_name': best_model_name,
        'best_test_df': best_test_df,
        'date_ranges': {
            'train_start': str(dates_all.iloc[:train_end].min().date()),
            'train_end': str(dates_all.iloc[:train_end].max().date()),
            'val_start': str(dates_all.iloc[train_end:val_end].min().date()),
            'val_end': str(dates_all.iloc[train_end:val_end].max().date()),
            'test_start': str(dates_all.iloc[val_end:].min().date()),
            'test_end': str(dates_all.iloc[val_end:].max().date()),
        },
    }


st.title('Gold Price Prediction Dashboard')
st.caption('Streamlit version of your notebook using Yahoo Finance market data and leakage-safe feature engineering.')

with st.sidebar:
    st.header('Settings')
    start_date = st.date_input('Start date', value=date(2004, 1, 1))
    train_ratio = st.slider('Train ratio', 0.50, 0.85, 0.70, 0.05)
    val_ratio = st.slider('Validation ratio', 0.05, 0.30, 0.15, 0.05)
    test_ratio = round(1 - train_ratio - val_ratio, 2)
    st.write(f'Test ratio: {test_ratio:.2f}')
    run_btn = st.button('Run pipeline', type='primary')

if train_ratio + val_ratio >= 0.95:
    st.error('Train ratio + validation ratio should leave enough data for testing. Keep total below 0.95.')
    st.stop()

if run_btn:
    try:
        with st.spinner('Downloading data, building features, and training models...'):
            out = train_pipeline(str(start_date), train_ratio, val_ratio)

        clean_df = out['clean_df']
        metrics_df = out['metrics_df']
        forecast_df = out['forecast_df']
        best_model_name = out['best_model_name']
        best_test_df = out['best_test_df']
        ranges = out['date_ranges']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Rows after cleaning', f"{len(clean_df):,}")
        c2.metric('Features used', f"{len(out['feature_cols'])}")
        c3.metric('Best model', best_model_name)
        c4.metric('Latest INR/10g', f"₹{clean_df['gold_inr_10g'].iloc[-1]:,.2f}")

        st.subheader('Split summary')
        st.write({
            'train': f"{ranges['train_start']} to {ranges['train_end']}",
            'validation': f"{ranges['val_start']} to {ranges['val_end']}",
            'test': f"{ranges['test_start']} to {ranges['test_end']}",
        })

        st.subheader('Test metrics')
        st.dataframe(metrics_df.style.format({'RMSE': '{:,.2f}', 'MAE': '{:,.2f}', 'MAPE': '{:,.2f}%', 'R2': '{:.4f}'}), use_container_width=True)

        st.subheader('Next-day forecast by model')
        st.dataframe(forecast_df.style.format({'CurrentPrice': '₹{:,.2f}', 'PredictedReturnPct': '{:.3f}%', 'PredictedNextPrice': '₹{:,.2f}'}), use_container_width=True)

        latest_forecast = forecast_df.loc[forecast_df['Model'] == best_model_name].iloc[0]
        st.success(
            f"Best model: {best_model_name} | Forecast date: {latest_forecast['ForecastDate']} | "
            f"Predicted next price: ₹{latest_forecast['PredictedNextPrice']:,.2f}"
        )

        st.subheader('Historical gold price (INR per 10g)')
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=clean_df['date'], y=clean_df['gold_inr_10g'], mode='lines', name='Gold INR/10g'))
        fig_hist.update_layout(height=450, template='plotly_white', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader(f'{best_model_name}: Actual vs predicted on test set')
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=best_test_df['date'], y=best_test_df['Upper90'], mode='lines', name='Upper 90% band', line=dict(color='lightgray')))
        fig_test.add_trace(go.Scatter(x=best_test_df['date'], y=best_test_df['Lower90'], mode='lines', name='Lower 90% band', fill='tonexty', line=dict(color='lightgray')))
        fig_test.add_trace(go.Scatter(x=best_test_df['date'], y=best_test_df['ActualPrice'], mode='lines', name='Actual Price'))
        fig_test.add_trace(go.Scatter(x=best_test_df['date'], y=best_test_df['PredictedPrice'], mode='lines', name='Predicted Price'))
        fig_test.update_layout(height=500, template='plotly_white', xaxis_title='Date', yaxis_title='Gold Price (INR per 10g)')
        st.plotly_chart(fig_test, use_container_width=True)

        csv = best_test_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download test predictions CSV', data=csv, file_name='gold_test_predictions.csv', mime='text/csv')

    except Exception as e:
        st.exception(e)
else:
    st.info('Choose settings from the sidebar and click “Run pipeline”.')
