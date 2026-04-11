import warnings
warnings.filterwarnings('ignore')

from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Optional libraries
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
LOOKBACK_WINDOWS = [5, 10, 21, 63]
YAHOO_TICKERS = {
    'gold_usd_oz': 'GC=F',
    'usd_inr': 'INR=X',
    'crude_oil': 'CL=F',
    'silver_usd_oz': 'SI=F',
    'nifty50': '^NSEI',
    'sp500': '^GSPC',
}

st.set_page_config(page_title='Gold Price Prediction Dashboard', page_icon='📈', layout='wide')


def safe_metric(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(np.abs(np.array(y_true)) < 1e-9, np.nan, np.array(y_true))
    mape = float(np.nanmean(np.abs((np.array(y_true) - np.array(y_pred)) / denom)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


@st.cache_data(show_spinner=False)

def simplify_feature_name(name: str) -> str:
    mapping = {
        'gold_inr_10g': 'Gold Price (INR/10g)',
        'gold_usd_oz': 'Gold Price (USD/oz)',
        'usd_inr': 'USD/INR Exchange Rate',
        'crude_oil': 'Crude Oil Price',
        'silver_usd_oz': 'Silver Price',
        'nifty50': 'Nifty 50',
        'sp500': 'S&P 500',
        'gold_return_1d': 'Gold 1-Day Return',
        'gold_usd_ret_1d': 'Gold USD 1-Day Return',
        'usd_inr_ret_1d': 'USD/INR 1-Day Return',
        'crude_ret_1d': 'Crude Oil 1-Day Return',
        'silver_ret_1d': 'Silver 1-Day Return',
        'nifty_ret_1d': 'Nifty 50 1-Day Return',
        'sp500_ret_1d': 'S&P 500 1-Day Return',
        'rsi': 'RSI',
        'macd': 'MACD',
        'bb_upper': 'Bollinger Band Upper',
        'bb_middle': 'Bollinger Band Middle',
        'bb_lower': 'Bollinger Band Lower',
        'day_of_week': 'Day of Week',
        'month': 'Month',
        'quarter': 'Quarter',
    }
    if name in mapping:
        return mapping[name]

    clean = name
    clean = clean.replace('_ret_', ' Return ')
    clean = clean.replace('_return_', ' Return ')
    clean = clean.replace('_mean_', ' Avg ')
    clean = clean.replace('_std_', ' Volatility ')
    clean = clean.replace('_lag', ' Lag ')
    clean = clean.replace('_1d', ' 1D')
    clean = clean.replace('_5d', ' 5D')
    clean = clean.replace('_10d', ' 10D')
    clean = clean.replace('_21d', ' 21D')
    clean = clean.replace('_63d', ' 63D')
    clean = clean.replace('_', ' ')
    clean = ' '.join(clean.split())
    return clean.title()

def download_yahoo_series(ticker: str, col_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f'No data found for {ticker}')
    use_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        use_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    out = df[[use_col]].reset_index().rename(columns={'Date': 'date', use_col: col_name})
    out['date'] = pd.to_datetime(out['date']).dt.tz_localize(None)
    return out.sort_values('date').reset_index(drop=True)


@st.cache_data(show_spinner=True)
def load_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    market_frames = []
    for name, ticker in YAHOO_TICKERS.items():
        market_frames.append(download_yahoo_series(ticker, name, start_date, end_date))

    master = market_frames[0].copy()
    for frame in market_frames[1:]:
        master = master.merge(frame, on='date', how='left')

    price_cols = list(YAHOO_TICKERS.keys())
    master[price_cols] = master[price_cols].ffill()
    # Convert USD/oz to INR/10g
    master['gold_inr_10g'] = master['gold_usd_oz'] * master['usd_inr'] * (10 / 31.1035)
    return master.dropna(subset=['gold_inr_10g']).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_features(raw_df: pd.DataFrame):
    df = raw_df.copy().sort_values('date').reset_index(drop=True)

    # Targets
    df['target_return'] = df['gold_inr_10g'].pct_change().shift(-1)
    df['target_price'] = df['gold_inr_10g'].shift(-1)

    # Core returns
    return_map = {
        'gold_return_1d': 'gold_inr_10g',
        'gold_usd_ret_1d': 'gold_usd_oz',
        'usd_inr_ret_1d': 'usd_inr',
        'crude_ret_1d': 'crude_oil',
        'silver_ret_1d': 'silver_usd_oz',
        'nifty_ret_1d': 'nifty50',
        'sp500_ret_1d': 'sp500',
    }
    for ret_name, src in return_map.items():
        df[ret_name] = df[src].pct_change()

    base_returns = list(return_map.keys())
    for col in base_returns:
        for lag in [1, 5, 10]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
        for window in LOOKBACK_WINDOWS:
            df[f'{col}_mean_{window}d'] = df[col].rolling(window).mean()
            df[f'{col}_std_{window}d'] = df[col].rolling(window).std()

    df['rsi_14'] = compute_rsi(df['gold_inr_10g'], 14)
    macd, macd_signal, macd_hist = compute_macd(df['gold_inr_10g'])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    bb_mid = df['gold_inr_10g'].rolling(20).mean()
    bb_std = df['gold_inr_10g'].rolling(20).std()
    df['bb_mid'] = bb_mid
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    feature_cols = [
        c for c in df.columns
        if c not in ['date', 'target_return', 'target_price']
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    model_df = df.dropna(subset=feature_cols + ['target_return', 'target_price']).reset_index(drop=True)
    return df, model_df, feature_cols


@st.cache_resource(show_spinner=True)
def train_pipeline(start_date: str, end_date: str):
    raw_df = load_market_data(start_date, end_date)
    feature_frame_all, model_df, feature_cols = build_features(raw_df)

    X_all = model_df[feature_cols].copy()
    y_return = model_df['target_return'].copy()
    y_price = model_df['target_price'].copy()
    current_price = model_df['gold_inr_10g'].copy()
    dates = model_df['date'].copy()

    n = len(model_df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, X_val, X_test = X_all.iloc[:train_end], X_all.iloc[train_end:val_end], X_all.iloc[val_end:]
    y_train, y_val, y_test = y_return.iloc[:train_end], y_return.iloc[train_end:val_end], y_return.iloc[val_end:]
    y_val_price, y_test_price = y_price.iloc[train_end:val_end], y_price.iloc[val_end:]
    current_val, current_test = current_price.iloc[train_end:val_end], current_price.iloc[val_end:]
    dates_test = dates.iloc[val_end:]

    std_scaler = StandardScaler()
    X_train_std = std_scaler.fit_transform(X_train)
    X_val_std = std_scaler.transform(X_val)
    X_test_std = std_scaler.transform(X_test)

    def to_price(curr, pred_ret):
        return np.array(curr) * (1 + np.array(pred_ret))

    model_defs = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=SEED,
            n_jobs=-1,
        ),
    }
    if XGB_AVAILABLE:
        model_defs['XGBoost'] = XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=SEED,
        )
    if LGBM_AVAILABLE:
        model_defs['LightGBM'] = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            random_state=SEED,
            verbosity=-1,
        )

    trained_models = {}
    val_price_preds = {}
    test_price_preds = {}
    val_rows = []
    test_rows = []

    for name, model in model_defs.items():
        if name == 'LinearRegression':
            model.fit(X_train_std, y_train)
            pred_val_ret = model.predict(X_val_std)
            pred_test_ret = model.predict(X_test_std)
        else:
            model.fit(X_train, y_train)
            pred_val_ret = model.predict(X_val)
            pred_test_ret = model.predict(X_test)

        pred_val_price = to_price(current_val, pred_val_ret)
        pred_test_price = to_price(current_test, pred_test_ret)

        trained_models[name] = model
        val_price_preds[name] = pred_val_price
        test_price_preds[name] = pred_test_price

        val_metrics = safe_metric(y_val_price, pred_val_price)
        test_metrics = safe_metric(y_test_price, pred_test_price)
        val_rows.append({'Model': name, **val_metrics})
        test_rows.append({'Model': name, **test_metrics})

    val_results = pd.DataFrame(val_rows).sort_values('RMSE').reset_index(drop=True)
    test_results = pd.DataFrame(test_rows).sort_values('RMSE').reset_index(drop=True)
    comparison = val_results.merge(test_results, on='Model', suffixes=('_val', '_test')).sort_values('RMSE_test').reset_index(drop=True)

    eligible = val_results[val_results['R2'] > 0].copy().sort_values('RMSE').head(3)
    eligible['inv_rmse'] = 1 / eligible['RMSE']
    eligible['Weight'] = eligible['inv_rmse'] / eligible['inv_rmse'].sum()
    ensemble_weights = eligible[['Model', 'RMSE', 'Weight']].reset_index(drop=True)

    ensemble_val = np.zeros(len(X_val))
    ensemble_test = np.zeros(len(X_test))
    for _, row in ensemble_weights.iterrows():
        ensemble_val += val_price_preds[row['Model']] * row['Weight']
        ensemble_test += test_price_preds[row['Model']] * row['Weight']

    val_results = pd.concat([val_results, pd.DataFrame([{'Model': 'Ensemble', **safe_metric(y_val_price, ensemble_val)}])], ignore_index=True)
    test_results = pd.concat([test_results, pd.DataFrame([{'Model': 'Ensemble', **safe_metric(y_test_price, ensemble_test)}])], ignore_index=True)

    latest_cols = ['date', 'gold_inr_10g'] + [c for c in feature_cols if c != 'gold_inr_10g']
    latest_block = feature_frame_all[latest_cols].dropna().copy()
    latest_row = latest_block.tail(1)
    latest_features = latest_row[feature_cols]
    latest_price = float(latest_row['gold_inr_10g'].iloc[0])
    latest_date = pd.to_datetime(latest_row['date'].iloc[0])

    latest_predictions = []
    for name, model in trained_models.items():
        if name == 'LinearRegression':
            pred_ret = float(model.predict(std_scaler.transform(latest_features))[0])
        else:
            pred_ret = float(model.predict(latest_features)[0])
        pred_price = latest_price * (1 + pred_ret)
        latest_predictions.append({'Model': name, 'Predicted Price INR10g': pred_price, 'Predicted Return %': pred_ret * 100})

    forecast_df = pd.DataFrame(latest_predictions)
    ensemble_forecast_price = 0.0
    for _, row in ensemble_weights.iterrows():
        model_name = row['Model']
        model_pred = forecast_df.loc[forecast_df['Model'] == model_name, 'Predicted Price INR10g'].iloc[0]
        ensemble_forecast_price += model_pred * row['Weight']
    ensemble_forecast_ret = ((ensemble_forecast_price / latest_price) - 1) * 100
    forecast_df = pd.concat([
        forecast_df,
        pd.DataFrame([{'Model': 'Ensemble', 'Predicted Price INR10g': ensemble_forecast_price, 'Predicted Return %': ensemble_forecast_ret}])
    ], ignore_index=True)

    # Direction classifier
    direction_target = (y_return > 0).astype(int)
    y_train_dir = direction_target.iloc[:train_end]
    y_val_dir = direction_target.iloc[train_end:val_end]
    y_test_dir = direction_target.iloc[val_end:]
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_std, y_train_dir)
    latest_prob_up = float(clf.predict_proba(std_scaler.transform(latest_features))[0, 1])
    latest_direction = 'UP' if latest_prob_up >= 0.5 else 'DOWN'
    direction_acc = accuracy_score(y_test_dir, clf.predict(X_test_std))

    # Range from best weighted model validation residuals
    range_model_name = ensemble_weights.iloc[0]['Model']
    val_residuals = y_val_price.values - val_price_preds[range_model_name]
    residual_std = float(np.nanstd(val_residuals))
    forecast_df['Lower Bound'] = forecast_df['Predicted Price INR10g'] - residual_std
    forecast_df['Upper Bound'] = forecast_df['Predicted Price INR10g'] + residual_std
    forecast_df['Predicted Date'] = latest_date + pd.Timedelta(days=1)

    # Strategy on test period using top weighted model
    strategy_model_name = range_model_name
    strategy_model = trained_models[strategy_model_name]
    if strategy_model_name == 'LinearRegression':
        strategy_pred_ret = strategy_model.predict(X_test_std)
    else:
        strategy_pred_ret = strategy_model.predict(X_test)
    threshold = 0.002
    strategy_signal = np.where(strategy_pred_ret > threshold, 1, np.where(strategy_pred_ret < -threshold, -1, 0))
    actual_test_ret = y_test.values
    strategy_daily_return = strategy_signal * actual_test_ret
    strategy_cum = (1 + pd.Series(strategy_daily_return, index=dates_test)).cumprod() - 1
    buy_hold_cum = (1 + pd.Series(actual_test_ret, index=dates_test)).cumprod() - 1

    # Risk label
    risk_df = raw_df[['date', 'gold_inr_10g']].copy()
    risk_df['gold_return_1d'] = risk_df['gold_inr_10g'].pct_change()
    risk_df['rolling_volatility_14d'] = risk_df['gold_return_1d'].rolling(14).std()
    latest_vol = float(risk_df['rolling_volatility_14d'].dropna().iloc[-1])
    q33 = float(risk_df['rolling_volatility_14d'].dropna().quantile(0.33))
    q66 = float(risk_df['rolling_volatility_14d'].dropna().quantile(0.66))
    if latest_vol <= q33:
        risk_level = 'Low Risk'
        risk_message = 'Recent volatility is relatively low.'
    elif latest_vol <= q66:
        risk_level = 'Medium Risk'
        risk_message = 'Recent volatility is moderate.'
    else:
        risk_level = 'High Risk'
        risk_message = 'Recent volatility is elevated, so forecast uncertainty is higher.'

    # Feature importance
    importance_model = trained_models['RandomForest'] if 'RandomForest' in trained_models else trained_models[ensemble_weights.iloc[0]['Model']]
    if hasattr(importance_model, 'feature_importances_'):
        fi = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance_model.feature_importances_}).sort_values('Importance', ascending=False).reset_index(drop=True)
    else:
        fi = pd.DataFrame({'Feature': X_train.columns, 'Importance': np.abs(np.nan_to_num(np.array(X_train).mean(axis=0)))}).sort_values('Importance', ascending=False).reset_index(drop=True)

    fi['Feature Display'] = fi['Feature'].apply(simplify_feature_name)
    top_features = fi['Feature Display'].head(3).tolist()
    if latest_prob_up >= 0.60:
        interpretation = 'Model shows a relatively strong upward bias for the next session.'
    elif latest_prob_up >= 0.50:
        interpretation = 'Model shows a mild upward bias for the next session.'
    elif latest_prob_up <= 0.40:
        interpretation = 'Model shows a relatively stronger downside bias for the next session.'
    else:
        interpretation = 'Model outlook is mixed and close to neutral.'

    return {
        'raw_df': raw_df,
        'feature_frame_all': feature_frame_all,
        'model_df': model_df,
        'feature_cols': feature_cols,
        'comparison': comparison,
        'val_results': val_results,
        'test_results': test_results,
        'trained_models': trained_models,
        'ensemble_weights': ensemble_weights,
        'forecast_df': forecast_df.sort_values('Predicted Price INR10g', ascending=False).reset_index(drop=True),
        'latest_price': latest_price,
        'latest_date': latest_date,
        'direction_prob_up': latest_prob_up,
        'direction_label': latest_direction,
        'direction_accuracy': direction_acc,
        'risk_level': risk_level,
        'risk_message': risk_message,
        'latest_volatility': latest_vol,
        'feature_importance': fi,
        'interpretation': interpretation,
        'top_features': top_features,
        'dates_test': dates_test,
        'y_test_price': y_test_price,
        'test_price_preds': test_price_preds,
        'strategy_model_name': strategy_model_name,
        'strategy_cum': strategy_cum,
        'buy_hold_cum': buy_hold_cum,
    }


def main():
    st.title('📈 Gold Price Prediction & Trading Strategy Dashboard')
    st.caption('Built for project expo: next-day INR gold prediction using multi-model ML, ensemble learning, risk labeling, and simple trading signals.')

    with st.sidebar:
        st.header('Settings')
        years_back = st.slider('Years of history to use', min_value=5, max_value=22, value=15)
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * years_back)
        st.write(f'Using data from **{start_date}** to **{end_date}**')
        st.info('Click the button below to run training. This avoids re-training on every widget refresh.')
        run_model = st.button('Run prediction pipeline', type='primary')

    if not run_model:
        st.warning('Choose settings and click **Run prediction pipeline**.')
        st.stop()

    result = train_pipeline(str(start_date), str(end_date + timedelta(days=1)))
    forecast_df = result['forecast_df']

    ensemble_row = forecast_df[forecast_df['Model'] == 'Ensemble'].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Current Price (INR/10g)', f"₹{result['latest_price']:,.2f}")
    c2.metric('Predicted Price', f"₹{ensemble_row['Predicted Price INR10g']:,.2f}", f"{ensemble_row['Predicted Return %']:.2f}%")
    c3.metric('Direction', result['direction_label'], f"Up probability {result['direction_prob_up'] * 100:.1f}%")
    c4.metric('Risk Level', result['risk_level'], f"Volatility {result['latest_volatility']:.4f}")

    st.markdown(f"**Prediction date:** {pd.to_datetime(ensemble_row['Predicted Date']).date()}  ")
    st.markdown(f"**Confidence range:** ₹{ensemble_row['Lower Bound']:,.2f} to ₹{ensemble_row['Upper Bound']:,.2f}")
    st.markdown(f"**Interpretation:** {result['interpretation']} Top drivers right now include **{', '.join(result['top_features'])}**.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Dashboard', 'Model Results', 'Forecast Table', 'Feature Importance', 'About Project'])

    with tab1:
        st.subheader('Actual vs Predicted Prices')
        strategy_model = result['strategy_model_name']
        plot_df = pd.DataFrame({
            'date': result['dates_test'],
            'Actual Price': result['y_test_price'].values,
            f'Predicted Price ({strategy_model})': result['test_price_preds'][strategy_model],
        })
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['Actual Price'], mode='lines', name='Actual Price'))
        fig1.add_trace(go.Scatter(x=plot_df['date'], y=plot_df[f'Predicted Price ({strategy_model})'], mode='lines', name=f'Predicted Price ({strategy_model})'))
        fig1.update_layout(height=420, xaxis_title='Date', yaxis_title='INR per 10g')
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader('Strategy vs Buy & Hold')
        strat_df = pd.DataFrame({
            'date': result['strategy_cum'].index,
            'Strategy Return': result['strategy_cum'].values * 100,
            'Buy & Hold Return': result['buy_hold_cum'].values * 100,
        })
        fig2 = px.line(strat_df, x='date', y=['Strategy Return', 'Buy & Hold Return'], height=420)
        fig2.update_layout(yaxis_title='Cumulative Return %', xaxis_title='Date')
        st.plotly_chart(fig2, use_container_width=True)

        st.info(result['risk_message'])

    with tab2:
        st.subheader('Validation vs Test Performance')
        st.dataframe(result['comparison'], use_container_width=True)
        st.subheader('Ensemble Weights')
        st.dataframe(result['ensemble_weights'], use_container_width=True)
        st.caption(f"Direction classification accuracy on test set: {result['direction_accuracy'] * 100:.2f}%")

    with tab3:
        st.subheader('Latest Next-Day Forecasts')
        st.dataframe(forecast_df, use_container_width=True)

    with tab4:
        st.subheader('Top Feature Importance')
        fi = result['feature_importance'].head(15)
        fig3 = px.bar(fi.sort_values('Importance'), x='Importance', y='Feature Display', orientation='h', height=520)
        st.plotly_chart(fig3, use_container_width=True)

    with tab5:
        st.markdown('''
### What this app does
- Predicts **next-day gold price in INR per 10 grams**
- Uses historical market drivers like **gold, USD/INR, crude oil, silver, Nifty50, and S&P 500**
- Engineers features such as **returns, lags, rolling averages, volatility, RSI, MACD, and Bollinger Bands**
- Trains multiple models and creates a **weighted ensemble**
- Adds **direction prediction, confidence range, risk label, and a simple trading strategy**


### Limitations
- Market shocks and sudden news are not directly modeled
- Backtest does not include real transaction costs
- It is a research/demo app, not financial advice
        ''')


if __name__ == '__main__':
    main()
