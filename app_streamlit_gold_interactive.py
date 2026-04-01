
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gold Dashboard", layout="wide")

st.title("🟡 Gold Price Prediction Dashboard (Interactive)")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
n_estimators = st.sidebar.slider("Model Complexity (Trees)", 50, 300, 100)

@st.cache_data
def load_data(start):
    data = yf.download("GC=F", start=start)
    data = data[['Close']].dropna()
    data['Return'] = data['Close'].pct_change()
    data['Lag1'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    return data

df = load_data(start_date)

# Train/Test split
split = int(len(df)*0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[['Lag1']]
y_train = train['Close']
X_test = test[['Lag1']]
y_test = test['Close']

model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

# KPI Section
col1, col2, col3 = st.columns(3)

col1.metric("📊 Data Points", len(df))
col2.metric("📉 MAE", f"{mae:.2f}")
col3.metric("📈 RMSE", f"{rmse:.2f}")

st.markdown("---")

# Interactive chart selector
chart_option = st.selectbox("Select Visualization", ["Price Trend", "Prediction vs Actual"])

if chart_option == "Price Trend":
    st.subheader("📈 Gold Price Trend")
    st.line_chart(df['Close'])
    st.caption("Gold prices show long-term upward momentum.")

elif chart_option == "Prediction vs Actual":
    st.subheader("🤖 Model Performance")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual")
    ax.plot(preds, label="Predicted")
    ax.legend()
    st.pyplot(fig)
    st.caption("Model predictions closely follow actual values.")

# Next prediction
latest = df[['Lag1']].iloc[-1:]
next_pred = model.predict(latest)[0]

st.markdown("---")
st.subheader("🔮 Next Day Prediction")

st.success(f"Predicted Gold Price: ₹{next_pred:,.2f}")

# Download
download_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
st.download_button("⬇️ Download Predictions", download_df.to_csv().encode(), "predictions.csv")

st.markdown("----")
st.info("Built by Chinmay Kulkarni | PGDM FinTech")
