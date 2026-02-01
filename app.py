import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Stock Forecasting App",
    page_icon="📈",
    layout="wide"
)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ App Controls")
st.sidebar.markdown("""
**Model:** ARIMAX  
**Horizon:** 30 Days  
**Frequency:** Daily  
""")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Cleaned CSV",
    type=["csv"]
)

st.sidebar.info(
    "CSV must be cleaned and contain\n"
    "macroeconomic + stock columns."
)

# ---------------- Main Title ----------------
st.title(" Stock Price Forecasting Dashboard")
st.markdown(
    """
    This dashboard forecasts **next 30 days stock prices**
    using an **ARIMAX model with macroeconomic indicators**.
    """
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    with open("final_arimax_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- Upload Validation ----------------
if uploaded_file is None:
    st.warning("👈 Please upload a cleaned CSV file from the sidebar.")
    st.stop()

# ---------------- Load Data ----------------
df = pd.read_csv(uploaded_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")
df = df.resample("D").mean().dropna()

# ---------------- KPIs ----------------
col1, col2, col3 = st.columns(3)

col1.metric("📅 Last Date", df.index[-1].strftime("%Y-%m-%d"))
col2.metric("💰 Last Stock Price", round(df["stock_price"].iloc[-1], 2))
col3.metric("📊 Records Used", df.shape[0])

# ---------------- Data Preview ----------------
with st.expander("🔍 View Data Preview"):
    st.dataframe(df.tail())

# ---------------- Feature Engineering ----------------
base_exog = [
    "nasdaq_index", "sp500_index", "market_sentiment",
    "inflation_rate", "unemployment_rate", "interest_rate"
]

for col in base_exog:
    df[col+"_lag1"] = df[col].shift(1)
    df[col+"_lag2"] = df[col].shift(2)

df["stock_lag1"] = df["stock_price"].shift(1)
df["stock_lag2"] = df["stock_price"].shift(2)
df["stock_ma7"] = df["stock_price"].rolling(7).mean()
df.dropna(inplace=True)

exog_features = (
    base_exog +
    [c+"_lag1" for c in base_exog] +
    [c+"_lag2" for c in base_exog] +
    ["stock_lag1", "stock_lag2", "stock_ma7"]
)

# ---------------- Forecast ----------------
st.subheader("🔮 30-Day Forecast")

future_days = 30
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1),
                             periods=future_days, freq="D")

last = df.iloc[-1]
future_X = pd.DataFrame(index=future_dates)

for col in base_exog:
    future_X[col] = last[col]
    future_X[col+"_lag1"] = last[col+"_lag1"]
    future_X[col+"_lag2"] = last[col+"_lag2"]

future_X["stock_lag1"] = last["stock_price"]
future_X["stock_lag2"] = df["stock_price"].iloc[-2]
future_X["stock_ma7"] = df["stock_price"].tail(7).mean()

future_X = future_X[exog_features]

forecast_log = model.forecast(steps=future_days, exog=future_X)
forecast_price = np.exp(forecast_log)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Price": forecast_price.values
})

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index[-100:], df["stock_price"].tail(100), label="Actual")
ax.plot(forecast_df["Date"], forecast_df["Forecasted Price"],
        linestyle="--", label="Forecast")

ax.set_title("Stock Price Forecast (ARIMAX)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ---------------- Download ----------------
st.subheader("📥 Download Forecast")
st.download_button(
    "Download CSV",
    forecast_df.to_csv(index=False),
    "30_day_forecast.csv",
    "text/csv"
)