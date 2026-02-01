import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Stock Price Forecasting",
    layout="wide"
)

st.title("📈 Stock Price Forecasting Dashboard (ARIMAX)")
st.caption("Historical Trend + 30-Day Forecast using Macroeconomic Indicators")

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("final_arimax_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------------------------------
# LOAD DATA (AUTO – NO UPLOAD)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")   # keep this file in repo
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.resample("D").mean().dropna()
    return df

df = load_data()

# --------------------------------------------------
# DEFINE EXOG FEATURES (IMPORTANT)
# --------------------------------------------------
base_exog = [
    "nasdaq_index",
    "sp500_index",
    "market_sentiment",
    "inflation_rate",
    "unemployment_rate",
    "interest_rate"
]

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
for col in base_exog:
    df[col + "_lag1"] = df[col].shift(1)
    df[col + "_lag2"] = df[col].shift(2)

df["stock_lag1"] = df["stock_price"].shift(1)
df["stock_lag2"] = df["stock_price"].shift(2)
df["stock_ma7"]  = df["stock_price"].rolling(7).mean()

df.dropna(inplace=True)

exog_features = (
    base_exog +
    [c + "_lag1" for c in base_exog] +
    [c + "_lag2" for c in base_exog] +
    ["stock_lag1", "stock_lag2", "stock_ma7"]
)

# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("📅 Last Available Date", str(df.index[-1].date()))
col2.metric("📊 Last Stock Price", f"{df['stock_price'].iloc[-1]:.2f}")
col3.metric("📈 Total Records", len(df))

# --------------------------------------------------
# HISTORICAL TREND
# --------------------------------------------------
st.subheader("📉 Historical Stock Price Trend")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df.index, df["stock_price"], label="Actual Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("Stock Price")
ax1.grid(True)
ax1.legend()

st.pyplot(fig1)

# --------------------------------------------------
# FORECAST BUTTON
# --------------------------------------------------
st.subheader("🔮 30-Day Forecast")

if st.button("🚀 Forecast Next 30 Days"):

    future_days = 30
    future_dates = pd.date_range(
        start=df.index.max() + pd.Timedelta(days=1),
        periods=future_days,
        freq="D"
    )

    last_row = df.iloc[-1]
    future_X = pd.DataFrame(index=future_dates)

    # ---- Generate FUTURE exogenous values (non-flat) ----
    for col in base_exog:
        noise = np.random.normal(
            loc=0,
            scale=0.01 * abs(last_row[col]),
            size=future_days
        )
        future_X[col] = last_row[col] + noise
        future_X[col + "_lag1"] = last_row[col]
        future_X[col + "_lag2"] = last_row[col + "_lag1"]

    future_X["stock_lag1"] = last_row["stock_price"]
    future_X["stock_lag2"] = df["stock_price"].iloc[-2]
    future_X["stock_ma7"]  = df["stock_price"].tail(7).mean()

    future_X = future_X[exog_features]

    # ---- Forecast ----
    forecast_log = model.forecast(steps=future_days, exog=future_X)
    forecast_price = np.exp(forecast_log)

    forecast_df = pd.DataFrame(
        {"Forecasted Price": forecast_price.values},
        index=future_dates
    )

    # --------------------------------------------------
    # ACTUAL + FORECAST GRAPH
    # --------------------------------------------------
    st.subheader("📈 Actual vs Forecast")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df.index[-120:], df["stock_price"].tail(120),
             label="Actual (Last 120 Days)")
    ax2.plot(forecast_df.index, forecast_df["Forecasted Price"],
             linestyle="--", color="red", label="Forecast (Next 30 Days)")

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stock Price")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)

    # --------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------
    st.subheader("📄 Forecast Table")
    st.dataframe(forecast_df)

    # --------------------------------------------------
    # DOWNLOAD BUTTON
    # --------------------------------------------------
    csv = forecast_df.reset_index().rename(
        columns={"index": "Date"}
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download 30-Day Forecast CSV",
        data=csv,
        file_name="30_day_stock_forecast.csv",
        mime="text/csv"
    )

    st.success("✅ Forecast generated successfully")