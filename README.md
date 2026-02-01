# 📈 Stock Price Forecasting using ARIMAX

This project forecasts future stock prices using an **ARIMAX
(Autoregressive Integrated Moving Average with Exogenous variables)**
model incorporating macroeconomic indicators.

---

## 🚀 Features
- Time series forecasting (30 days ahead)
- Uses macroeconomic variables
- ARIMAX model built using `statsmodels`
- Interactive Streamlit dashboard
- CSV download of forecasts

---

## 🧠 Model Used
**ARIMAX**  
Chosen after comparison with ARIMA, SARIMA, VAR, Random Forest,
XGBoost, and LSTM.

**Why ARIMAX?**
- Incorporates external economic factors
- Interpretable statistical model
- Stable performance on time series data

---

## 📊 Input Data
The app requires a **cleaned CSV file** containing:
- `stock_price`
- `nasdaq_index`
- `sp500_index`
- `market_sentiment`
- `inflation_rate`
- `unemployment_rate`
- `interest_rate`
- `timestamp`

Preprocessing consistency is required for accurate forecasting.

---

## 🖥️ Deployment
The model is deployed using **Streamlit**.

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py