import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from models.random_forest_model import train_random_forest
from sqlalchemy import create_engine
from urllib.parse import quote

st.set_page_config(layout="wide")
st.title("🔮 Iron Ore Price Forecasting App")

# Upload CSV or use default
uploaded_file = st.file_uploader("Upload your Iron Ore CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")
else:
    st.warning("Using default dataset: Data_set_Iron.csv")
    df = pd.read_csv("Data_set_Iron.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Data preprocessing
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values(by='Date')
df = df.drop(columns=['Open', 'High', 'Low'], errors='ignore')
df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100
df["Vol."] = df["Vol."].astype(str).str.replace("K", "", regex=True)
df["Vol."] = pd.to_numeric(df["Vol."], errors="coerce") * 1000
df["Vol."].fillna(df["Vol."].median(), inplace=True)
df["Price"].fillna(method="bfill", inplace=True)
df["Change %"].fillna(0, inplace=True)
df['SMA_30'] = df['Price'].rolling(window=30).mean()
df['EMA_30'] = df['Price'].ewm(span=30, adjust=False).mean()
df['month'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter
df['day_of_week'] = df['Date'].dt.dayofweek
df.fillna(method="bfill", inplace=True)

# Visualize Raw Price Trend
st.subheader("📈 Iron Ore Market Price Trend")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Date'], df['Price'], label="Market Price", color='teal')
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.set_title("Iron Ore Price Trend Over Time")
ax.grid(True)
st.pyplot(fig)

# Forecasting with Random Forest
if st.button("📊 Run Random Forest Forecast"):
    st.info("Training model... this may take a few seconds.")
    results = train_random_forest(df)
    st.success("✅ Model Trained and Forecasted Successfully!")

    st.subheader("📌 Evaluation Metrics")
    st.write(f"**RMSE:** {results['rmse']:.2f}")
    st.write(f"**MAE:** {results['mae']:.2f}")
    st.write(f"**MAPE:** {results['mape']:.2f}%")

    st.download_button("📥 Download Model", data=open(results['model_path'], 'rb'), file_name="random_forest_model.pkl")
    st.download_button("📥 Download Scaler", data=open(results['scaler_path'], 'rb'), file_name="scaler.pkl")
