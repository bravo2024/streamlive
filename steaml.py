from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from nsepy import get_history

# Function to load stock data using NSEpy
def load_data(symbol, timeframe, num_days=100):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    df = get_history(symbol=symbol, start=start_date, end=end_date, index=True)
    return df[['Date', 'Close']]  # Select 'Date' and 'Close' columns

# Function to train Prophet model
def train_model(df):
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df)
    return model

# Function to make predictions with Prophet model
def predict(model, future):
    forecast = model.predict(future)
    return forecast

# Function to display results
def display_results(df, forecast):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df['Close'], label='Actual', color='blue')
    ax.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Actual vs. Predicted Closing Prices')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Live Stock Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., SBIN for State Bank of India):")
    timeframe = st.selectbox("Select Timeframe", ['1d', '1wk', '1mo'])

    if symbol:
        df = load_data(symbol, timeframe)
        if not df.empty:
            model = train_model(df)
            forecast_periods = st.slider("Select Number of Forecast Periods", min_value=1, max_value=365, value=3)
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = predict(model, future)
            display_results(df, forecast)

if __name__ == "__main__":
    main()
