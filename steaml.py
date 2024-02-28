from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import yfinance as yf

# Function to load stock data using Yahoo Finance
def load_data(symbol, timeframe):
    end_date = datetime.now()
    if timeframe in ['1m', '5m', '15m', '30m', '1h']:  # Intraday timeframes
        start_date = end_date - timedelta(days=1)  # 1 day of data
    else:  # Higher timeframes
        start_date = end_date - timedelta(days=365)  # 1 year of data
    data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
    return data[['Close']]  # Select 'Close' column only

# Function to train Prophet model
def train_model(df):
    df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df)
    return model

# Function to make predictions with Prophet model
def predict(model, future):
    forecast = model.predict(future)
    return forecast

# Function to display results
def display_results(df, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
    fig.update_layout(title='Actual vs. Predicted Closing Prices')
    st.plotly_chart(fig)

def main():
    st.title("Live Stock Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple Inc.):")
    timeframe = st.selectbox("Select Timeframe", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'])

    if symbol:
        df = load_data(symbol, timeframe)
        if not df.empty:
            model = train_model(df)
            if timeframe in ['1m', '5m', '15m', '30m', '1h']:
                future = model.make_future_dataframe(periods=10, freq='T')  # Adjust periods and frequency as needed
            else:
                future = model.make_future_dataframe(periods=10, freq='D')  # Adjust periods and frequency as needed
            forecast = predict(model, future)
            display_results(df, forecast)

if __name__ == "__main__":
    main()
