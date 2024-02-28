from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import yfinance as yf

# Function to load stock data using Yahoo Finance
def load_data(symbol, timeframe, periods):
    end_date = datetime.now()
    if timeframe in ['1m', '5m', '15m', '30m', '1h']:  # Intraday timeframes
        start_date = end_date - timedelta(days=1)  # 1 day of data
    else:  # Higher timeframes
        start_date = end_date - timedelta(days=365)  # 1 year of data
    data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
    return data[-periods:]

# Function to train Prophet model
def train_model(df):
    df = df.reset_index()
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'ds', 'Close': 'y'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    
    # Check if 'ds' column contains datetime objects
    if isinstance(df['ds'].iloc[0], pd.Timestamp):
        df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone information if present
    else:
        df['ds'] = pd.to_datetime(df['ds'])  # Convert to datetime if it's not
    
    # Train Prophet model
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
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
    fig.update_layout(title='Actual vs. Predicted Closing Prices')
    return fig

def main():
    st.title("Live Stock Analysis")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple Inc.):")
    timeframe = st.selectbox("Select Timeframe", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'])
    periods = st.slider("Select Number of Periods", min_value=1, max_value=365, value=3)

    if symbol:
        df = load_data(symbol, timeframe, periods)
        if not df.empty:
            model = train_model(df)
            future = model.make_future_dataframe(periods=periods, freq='D' if timeframe in ['1d', '1wk', '1mo'] else 'T')
            forecast = predict(model, future)
            fig = display_results(df, forecast)
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
