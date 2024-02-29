from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import yfinance as yf
import pytz

# Load stock data using Yahoo Finance
def load_data(symbol, timeframe, periods=30, timezone='UTC'):
    # Convert end date to the specified timezone
    end_date = datetime.now(pytz.timezone(timezone))
    
    # Adjust start date accordingly
    start_date = calculate_start_date(timeframe, periods, end_date)
    
    # Load data using adjusted start and end dates
    data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
    return data

# Calculate start date based on timeframe and periods
def calculate_start_date(timeframe, periods, end_date):
    if timeframe in ['1m', '5m', '15m', '30m', '1h']:
        return end_date - timedelta(days=1)
    elif timeframe == '1d':
        return end_date - timedelta(days=periods)
    elif timeframe == '1wk':
        return end_date - timedelta(weeks=periods)
    elif timeframe == '1mo':
        return end_date - relativedelta(months=periods)

# Train Prophet model
def train_model(df):
    df = prepare_dataframe(df)
    model = Prophet()
    model.fit(df)
    return model

# Prepare DataFrame for Prophet model
def prepare_dataframe(df):
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'ds', 'Close': 'y'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    if isinstance(df['ds'].iloc[0], pd.Timestamp):
        df['ds'] = df['ds'].dt.tz_localize(None)
    else:
        df['ds'] = pd.to_datetime(df['ds'])
    return df

# Make predictions with Prophet model
def predict(model, future):
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat'] = forecast['yhat'].rolling(window=5, min_periods=1).mean()
    recent_data = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=7)]
    min_close = recent_data['yhat'].min()
    floor_value = max(0.01, 0.05 * min_close)
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, floor_value))
    return forecast

# Display results
def display_results(df, forecast):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
    fig.update_layout(title='Actual vs. Predicted Closing Prices',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      legend_title='Data',
                      xaxis_rangeslider_visible=False)
    return fig

# Main function
def main():
    st.title("Live Stock Analysis")

    symbols = [
        '^NSEI',
        'BHARTIARTL.NS', 'HDFCLIFE.NS', 'BRITANNIA.NS', 'TCS.NS', 'LTIM.NS',
        'ITC.NS', 'CIPLA.NS', 'TECHM.NS', 'NTPC.NS', 'BAJFINANCE.NS',
        'BAJAJFINSV.NS', 'HINDALCO.NS', 'LT.NS', 'NESTLEIND.NS', 'HEROMOTOCO.NS',
        'TATACONSUM.NS', 'ONGC.NS', 'KOTAKBANK.NS', 'ULTRACEMCO.NS', 'TITAN.NS',
        'COALINDIA.NS', 'RELIANCE.NS', 'TATASTEEL.NS', 'ADANIENT.NS', 'WIPRO.NS',
        'INDUSINDBK.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'APOLLOHOSP.NS'
    ]
    
    symbol = st.selectbox("Select Stock Symbol", symbols)
    timeframe = st.selectbox("Select Timeframe", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'])
    future_periods = st.slider("Select Number of Future Periods", min_value=1, max_value=365, value=10)
    timezone = st.selectbox("Select Timezone", pytz.all_timezones)

    if symbol:
        df = load_data(symbol, timeframe, timezone=timezone)  
        if not df.empty:
            model = train_model(df)
            future = model.make_future_dataframe(periods=future_periods)
            forecast = predict(model, future)
            fig = display_results(df, forecast)
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
