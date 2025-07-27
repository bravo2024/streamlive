# from datetime import datetime, timedelta
# import streamlit as st
# import pandas as pd
# from prophet import Prophet
# import plotly.graph_objects as go
# import yfinance as yf
# import pytz

# # Load stock data using Yahoo Finance
# # def load_data(symbol, timeframe, periods=30, timezone='UTC'):
# #     # Convert end date to the specified timezone
# #     end_date = datetime.now(pytz.timezone(timezone))
    
# #     # Adjust start date accordingly
# #     start_date = calculate_start_date(timeframe, periods, end_date)
    
# #     # Load data using adjusted start and end dates
# #     data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
# #     return data


# def load_data(symbol, timeframe, periods=30, timezone='UTC'):
#     # Current time in UTC
#     end_date = datetime.now()
    
#     # Determine start_date based on timeframe and period count
#     if timeframe in ['1m', '5m', '15m', '30m', '1h']:  # Intraday (Yahoo limits to 7 days max)
#         start_date = end_date - timedelta(days=1)
#     elif timeframe == '1d':
#         start_date = end_date - timedelta(days=periods)
#     elif timeframe == '1wk':
#         start_date = end_date - timedelta(weeks=periods)
#     elif timeframe == '1mo':
#         start_date = end_date - relativedelta(months=periods)
#     else:
#         raise ValueError(f"Unsupported timeframe: {timeframe}")
    
#     # Download data from Yahoo Finance
#     data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
#     print(data.head(20))
#     # If the data has MultiIndex columns (e.g., from multiple tickers), flatten it
#     if isinstance(data.columns, pd.MultiIndex):
#         data.columns = data.columns.get_level_values(0)

#     # Convert timezone from UTC to IST (Asia/Kolkata)
#     if data.index.tz is None:
#         data.index = data.index.tz_localize('UTC')
#     data.index = data.index.tz_convert('Asia/Kolkata')

#     # Reset index so datetime is a column
#     data.reset_index(inplace=True)

#     # Validate essential columns
#     if 'Close' not in data.columns:
#         raise ValueError("Missing 'Close' column in the downloaded data.")
#     if not any(col in data.columns for col in ['Date', 'Datetime']):
#         raise ValueError("Expected a 'Date' or 'Datetime' column in the data.")
#     print(data)
#     return data

# # Calculate start date based on timeframe and periods
# def calculate_start_date(timeframe, periods, end_date):
#     if timeframe in ['1m', '5m', '15m', '30m', '1h']:
#         return end_date - timedelta(days=1)
#     elif timeframe == '1d':
#         return end_date - timedelta(days=periods)
#     elif timeframe == '1wk':
#         return end_date - timedelta(weeks=periods)
#     elif timeframe == '1mo':
#         return end_date - relativedelta(months=periods)

# # Train Prophet model
# def train_model(df):
#     df = prepare_dataframe(df)
#     model = Prophet()
#     model.fit(df)
#     return model

# # Prepare DataFrame for Prophet model
# def prepare_dataframe(df):
#     if 'Datetime' in df.columns:
#         df = df.rename(columns={'Datetime': 'ds', 'Close': 'y'})
#     elif 'Date' in df.columns:
#         df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
#     else:
#         st.error("DataFrame must contain either 'Datetime' or 'Date' column.")
#         return pd.DataFrame()
    
#     if 'ds' not in df.columns:
#         st.error("DataFrame must contain a 'ds' column.")
#         return pd.DataFrame()
    
#     if 'Close' not in df.columns:
#         st.error("DataFrame must contain a 'Close' column.")
#         return pd.DataFrame()
    
#     if isinstance(df['ds'].iloc[0], pd.Timestamp):
#         df['ds'] = df['ds'].dt.tz_localize(None)
#     else:
#         df['ds'] = pd.to_datetime(df['ds'])
    
#     return df

# # Make predictions with Prophet model
# def predict(model, future):
#     forecast = model.predict(future)
#     forecast['yhat'] = forecast['yhat'].clip(lower=0)
#     forecast['yhat'] = forecast['yhat'].rolling(window=5, min_periods=1).mean()
#     recent_data = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=7)]
#     min_close = recent_data['yhat'].min()
#     floor_value = max(0.01, 0.05 * min_close)
#     forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, floor_value))
#     return forecast

# # Display results
# def display_results(df, forecast):
#     fig = go.Figure()
#     fig.add_trace(go.Candlestick(x=df.index,
#                     open=df['Open'],
#                     high=df['High'],
#                     low=df['Low'],
#                     close=df['Close'], name='Actual'))
#     fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
#     fig.update_layout(title='Actual vs. Predicted Closing Prices',
#                       xaxis_title='Date',
#                       yaxis_title='Price',
#                       legend_title='Data',
#                       xaxis_rangeslider_visible=False)
#     return fig

# # Main function
# def main():
#     st.title("Live Stock Analysis")

#     symbols = [
#         '^NSEI',
#         'BHARTIARTL.NS', 'HDFCLIFE.NS', 'BRITANNIA.NS', 'TCS.NS', 'LTIM.NS',
#         'ITC.NS', 'CIPLA.NS', 'TECHM.NS', 'NTPC.NS', 'BAJFINANCE.NS',
#         'BAJAJFINSV.NS', 'HINDALCO.NS', 'LT.NS', 'NESTLEIND.NS', 'HEROMOTOCO.NS',
#         'TATACONSUM.NS', 'ONGC.NS', 'KOTAKBANK.NS', 'ULTRACEMCO.NS', 'TITAN.NS',
#         'COALINDIA.NS', 'RELIANCE.NS', 'TATASTEEL.NS', 'ADANIENT.NS', 'WIPRO.NS',
#         'INDUSINDBK.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'APOLLOHOSP.NS'
#     ]
    
#     symbol = st.selectbox("Select Stock Symbol", symbols)
#     timeframe = st.selectbox("Select Timeframe", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'])
#     future_periods = st.slider("Select Number of Future Periods", min_value=1, max_value=365, value=10)
#     timezone = st.selectbox("Select Timezone", pytz.all_timezones)

#     if symbol:
#         df = load_data(symbol, timeframe, timezone=timezone)  
#         if not df.empty:
#             model = train_model(df)
#             future = model.make_future_dataframe(periods=future_periods)
#             forecast = predict(model, future)
#             fig = display_results(df, forecast)
#             st.plotly_chart(fig)

# if __name__ == "__main__":
#     main()


from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import yfinance as yf
import pytz
from dateutil.relativedelta import relativedelta

# ---------------------- Load Stock Data ----------------------
def load_data(symbol, timeframe, periods=30, timezone='UTC'):
    end_date = datetime.now()

    if timeframe in ['1m', '5m', '15m', '30m', '1h']:
        start_date = end_date - timedelta(days=1)
    elif timeframe == '1d':
        start_date = end_date - timedelta(days=periods)
    elif timeframe == '1wk':
        start_date = end_date - timedelta(weeks=periods)
    elif timeframe == '1mo':
        start_date = end_date - relativedelta(months=periods)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)

    if data.empty:
        raise ValueError("No data downloaded. Check symbol or timeframe.")

    # If MultiIndex, flatten it
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Localize and convert timezone
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert(timezone)

    # Reset index and ensure datetime column
    data.reset_index(inplace=True)
    if 'index' in data.columns:
        data.rename(columns={'index': 'Datetime'}, inplace=True)
    elif 'Date' in data.columns:
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
    elif 'Datetime' not in data.columns:
        raise ValueError("No datetime column found after reset_index.")

    # Validate required columns
    if 'Close' not in data.columns:
        raise ValueError("Missing 'Close' column in the downloaded data.")

    return data

# ---------------------- Prepare Data ----------------------
def prepare_dataframe(df):
    if 'Datetime' not in df.columns or 'Close' not in df.columns:
        st.error("Data must contain 'Datetime' and 'Close' columns.")
        return pd.DataFrame()

    df = df.rename(columns={'Datetime': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    if isinstance(df['ds'].iloc[0], pd.Timestamp):
        df['ds'] = df['ds'].dt.tz_localize(None)

    return df

# ---------------------- Train Prophet ----------------------
def train_model(df):
    df = prepare_dataframe(df)
    if df.empty:
        return None
    model = Prophet()
    model.fit(df)
    return model

# ---------------------- Predict ----------------------
def predict(model, future):
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat'] = forecast['yhat'].rolling(window=5, min_periods=1).mean()
    recent_data = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=7)]
    min_close = recent_data['yhat'].min()
    floor_value = max(0.01, 0.05 * min_close)
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, floor_value))
    return forecast

# ---------------------- Plot Results ----------------------
def display_results(df, forecast):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Datetime'],
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

# ---------------------- Streamlit App ----------------------
def main():
    st.title("📈 Live Stock Forecasting App")

    symbols = [
        '^NSEI',
        'BHARTIARTL.NS', 'HDFCLIFE.NS', 'BRITANNIA.NS', 'TCS.NS', 'LTIM.NS',
        'ITC.NS', 'CIPLA.NS', 'TECHM.NS', 'NTPC.NS', 'BAJFINANCE.NS',
        'BAJAJFINSV.NS', 'HINDALCO.NS', 'LT.NS', 'NESTLEIND.NS', 'HEROMOTOCO.NS',
        'TATACONSUM.NS', 'ONGC.NS', 'KOTAKBANK.NS', 'ULTRACEMCO.NS', 'TITAN.NS',
        'COALINDIA.NS', 'RELIANCE.NS', 'TATASTEEL.NS', 'ADANIENT.NS', 'WIPRO.NS',
        'INDUSINDBK.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'APOLLOHOSP.NS'
    ]
    
    symbol = st.selectbox("📊 Select Stock Symbol", symbols)
    timeframe = st.selectbox("🕒 Select Timeframe", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'])
    future_periods = st.slider("🔮 Forecast Period (Days)", min_value=1, max_value=365, value=10)
    timezone = st.selectbox("🌍 Select Timezone", pytz.all_timezones, index=pytz.all_timezones.index('Asia/Kolkata'))

    if symbol:
        try:
            df = load_data(symbol, timeframe, timezone=timezone)  
            if not df.empty:
                model = train_model(df)
                if model:
                    future = model.make_future_dataframe(periods=future_periods)
                    forecast = predict(model, future)
                    fig = display_results(df, forecast)
                    st.plotly_chart(fig)
                else:
                    st.error("Model training failed.")
            else:
                st.warning("No data loaded.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

