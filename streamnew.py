from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
from pytz import timezone



# Function to load stock data using Yahoo Finance
# Function to load stock data using Yahoo Finance
# Function to load stock data using Yahoo Finance
# Function to load stock data using Yahoo Finance
def load_data(symbol, timeframe, num_days=30):
    end_date = datetime.now()
    if timeframe in ['1m', '5m', '15m', '30m', '1h']:  # Intraday timeframes
        start_date = end_date - timedelta(days=30)  # 1 day of data
    elif timeframe == '1d':
        start_date = end_date - timedelta(days=periods)  # Number of days selected by the user
    elif timeframe == '1wk':
        start_date = end_date - timedelta(weeks=periods)  # Number of weeks selected by the user
    elif timeframe == '1mo':
        start_date = end_date - relativedelta(months=periods)  # Number of months selected by the user
    
    
    data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
    #end_date = datetime.now()
    #start_date = end_date - timedelta(days=num_days)
    #df = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
    #data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    # Reset index for compatibility with Prophet
    #data.reset_index(inplace=True)

    return data
    
    
    
    #return df




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
# Function to make predictions with Prophet model
# Function to make predictions with Prophet model
# Function to make predictions with Prophet model
# Function to make predictions with Prophet model
# Function to make predictions with Prophet model
def predict(model, future, floor_percentage=0.05):
    forecast = model.predict(future)
    
    # Adjust forecasted values to prevent negative values and ensure smoothness
    forecast['yhat'] = forecast['yhat'].clip(lower=0)  # Clip negative values to zero
    
    # Ensure smoothness of the forecast by removing sharp changes
    forecast['yhat'] = forecast['yhat'].rolling(window=5, min_periods=1).mean()  # Smooth forecast with rolling mean
    
    # Adjust floor value dynamically based on recent historical data
    recent_data = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=7)]  # Recent data (last 7 days)
    min_close = recent_data['yhat'].min()  # Minimum forecasted close value in the recent data
    floor_value = max(0.01, floor_percentage * min_close)  # Minimum floor value as 0.01 or a percentage of the minimum value
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, floor_value))
    
    return forecast





# Function to display results
def display_results(df, forecast):
    #fig = go.Figure()
    #fig.add_trace(go.Candlestick(x=df.index,
   #                 open=df['Open'],
    #                high=df['High'],
    #                low=df['Low'],
    #                close=df['Close'], name='Actual'))
    #fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
    #fig.update_layout(title='Actual vs. Predicted Closing Prices')

   # #fig = go.Figure()
    ##fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Actual'))
   # #fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
   # #fig.update_layout(title='Actual vs. Predicted Closing Prices')
    #st.plotly_chart(fig)
    #st.subheader('Last 10 Actual Values')
    #st.write(df.tail(10))

    #st.subheader('Last 10 Predicted Values')
    #st.write(forecast[['ds', 'yhat']].tail(10))
    ##return fig
    market_open_time = pd.Timestamp('09:30').time()  # Market open time (e.g., 9:30 AM)
    market_close_time = pd.Timestamp('16:00').time()  # Market close time (e.g., 4:00 PM)

    # Filter out data for non-trading hours
    df = df.between_time(market_open_time, market_close_time)
    forecast = forecast[(forecast['ds'].dt.time >= market_open_time) & (forecast['ds'].dt.time <= market_close_time)]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
    fig.update_layout(title='Actual vs. Predicted Closing Prices')
    return fig

#def display_last_values(df, forecast, future_periods):
    # Extract future predicted closing prices
    #future_predicted = forecast[['ds', 'yhat']].tail(future_periods)

    # Rename columns for clarity
    #future_predicted.rename(columns={'ds': 'Future Date', 'yhat': 'Future Predicted Close'}, inplace=True)

    # Display future predicted values
    #st.subheader(f'Future {future_periods} Predicted Closing Values')
    #st.write(future_predicted)

def calculate_signal(df, forecast, future_periods):
    # Calculate thresholds for buy and sell signals
    buy_threshold = forecast['yhat'].quantile(0.75)
    sell_threshold = forecast['yhat'].quantile(0.25)
    
    # Extract the last predicted closing prices
    last_predicted = forecast[['ds', 'yhat']].tail(future_periods)

    # Add buy/sell signal based on thresholds
    last_predicted['Signal'] = 'Hold'
    last_predicted.loc[last_predicted['yhat'] > buy_threshold, 'Signal'] = 'Buy'
    last_predicted.loc[last_predicted['yhat'] < sell_threshold, 'Signal'] = 'Sell'

    return last_predicted


def display_last_values(df, forecast, future_periods, timeframe,model):
    if timeframe in ['1m', '5m', '15m', '30m', '1h']:
        # For intraday time frames, adjust future dates to align with the desired time interval
        last_date = forecast['ds'].iloc[-1]  # Get the last date in the forecast
        freq = {'1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T', '1h': 'H'}[timeframe]  # Mapping from time frame to frequency
        future = pd.date_range(start=last_date, periods=future_periods, freq=freq)  # Generate future dates
        future_forecast = pd.DataFrame({'ds': future})

        # Make predictions for the future dates
        future_forecast['yhat'] = predict(model, future_forecast)['yhat']

        # Combine the future forecast with the existing forecast
        forecast = pd.concat([forecast, future_forecast], ignore_index=True)

    else:
        # For daily or other time frames, display all future predicted values
        forecast = forecast.tail(future_periods)  # Limit the forecast to the specified number of future periods

    # Extract last 20 predicted closing prices
    last_predicted = forecast[['ds', 'yhat']].tail(20)  # Always extract last 20 values

    # Calculate buy/sell signal
    signal_df = calculate_signal(df, forecast, future_periods)

    # Merge predicted closing prices with buy/sell signal
    last_predicted = pd.merge(last_predicted, signal_df[['ds', 'Signal']], left_on='ds', right_on='ds', how='left')

    # Rename columns for clarity
    last_predicted.rename(columns={'ds': 'Predicted Date', 'yhat': 'Predicted Close'}, inplace=True)

    # Reset index of the DataFrame
    last_predicted.reset_index(drop=True, inplace=True)

    # Display last 20 predicted closing values with buy/sell signal
    st.subheader('Last 20 Predicted Closing Values with Buy/Sell Signal')
    st.write(last_predicted)

    # Display future predicted closing values
    future_predicted = forecast[['ds', 'yhat']].tail(future_periods)
    future_predicted.rename(columns={'ds': 'Future Date', 'yhat': 'Future Predicted Close'}, inplace=True)
    st.subheader(f'Future {future_periods} Predicted Closing Values')
    st.write(future_predicted)

"""
"""
#def apply_leading_indicators(df):
    # Calculate indicators
   # df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)
   # df['EMA_50'] = ta.EMA(df['Close'], timeperiod=50)
   # df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14)
   # df['MACD'], df['MACD Signal'], _ = ta.MACD(df['Close'])

    # Generate buy/sell signals based on indicator values
   # df['Signal'] = 'Hold'
    # Example: Buy when SMA_20 crosses above EMA_50, RSI > 30, and MACD > Signal
  #  buy_condition = (df['SMA_20'] > df['EMA_50']) & (df['RSI_14'] > 50) & (df['MACD'] > df['MACD Signal'])
  #  df.loc[buy_condition, 'Signal'] = 'Buy'
    # Example: Sell when SMA_20 crosses below EMA_50, RSI < 70, and MACD < Signal
   # sell_condition = (df['SMA_20'] < df['EMA_50']) & (df['RSI_14'] < 50) & (df['MACD'] < df['MACD Signal'])
   # df.loc[sell_condition, 'Signal'] = 'Sell'
    #st.write(df.tail(10))
   # return df[['Close', 'Signal','SMA_20','RSI_14']]  # Return DataFrame with Date, Close, and Signal columns

# Assuming you already have a DataFrame named 'model
"""
"""
def apply_leading_indicators(df):
    """
    Calculate leading indicators including SMA, EMA, RSI, MACD, and additional indicators suitable for smaller time frames.
    Generate buy/sell signals based on indicator values.

    Parameters:
    - df (DataFrame): DataFrame containing historical stock data with 'Close' prices.

    Returns:
    - DataFrame: DataFrame with added indicators, buy/sell signals, and their values.
    """
    # Calculate indicators
    df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)
    df['EMA_50'] = ta.EMA(df['Close'], timeperiod=50)
    df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = ta.MACD(df['Close'])

    # Additional indicators suitable for smaller time frames
    df['Stochastic'] = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowd_period=3)[0]
    df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Generate buy/sell signals based on indicator values
    df['Signal'] = 'Hold'
    # Example: Buy when SMA_20 crosses above EMA_50, RSI > 30, and MACD > Signal
    buy_condition = (df['SMA_20'] > df['EMA_50']) & (df['RSI_14'] > 30) & (df['MACD'] > df['MACD_Signal'])
    df.loc[buy_condition, 'Signal'] = 'Buy'
    # Example: Sell when SMA_20 crosses below EMA_50, RSI < 70, and MACD < Signal
    sell_condition = (df['SMA_20'] < df['EMA_50']) & (df['RSI_14'] < 70) & (df['MACD'] < df['MACD_Signal'])
    df.loc[sell_condition, 'Signal'] = 'Sell'

    # Extract only the indicator columns and 'Signal'
    indicators_signal_df = df[['SMA_20', 'EMA_50', 'RSI_14', 'MACD', 'Stochastic', 'CCI', 'ATR', 'Signal']]

    # Replace buy/sell/hold with numeric values
    indicators_signal_df['Signal'] = indicators_signal_df['Signal'].replace({'Buy': 1, 'Sell': -1, 'Hold': 0})

    # Calculate correlation matrix
    corr_matrix = indicators_signal_df.corr()

    # Plot heatmap with buy/sell/hold annotations
    st.write("### Signal Heatmap")
    fig, ax = plt.subplots()
    heatmap = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)

    # Customize ticks and labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns)
    ax.set_yticklabels(corr_matrix.index)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add color bar
    plt.colorbar(heatmap)

    # Add title
    plt.title('Signal Heatmap')

    # Add buy/sell/hold annotations
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.5:
                text = ax.text(j, i, 'Buy', ha='center', va='center', color='white')
            elif corr_matrix.iloc[i, j] < -0.5:
                text = ax.text(j, i, 'Sell', ha='center', va='center', color='white')
            else:
                text = ax.text(j, i, 'Hold', ha='center', va='center', color='black')

    # Show the plot
    st.pyplot(fig)

def generate_signal_plots(df):
    """
    Calculate leading indicators including SMA, EMA, RSI, MACD, and additional indicators suitable for smaller time frames.
    Generate buy/sell signals based on indicator values.
    Generate line plots for each indicator with buy/sell/hold signals.

    Parameters:
    - df (DataFrame): DataFrame containing historical stock data with 'Close' prices.

    Returns:
    - None
    """
    # Calculate indicators

    st.write(df.tail(4))
    if 'Close' not in df.columns:
        st.error("DataFrame must contain a 'Close' column.")
        return

    # Calculate indicators
    df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)
    df['EMA_50'] = ta.EMA(df['Close'], timeperiod=50)
    df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = ta.MACD(df['Close'])

    # Additional indicators suitable for smaller time frames
    df['Stochastic'] = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowd_period=3)[0]
    df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)
    df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Generate buy/sell signals based on indicator values
    df['Signal'] = 'Hold'
    # Example: Buy when SMA_20 crosses above EMA_50, RSI > 30, and MACD > Signal
    buy_condition = (df['SMA_20'] > df['EMA_50']) & (df['RSI_14'] > 30) & (df['MACD'] > df['MACD_Signal'])
    df.loc[buy_condition, 'Signal'] = 'Buy'
    # Example: Sell when SMA_20 crosses below EMA_50, RSI < 70, and MACD < Signal
    sell_condition = (df['SMA_20'] < df['EMA_50']) & (df['RSI_14'] < 70) & (df['MACD'] < df['MACD_Signal'])
    df.loc[sell_condition, 'Signal'] = 'Sell'

    # Plot each indicator with buy/sell/hold signals
    st.write("### Signal Plots")
    for col in ['SMA_20', 'EMA_50', 'RSI_14', 'MACD', 'Stochastic', 'CCI', 'ATR']:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[col], label=col)
        for index, row in df.iterrows():
            if row['Signal'] == 'Buy':
                plt.scatter(index, row[col], color='green', label='Buy', marker='^', s=100)
            elif row['Signal'] == 'Sell':
                plt.scatter(index, row[col], color='red', label='Sell', marker='v', s=100)
            else:
                plt.scatter(index, row[col], color='blue', label='Hold', marker='o', s=50)
        plt.title(f'{col} with Buy/Sell/Hold Signals')
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(plt)






def main():
    st.title("Live Stock Analysis")


    
    #symbol = st.text_input("Enter Stock Symbol (e.g., AAPL for Apple Inc.):")
    symbols = [
        '^NSEI',
        'BHARTIARTL.NS', 'HDFCLIFE.NS', 'BRITANNIA.NS', 'TCS.NS', 'LTIM.NS',
        'ITC.NS', 'CIPLA.NS', 'TECHM.NS', 'NTPC.NS', 'BAJFINANCE.NS',
        'BAJAJFINSV.NS', 'HINDALCO.NS', 'LT.NS', 'NESTLEIND.NS', 'HEROMOTOCO.NS',
        'TATACONSUM.NS', 'ONGC.NS', 'KOTAKBANK.NS', 'ULTRACEMCO.NS', 'TITAN.NS',
        'COALINDIA.NS', 'RELIANCE.NS', 'TATASTEEL.NS', 'ADANIENT.NS', 'WIPRO.NS',
        'INDUSINDBK.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'APOLLOHOSP.NS'
    ]
    select_option = st.radio("Select Input Method", ["Dropdown", "Type"])
    
    if select_option == "Dropdown":
        symbol = st.selectbox("Select Stock Symbol", symbols)
    else:
        symbol = st.text_input("Enter Stock Symbol")

    
    timeframe = st.selectbox("Select Timeframe", ['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'])
    future_periods = st.slider("Select Number of Future Periods", min_value=1, max_value=365, value=10)

    if symbol:
        df = load_data(symbol, timeframe)  # Ensure at least 100 historical data points
        if not df.empty:
            model = train_model(df)
            future = model.make_future_dataframe(periods=future_periods)
            forecast = predict(model, future)
            fig = display_results(df, forecast)
            st.plotly_chart(fig)
            display_last_values(df, forecast,future_periods,timeframe,model)
            # signal_table = apply_leading_indicators(df)
            # st.table(signal_table)
            generate_signal_plots(df)
            #signals = [
              ##  ('RSI', 'Sell'),
               # ('Moving Average', 'Buy'),
               # ('Stochastic Oscillator', 'Sell'),
             #   ('Bollinger Bands', 'Hold')
           # ]
            #signal_df_styled = generate_signal_table(signals)
            #st.subheader('Buy/Sell Signals for Leading and Lagging Indicators')
            #st.write(signal_df_styled)

if __name__ == "__main__":
    main()
