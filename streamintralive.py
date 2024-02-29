import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt


@st.cache
def load_data(symbol, timeframe, periods=30):
    end_date = pd.Timestamp.now()
    if timeframe in ['1m', '5m', '15m', '30m', '1h']:  
        start_date = end_date - pd.Timedelta(days=1)  
    elif timeframe == '1d':
        start_date = end_date - pd.Timedelta(days=periods)  
    elif timeframe == '1wk':
        start_date = end_date - pd.Timedelta(weeks=periods)  
    
    data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
    return data


@st.cache
def train_model(df):
    df = df.reset_index()
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'ds', 'Close': 'y'})
    elif 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    if isinstance(df['ds'].iloc[0], pd.Timestamp):
        df['ds'] = df['ds'].dt.tz_localize(None)  
    else:
        df['ds'] = pd.to_datetime(df['ds'])  

    model = Prophet()
    model.fit(df)
    return model


@st.cache
def predict(model, future, floor_percentage=0.05):
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)  
    forecast['yhat'] = forecast['yhat'].rolling(window=5, min_periods=1).mean()  

    recent_data = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=7)]  
    min_close = recent_data['yhat'].min()  
    floor_value = max(0.01, floor_percentage * min_close)  
    forecast['yhat'] = forecast['yhat'].apply(lambda x: max(x, floor_value))
    
    return forecast


def display_results(df, forecast):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
    fig.update_layout(
        title='Actual vs. Predicted Closing Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Data',
        xaxis_rangeslider_visible=False
    )
    return fig


def generate_signal_plots(df):
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
        df = load_data(symbol, timeframe)  
        if not df.empty:
            model = train_model(df)
            future = model.make_future_dataframe(periods=future_periods)
            forecast = predict(model, future)
            fig = display_results(df, forecast)
            st.plotly_chart(fig)
            generate_signal_plots(df)

if __name__ == "__main__":
    main()
