import datetime as dt
import os

os.environ["STREAMLIT_LOGGER_LEVEL"] = "error"

import logging
import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from PIL import Image
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

st.write(
    """
# Cryptocurrency Dashboard Application
Visually show data on crypto (BTC, ETH and more) from 01/01/2019 to the most recent date
and predicts data on the cryptocurrencies mentioned above.
"""
)

image = Image.open("./Dashboard.png")
st.image(image, use_container_width=True)

st.sidebar.header("User input")


def get_input():
    crypto_symbol = st.sidebar.text_input("Crypto Symbol", "BTC")
    return crypto_symbol


def get_crypto_name(symbol):
    symbol = symbol.upper()
    if symbol == "BTC":
        return "Bitcoin"
    elif symbol == "ETH":
        return "Ethereum"
    elif symbol == "DOGE":
        return "Dogecoin"
    elif symbol == "SOL":
        return "Solana"
    elif symbol == "XRP":
        return "XRP"
    elif symbol == "BCH":
        return "Bitcoin Cash"
    elif symbol == "XLM":
        return "Stellar"
    elif symbol == "ADA":
        return "Cardano"
    elif symbol == "AVAX":
        return "Avalanche"
    elif symbol == "LINK":
        return "Chainlink"
    elif symbol == "LTC":
        return "Litecoin"
    else:
        return "None"


def generate_predictions(data):
    df = data[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    new_names = {"Date": "ds", "Close": "y"}
    df.rename(columns=new_names, inplace=True)
    model = Prophet(seasonality_mode="multiplicative")
    model.fit(df)
    future_data = model.make_future_dataframe(periods=365)
    forecast = model.predict(future_data)
    result = forecast[["ds", "yhat"]].copy()
    # CONVERT TO STRING IMMEDIATELY TO AVOID ARROW BULLSHIT
    result["ds"] = pd.to_datetime(result["ds"]).dt.strftime("%Y-%m-%d")
    return result


symbol = get_input()
crypto_name = get_crypto_name(symbol)
conversion_currency = "USD"
start = dt.datetime(2019, 1, 1)
end = dt.datetime.now()

ticker = yf.Ticker(f"{symbol}-{conversion_currency}")
data = ticker.history(start=start, end=end)
data.reset_index(inplace=True)
data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)

predictions = generate_predictions(data)
predictions.columns = ["Date", "Predictions"]

# Convert data Date to strings for display and merging
data_display = data.copy()
data_display["Date"] = data_display["Date"].dt.strftime("%Y-%m-%d")

# Merge (predictions already has string dates)
mix = pd.merge(data_display, predictions, how="inner", on="Date")
combined_data = mix[["Close", "Predictions"]]

data_stats = data.drop(columns=["Date"]).describe()

fig = go.Figure(
    data=[
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            increasing_line_color="green",
            decreasing_line_color="red",
        )
    ]
)

st.header(crypto_name + " Data")
st.write(data_display)

st.header(crypto_name + " Data Statistics")
st.write(data_stats)

st.header(crypto_name + " Close Price")
st.line_chart(data.set_index("Date")["Close"])

st.header(crypto_name + " Close Price and Predictions")
st.write(combined_data)

st.header(crypto_name + " Price Predictions")
st.write(predictions)

st.header(crypto_name + " Volume")
st.bar_chart(data.set_index("Date")["Volume"])

st.header(crypto_name + " Candle Stick")
st.plotly_chart(fig)
