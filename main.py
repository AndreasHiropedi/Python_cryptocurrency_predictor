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
# Cryptocurrency Dashboard
Visualize data about different cryptocurrencies (BTC, ETH and many more) from 01/01/2019 
up until today and also show how a simple model's predictions for these cryptocurrencies 
compare to the real data.

DISCLAIMER: This is for fun and educational purposes only and should not be treated as
financial advice for trading cryptocurrencies.
"""
)

image = Image.open("./Dashboard.png")
st.image(image, use_container_width=True)

st.sidebar.header("Select Cryptocurrency")


def get_input():
    crypto_options = {
        "Bitcoin (BTC)": "BTC",
        "Ethereum (ETH)": "ETH",
        "Dogecoin (DOGE)": "DOGE",
        "Solana (SOL)": "SOL",
        "XRP (XRP)": "XRP",
        "Bitcoin Cash (BCH)": "BCH",
        "Stellar (XLM)": "XLM",
        "Cardano (ADA)": "ADA",
        "Avalanche (AVAX)": "AVAX",
        "Chainlink (LINK)": "LINK",
        "Litecoin (LTC)": "LTC",
    }

    selected = st.sidebar.selectbox("Cryptocurrency", list(crypto_options.keys()))
    return crypto_options[selected]


def get_crypto_name(symbol):
    symbol = symbol.upper()
    crypto_names = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "DOGE": "Dogecoin",
        "SOL": "Solana",
        "XRP": "XRP",
        "BCH": "Bitcoin Cash",
        "XLM": "Stellar",
        "ADA": "Cardano",
        "AVAX": "Avalanche",
        "LINK": "Chainlink",
        "LTC": "Litecoin",
    }
    return crypto_names.get(symbol, "None")


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
    result["ds"] = pd.to_datetime(result["ds"]).dt.strftime("%Y-%m-%d")
    return result


symbol = get_input()
crypto_name = get_crypto_name(symbol)
conversion_currency = "USD"
start = dt.datetime(2024, 1, 1)
end = dt.datetime.now()

ticker = yf.Ticker(f"{symbol}-{conversion_currency}")
data = ticker.history(start=start, end=end)
data.reset_index(inplace=True)
data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)

predictions = generate_predictions(data)
predictions.columns = ["Date", "Predictions"]

data_display = data.copy()
data_display["Date"] = data_display["Date"].dt.strftime("%Y-%m-%d")

mix = pd.merge(data_display, predictions, how="inner", on="Date")
combined_data = mix[["Date", "Close", "Predictions"]].copy()

data_stats = data.drop(columns=["Date"]).describe()

# Candlestick chart with dark theme colors
fig_candle = go.Figure(
    data=[
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            increasing_line_color="#00ff88",  # Bright green
            decreasing_line_color="#ff4444",  # Bright red
        )
    ]
)
fig_candle.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ffffff"),
    xaxis=dict(gridcolor="#333333"),
    yaxis=dict(gridcolor="#333333"),
)

# Close price chart with neon cyan
fig_close = go.Figure()
fig_close.add_trace(
    go.Scatter(
        x=data["Date"],
        y=data["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#00d9ff", width=2),  # Neon cyan
    )
)
fig_close.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ffffff"),
    xaxis=dict(gridcolor="#333333"),
    yaxis=dict(gridcolor="#333333"),
)

# Volume chart with purple gradient
fig_volume = go.Figure()
fig_volume.add_trace(
    go.Bar(
        x=data["Date"],
        y=data["Volume"],
        name="Volume",
        marker_color="#a855f7",  # Purple
    )
)
fig_volume.update_layout(
    xaxis_title="Date",
    yaxis_title="Volume",
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ffffff"),
    xaxis=dict(gridcolor="#333333"),
    yaxis=dict(gridcolor="#333333"),
)

# Actual vs Predicted chart with vibrant colors
fig_predictions = go.Figure()

# Add actual close prices - Neon cyan
fig_predictions.add_trace(
    go.Scatter(
        x=combined_data["Date"],
        y=combined_data["Close"],
        mode="lines",
        name="Actual Close Price",
        line=dict(color="#00d9ff", width=2.5),  # Neon cyan
    )
)

# Add predictions for historical dates - Bright green
fig_predictions.add_trace(
    go.Scatter(
        x=combined_data["Date"],
        y=combined_data["Predictions"],
        mode="lines",
        name="Model Fit",
        line=dict(color="#1eff00", width=2.5),  # Bright green
    )
)

# Add future predictions - Neon orange/yellow
future_predictions = predictions[~predictions["Date"].isin(data_display["Date"])]
fig_predictions.add_trace(
    go.Scatter(
        x=future_predictions["Date"],
        y=future_predictions["Predictions"],
        mode="lines",
        name="Future Predictions",
        line=dict(color="#ffaa00", width=2.5),  # Neon orange
    )
)

fig_predictions.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#ffffff"),
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ffffff"),
    xaxis=dict(gridcolor="#333333"),
    yaxis=dict(gridcolor="#333333"),
)

# Display sections
st.header(crypto_name + " Data")
st.write(data_display.sort_values("Date", ascending=False).reset_index(drop=True))

st.header(crypto_name + " Data Statistics")
st.write(data_stats)

st.header(crypto_name + " Close Price Over Time")
st.plotly_chart(fig_close, use_container_width=True)

st.header(crypto_name + " Actual vs Predicted Prices")
st.plotly_chart(fig_predictions, use_container_width=True)

st.header("Combined Data and Predictions Table")
st.write(combined_data.sort_values("Date", ascending=False).reset_index(drop=True))

st.header(crypto_name + " Trading Volume")
st.plotly_chart(fig_volume, use_container_width=True)

st.header(crypto_name + " Price Movement")
st.plotly_chart(fig_candle, use_container_width=True)
