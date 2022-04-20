import streamlit as st
import pandas as pd
import pandas_datareader as web
import datetime as dt
import plotly.graph_objects as go
from PIL import Image
from fbprophet import Prophet


# this sets up the header and description of the dashboard
st.write("""
# Cryptocurrency Dashboard Application
Visually show data on crypto (BTC, DOGE & ETH) from 01/01/2019 to the most recent date
and predicts data on the cryptocurrencies mentioned above.
""")

# this is an image that fits the dashboard really well
# it is solely used for aesthetics
image = Image.open('/Users/ANDIhiropedi/Desktop/Careers/CS stuff/Crypto Predictor/Dashboard.png')
st.image(image, use_column_width = True)

# this sidebar allows the user to restrict the data displayed for a certain cryptocurrency
st.sidebar.header("User input")


# this retrieves the user and input in the sidebar
def get_input():
    crypto_symbol = st.sidebar.text_input("Crypto Symbol", "BTC")
    return crypto_symbol


# this gets the appropriate name of the cryptocurrency, based on the symbol
# used for aesthetics
def get_crypto_name(symbol):
    symbol = symbol.upper()
    if symbol == "BTC":
        return "Bitcoin"
    elif symbol == "ETH":
        return "Ethereum"
    elif symbol == "DOGE":
        return "Dogecoin"
    else:
        return "None"


# this function generates the predictions for the appropriate cryptocurrency
def generate_predictions(data):
    df = data[['Date', 'Close']]
    new_names = {
        'Date': 'ds',
        'Close': 'y'
        }
    df.rename(columns = new_names, inplace = True)
    model = Prophet(seasonality_mode = "multiplicative")
    model.fit(df)
    future_data = model.make_future_dataframe(periods = 365)
    forecast = model.predict(future_data)
    return forecast[['ds','yhat']]


# this retrieves all the necessary information from the functions above
symbol = get_input()
crypto_name = get_crypto_name(symbol)
# the dashboard will only account for conversions in US dollars
conversion_currency = "USD"
# and will only account for data from 19/06/2019, for more accurate predictions
start = dt.datetime(2019,1,1)
end = dt.datetime.now()
data = web.DataReader(f'{symbol}-{conversion_currency}', 'yahoo', start, end)
data.reset_index(inplace = True)
predictions = generate_predictions(data)
new_names = {
        'ds': 'Date',
        'yhat': 'Predictions'
        }
predictions.rename(columns = new_names, inplace = True)
mix = pd.merge(data, predictions, how="inner", on = 'Date')
combined_data = mix[['Close','Predictions']]


# this is a special figure used to display the data for the selecte cryptocurrency
# makes the dashboard more interactive
fig = go.Figure(
    data = [go.Candlestick(
        x = data.index,
        open = data['Open'],
        high = data['High'],
        low = data['Low'],
        close = data['Close'],
        increasing_line_color = 'green',
        decreasing_line_color = 'red')])


# The following code displays some more graphs and sttistics about the cryptocurrency.
# Also displays the predictions for the close price in USD of the selected crypto.
# Used to make the dashboard more fun and interactive.
st.header(crypto_name + " Data")
st.write(data)

st.header(crypto_name + " Data Statistics")
st.write(data.describe())

st.header(crypto_name + " Close Price")
st.line_chart(data['Close'])

st.header(crypto_name + " Close Price and Predictions")
st.write(combined_data)

st.header(crypto_name + "Price Predictions")
st.write(predictions)

st.header(crypto_name + " Volume")
st.bar_chart(data['Volume'])

st.header(crypto_name + " Candle Stick")
st.plotly_chart(fig)

