import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import statsmodels.api as sm
from pmdarima.arima import auto_arima

app = 'Stock Market Forecaster'

st.title(app)
st.subheader("Forecasting Stock Price of Selected Company")
st.image("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
st.sidebar.header('Select Parameters')

sdate = st.sidebar.date_input('Start Date', datetime.date(2014, 1, 1))
edate = st.sidebar.date_input('End Date', datetime.date.today())

ticker_list = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOG', 'META', 'TSM', 'NVDA', 'NFLX', 'AMD']
ticker = st.sidebar.selectbox('Select Company', ticker_list)

data = yf.download(ticker, start=sdate, end=edate)
data.insert(0, 'Date', data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', sdate, 'to', edate)
st.write(data)

st.header('Data Visualization')
st.subheader('Plot of the Data')
fig = px.line(data, x='Date', y=data.columns, title='Stock Price of Selected Company', template='plotly_dark', width=950, height=600)
st.plotly_chart(fig)

column = st.selectbox('Select the Column to Forecast', data.columns[1:])

data = data[['Date', column]]
st.write('Selected Column')
st.write(data)

st.header('Is Data Stationary?')
st.write(sm.tsa.adfuller(data[column])[1] < 0.05)

st.header('Decomposed Data')
decomposition = sm.tsa.seasonal_decompose(data[column], model='additive', period=30)
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title="Trend", width=950, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title="Seasonality", width=950, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title="Residuals", width=950, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='red', line_dash='dot'))

st.write("---")

st.write("<p style='color:#336fbb; font-size:50px; font-weight:bold;'>Forecasting the Data</p>", unsafe_allow_html=True)
forecast_period = st.number_input('Enter the number of days to forecast', 1, 365, 30)

# Perform auto-arima to automatically select the best parameters
auto_model = auto_arima(data[column], seasonal=True, m=12, trace=True)

# Get the best parameters
best_params = auto_model.get_params()

# Fit the final model with the best parameters
final_model = sm.tsa.statespace.SARIMAX(data[column], order=(best_params['order']), seasonal_order=(best_params['seasonal_order']))
final_model = final_model.fit()

# Make predictions
predictions = final_model.forecast(steps=forecast_period)

# Create dataframe for predictions
forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=forecast_period + 1)
forecast_data = pd.DataFrame({'Date': forecast_dates[1:], 'Predicted': predictions})

st.write("## Predicted Data", forecast_data)

st.write("---")

# Plot actual vs predicted
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast_data['Date'], y=forecast_data["Predicted"], mode='lines', name='Predicted', line=dict(color='red')))
fig.update_layout(title='Actual vs. Predicted', xaxis_title='Date', yaxis_title='Price', width=950, height=600)
st.plotly_chart(fig)

st.write("---")

st.write("Connect with me on Social Media")

#insta_logo = "https://icones.pro/wp-content/uploads/2021/02/instagram-logo-icone
