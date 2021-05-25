# import streamlit

# step 1: setup basic streamlit app
# step 2: use a generic time series dataset (see what datasets python has)
# step 3: build forecast using the naive mean, GreyKite, FB Prophet, Python's auto.arima version, 
#         and a deep learning (LSTM?) model
# step 4: save off(pickle?) the 3 models built above, and deploy them into GCP. then can query from hear
# step 5: compare evaluation metrics of 3 models [MAPE horizon 1, MAPE horizon 2, training time, inference time, model size]
# step 6: allow users to input their own datasets and specify parameters (horizon, CV window, other?)

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pmdarima as pm
from pmdarima.model_selection import train_test_split
from prophet import Prophet




# TODO: allow pass in CSV, with first two columns being (1) a Datetime field and (2) values
# TODO: get seasonality and frequency from a streamlit dropdown
# TODO: get horizon from a streamlit dropdown


y = pm.datasets.load_wineind()
train, test = train_test_split(y, train_size=150)

arima = pm.auto_arima(train, seasonal=True, m=12)
forecasts = arima.predict(test.shape[0])  # predict N steps into the future

# Visualize the forecasts (blue=train, green=forecasts)
x = np.arange(y.shape[0])
fig, ax = plt.subplots()

ax.plot(x[:150], train, c='blue')

arima = pm.auto_arima(train, seasonal=True, m=12)
forecasts = arima.predict(test.shape[0])  # predict N steps into the future
ax.plot(x[150:], forecasts, c='green')

prophet = Prophet().fit()
forecasts = arima.predict(test.shape[0])  # predict N steps into the future

st.pyplot(fig)

#st.line_chart(np.concatenate([train, forecasts]))
