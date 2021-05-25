# import streamlit

# step 1: setup basic streamlit app
# step 2: use a generic time series dataset (see what datasets python has)
# step 3: build forecast using the naive mean, GreyKite, FB Prophet, Python's auto.arima version, 
#         and a deep learning (LSTM?) model
# step 4: save off(pickle?) the 3 models built above, and deploy them into GCP. then can query from hear
# step 5: compare evaluation metrics of 3 models
# step 6: allow users to input their own datasets and specify parameters (horizon, CV window, other?)

import streamlit as st
import numpy as np
import pandas as pd

def from_data_file(filename):
    