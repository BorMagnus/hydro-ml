import os
import random
import sys
from functools import partial
from typing import List

import pandas as pd
import ray
import streamlit as st
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Make sure that the src folder is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data import Data
from train import train_model


def describe_dataframe(dataframe):
    st.header("Data Statistics")
    st.write(dataframe.describe())


st.title("Hydrological Forecasting Tool")
st.header("What does the tool do?")
st.markdown("This tool allows you to experiment with various machine learning models for predicting hydrological forecasting. With a simple and user-friendly interface, you can upload your data, select the forecasting model, and view the results.")
st.header("How does it work?")
st.write("The application is divided into three main sections:")
st.markdown("- **App/Home page:** Here, you can upload your dataset in CSV format. Please ensure that your data is prepared and cleaned before upload.")
st.markdown("- **Training page:** Select your datetime and target variables, then choose the machine learning model for your forecast. You can opt for FCN, LSTM, LSTM with temporal attention, or LSTM with spatio-temporal attention models. The app also allows you to define the variables for hyper-parameter search, providing you with a comprehensive model comparison. The training progress is displayed visually for your convenience.")
st.markdown("- **Visualization page:** Upon successful model training, this page presents the results in an easily digestible format. You will see graphical plots of your data, predicted versus actual values, and a table detailing the performance metrics of the models.")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file to upload",
    type=["csv"],
    help="The uploaded file need to be an CSV file with the target and datetime values in it.",
)
if uploaded_file:
    dataframe = pd.read_csv(uploaded_file)
    if "df" not in st.session_state:
        st.session_state["df"] = dataframe
    if "file_name" not in st.session_state:
        st.session_state["file_name"] = uploaded_file.name

    describe_dataframe(dataframe)
