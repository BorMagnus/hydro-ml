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


st.title("Inflow forecasting")
st.text("This is an app to create inflow forecast")
st.text("Info page!")

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
