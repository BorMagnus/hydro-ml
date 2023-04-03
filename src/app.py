import streamlit as st
import pandas as pd

import os
import sys

# Get the absolute path of the parent directory (Hydro-ML)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the root directory to sys.path
sys.path.append(root_dir)


import plotly.express as px


def describe_dataframe(dataframe):
    st.header('Data Statistics')
    st.write(dataframe.describe())


st.title('Inflow forecasting')
st.text('This is an app to create inflow forecast')
st.text('Info page!')


uploaded_file = st.sidebar.file_uploader("Select file") #TODO Do not need file_name for application?
if uploaded_file:

    dataframe = pd.read_csv(uploaded_file)
    st.session_state['df'] = dataframe
    st.session_state['file_name'] = uploaded_file.name

    describe_dataframe(dataframe)
    #data = Data(data_file, datetime_variable)
    