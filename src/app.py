import streamlit as st
import pandas as pd

import plotly.express as px

from data import Data


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
    
