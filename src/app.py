import streamlit as st
import pandas as pd

import plotly.express as px

from data import Data

def stats(dataframe):
    st.header('Data Statistics')
    st.write(dataframe.describe())

def plot(dataframe):
    x_axis_val = 'Datetime'
    y_axis_val = st.selectbox('Select Y-Axis Value', options=dataframe.columns)

    plot = px.line(dataframe, x=x_axis_val, y=y_axis_val)
    st.plotly_chart(plot)


st.title('Inflow forecasting')
st.text('This is an app to create inflow forecast')

st.sidebar.title('Navigation')

options = st.sidebar.radio('Pages', options=['Home', 'Data Statistics', 'Data Header', 'Plot'])

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file:
    dataframe = pd.read_csv(uploaded_file)
    st.session_state['df'] = dataframe
    
if options == 'Data Statistics':
    stats(dataframe)
elif options == 'Plot':
    plot(dataframe)