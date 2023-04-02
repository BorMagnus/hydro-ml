import streamlit as st

dataframe = st.session_state['df']
st.header('Data Statistics')
st.write(dataframe.describe())