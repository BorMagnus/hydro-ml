import plotly.express as px
import streamlit as st


dataframe = st.session_state['df']
x_axis_val = 'Datetime'
y_axis_val = st.selectbox('Select Y-Axis Value', options=dataframe.columns)

plot = px.line(dataframe, x=x_axis_val, y=y_axis_val)
st.plotly_chart(plot)

results = st.session_state['analysis']
st.header('Training results')
df = results.results_df
st.write(df[['config/arch', 'train_loss', 'val_loss', 'test_loss', 'config/variables']].sort_values('test_loss'))
