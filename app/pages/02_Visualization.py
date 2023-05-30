import os
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from src.evaluate import box_plot, calculate_model_metrics, plot_pred_actual


def main():
    st.header("Data Visualization")
    dataframe = st.session_state["df"]
    x_axis_val = "Datetime"
    y_axis_val = st.selectbox("Select Y-Axis Value", options=dataframe.columns)

    plot = px.line(dataframe, x=x_axis_val, y=y_axis_val)
    st.plotly_chart(plot)
    

if __name__ == "__main__":
    st.title("Visualization")

    if "df" and "file_name" in st.session_state:
        dataframe = st.session_state["df"]
        file_name = st.session_state["file_name"]
        main()
    else:
        st.error("Need to upload file!")

    # List all model directories
    ray_results = Path("../ray_results/")
    if not os.path.exists(ray_results):
        os.makedirs(ray_results)
    model_dirs = [d for d in ray_results.iterdir() if d.is_dir()]

    # Get the experiment names from the full paths
    experiments = [os.path.basename(d) for d in model_dirs]
    st.header("Training results")
    if model_dirs:
        with st.form(key='my_form'):

            col1, col2 = st.columns(2)

            experiment = col1.selectbox("Select experiment:", experiments)
            best = col2.number_input("Number of models", key="num_models", step=1, min_value=1, value=5)
            # Every form must have a submit button.
            submitted = st.form_submit_button("Get Results")
            if submitted:
                model_dfs, parameters = calculate_model_metrics(model_dirs, experiment, best)
                
                # concatenate the dataframes
                df_concat_avg_w_var = pd.concat([model_dfs[k] for k in model_dfs.keys() if experiment in k])

                # calculate the mean of each evaluation metric
                df_avg_w_var = df_concat_avg_w_var.groupby(['model', 'variables']).mean()
                model_var_counts = df_concat_avg_w_var.groupby(['model', 'variables']).size().reset_index(name='counts')
                df_avg_w_var = df_avg_w_var.reset_index()  # reset index so that 'model' and 'variables' become regular columns
                df_avg_w_var = pd.merge(df_avg_w_var, model_var_counts, on=['model', 'variables'])
                
                st.header(f"Best {best} performing models")
                st.write(df_avg_w_var.sort_values("test_mae"))
    
                st.header("Plot of the best performing model")
                st.plotly_chart(plot_pred_actual(model_dirs, experiment))
    else:
        st.error("Need to train a model first!")