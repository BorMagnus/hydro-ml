import plotly.express as px
import streamlit as st


def main():
    dataframe = st.session_state["df"]
    x_axis_val = "Datetime"
    y_axis_val = st.selectbox("Select Y-Axis Value", options=dataframe.columns)

    plot = px.line(dataframe, x=x_axis_val, y=y_axis_val)
    st.plotly_chart(plot)

    results = st.session_state["analysis"]
    st.header("Training results")
    df = results.results_df
    st.write(
        df[
            [
                "config/model",
                "train_loss",
                "val_loss",
                "test_loss",
                "time_total_s",
                "config/data/variables",
            ]
        ].sort_values("test_loss")
    )


if __name__ == "__main__":
    st.title("Visualization")

    if "df" and "file_name" in st.session_state:
        dataframe = st.session_state["df"]
        file_name = st.session_state["file_name"]
        main()
    else:
        st.error("Need to upload file!")
