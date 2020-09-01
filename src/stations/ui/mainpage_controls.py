import streamlit as st

from stations.ui.ifu_data import Resolution


def select_data_resolution():
    return st.sidebar.selectbox(
        "Data resolution",
        Resolution.names(),
        format_func=lambda x: x.value.replace("_", " ").capitalize(),
    )


def select_max_station_distance():
    return st.sidebar.slider(
        "Distance to IFU [km]", min_value=10, max_value=150, value=50, step=5
    )


def select_observation_years():
    return st.sidebar.slider("Data coverage [year]", 1990, 2020, (2010, 2019))
