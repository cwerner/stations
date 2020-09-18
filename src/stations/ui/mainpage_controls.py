import streamlit as st
from wetterdienst.enumerations.time_resolution_enumeration import TimeResolution


def select_data_resolution():
    return st.sidebar.selectbox(
        "Data resolution",
        [
            x
            for x in list(TimeResolution)
            if x in [TimeResolution.MINUTES_10, TimeResolution.HOURLY]
        ],
        format_func=lambda x: x.value.replace("_", " ").lower(),
    )


def select_max_station_distance():
    return st.sidebar.slider(
        "Distance to IFU [km]", min_value=10, max_value=150, value=50, step=5
    )


def select_observation_years():
    return st.sidebar.slider("Data coverage [year]", 1990, 2020, (2010, 2019))
