from typing import Tuple

import numpy as np
import streamlit as st

from stations.helper.custom_types import StationsType


@st.cache
def compute_center_coordinate(stations: StationsType) -> Tuple[float, float]:
    lat = np.array([x["geo_lat"] for x in stations]).mean()
    lon = np.array([x["geo_lon"] for x in stations]).mean()
    return float(lat), float(lon)


@st.cache
def compute_bounds(
    stations: StationsType,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    min_lat = np.min(np.array([x["geo_lat"] for x in stations]))
    min_lon = np.min(np.array([x["geo_lon"] for x in stations]))
    max_lat = np.max(np.array([x["geo_lat"] for x in stations]))
    max_lon = np.max(np.array([x["geo_lon"] for x in stations]))
    return (float(min_lat), float(min_lon)), (float(max_lat), float(max_lon))
