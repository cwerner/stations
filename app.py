import datetime
from typing import Collection, Tuple

import streamlit as st
from dwdweather import DwdWeather
from streamlit_folium import folium_static

from stations.helper.custom_types import StationsType
from stations.ui.ifu_data import Resolution, ifu, tereno_stations
from stations.ui.mainpage_controls import (
    select_data_resolution,
    select_max_station_distance,
    select_observation_years,
)
from stations.ui.mainpage_map import create_map
from stations.ui.metrics import REGISTRY

METRICS = REGISTRY.get_metrics()


@st.cache
@METRICS.REQUEST_TIME.time()
def find_close_stations(
    dist: int = 50, res: Resolution = Resolution.HOURLY
) -> StationsType:
    """Find closest stations (dist: radius in km)"""

    dwd = DwdWeather(resolution=res.value)
    return dwd.nearest_station(
        lat=ifu["geo_lat"], lon=ifu["geo_lon"], surrounding=dist * 1000
    )


# @st.cache
def fetch_data(res: Resolution = Resolution.HOURLY):

    # Create client object.
    dwd = DwdWeather(resolution=res.value)

    # Find closest station to position.
    nearest = dwd.nearest_station(lat=ifu["geo_lat"], lon=ifu["geo_lon"])

    st.write(nearest)

    # The hour you're interested in.
    # The example is 2014-03-22 12:00 (UTC).
    query_hour = datetime.datetime(2014, 3, 22, 12, 10)

    result = dwd.query(station_id=nearest["station_id"], timestamp=query_hour)
    return result


def filter_by_dates(stations: StationsType, start: int, end: int) -> StationsType:
    filtered = []
    for station in stations:
        start_date = datetime.datetime.strptime(str(station["date_start"]), "%Y%m%d")
        if start_date.day != 1 or start_date.month != 1:
            start_year = start_date.year + 1
        else:
            start_year = start_date.year

        end_date = datetime.datetime.strptime(str(station["date_end"]), "%Y%m%d")
        end_year = end_date.year

        if start_year <= start and end_year >= end:
            filtered.append(station)
    return filtered


def create_sidebar() -> Tuple[Resolution, int, Collection[int]]:
    res = select_data_resolution()
    dist = select_max_station_distance()
    years = select_observation_years()
    return res, dist, years


def create_mainpage(
    filtered_stations: StationsType, tereno_stations: StationsType, dist: int
):
    st.title("Stations near IMK-IFU/ KIT 🏔🌦")
    st.write(f"Number of stations: {len(filtered_stations)}")

    station_map = create_map(filtered_stations, tereno_stations, dist)
    folium_static(station_map)


def main():

    # site config
    st.beta_set_page_config(
        page_title="DWD Stations",
        initial_sidebar_state="expanded",
    )

    data_resolution, max_station_distance, observation_years = create_sidebar()

    closest_stations = find_close_stations(
        dist=max_station_distance, res=data_resolution
    )
    filtered_stations = filter_by_dates(closest_stations, *observation_years)

    create_mainpage(filtered_stations, tereno_stations, max_station_distance)


if __name__ == "__main__":
    main()
