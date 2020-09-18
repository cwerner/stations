from datetime import datetime, timedelta
from typing import Collection, Tuple

import streamlit as st
from streamlit_folium import folium_static
from wetterdienst import Parameter, PeriodType, TimeResolution, get_nearby_stations

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
    dist: int = 50,
    res: TimeResolution = TimeResolution.HOURLY,
    start: datetime = datetime(2020, 1, 1),
    end: datetime = datetime(2020, 1, 31),
) -> StationsType:
    """Find closest stations (dist: radius in km)"""

    stations_df = get_nearby_stations(
        ifu["geo_lat"],
        ifu["geo_lon"],
        start,
        end,
        Parameter.TEMPERATURE_AIR,
        res,
        PeriodType.RECENT,
        max_distance_in_km=dist,
    )

    # TODO: use wetterdienst format instead of converting to dwdweather2 conventions
    filtered_stations = []
    for _, row in stations_df.iterrows():
        data = {
            "station_id": row.STATION_ID,
            "geo_lat": row.LAT,
            "geo_lon": row.LON,
            "date_start": row.FROM_DATE.strftime("%Y%m%d"),
            "date_end": row.TO_DATE.strftime("%Y%m%d"),
            "name": row.STATION_NAME,
        }
        filtered_stations.append(data)

    return filtered_stations


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

    today = datetime(
        datetime.now().date().year,
        datetime.now().date().month,
        datetime.now().date().day,
    )
    before_1week = today - timedelta(days=7)

    end_date = (
        today
        if observation_years[1] == before_1week.year
        else datetime(observation_years[1], 12, 31)
    )
    print(f"X: {end_date}")
    closest_stations = find_close_stations(
        dist=max_station_distance,
        res=data_resolution,
        start=datetime(observation_years[0], 1, 1),
        end=end_date,
    )
    create_mainpage(closest_stations, tereno_stations, max_station_distance)


if __name__ == "__main__":
    main()
