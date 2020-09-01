import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium import plugins

from stations.helper.custom_types import StationsType
from stations.helper.spatial import compute_bounds, compute_center_coordinate
from stations.ui.ifu_data import ifu


@st.cache
def create_chart(df: pd.DataFrame) -> alt.Chart:
    """Create (dummy) charts for popup items"""
    chart = alt.Chart(df).mark_line().encode(x="a", y="b").properties(height=100)
    return chart.to_json()


def define_kit_marker():
    icon_url = (
        "https://www.kit-ausbildung.de/typo3conf/ext/"
        + "dp_contentelements/Resources/Public/img/kit-logo-without-text.svg"
    )
    kit_icon = folium.features.CustomIcon(icon_url, icon_size=(32, 32))

    kit_info = """<b>KIT Campus Alpin</b></br>
<center>
<img src="https://www.imk-ifu.kit.edu/img/gesamtansicht_IFU.gif" width="100px"/>
</center>
<a href="https://www.imk-ifu.kit.edu" target="_blank">Homepage</a>
"""

    return folium.Marker(
        (ifu["geo_lat"], ifu["geo_lon"]),
        tooltip=kit_info,
        popup=kit_info,
        icon=kit_icon,
    )


def create_map(stations: StationsType, tereno_stations: StationsType, dist: int):
    m = folium.Map(location=compute_center_coordinate(stations), tiles=None)
    folium.TileLayer("Stamen Toner", name="Stamen Toner").add_to(m)
    folium.TileLayer("Stamen Terrain", name="Stamen Terrain").add_to(m)
    folium.TileLayer("Stamen Watercolor", name="Stamen Watercolor").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    feature_group_tereno = folium.FeatureGroup("TERENO Sites")
    feature_group_dwd = folium.FeatureGroup("DWD Sites", control=False)

    define_kit_marker().add_to(m)

    for station in tereno_stations:
        folium.Marker(
            (station["geo_lat"], station["geo_lon"]),
            tooltip=f"{station['name']} (TERENO)",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(feature_group_tereno)

    # dwd stations
    for station in stations:
        dummy_df = pd.DataFrame(
            {"a": range(100), "b": np.cumsum(np.random.normal(0, 0.1, 100))}
        )

        folium.Marker(
            (station["geo_lat"], station["geo_lon"]),
            tooltip=f"{station['name']} (id:{station['station_id']})",
            # popup = f"{station['name']} (id:{station['station_id']})",
            popup=folium.Popup(max_width=300).add_child(
                folium.VegaLite(create_chart(dummy_df), width=300, height=100)
            ),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(feature_group_dwd)

    # distance circle
    folium.Circle(
        radius=dist * 1000,
        location=(ifu["geo_lat"], ifu["geo_lon"]),
        dash_array="5",
        tooltip=f"{dist} km to IFU",
        color="crimson",
        fill=False,
    ).add_to(m)

    # fit bounds
    bounds = compute_bounds(stations)
    m.fit_bounds(bounds)

    feature_group_tereno.add_to(m)
    feature_group_dwd.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    plugins.Fullscreen(
        position="topright",
        title="Expand me",
        title_cancel="Exit me",
        force_separate_button=True,
    ).add_to(m)

    return m
