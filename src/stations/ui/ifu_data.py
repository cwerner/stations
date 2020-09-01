from enum import Enum

# ifu
ifu = {"name": "IFU", "geo_lat": 47.476180, "geo_lon": 11.063350}

# tereno stations
tereno_stations = [
    {"name": "Fendth", "geo_lat": 47.83243, "geo_lon": 11.06111},
    {"name": "Grasswang", "geo_lat": 47.57026, "geo_lon": 11.03189},
    {"name": "Rottenbuch", "geo_lat": 47.73032, "geo_lon": 11.03189},
]


class Resolution(Enum):
    TENMIN = "10_minutes"
    HOURLY = "hourly"
    DAILY = "daily"

    @staticmethod
    def names():
        return list(map(lambda c: c, Resolution))
