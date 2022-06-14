from io import BytesIO
from typing import Tuple
import requests
import math
import numpy as np
from matplotlib.image import imread

from lidarNet.utils.geo_utils import convert_to_epsg4326

ZOOM = 18
URL = "http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z=" + str(ZOOM)
TILE_SIZE = 256


def project(lat_lng, tile_size):
    sin_y = math.sin((lat_lng[0] * math.pi) / 180)
    sin_y = min(max(sin_y, -0.9999), 0.9999)

    return (
        tile_size * (0.5 + lat_lng[1] / 360),
        tile_size * (0.5 - math.log((1 + sin_y) / (1 - sin_y)) / (4 * math.pi))
    )


def tile_coords(wrld_coords, tile_size, zoom):
    scale = 1 << zoom
    return (
        math.floor((wrld_coords[0] * scale) / tile_size),
        math.floor((wrld_coords[1] * scale) / tile_size)
    )


def latlong_to_tile(lat_lng, tile_size, zoom):
    return tile_coords(project(lat_lng, tile_size), tile_size, zoom)


def get_tile(lats: Tuple[float, float], longs: Tuple[float, float], src_crs) -> np.ndarray:
    # Google MT server can't be used directly like this according to rules, so don't use this.
    lat_long = convert_to_epsg4326(lats, longs, src_crs)
    lat, long = latlong_to_tile(lat_long, TILE_SIZE, ZOOM)
    url = URL.replace("{x}", str(lat)).replace("{y}", str(long))
    r = requests.get(url)
    if not r.status_code == 200:
        raise ConnectionError(f"Failed to connect to tile server, error code: {r.status_code}")
    b = BytesIO(r.content)
    b.name = "placeholder.jpeg"
    return imread(b)


def get_tiles(lat_long: Tuple[float, float], n: int) -> np.ndarray:
    latlong_to_tile(lat_long, TILE_SIZE, 18)
