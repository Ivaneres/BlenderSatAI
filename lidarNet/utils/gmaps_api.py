from io import BytesIO

import numpy as np
import requests
from matplotlib.image import imread

from lidarNet.utils.geo_utils import LatLong

API_KEY = "AIzaSyB_MQrIkzvHx6YttJA8jcJFkiQVJ9PZ6nU"
RES = 640
CROP = 25
SATELLITE_IMG_DIM = RES - CROP * 2


def get_gmap_satellite_image(center: LatLong, zoom: int) -> np.ndarray:
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={center.lat},{center.long}&zoom={zoom}&size={RES}x{RES}&maptype=satellite&key={API_KEY}"
    r = requests.get(url)
    if r.status_code != 200:
        raise ConnectionError(f"Failed to connect to gmaps API, status code {r.status_code}")
    b = BytesIO(r.content)
    b.name = "placeholder.jpeg"
    img = imread(b)
    img = img[CROP:RES - CROP, CROP:RES - CROP]
    return img
