import math
from typing import Tuple

from lidarNet.utils.geo_utils import Point, LatLong

MERCATOR_RANGE = 256
pixel_origin = Point(MERCATOR_RANGE / 2, MERCATOR_RANGE / 2)
pixels_per_long_degree = MERCATOR_RANGE / 360
pixels_per_long_radian = MERCATOR_RANGE / (2 * math.pi)


def __bound(value, low, high):
    return max(low, min(high, value))


def from_latlong_to_point(lat_long: LatLong) -> Point:
    x = pixel_origin.x + lat_long.long * pixels_per_long_degree
    sin_y = __bound(math.sin(math.radians(lat_long.lat)), -0.9999, 0.9999)
    y = pixel_origin.y + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -pixels_per_long_radian
    return Point(x, y)


def from_point_to_latlong(point: Point) -> LatLong:
    long = (point.x - pixel_origin.x) / pixels_per_long_degree
    lat_radians = (point.y - pixel_origin.y) / -pixels_per_long_radian
    lat = math.degrees(2 * math.atan(math.exp(lat_radians)) - math.pi / 2)
    return LatLong(lat, long)


def get_center_from_nw_corner(corner: LatLong, zoom: int, width: int, height: int) -> LatLong:
    scale = 2 ** zoom
    corner_point = from_latlong_to_point(corner)
    corner_point.x += (width / 2) / scale
    corner_point.y += (height / 2) / scale
    return from_point_to_latlong(corner_point)


def get_bounds_from_nw_corner(corner: LatLong, zoom: int, width: int, height: int) -> Tuple[
    LatLong, LatLong, LatLong, LatLong]:
    scale = 2 ** zoom
    nw_point = from_latlong_to_point(corner)
    ne_point = Point(nw_point.x, nw_point.y + height / scale)
    se_point = Point(nw_point.x + width / scale, nw_point.y + height / scale)
    sw_point = Point(nw_point.x + width / scale, nw_point.y)
    return tuple([corner] + list(map(from_point_to_latlong, [ne_point, sw_point, se_point])))
