import math
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, List

import numpy as np
import rasterio
from osgeo import gdalconst, gdal
from pyproj.transformer import TransformerGroup, Transformer
from rasterio.control import GroundControlPoint as GCP


@dataclass
class LatLong:
    lat: float
    long: float

    def __str__(self):
        return f"{self.lat},{self.long}"

    def to_tuple(self):
        return self.long, self.lat


@dataclass
class Point:
    x: float
    y: float


def convert_crs(coords: Tuple[float, float], src_crs: str, target_crs: str) -> Tuple[float, float]:
    tg = TransformerGroup(27700, 4326)
    transformer = Transformer.from_crs(src_crs, target_crs)
    # print(tg.unavailable_operations[0].name)
    tg.download_grids(verbose=True)
    return transformer.transform(*coords)


def meters_per_px(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(lat * math.pi / 180) / math.pow(2, zoom)


def load_lidar_data(fp: str, fill_empty=False, fill_method="mean") -> Tuple[rasterio.DatasetReader, np.ndarray]:
    raster = rasterio.open(fp)
    raster_data = raster.read(1)
    if fill_empty:
        if fill_method == "mean":
            raster_data[raster_data < 0] = raster_data[raster_data >= 0].mean()
    return raster, raster_data


def visualise_height_data(visible, height):
    alpha = 0.5
    boundaries = [3.1630, 10]
    visible[(height > boundaries[0]) & (height < boundaries[1]), 0] += alpha * 1.0
    visible[(height > boundaries[0]) & (height < boundaries[1]), 1] += alpha * 0.5
    visible[height <= boundaries[0], 1] += alpha * 1.0
    visible[height >= boundaries[1], 0] += alpha * 1.0


def gcps_from_bounds(bounds, dims):
    return (
        GCP(0, 0, *bounds[0].to_tuple()),
        GCP(0, dims[1], *bounds[2].to_tuple()),
        GCP(dims[0], 0, *bounds[1].to_tuple()),
        GCP(dims[0], dims[1], *bounds[3].to_tuple())
    )


def get_bounds_from_raster(raster: rasterio.DatasetReader) -> Tuple[LatLong, LatLong, LatLong, LatLong]:
    """
    :returns: NW, NE, SW, SE corners
    """
    transform = raster.transform
    nw = transform * (0, 0)
    ne = transform * (raster.width, 0)
    sw = transform * (0, raster.height)
    se = transform * (raster.width, raster.height)
    return nw, ne, sw, se


def create_raster_from_transform(transform, crs, data, fp, dims, channels: int):
    raster_writer = rasterio.open(
        fp,
        "w",
        driver="GTiff",
        height=dims[0],
        width=dims[1],
        count=channels,
        dtype=data.dtype,
        crs=crs,
        transform=transform
    )

    raster_writer.write(data)
    raster_writer.close()


def align_rasters(raster_in: gdal.Dataset, raster_match_fp, raster_out_fp):
    src_proj = raster_in.GetProjection()

    match_ds = gdal.Open(raster_match_fp, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    dst = gdal.GetDriverByName('GTiff').Create(raster_out_fp, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    gdal.ReprojectImage(raster_in, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)
    del dst


def fiona_shape(points: List[LatLong]) -> Dict:
    return {
        "geometry": {
            "type": "LineString",
            "coordinates": [p.to_tuple() for p in points]
        }
    }


def line_func_from_two_points(point1: Point, point2: Point) -> Callable[[float], float]:
    """
    :returns: mx + c
    """
    m = (point1.y - point2.y) / (point1.x - point2.x)
    c = point1.y - m * point1.x

    def line_func(x: float) -> float:
        return m * x + c

    return line_func


def reverse_line_func_from_two_points(point1: Point, point2: Point) -> Callable[[float], float]:
    """
    :returns: (y - c) / m
    """
    m = (point1.y - point2.y) / (point1.x - point2.x)
    c = point1.y - m * point1.x

    def reverse_line_func(y: float) -> float:
        return (y - c) / m

    return reverse_line_func
