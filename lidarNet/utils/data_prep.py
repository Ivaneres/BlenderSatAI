from typing import Tuple, Dict

import numpy as np
import rasterio
from osgeo import gdalconst, gdal

from lidarNet.utils.geo_utils import load_lidar_data, convert_crs, LatLong, gcps_from_bounds, \
    create_raster_from_transform, align_rasters, get_bounds_from_raster, Point, fiona_shape, line_func_from_two_points, \
    reverse_line_func_from_two_points
from lidarNet.utils.gmaps_api import SATELLITE_IMG_DIM, get_gmap_satellite_image
from lidarNet.utils.mercator import get_center_from_nw_corner, get_bounds_from_nw_corner, from_latlong_to_point, \
    from_point_to_latlong

import fiona
from fiona.crs import from_epsg


def create_dataset(ndsm_fp: str, rgb_out_dir: str, lidar_out_dir: str) -> None:
    ZOOM = 18
    gmaps_dims = (SATELLITE_IMG_DIM,) * 2

    ndsm = rasterio.open(ndsm_fp)
    ndsm_bounds = get_bounds_from_raster(ndsm)
    ndsm_bounds = list(map(lambda x: LatLong(*convert_crs(x, ndsm.crs, "EPSG:4326")), ndsm_bounds))
    ndsm_bounds_points = list(map(from_latlong_to_point, ndsm_bounds))
    ndsm_nw_point, ndsm_ne_point, ndsm_sw_point, ndsm_se_point = ndsm_bounds_points

    ndsm_top_line_func = line_func_from_two_points(ndsm_nw_point, ndsm_ne_point)
    ndsm_bottom_line_func = line_func_from_two_points(ndsm_sw_point, ndsm_se_point)
    ndsm_right_reverse_line_func = reverse_line_func_from_two_points(ndsm_ne_point, ndsm_se_point)
    ndsm_left_reverse_line_func = reverse_line_func_from_two_points(ndsm_nw_point, ndsm_sw_point)

    gmaps_scale = 2 ** ZOOM
    gmaps_tile_width = gmaps_dims[0] / gmaps_scale
    gmaps_tile_height = gmaps_dims[1] / gmaps_scale
    tiling_y_delta = ndsm_top_line_func(ndsm_nw_point.x + gmaps_tile_width) - ndsm_nw_point.y

    def check_x_within_bounds_right(nw_p: Point):
        return (nw_p.x + gmaps_tile_width < ndsm_right_reverse_line_func(nw_p.y) and
                nw_p.x + gmaps_tile_width < ndsm_right_reverse_line_func(nw_p.y + gmaps_tile_height))

    def check_x_within_bounds_left(nw_p: Point):
        return nw_p.x > ndsm_left_reverse_line_func(nw_p.y + gmaps_tile_height)

    def check_y_within_bounds(nw_p: Point):
        return (nw_p.y + gmaps_tile_height < ndsm_bottom_line_func(
            nw_p.x) and nw_p.y + gmaps_tile_height < ndsm_bottom_line_func(nw_p.x + gmaps_tile_width))

    ndsm_gdal = gdal.Open(ndsm_fp, gdalconst.GA_ReadOnly)

    fiona_epsg = from_epsg(4326)
    schema = {
        "geometry": "LineString"
    }
    with fiona.open("./temp.shp", "w", crs=fiona_epsg, driver="ESRI Shapefile", schema=schema) as shpfile:
        cur_tile_nw_point = Point(ndsm_nw_point.x, ndsm_nw_point.y)
        column_counter = 0
        while check_x_within_bounds_right(cur_tile_nw_point):
            if not check_x_within_bounds_left(cur_tile_nw_point):
                cur_tile_nw_point.x += gmaps_tile_width
                continue
            cur_tile_nw_point.y += tiling_y_delta
            start_y = cur_tile_nw_point.y
            row_counter = 0
            while check_y_within_bounds(cur_tile_nw_point):
                if not check_x_within_bounds_right(cur_tile_nw_point):
                    break

                gmaps_center = get_center_from_nw_corner(
                    from_point_to_latlong(cur_tile_nw_point),
                    ZOOM,
                    SATELLITE_IMG_DIM,
                    SATELLITE_IMG_DIM
                )

                gmaps_bounds = get_bounds_from_nw_corner(
                    from_point_to_latlong(cur_tile_nw_point),
                    ZOOM,
                    SATELLITE_IMG_DIM,
                    SATELLITE_IMG_DIM
                )

                gmaps_nw, gmaps_ne, gmaps_sw, gmaps_se = gmaps_bounds
                shpfile.write(fiona_shape([gmaps_nw, gmaps_ne, gmaps_se, gmaps_sw, gmaps_nw]))

                gcps = gcps_from_bounds(gmaps_bounds, gmaps_dims)
                transform = rasterio.transform.from_gcps(gcps)
                gmaps_tile_np = get_gmap_satellite_image(gmaps_center, ZOOM)

                rgb_tile_fp = f"{rgb_out_dir}/col{column_counter}_row{row_counter}_rgb.tif"
                ndsm_tile_fp = f"{lidar_out_dir}/col{column_counter}_row{row_counter}_ndsm.tif"

                create_raster_from_transform(
                    transform,
                    rasterio.CRS(init="EPSG:4326"),
                    np.transpose(gmaps_tile_np[..., :3], axes=(2, 0, 1)),
                    rgb_tile_fp,
                    gmaps_dims
                )

                align_rasters(ndsm_gdal, rgb_tile_fp, ndsm_tile_fp)

                cur_tile_nw_point.y += gmaps_tile_height
                row_counter += 1
            cur_tile_nw_point.x += gmaps_tile_width
            cur_tile_nw_point.y = start_y
            print(f"Column {column_counter}")
            column_counter += 1

    print(f"Total tiles (approx): {column_counter * row_counter}")


create_dataset("./salisbury/salisbury_ndsm_clipped.tif", "./lidarNet/data/salisbury/rgb", "./lidarNet/data/salisbury/lidar")
