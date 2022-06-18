import shutil

import numpy as np
import rasterio
import torch
from matplotlib import pyplot as plt, gridspec
from torch.utils.data import RandomSampler
from torchvision.transforms import transforms

from lidarNet.dataset import LidarEnlarge
from lidarNet.utils.geo_utils import visualise_height_data, create_raster_from_transform, LatLong, gcps_from_bounds
from lidarNet.utils.gmaps_api import get_gmap_satellite_image
from lidarNet.utils.mercator import get_bounds_from_nw_corner, from_latlong_to_point, from_point_to_latlong


def crop_masks(masks, dims):
    # dims smaller than masks
    if masks.shape[-2] < dims[0] or masks.shape[-1] < dims[1]:
        raise ValueError("Dims must be smaller than masks")
    is_batch = False
    if len(masks.shape) == 4:
        # batch, channels, y, x
        is_batch = True
    h_old, w_old = masks.shape[-2:]
    h_new, w_new = dims
    h_diff = h_old - h_new
    w_diff = w_old - w_new
    top, bottom = h_diff // 2, h_diff - h_diff // 2
    left, right = w_diff // 2, w_diff - w_diff // 2
    if not is_batch:
        return masks[:, top: h_old - bottom, left: w_old - right]
    return masks[:, :, top: h_old - bottom, left: w_old - right]


def visualise_test(mdl, test_data, device):
    mdl.eval()
    n_col = 6
    n_row = 1

    r = RandomSampler(test_data, num_samples=n_col)

    n_col //= 2
    n_row *= 2
    fig = plt.figure(figsize=(n_row + 1, n_col + 2))
    axs = []
    gs = gridspec.GridSpec(n_col, n_row, wspace=0.0, hspace=0.0,
       top=1. - 0.5 / (n_col + 1), bottom=0.5 / (n_col + 1),
       left=0.5 / (n_row + 1), right=1 - 0.5 / (n_row + 1)
    )

    for i in range(n_col):
        for j in range(n_row):
            ax = plt.subplot(gs[i, j])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            axs.append(ax)

    for i, d_idx in enumerate(r):
        test_in, test_gt = test_data[d_idx]
        test_in = test_in.to(device)

        with torch.no_grad():
            outputs = mdl(test_in.unsqueeze(dim=0)).squeeze(dim=0)
        test_gt = crop_masks(np.expand_dims(test_gt, axis=0), outputs.shape[-2:])
        test_net = outputs[0].cpu()
        print(test_net.shape)

        test_in = crop_masks(test_in, (325, 325))
        test_in = transforms.Resize((650, 650))(test_in)
        test_in = crop_masks(test_in, outputs.shape[-2:])

        test_in = test_in.cpu()
        test_in_clone = test_in.clone()
        test_in[0][test_gt == 0.0] += 0.3
        test_in[0][test_in[0] > 1.0] = 1.0
        test_in_clone[0][test_net < 0.5] += 0.3
        test_in_clone[0][test_in_clone[0] > 1.0] = 1.0
        visible = test_in.permute(1, 2, 0).detach().numpy()
        height_preds = test_net.detach().numpy()
        visualise_height_data(visible, height_preds)
        axs[i].imshow(visible)
        # axs[i, 0].imshow(test_gt)
        # axs[(i * 2) + 1].imshow(test_net)


def test_from_raster(col: int, row: int, rgb_dir: str, lidar_dir: str, out_dir: str, mdl, transform, device):
    rgb_raster_fp = f"{rgb_dir}/col{col}_row{row}_rgb.tif"
    lidar_raster_fp = f"{lidar_dir}/col{col}_row{row}_ndsm.tif"
    rgb_raster = rasterio.open(rgb_raster_fp)

    mdl.eval()
    test_in = np.transpose(rgb_raster.read((3, 2, 1)), axes=(1, 2, 0))
    img = np.transpose(np.copy(test_in), axes=(2, 0, 1))
    test_in = transform(test_in)
    test_in = test_in.to(device)

    with torch.no_grad():
        outputs = mdl(test_in.unsqueeze(dim=0)).squeeze(dim=0)
    height_preds = outputs[0].cpu()
    height_preds = transforms.Resize(img.shape[-2:])(torch.unsqueeze(height_preds, dim=0))

    create_raster_from_transform(
        rgb_raster.transform,
        rgb_raster.crs,
        height_preds.detach().numpy(),
        f"./{out_dir}/model_preds_20_epoch_transconv.tif",
        height_preds.shape[-2:],
        channels=1
    )

    shutil.copy(rgb_raster_fp, f"{out_dir}/rgb.tif")
    shutil.copy(lidar_raster_fp, f"{out_dir}/gt.tif")


def run_from_coords(coords: LatLong, mdl, transform, device, out_dir, test_name, gmaps_dim=590):
    ZOOM = 18
    tile = get_gmap_satellite_image(coords, ZOOM)
    tile = np.transpose(tile[..., :3], axes=(2, 0, 1))

    scale = 2 ** ZOOM
    gmaps_dim_mercator = gmaps_dim / scale

    center_point = from_latlong_to_point(coords)
    center_point.x -= gmaps_dim_mercator
    center_point.y -= gmaps_dim_mercator

    gmaps_bounds = get_bounds_from_nw_corner(
        from_point_to_latlong(center_point),
        ZOOM,
        gmaps_dim,
        gmaps_dim
    )
    gcps = gcps_from_bounds(gmaps_bounds, (gmaps_dim,) * 2)
    gmaps_transform = rasterio.transform.from_gcps(gcps)

    create_raster_from_transform(
        gmaps_transform,
        rasterio.CRS(init="EPSG:4326"),
        tile,
        f"{out_dir}/rgb_{test_name}.tif",
        (gmaps_dim,) * 2,
        channels=3
    )

    mdl.eval()
    test_in = np.transpose(tile, axes=(1, 2, 0))
    img = np.transpose(np.copy(test_in), axes=(2, 0, 1))
    test_in = transform(test_in)
    test_in = test_in.to(device)

    with torch.no_grad():
        outputs = mdl(test_in.unsqueeze(dim=0)).squeeze(dim=0)
    height_preds = outputs[0].cpu()
    height_preds = transforms.Resize(img.shape[-2:])(torch.unsqueeze(height_preds, dim=0))

    create_raster_from_transform(
        gmaps_transform,
        rasterio.CRS(init="EPSG:4326"),
        height_preds,
        f"{out_dir}/heights_{test_name}.tif",
        (gmaps_dim,) * 2,
        channels=1
    )

