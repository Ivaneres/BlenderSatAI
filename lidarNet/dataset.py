import os
import re

import numpy as np
import rasterio
import torch.nn.functional as F
from torch.utils.data import Dataset


class LidarEnlarge:

    def __init__(self, new_dim):
        self.new_dim = new_dim

    def __call__(self, x):
        h_old, w_old = x.shape[1:3]
        new_dim = self.new_dim
        h_new, w_new = new_dim
        if h_old > h_new or w_old > w_new:
            raise ValueError("New dimensions cannot be smaller than old dimensions")
        incr_top, inc_bottom = np.ceil((h_new - h_old) / 2).astype(int), np.floor((h_new - h_old) / 2).astype(int)
        incr_left, incr_right = np.ceil((w_new - w_old) / 2).astype(int), np.floor((w_new - w_old) / 2).astype(int)
        return F.pad(x, (incr_top, inc_bottom, incr_left, incr_right), "reflect")


class LidarNetDataset(Dataset):
    def __init__(self, rgb_dir, lidar_dir, transform=None, target_transform=None):
        self.img_dir = rgb_dir
        self.lidar_dir = lidar_dir
        self.rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".tif")]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_filename = self.rgb_files[idx]
        lidar_filename = re.sub(r"rgb", "ndsm", rgb_filename)

        lidar_filepath = os.path.join(self.lidar_dir, lidar_filename)
        rgb_filepath = os.path.join(self.img_dir, rgb_filename)

        with rasterio.open(lidar_filepath) as lidar_rio:
            lidar = lidar_rio.read(1)
        with rasterio.open(rgb_filepath) as rgb_rio:
            rgb = np.transpose(rgb_rio.read((3, 2, 1)), axes=(1, 2, 0))

        if self.transform:
            rgb = self.transform(rgb)
        if self.target_transform:
            lidar = self.target_transform(lidar)
        return rgb, lidar


def calculate_unet_output_dim(input_dim, num_layers):
    cur_dim = input_dim
    for _ in range(num_layers):
        cur_dim -= 4
        cur_dim //= 2
    for _ in range(num_layers):
        cur_dim -= 4
        cur_dim *= 2
    cur_dim -= 4
    return cur_dim


def calculate_input_dim(target_dim, num_layers):
    cur_input_dim = target_dim
    output_dim = calculate_unet_output_dim(cur_input_dim, num_layers)
    while output_dim < target_dim:
        cur_input_dim += 1
        output_dim = calculate_unet_output_dim(cur_input_dim, num_layers)
    return cur_input_dim - 1
