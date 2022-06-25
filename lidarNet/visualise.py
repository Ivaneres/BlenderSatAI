import matplotlib.pyplot as plt
import numpy as np
import rasterio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision import transforms

from lidarNet.utils.ml_utils import crop_masks


def vis_row_2(sat_fp, lidar_fp):
    sat_np = rasterio.open(sat_fp).read((3, 2, 1))
    gt_np = rasterio.open(lidar_fp).read(1)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.transpose(sat_np, axes=(1, 2, 0)))
    im = axs[1].imshow(gt_np)

    cbar_ax = fig.add_axes([0.91, 0.235, 0.02, 0.52])
    cbar = fig.colorbar(im, cax=cbar_ax)

    for ax in axs:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def vis_row_3(sat, gt, preds):
    sat = rasterio.open(sat).read((3, 2, 1))
    gt = rasterio.open(gt).read(1)
    preds = rasterio.open(preds).read(1)

    gt[gt < -1] = gt[gt > 0].mean()

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(np.transpose(sat, axes=(1, 2, 0)))
    axs[1].imshow(gt)
    im = axs[2].imshow(preds)

    cbar_ax = fig.add_axes([0.91, 0.32, 0.02, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax)

    for ax in axs:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    vis_row_3("./testing/rgb_rah.tif", "./testing/gt_rah.tif", "./testing/heights_rah.tif")
