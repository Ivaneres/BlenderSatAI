import pandas as pd
from matplotlib import pyplot as plt

from utils.ml import load_dir_filenames, display_tiff
from shapely import wkt
import geopandas as gpd

from utils.spacenet import filename_to_id

rgb_images_dir = "./datasets/AOI_3_Paris_Train/RGB-PanSharpen"
rgb_images_list = load_dir_filenames(rgb_images_dir, ".tif")
summary = pd.read_csv("./datasets/AOI_3_Paris_Train/summaryData/AOI_3_Paris_Train_Building_Solutions.csv", delimiter=",")

sample_img_path = rgb_images_list[0]
img_id = filename_to_id(sample_img_path)
for polygon_str in summary[summary["ImageId"] == img_id]["PolygonWKT_Geo"]:
    polygon = wkt.loads(polygon_str)
    g_ply = gpd.GeoSeries(polygon)
    g_ply.plot()
plt.show()

# display_tiff(rgb_images_list[40])
# display_tiff("/home/ivan/PycharmProjects/SatDetect/datasets/AOI_3_Paris_Train/PAN/PAN_AOI_3_Paris_img345.tif")
