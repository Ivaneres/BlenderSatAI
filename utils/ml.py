import os
from typing import List
import tifffile as tiff
import matplotlib.pyplot as plt


def load_dir_filenames(path: str, extension: str) -> List[str]:
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(extension)]


def display_tiff(path: str) -> None:
    img = tiff.imread(path)
    tiff.imshow(img)
