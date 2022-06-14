from argparse import ArgumentParser

import numpy as np

from lidarNet.utils.geo_utils import load_lidar_data


def np_to_obj(arr: np.ndarray) -> str:
    assert len(arr.shape) == 2
    res = []
    INDEX_SCALE = 0.05
    for y, x in np.ndindex(*arr.shape):
        res.append(f"v {(x * INDEX_SCALE):.3f} {(arr[y, x] * INDEX_SCALE):.3f} {(y * INDEX_SCALE):.3f} 1.0")
    vertex_count = len(res)
    print(f"Vertex count: {vertex_count}")
    res.append("")
    for y, x in np.ndindex(arr.shape[0] - 1, arr.shape[1] - 1):
        count = y * arr.shape[1] + x + 1
        res.append(f"f {count} {count + 1} {count + arr.shape[1] + 1} {count + arr.shape[1]}")
    # print(f"Largest count: {count + arr.shape[1] + 1}")
    print(f"Face count: {len(res) - vertex_count - 1}")
    return "\n".join(res)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file. Expects geotiff.", required=True)
    parser.add_argument("-o", "--output", help="Output mesh. .obj extension.", required=True)

    args = parser.parse_args()

    if not args.input.endswith(".tif"):
        raise ValueError("Expected input with .tif extension")

    ndsm_obj, ndsm = load_lidar_data(args.input)
    ndsm[ndsm < 0] = 0
    a = np_to_obj(ndsm)
    out_fp = args.output
    if not out_fp.endswith(".obj"):
        out_fp += ".obj"
    with open(out_fp, "w") as fp:
        fp.write(a)
