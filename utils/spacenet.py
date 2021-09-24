import os


def filename_to_id(filename: str) -> str:
    return "_".join(os.path.basename(filename).split("_")[1:]).split(".")[0]
