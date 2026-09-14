"""Read the intri.yml/extri.yml camera pairs used by multi-view datasets."""

from os.path import join

from utils.base_utils import dotdict
from utils.ext_utils import load_ext

# Build before DataLoader workers fork; retain the existing extension/cache name.
CAMERA_IO = load_ext("cpp/easy_utils_ext.cpp")


def read_camera_minimal(intri_path: str, extri_path: str = None) -> dotdict:
    if extri_path is None:
        extri_path = join(intri_path, "extri.yml")
        intri_path = join(intri_path, "intri.yml")
    return dotdict(CAMERA_IO.read_camera_minimal(intri_path, extri_path))
