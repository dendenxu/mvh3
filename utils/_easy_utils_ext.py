from utils.ext_utils import load_ext
_EXT = load_ext('cpp/easy_utils_ext.cpp')  # must call this at import time to avoid strange errors with pytorch multiprocess dataloader (especially with fork)


def read_camera_cpp_minimal(intri_path: str, extri_path: str):
    return _EXT.read_camera_minimal(intri_path, extri_path)


def read_camera_cpp(intri_path: str, extri_path: str):
    return _EXT.read_camera(intri_path, extri_path)


def write_camera_cpp(cameras, path: str, intri_name: str = "", extri_name: str = ""):
    return _EXT.write_camera(cameras, path, intri_name, extri_name)
