from utils.ext_utils import load_ext
_EXT = load_ext("cpp/random_video_ext.cpp")


def randomly_construct_video_np_cpp(
    length: int,
    n_views: int,
    n_frames: int,
    frame_velo_min: float = 1.0,
    frame_velo_max: float = 1.0,
    view_velo_min: float = -4.0,
    view_velo_max: float = 4.0,
    frame_acc_min: float = -1.0,
    frame_acc_max: float = 1.0,
    view_acc_min: float = -1.0,
    view_acc_max: float = 1.0,
    frame_velo_buffer_min: float = -4.0,
    frame_velo_buffer_max: float = 4.0,
    view_velo_buffer_min: float = -4.0,
    view_velo_buffer_max: float = 4.0,
    acc_update_iter: int = 25,
    drag_coefficient: float = 0.8,
    seed: int = -1,
):
    """
    Returns an int64 NumPy array of shape (length, 2): [view_idx, frame_idx].
    """
    return _EXT.randomly_construct_video_np(
        int(length),
        int(n_views),
        int(n_frames),
        float(frame_velo_min),
        float(frame_velo_max),
        float(view_velo_min),
        float(view_velo_max),
        float(frame_acc_min),
        float(frame_acc_max),
        float(view_acc_min),
        float(view_acc_max),
        float(frame_velo_buffer_min),
        float(frame_velo_buffer_max),
        float(view_velo_buffer_min),
        float(view_velo_buffer_max),
        int(acc_update_iter),
        float(drag_coefficient),
        int(seed),
    )
