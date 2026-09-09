"""Real-video sampling and the released H3 VAE's temporal geometry."""

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch


@dataclass(frozen=True)
class TemporalLayout:
    source_frames: int
    padded_frames: int
    camera_frames: torch.Tensor
    rotary_frames: torch.Tensor
    valid: torch.Tensor


def temporal_layout(num_frames: int) -> TemporalLayout:
    """Keep all source frames despite H3 dropping its final three encoder latents.

    Encoder chunks produce anchors [0,4,8,12,16] every 17 pixel frames. The
    native rotary clock uses interval starts [0,1,5,9,13]; camera anchors use
    causal interval ends. Arbitrary lengths are padded to 17*n+5 first.
    """
    if num_frames < 1:
        raise ValueError("A clip needs at least one frame")
    if num_frames == 1:
        return TemporalLayout(1, 1, torch.tensor([0]), torch.tensor([0.0], dtype=torch.float64), torch.tensor([True]))
    chunks = max(0, math.ceil((num_frames - 5) / 17))
    padded_frames, latents = 17 * chunks + 5, 5 * chunks + 2
    ids = torch.arange(latents)
    anchors = 17 * (ids // 5) + 4 * (ids % 5)
    starts = 17 * (ids // 5) + torch.tensor([0, 1, 5, 9, 13])[ids % 5]
    # Include a partial final interval; its camera is anchored to the last real frame.
    valid = starts < num_frames
    return TemporalLayout(num_frames, padded_frames, anchors.clamp_max(num_frames - 1), starts.double(), valid)


def pad_video(pixels: torch.Tensor, layout: TemporalLayout):
    if pixels.shape[2] != layout.source_frames:
        raise ValueError("Video and temporal layout have different frame counts")
    count = layout.padded_frames - layout.source_frames
    return torch.cat((pixels, pixels[:, :, -1:].expand(-1, -1, count, -1, -1)), dim=2) if count else pixels


def read_parquet_clip(parquet, row_index, views, num_frames, fps=16, height=448, width=832):
    """Read a real multi-camera MP4 row for bounded integration verification.

    This reader supports the native synchronized per-camera MP4 representation.
    It does not replace WorldGen's augmented-game/static-self-view sampler.
    """
    import cv2
    import pyarrow.parquet as pq
    from scipy.spatial.transform import Rotation

    if row_index < 0 or num_frames < 1 or fps <= 0 or min(height, width) <= 0:
        raise ValueError("Row, frame count, cadence and dimensions must be valid")
    if not views or len(set(views)) != len(views) or min(views) < 0:
        raise ValueError("Select distinct nonnegative source views")
    parquet = Path(parquet)
    reader = pq.ParquetFile(parquet)
    required = {"video_path", "caption", "pose", "num_frames", "fps"}
    if not required.issubset(reader.schema_arrow.names):
        raise ValueError("This probe requires native per-camera MP4 rows with explicit pose/fps/frame counts")
    columns = [key for key in ("video_path", "caption", "pose", "num_frames", "fps", "height", "width", "frame_start") if key in reader.schema_arrow.names]
    offset, row = 0, None
    for batch in reader.iter_batches(batch_size=1, columns=columns):
        if offset == row_index:
            row = batch.to_pylist()[0]
            break
        offset += 1
    if row is None:
        raise IndexError(row_index)
    if isinstance(row["num_frames"], list) or isinstance(row["video_path"], list):
        raise NotImplementedError("Per-camera length/path arrays need the full WorldGen family adapter")
    if row.get("frame_start"):
        raise NotImplementedError("Windowed rows need explicit absolute-video versus relative-pose indexing")
    source_frames = int(row["num_frames"])
    if source_frames < 1:
        raise ValueError("Source frame count must be positive")
    poses = np.asarray(row["pose"], dtype=np.float32).reshape(-1, source_frames, 10)
    if max(views) >= poses.shape[0] or not np.isfinite(poses).all() or (poses[..., :2] <= 0).any():
        raise ValueError("Source poses or selected views are invalid")
    source_fps = float(row["fps"])
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Source FPS must be finite and positive")
    sample_indices = np.rint(np.arange(num_frames) * source_fps / fps).astype(np.int64)
    if sample_indices[-1] >= source_frames:
        raise ValueError("The requested physical window exceeds the source clip")
    root = parquet.parent / row["video_path"]
    clips, selected_poses, source_paths = [], [], []
    for view in views:
        path = root / f"{view}.mp4" if root.is_dir() else root
        if not root.is_dir() and view != 0:
            raise ValueError("Single MP4 rows expose only one view")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        decoded = []
        try:
            for frame in range(int(sample_indices[-1]) + 1):
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"Premature video EOF: {path}, frame={frame}")
                decoded.append(image)
        finally:
            cap.release()
        input_h, input_w = decoded[0].shape[:2]
        scale = max(height / input_h, width / input_w)
        resized_h, resized_w = max(height, round(input_h * scale)), max(width, round(input_w * scale))
        top, left = (resized_h - height) // 2, (resized_w - width) // 2
        pixels = []
        for index in sample_indices:
            image = cv2.resize(decoded[index], (resized_w, resized_h), interpolation=cv2.INTER_AREA)
            pixels.append(cv2.cvtColor(image[top:top+height, left:left+width], cv2.COLOR_BGR2RGB))
        camera = poses[view, sample_indices].copy()
        camera[:, 0] *= resized_w / input_w / width
        camera[:, 1] *= resized_h / input_h / height
        camera[:, 2] = (camera[:, 2] * resized_w / input_w - left) / width
        camera[:, 3] = (camera[:, 3] * resized_h / input_h - top) / height
        clips.append(torch.from_numpy(np.stack(pixels)).permute(3, 0, 1, 2).unsqueeze(0))
        selected_poses.append(camera)
        source_paths.append(str(path))
    camera = np.stack(selected_poses)
    reference_rotation = Rotation.from_rotvec(camera[0, 0, 4:7]).as_matrix()
    reference_center = camera[0, 0, 7:10].copy()
    rotations = Rotation.from_rotvec(camera[..., 4:7].reshape(-1, 3)).as_matrix()
    camera[..., 4:7] = Rotation.from_matrix(reference_rotation.T @ rotations).as_rotvec().reshape(camera.shape[:-1] + (3,))
    camera[..., 7:10] = (camera[..., 7:10] - reference_center) @ reference_rotation
    return {
        "pixels": clips, "pose": torch.from_numpy(camera), "caption": row["caption"],
        "source_paths": source_paths, "source_view_count": poses.shape[0], "source_frames": source_frames,
        "sample_indices": sample_indices.tolist(), "fps": fps,
    }
