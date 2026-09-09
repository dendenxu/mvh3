"""Validation video, caption/pose metadata, and asynchronous HDFS mirroring."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from utils.video import write_video

_mirror = None


def write_visualization(outputs, document, video, cfg, step, rank, index=0, stage=None):
    global _mirror
    directory = Path(cfg.vis_dir or
                     (Path(cfg.h3.logdir) / "vis")) / f"step_{step:09d}" / f"rank{rank}" / f"sample{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    for view_index, (latent, view) in enumerate(zip(outputs, document["views"])):
        generated = video.decode(latent, view["height"], view["width"], view["source_frames"])
        panels = [generated]
        if cfg.vis_gt:
            panels.append(video.decode(view["latent"], view["height"], view["width"], view["source_frames"]))
        if cfg.vis_pose:
            h, w = generated.shape[-2:]
            panel = np.zeros((h, w, 3), dtype=np.uint8)
            points = view["pose"][:, 7:10].numpy()[:, [0, 2]]
            points = (points - points.min(0)) / (np.ptp(points, axis=0).max() + 1e-6)
            points = (points * np.array([w * .8, h * .8]) + np.array([w * .1, h * .1])).astype(np.int32)
            cv2.polylines(panel, [points], False, (70, 210, 210), 2)
            panels.append(
                torch.from_numpy(panel).permute(2, 0, 1).float().div(255)[None].expand(len(generated), -1, -1, -1))
        frames = torch.cat(panels, -1).permute(0, 2, 3, 1).mul(255).byte().numpy()
        write_video(str(directory / f"view{view_index:03d}.mp4"), frames, fps=view["fps"])
        meta = {key: value for key, value in view.items() if isinstance(value, (str, float, int))}
        meta.update(step=step, source=document["source"], stage=cfg.h3.stage if stage is None else stage)
        (directory / f"view{view_index:03d}.json").write_text(json.dumps(meta, indent=2) + "\n")
        if cfg.decode_offload:
            torch.cuda.empty_cache()
    if cfg.vis_copy_to_hdfs:

        def mirror():
            destination = f"{cfg.vis_hdfs_dir.rstrip('/')}/{Path(cfg.h3.logdir).name}/vis/step_{step:09d}/rank{rank}/sample{index:03d}"
            for attempt in range(int(cfg.vis_hdfs_copy_retries) + 1):
                try:
                    subprocess.run([cfg.vis_hdfs_cli, "dfs", "-mkdir", "-p", destination],
                                   timeout=cfg.vis_hdfs_copy_timeout,
                                   check=True,
                                   capture_output=True)
                    subprocess.run(
                        [cfg.vis_hdfs_cli, "dfs", "-put", "-f", *map(str, directory.iterdir()), destination],
                        timeout=cfg.vis_hdfs_copy_timeout,
                        check=True,
                        capture_output=True)
                    break
                except (OSError, subprocess.SubprocessError) as error:
                    print(f"Visualization mirror attempt {attempt+1} failed: {type(error).__name__}", flush=True)
                    if attempt < int(cfg.vis_hdfs_copy_retries):
                        time.sleep(min(2**attempt, 30))

        if _mirror is None:
            _mirror = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-vis-copy")
        _mirror.submit(mirror)
