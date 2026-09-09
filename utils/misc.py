import numpy as np
import random
import torch
import os


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict


def find_latest_checkpoint(logdir):
    """Find the latest checkpoint in the logdir.

    Prefer ``latest.pt`` when present — it may be a symlink pointing into another
    experiment's ckpt dir (useful for bootstrapping a new run from an existing
    checkpoint without copying files). ``os.path.exists`` follows symlinks, so a
    dangling symlink falls through to the directory scan below.
    """
    if not os.path.exists(logdir):
        return ''

    latest_file = os.path.join(logdir, "latest.pt")
    if os.path.exists(latest_file):
        return latest_file

    checkpoint_dirs = []
    for item in os.listdir(logdir):
        if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
            try:
                # Extract step number from directory name
                step_str = item.replace("checkpoint_model_", "")
                step = int(step_str)
                checkpoint_path = os.path.join(logdir, item, "model.pt")
                if os.path.exists(checkpoint_path):
                    checkpoint_dirs.append((step, checkpoint_path))
            except ValueError:
                continue

    if not checkpoint_dirs:
        return ''

    # Sort by step number and return the latest one
    checkpoint_dirs.sort(key=lambda x: x[0])
    latest_step, latest_path = checkpoint_dirs[-1]
    return latest_path


def bytesafe_filename_cap(s: str, max_bytes: int = 80) -> str:
    """Sanitize `s` for use in a filename and truncate to `max_bytes` UTF-8 bytes.

    Linux filesystems (ext4/xfs/tmpfs) cap each path component at 255 bytes, so
    char-based truncation like ``s[:128]`` silently busts the limit for Chinese
    prompts (each char = 3–4 bytes in UTF-8). We strip path-hostile chars, then
    encode + slice by bytes, and decode back ignoring any trailing partial
    codepoint.
    """
    cleaned = []
    for c in s:
        if c in '/\\\x00' or ord(c) < 0x20:
            cleaned.append('_')
        else:
            cleaned.append(c)
    return ''.join(cleaned).encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
