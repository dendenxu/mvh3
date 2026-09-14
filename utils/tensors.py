"""Convert nested camera arrays between NumPy and Torch without changing their math."""

from typing import Dict, List, Union

import torch
import numpy as np

from utils.base_utils import dotdict


def to_tensor(batch, non_recursive_list: bool = False) -> Union[torch.Tensor, dotdict[str, torch.Tensor]]:
    if isinstance(batch, (tuple, list)) and not non_recursive_list:
        batch = [to_tensor(b, non_recursive_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_tensor(v, non_recursive_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        pass
    else:  # numpy and others
        batch = torch.as_tensor(batch)
    return batch


def to_numpy(
    batch, non_blocking=False, ignore_list: bool = False
) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_numpy(b, non_blocking, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_numpy(v, non_blocking, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to("cpu", non_blocking=non_blocking).numpy()
    else:  # numpy and others
        batch = np.asarray(batch)
    return batch


def as_numpy_func(func):

    def wrapper(*args, **kwargs):
        args = to_tensor(args)
        kwargs = to_tensor(kwargs)
        ret = func(*args, **kwargs)
        return to_numpy(ret)

    return wrapper
