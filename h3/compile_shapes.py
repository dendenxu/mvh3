"""Bound static training shapes without truncating observations or captions."""

from dataclasses import fields, replace, is_dataclass

import torch


def pad_rows(value, dimension, multiple, *, repeat=False):
    if not multiple:
        return value
    padding = -value.shape[dimension] % multiple
    if not padding:
        return value
    shape = list(value.shape)
    shape[dimension] = padding
    tail = value.narrow(dimension, 0, 1).expand(shape) if repeat else value.new_zeros(shape)
    return torch.cat((value, tail), dim=dimension)


def pad_camera(camera, multiple):
    if camera is None or not multiple:
        return camera
    if isinstance(camera, torch.Tensor):
        # Camera indices only reference original rows. Repeated rows keep
        # rotations/projections valid without adding camera observations.
        return pad_rows(camera, 1, multiple, repeat=True)
    if is_dataclass(camera):
        return replace(
            camera,
            **{field.name: pad_camera(getattr(camera, field.name), multiple) for field in fields(camera)},
        )
    return camera
