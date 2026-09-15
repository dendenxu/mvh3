"""Training shape buckets and explicit compiler dimension contracts."""

from dataclasses import fields, replace, is_dataclass

import torch


def mark_tensor_dimensions(value, dynamic=()):
    """Vary only the named axes; batch, heads and feature widths stay fixed."""
    if value is None:
        return
    for dimension in range(value.ndim):
        if dimension in dynamic:
            torch._dynamo.mark_dynamic(value, dimension)
        else:
            torch._dynamo.mark_static(value, dimension)


def mark_camera_dimensions(camera):
    if isinstance(camera, torch.Tensor):
        mark_tensor_dimensions(camera, (1,))
    elif is_dataclass(camera):
        for field in fields(camera):
            mark_camera_dimensions(getattr(camera, field.name))


def mark_layout_dimensions(layout):
    for value in (layout.kind, layout.chunk, layout.scope, layout.active):
        mark_tensor_dimensions(value, (0,))
    if layout.history_dropout is not None:
        mark_tensor_dimensions(layout.history_dropout, (0, 1))
        mark_tensor_dimensions(layout._history_dropout_flat, (0,))
    mark_tensor_dimensions(layout.mask_dimensions)


def mark_mask_dimensions(mask):
    # The block grid grows with the sequence; its batch/head axes remain fixed.
    for name in (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
    ):
        value = getattr(mask, name)
        if value is not None:
            mark_tensor_dimensions(value, tuple(range(2, value.ndim)))


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
