"""Small, explicit config/checkpoint helpers for ordinary PyTorch H3 modules."""

import inspect
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


class ModelConfig(dict):

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def model_config(arguments):
    return ModelConfig({key: value for key, value in arguments.items() if key not in ("self", "__class__")})


def get_parameter_dtype(module):
    # FSDP gathered weights can be Tensor views absent from named_parameters().
    for child in module.modules():
        weight = getattr(child, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight.dtype
    return next(module.parameters()).dtype


def set_gradient_checkpointing(module, enabled=True, function=None):
    function = function or (lambda fn, *args, **kwargs: checkpoint(fn, *args, use_reentrant=False, **kwargs))
    for child in module.modules():
        if hasattr(child, "gradient_checkpointing"):
            child.gradient_checkpointing = enabled
            child._gradient_checkpointing_func = function


def load_local_model(cls, path, *, torch_dtype=None, local_files_only=True):
    """Strictly load a local converted checkpoint; never download or infer a model."""
    from safetensors import safe_open
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    keys = inspect.signature(cls.__init__).parameters
    unknown = {key for key in config if not key.startswith("_") and key not in keys}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} configuration: {sorted(unknown)}")
    with torch.device("meta"):
        model = cls(**{key: value for key, value in config.items() if key in keys})
    # Materialize nonpersistent RoPE buffers from a small empty constructor.
    buffers = dict(model.named_buffers())
    model.to_empty(device="cpu")
    for name in buffers:
        owner_name, leaf = name.rsplit(".", 1)
        owner = model.get_submodule(owner_name)
        if leaf == "inv_freq" and hasattr(owner, "reset_parameters"):
            owner.reset_parameters()
        else:
            raise ValueError(f"No initializer for nonpersistent buffer {name}")
    expected = model.state_dict()
    index_file = path / "diffusion_pytorch_model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text())["weight_map"]
    else:
        filename = "diffusion_pytorch_model.safetensors"
        with safe_open(path / filename, framework="pt") as reader:
            index = {key: filename for key in reader.keys()}
    if set(index) != set(expected):
        raise ValueError(
            f"Checkpoint keys differ: missing={set(expected)-set(index)}, extra={set(index)-set(expected)}")
    seen = set()
    for filename in sorted(set(index.values())):
        with safe_open(path / filename, framework="pt") as reader:
            state = {}
            for key in reader.keys():
                if key in seen or index.get(key) != filename:
                    raise ValueError(f"Duplicate or misindexed tensor: {key}")
                tensor = reader.get_tensor(key)
                if tensor.shape != expected[key].shape:
                    raise ValueError(f"Wrong checkpoint shape for {key}")
                state[key] = tensor.to(torch_dtype) if torch_dtype is not None else tensor
                seen.add(key)
            model.load_state_dict(state, strict=False, assign=True)
    if seen != set(expected):
        raise ValueError("Incomplete checkpoint")
    return model
