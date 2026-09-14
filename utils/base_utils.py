from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Callable, Dict, Mapping, TypeVar

if TYPE_CHECKING:
    import torch

# these are generic type vars to tell mapping to accept any type vars when creating a type
KT = TypeVar("KT")  # key type
VT = TypeVar("VT")  # value type


def type_to_torch_dtype(type):
    import numpy as np
    import torch

    if not hasattr(type_to_torch_dtype, "dtype_map"):
        type_to_torch_dtype.dtype_map = {
            int: torch.int64,
            float: torch.float32,
            bool: torch.bool,
            complex: torch.complex64,
            np.dtype("float32"): torch.float32,
            np.dtype("float64"): torch.float64,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
            np.dtype("uint8"): torch.uint8,
            np.dtype("bool"): torch.bool,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
        }
    return type_to_torch_dtype.dtype_map[type]


def return_dotdict(func: Callable):

    def inner(*args, **kwargs):
        return dotdict(func(*args, **kwargs))

    return inner


class dotdict(dict, Dict[KT, VT]):
    """
    This is the default data passing object used throughout the codebase
    Main function: dot access for dict values & dict like merging and updates

    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = make_dotdict() or d = make_dotdict{'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def update(self, dct: Dict = None, **kwargs):  # noqa: C901
        dct = copy(dct)  # avoid modifying the original dict, use super's copy to avoid recursion

        # Handle different arguments
        if dct is None:
            dct = kwargs
        elif isinstance(dct, Mapping):
            dct.update(kwargs)
        else:
            super().update(dct, **kwargs)
            return

        # Recursive updates
        for k, v in dct.items():
            if k in self:
                # Handle type conversions
                target_type = type(self[k])
                if not isinstance(v, target_type):
                    # Note: bool('False') will be True
                    if target_type == bool and isinstance(v, str):
                        dct[k] = v == "True"
                    else:
                        import numpy as np

                        # Lazy imports
                        import torch

                        if isinstance(v, torch.Tensor) and issubclass(target_type, np.ndarray):
                            dct[k] = v
                        elif isinstance(v, torch.Tensor) and not issubclass(target_type, torch.Tensor):
                            dct[k] = v.type(type_to_torch_dtype(target_type))
                        elif isinstance(v, np.ndarray) and not issubclass(target_type, np.ndarray):
                            dct[k] = v.astype(target_type)
                        else:
                            dct[k] = target_type(v)

                if isinstance(v, dict):
                    self[k].update(v)  # recursion from here
                else:
                    self[k] = v
            else:
                if isinstance(v, dict):
                    self[k] = dotdict(v)  # recursion?
                elif isinstance(v, list):
                    self[k] = [dotdict(x) if isinstance(x, dict) else x for x in v]
                else:
                    self[k] = v
        return self

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    copy = return_dotdict(dict.copy)
    fromkeys = return_dotdict(dict.fromkeys)

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError as e:
            raise AttributeError(e)

    # AttributeError is required for pickle and DataLoader attribute probing.
    __getattr__: Callable[..., "torch.Tensor"] = __getitem__  # type: ignore # overidden dict.__getitem__
    __getattribute__: Callable[..., "torch.Tensor"]  # type: ignore
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @property
    def meta(self) -> dotdict:
        # Special variable used for storing cpu tensor in batch
        if "meta" not in self:
            self.meta = dotdict()
        return self.__getitem__("meta")

    @meta.setter
    def meta(self, meta):
        self.__setitem__("meta", meta)

    @property
    def output(self) -> dotdict:  # late annotation needed for this
        # Special entry for storing output tensor in batch
        if "output" not in self:
            self.output = dotdict()
        return self.__getitem__("output")

    @output.setter
    def output(self, output):
        self.__setitem__("output", output)

    @property
    def persistent(self) -> dotdict:  # late annotation needed for this
        # Special entry for storing persistent tensor in batch
        if "persistent" not in self:
            self.persistent = dotdict()
        return self.__getitem__("persistent")

    @persistent.setter
    def persistent(self, persistent):
        self.__setitem__("persistent", persistent)

    @property
    def type(self) -> str:  # late annotation needed for this
        # Special entry for type based construction system
        return self.__getitem__("type")

    @type.setter
    def type(self, type):
        self.__setitem__("type", type)

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, dotdict):
                v = v.to_dict()  # recursion point
            out[k] = v
        return out
