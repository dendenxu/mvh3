"""
Contains modified code from https://github.com/Stonesjtu/calmsize and https://github.com/Stonesjtu/pytorch_memlab
"""

import gc
import math
from collections import defaultdict
from math import isnan
from typing import List, Optional, Tuple

import torch
# from torch import nn
# from typing import List
# from easyvolcap.utils.console import *

LEN = 79

# some pytorch low-level memory management constant
# the minimal allocate memory size (Byte)
PYTORCH_MIN_ALLOCATE = 2**9
# the minimal cache memory size (Byte)
PYTORCH_MIN_CACHE = 2**20


def calc_memory_usage(tensor: torch.Tensor):
    # numel = tensor.numel()
    element_size = tensor.element_size()
    fact_numel = tensor.storage().size()
    fact_memory_size = fact_numel * element_size
    # since pytorch allocate at least 512 Bytes for any tensor, round
    # up to a multiple of 512
    memory_size = (
        math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE) * PYTORCH_MIN_ALLOCATE
    )
    return memory_size


def check_pinned_memory_usage():
    objects = gc.get_objects()
    tensors = [
        obj for obj in objects if isinstance(obj, torch.Tensor) and obj.is_pinned()
    ]
    return sum(calc_memory_usage(tensor) for tensor in tensors)


def check_device_memory_usage(device: torch.device = torch.device("cuda")):  # noqa: B008
    objects = gc.get_objects()
    tensors = [
        obj for obj in objects if isinstance(obj, torch.Tensor) and obj.device == device
    ]
    return sum(calc_memory_usage(tensor) for tensor in tensors)


def check_cpu_memory_usage():
    return check_device_memory_usage(torch.device("cpu"))


def check_cuda_memory_usage():
    return check_device_memory_usage(torch.device("cuda"))


# encoding=utf-8
# Modified by Kaiyu Shi
# Date:2019-05-24

traditional = [
    (1024**5, "P"),
    (1024**4, "T"),
    (1024**3, "G"),
    (1024**2, "M"),
    (1024**1, "K"),
    (1024**0, "B"),
]

alternative = [
    (1024**5, " PB"),
    (1024**4, " TB"),
    (1024**3, " GB"),
    (1024**2, " MB"),
    (1024**1, " KB"),
    (1024**0, (" byte", " bytes")),
]

verbose = [
    (1024**5, (" petabyte", " petabytes")),
    (1024**4, (" terabyte", " terabytes")),
    (1024**3, (" gigabyte", " gigabytes")),
    (1024**2, (" megabyte", " megabytes")),
    (1024**1, (" kilobyte", " kilobytes")),
    (1024**0, (" byte", " bytes")),
]

iec = [
    (1024**5, "Pi"),
    (1024**4, "Ti"),
    (1024**3, "Gi"),
    (1024**2, "Mi"),
    (1024**1, "Ki"),
    (1024**0, ""),
]

si = [
    (1000**5, "P"),
    (1000**4, "T"),
    (1000**3, "G"),
    (1000**2, "M"),
    (1000**1, "K"),
    (1000**0, "B"),
]


class ByteSize:
    def __init__(self, num_bytes, system=traditional):
        self.num_bytes = num_bytes
        self.system = system
        self.amount = num_bytes
        self.unit = self.system[-1]  # lowest is pure Bytes
        self.find_largest_unit()

    def _find_largest_unit_pos(self, num_bytes):
        """Find the proper unit and corresponding amount

        This implementation only works for positive number
        """
        for factor, unit in self.system:  # noqa: B007
            if num_bytes >= factor:
                break
        self.amount = num_bytes / factor

        # singular and plural for a tuple
        if isinstance(unit, tuple):
            singular, multiple = unit
            if self.amount == 1:
                unit = singular
            else:
                unit = multiple
        self.unit = unit

    def find_largest_unit(self):
        num_bytes = self.num_bytes
        pos_bytes = abs(num_bytes)
        sign = int(num_bytes >= 0) * 2 - 1  # sign function
        self._find_largest_unit_pos(pos_bytes)
        self.amount *= sign

    def __str__(self):
        return str(int(round(self.amount))) + self.unit

    def __format__(self, formatstr):
        if formatstr:
            return self.amount.__format__(formatstr) + self.unit
        else:
            return str(self)

    def __repr__(self):
        return str(self) + "<ByteSize amount={}>".format(self.amount)

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        elif isinstance(other, ByteSize):
            return self.amount == other.amount
        else:
            return type(other)(self.amount) == other

    def __lt__(self, other):
        if isinstance(other, str):
            raise NotImplementedError(
                "Comparison between string and ByteSize not supported yet"
            )
        elif isinstance(other, ByteSize):
            return self.amount < other.amount
        else:
            return type(other)(self.amount) < other

    def __gt__(self, other):
        if isinstance(other, str):
            raise NotImplementedError(
                "Comparison between string and ByteSize not supported yet"
            )
        elif isinstance(other, ByteSize):
            return self.amount > other.amount
        else:
            return type(other)(self.amount) > other


def calmsize(bytes, system=traditional):
    """Human-readable file size.

    Using the traditional system, where a factor of 1024 is used::

    >>> size(10)
    '10B'
    >>> size(100)
    '100B'
    >>> size(2000000)
    '1M'

    Using the SI system, with a factor 1000::

    >>> size(10, system=si)
    '10B'
    >>> size(100, system=si)
    '100B'
    >>> size(1000, system=si)
    '1K'
    >>> size(2000000, system=si)
    '2M'

    """
    byte_size = ByteSize(bytes, system)
    return byte_size


def readable_size(num_bytes: int) -> str:
    return "" if isnan(num_bytes) else "{:.2f}".format(calmsize(num_bytes))


LEN = 79

# some pytorch low-level memory management constant
# the minimal allocate memory size (Byte)
PYTORCH_MIN_ALLOCATE = 2**9
# the minimal cache memory size (Byte)
PYTORCH_MIN_CACHE = 2**20


class MemReporter:
    """A memory reporter that collects tensors and memory usages

    Parameters:
        - model: an extra nn.Module can be passed to infer the name
        of Tensors
        - pre_collect: do a garbage collection before getting remaining
        Tensors, this gives cleaner outputs.
          Caution: This is an intrusive change to your original code.

    """

    def __init__(
        self, model: Optional[torch.nn.Module] = None, pre_collect: bool = False
    ):
        self.tensor_name = {}
        self.device_mapping = defaultdict(list)
        self.device_tensor_stat = {}
        # to numbering the unknown tensors
        self.name_idx = 0
        self.pre_collect = pre_collect

        tensor_names = defaultdict(list)
        if model is not None:
            assert isinstance(model, torch.nn.Module)
            # for model with tying weight, multiple parameters may share
            # the same underlying tensor
            for name, param in model.named_parameters():
                tensor_names[param].append(name)

        for param, name in tensor_names.items():
            self.tensor_name[id(param)] = "+".join(name)

    def _get_tensor_name(self, tensor: torch.Tensor) -> str:
        tensor_id = id(tensor)
        if tensor_id in self.tensor_name:
            name = self.tensor_name[tensor_id]
        # use numbering if no name can be inferred
        else:
            name = type(tensor).__name__ + str(self.name_idx)
            self.tensor_name[tensor_id] = name
            self.name_idx += 1
        return name

    def add_optimizer(self, optimizer: torch.optim.Optimizer):
        optimizer_name = optimizer.__class__.__name__
        for param, states in optimizer.state.items():
            param_name = self.tensor_name[id(param)]
            for name, tensor in states.items():
                self.tensor_name[id(tensor)] = f"{optimizer_name}.{param_name}.{name}"
            # self.tensor_name[id()]
            # print(states)

    def collect_tensor(self):
        """Collect all tensor objects tracked by python

        NOTICE:
            - the buffers for backward which is implemented in C++ are
            not tracked by python's reference counting.
            - the gradients(.grad) of Parameters is not collected, and
            I don't know why.
        """
        # FIXME: make the grad tensor collected by gc
        # Do a pre-garbage collect to eliminate python garbage objects
        if self.pre_collect:
            gc.collect()
        objects = gc.get_objects()
        tensors = [obj for obj in objects if isinstance(obj, torch.Tensor)]
        for t in tensors:
            self.device_mapping[t.device].append(t)

    def get_stats(self):
        """Get the memory stat of tensors and then release them

        As a memory profiler, we cannot hold the reference to any tensors, which
        causes possibly inaccurate memory usage stats, so we delete the tensors after
        getting required stats"""
        visited_data = {}
        self.device_tensor_stat.clear()

        def get_tensor_stat(tensor: torch.Tensor) -> List[Tuple[str, int, int, int]]:
            """Get the stat of a single tensor

            Returns:
                - stat: a tuple containing (tensor_name, tensor_size,
            tensor_numel, tensor_memory)
            """
            assert isinstance(tensor, torch.Tensor)

            name = self._get_tensor_name(tensor)
            if tensor.is_sparse:
                indices_stat = get_tensor_stat(tensor._indices())
                values_stat = get_tensor_stat(tensor._values())
                return indices_stat + values_stat

            numel = tensor.numel()
            element_size = tensor.element_size()
            fact_numel = tensor.untyped_storage().size()
            fact_memory_size = fact_numel * element_size
            # since pytorch allocate at least 512 Bytes for any tensor, round
            # up to a multiple of 512
            memory_size = (
                math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE)
                * PYTORCH_MIN_ALLOCATE
            )

            # tensor.storage should be the actual object related to memory
            # allocation
            data_ptr = tensor.untyped_storage().data_ptr()
            if data_ptr in visited_data:
                name = "{}(->{})".format(
                    name,
                    visited_data[data_ptr],
                )
                # don't count the memory for reusing same underlying storage
                memory_size = 0
            else:
                visited_data[data_ptr] = name

            size = tuple(tensor.size())
            # torch scalar has empty size
            if not size:
                size = (1,)

            return [(name, size, numel, memory_size)]

        for device, tensors in self.device_mapping.items():
            tensor_stats = []
            for tensor in tensors:
                if tensor.numel() == 0:
                    continue
                stat = get_tensor_stat(tensor)  # (name, shape, numel, memory_size)
                tensor_stats += stat
                if isinstance(tensor, torch.nn.Parameter):
                    if tensor.grad is not None:
                        # manually specify the name of gradient tensor
                        self.tensor_name[id(tensor.grad)] = "{}.grad".format(
                            self._get_tensor_name(tensor)
                        )
                        stat = get_tensor_stat(tensor.grad)
                        tensor_stats += stat

            self.device_tensor_stat[device] = tensor_stats

        self.device_mapping.clear()

    def print_stats(
        self, verbose: bool = False, target_device: Optional[torch.device] = None
    ) -> None:
        # header
        show_reuse = verbose
        template_format = "{:<40s}{:>20s}{:>10s}"
        print(template_format.format("Element type", "Size", "Used MEM"))
        for device, tensor_stats in self.device_tensor_stat.items():
            # By default, if the target_device is not specified,
            # print tensors on all devices
            if target_device is not None and device != target_device:
                continue
            print("-" * LEN)
            print("Storage on {}".format(device))
            total_mem = 0
            total_numel = 0
            for stat in tensor_stats:
                name, size, numel, mem = stat
                if not show_reuse:
                    name = name.split("(")[0]
                print(
                    template_format.format(
                        str(name),
                        str(size),
                        readable_size(mem),
                    )
                )
                total_mem += mem
                total_numel += numel

            print("-" * LEN)
            print(
                "Total Tensors: {} \tUsed Memory: {}".format(
                    total_numel,
                    readable_size(total_mem),
                )
            )

            if device != torch.device("cpu"):
                with torch.cuda.device(device):
                    memory_allocated = torch.cuda.memory_allocated()
                print(
                    "The allocated memory on {}: {}".format(
                        device,
                        readable_size(memory_allocated),
                    )
                )
                if memory_allocated != total_mem:
                    print(
                        "Memory differs due to the matrix alignment or"
                        " invisible gradient buffer tensors"
                    )
            print("-" * LEN)

    def report(
        self, verbose: bool = False, device: Optional[torch.device] = None
    ) -> None:
        """Interface for end-users to directly print the memory usage

        args:
            - verbose: flag to show tensor.storage reuse information
            - device: `torch.device` object, specify the target device
            to report detailed memory usage. It will print memory usage
            on all devices if not specified. Usually we only want to
            print the memory usage on CUDA devices.

        """
        self.collect_tensor()
        self.get_stats()
        self.print_stats(verbose, target_device=device)
