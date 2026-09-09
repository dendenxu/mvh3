from __future__ import annotations

import os
import sys
from os.path import dirname, join, basename, exists

import importlib
from torch.utils.cpp_extension import load


def extra_cflags() -> list[str]:
    # Torch's JIT extension builder forwards these directly to the compiler.
    # On Windows the compiler is MSVC cl.exe, so we must use MSVC-style flags.
    if sys.platform.startswith("win"):
        return ["/O2", "/std:c++17", "/EHsc", "/bigobj"]
    return ["-O3", "-std=c++17"]


def import_so_as_module(name: str, so_path: str):
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ext(src: str = 'cpp/easy_utils_ext.cpp'):
    this_dir = dirname(__file__)
    src = join(this_dir, src)
    name = 'mvh3_' + basename(src).replace('.', '_')
    # build_dir = join(this_dir, "_torch_extensions", name)
    # so_path = join(build_dir, f"{name}.so")

    _EXT = globals().get(name, None)
    if _EXT is not None:
        return _EXT

    # if exists(so_path) and not force_rebuild:
    #     _EXT = import_so_as_module(name, so_path)
    # else:
        # os.makedirs(build_dir, exist_ok=True)
    _EXT = load(
        name=name,
        sources=[src],
        extra_cflags=extra_cflags(),
        # build_directory=build_dir,
        with_cuda=False,
    )
    globals()[name] = _EXT
    return _EXT
