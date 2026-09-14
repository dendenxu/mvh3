from __future__ import annotations

import sys
from os.path import join, dirname, basename

from torch.utils.cpp_extension import load


def extra_cflags() -> list[str]:
    # Torch's JIT extension builder forwards these directly to the compiler.
    # On Windows the compiler is MSVC cl.exe, so we must use MSVC-style flags.
    if sys.platform.startswith("win"):
        return ["/O2", "/std:c++17", "/EHsc", "/bigobj"]
    return ["-O3", "-std=c++17"]


def load_ext(src: str = "cpp/easy_utils_ext.cpp"):
    this_dir = dirname(__file__)
    src = join(this_dir, src)
    name = "mvh3_" + basename(src).replace(".", "_")

    extension = globals().get(name, None)
    if extension is not None:
        return extension

    extension = load(
        name=name,
        sources=[src],
        extra_cflags=extra_cflags(),
        with_cuda=False,
    )
    globals()[name] = extension
    return extension
