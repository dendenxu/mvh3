#!/usr/bin/env python3
"""Run bounded tests with optional isolated dependencies next to this checkout."""

from pathlib import Path
import sys


root = Path(__file__).resolve().parents[1]
for candidate in (root.parent / "python_deps", root.parent / "diffusers" / "src", root):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([str(root / "tests"), "-q", *sys.argv[1:]]))
