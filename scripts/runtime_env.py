"""Resolve optional sibling dependencies without modifying the installed environment."""

from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
for candidate in (ROOT.parent / "python_deps", ROOT):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
