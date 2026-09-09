"""Use the pinned sibling dependencies when running from the downloaded bundle."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT.parent / "python_deps", ROOT.parent / "diffusers" / "src", ROOT):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

import torch

torch.set_num_threads(1)
