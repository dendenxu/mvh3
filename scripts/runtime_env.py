"""Resolve optional sibling dependencies without modifying the installed environment."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
for candidate in (ROOT.parent / "python_deps", ROOT):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))


def record_environment():
    """Check the existing FA4/Tracking runtime before starting all ranks."""
    import argparse
    import hashlib
    import json
    import socket
    from importlib import metadata

    import flash_attn
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    root = Path(flash_attn.__file__).resolve().parent.parent
    mismatches = [
        entry["path"]
        for entry in reference["files"]
        if not (root / entry["path"]).is_file()
        or hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]
    ]
    import wandb

    report = dict(
        host=socket.gethostname(),
        python=sys.executable,
        torch=torch.__version__,
        fa4_source_files=len(reference["files"]),
        fa4_mismatches=mismatches,
        byted_wandb=metadata.version("byted-wandb"),
        internal_tracking=bool(getattr(wandb, "_IS_TRACKING", False)),
        cuda_devices=torch.cuda.device_count(),
    )
    report["status"] = (
        "passed"
        if not mismatches
        and report["torch"] == "2.12.1+cu129"
        and report["internal_tracking"]
        and report["cuda_devices"] == 8
        else "failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise SystemExit("Worker runtime differs from the validated original WorldViews environment")


if __name__ == "__main__":
    record_environment()
