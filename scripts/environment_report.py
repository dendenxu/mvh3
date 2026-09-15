"""Check worker dependencies and installed FA4 sources before loading weights."""

import sys
import json
import socket
import hashlib
import argparse
from pathlib import Path
from importlib import metadata

import torch
import wandb
import flash_attn
from transformers import Qwen3VLProcessor


def record_environment():
    """Check the original runtime and construct the actual i2v processor once."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    # Processor construction resolves indirect tokenizer dependencies that a
    # plain Transformers import misses. Fail before loading the 33B model.
    processor = Qwen3VLProcessor.from_pretrained(args.checkpoint / "processor", local_files_only=True)
    reference = json.loads(args.reference.read_text())
    root = Path(flash_attn.__file__).resolve().parent.parent
    mismatches = [
        entry["path"]
        for entry in reference["files"]
        if not (root / entry["path"]).is_file()
        or hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]
    ]
    report = dict(
        host=socket.gethostname(),
        python=sys.executable,
        torch=torch.__version__,
        torch_num_threads=torch.get_num_threads(),
        fa4_source_files=len(reference["files"]),
        fa4_mismatches=mismatches,
        byted_wandb=metadata.version("byted-wandb"),
        pycountry=metadata.version("pycountry"),
        i2v_processor=type(processor).__name__,
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
