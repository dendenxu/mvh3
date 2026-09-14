"""WorldViews-compatible, globally synchronized HR save/visualization requests."""

import os
from pathlib import Path

import torch.distributed as dist

from utils.distributed import get_rank


class Requests:
    def __init__(self):
        self.directory = None
        trial = os.environ.get("ARNOLD_MONITOR_TRIAL_ID", "")
        if trial and trial != "unknown":
            root = os.environ.get("HR_STATE_ROOT") or os.environ.get("HR_ROOT") or str(Path(os.environ.get("BN", "/mnt/bn/foundation-ads3/zhenxu.zx")) / "hotreload")
            self.directory = Path(root) / trial
            group = os.environ.get("HR_CONTROL_GROUP", "").strip()
            if group:
                if not group.isascii() or not all(c.isalnum() or c in "_-" for c in group):
                    raise ValueError("Invalid HR_CONTROL_GROUP")
                self.directory = self.directory / "groups" / group
        self.last = {kind: self.read(kind) for kind in ("save", "vis")} if get_rank() == 0 else {}

    def read(self, kind):
        if self.directory is not None:
            try:
                return (self.directory / f"{kind}_request").read_text().strip() or None
            except OSError:
                pass
        return None

    def poll(self):
        requested = [[]]
        if get_rank() == 0:
            for kind in ("save", "vis"):
                token = self.read(kind)
                if token and token != self.last[kind]:
                    requested[0].append((kind, token))
                    self.last[kind] = token
        if dist.is_initialized():
            dist.broadcast_object_list(requested, src=0)
        return requested[0]

    def acknowledge(self, kind, token, step, status, checkpoint=""):
        if get_rank() == 0 and self.directory is not None:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                target = self.directory / f"{kind}_{'started' if status == 'started' else 'done'}"
                temporary = target.with_suffix(f".tmp.{os.getpid()}")
                temporary.write_text(f"token={token} status={status} step={step} ckpt={checkpoint}\n")
                os.replace(temporary, target)
            except OSError as error:
                print(f"HR {kind} acknowledgment failed: {type(error).__name__}", flush=True)
