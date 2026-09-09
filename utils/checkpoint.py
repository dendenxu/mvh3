"""Atomic distributed checkpoints of original trainable weights and FP32 AdamW."""

import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist

from utils.config import recipe_digest
from h3.distributed.fsdp import canonical_name

FLOW_CONVENTION = "h3_t1_clean_data_minus_noise"


def rng_state():
    return dict(python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(model, optimizer, cfg, step, stage, runtime, directory):
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    # A new generation keeps an interrupted re-save from mixing rank shards
    # with the previously committed checkpoint at the same global step.
    generation = [uuid.uuid4().hex[:12] if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(generation, src=0)
    path = Path(directory) / f"step_{step:09d}_{generation[0]}"
    path.mkdir(parents=True, exist_ok=True)
    weights = {canonical_name(n): p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    state = dict(flow_convention=FLOW_CONVENTION,
                 recipe=recipe_digest(cfg),
                 step=step,
                 stage=stage,
                 world_size=world,
                 fs_size=cfg.fs_size,
                 sp_size=cfg.sp_size,
                 weights=weights,
                 optimizer=optimizer.state_dict(),
                 rng=rng_state(),
                 runtime=runtime)
    target = path / f"rank{rank:05d}.pt"
    temp = target.with_suffix(f".{os.getpid()}.tmp")
    torch.save(state, temp)
    os.replace(temp, target)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        manifest = dict(step=step,
                        stage=stage,
                        world_size=world,
                        recipe=state["recipe"],
                        flow_convention=FLOW_CONVENTION,
                        files=[f"rank{r:05d}.pt" for r in range(world)])
        temporary = path / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, path / "manifest.json")
        latest = Path(directory) / "latest.json"
        temp = latest.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(path=path.name)) + "\n")
        os.replace(temp, latest)
    if dist.is_initialized():
        dist.barrier()
    return path


def load_checkpoint(model, optimizer, cfg, directory):
    path = Path(directory)
    if path.is_file() and path.name == "latest.json":
        path = path.parent / json.loads(path.read_text())["path"]
    manifest = json.loads((path / "manifest.json").read_text())
    world, rank = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)
    if manifest["world_size"] != world:
        raise ValueError("Sharded resume requires the saved world size; consolidate before changing topology")
    if manifest["recipe"] != recipe_digest(cfg):
        raise ValueError("WorldViews settings differ from the saved checkpoint")
    state = torch.load(path / f"rank{rank:05d}.pt", map_location="cpu", weights_only=False)
    if state.get("flow_convention") != FLOW_CONVENTION:
        raise ValueError("Checkpoint has an incompatible flow convention")
    if state["fs_size"] != cfg.fs_size or state["sp_size"] != cfg.sp_size:
        raise ValueError("FSDP/SP topology differs from the saved checkpoint")
    params = {canonical_name(n): p for n, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state["weights"].keys():
        raise ValueError("Trainable original-layer scope changed")
    with torch.no_grad():
        for name, p in params.items():
            value = state["weights"][name]
            if value.shape != p.shape:
                raise ValueError(f"Shard shape differs for {name}")
            p.copy_(value)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state
