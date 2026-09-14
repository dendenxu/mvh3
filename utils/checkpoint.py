"""Atomic distributed checkpoints of original trainable weights and FP32 AdamW."""

import json
import os
from pathlib import Path
import random
import uuid
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist

from utils.config import recipe_digest
from h3.utils.model import canonical_name
from utils.ema import decay_change_preserves_history

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


def save_checkpoint(model, optimizer, cfg, step, stage, runtime, directory, ema=None):
    if bool(cfg.ema_weight) != (ema is not None):
        raise ValueError("Checkpoint must include the configured EMA state")
    if ema is not None and (ema.swapped or ema.num_updates != step):
        raise ValueError("Save raw optimizer weights and matching EMA updates outside an EMA swap")
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
                 ema=ema.state_dict() if ema is not None else None,
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
                        ema=ema is not None,
                        ema_updates=ema.num_updates if ema is not None else 0,
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


def load_checkpoint(model, optimizer, cfg, directory, restore_random=True, ema=None, weights="raw",
                    ema_schedule_change=None):
    if weights not in ("raw", "ema") or (weights == "ema" and (optimizer is not None or ema is not None)):
        raise ValueError("EMA weights are for inference; optimizer resume requires raw weights")
    if ema_schedule_change is not None and (not isinstance(ema_schedule_change, str)
            or not ema_schedule_change.strip() or optimizer is None or ema is None):
        raise ValueError("An EMA schedule change requires an explicit reason and raw optimizer/EMA resume")
    path = Path(directory)
    if path.is_file() and path.name == "latest.json":
        path = path.parent / json.loads(path.read_text())["path"]
    manifest = json.loads((path / "manifest.json").read_text())
    world, rank = (dist.get_world_size(), dist.get_rank()) if dist.is_initialized() else (1, 0)
    if manifest["world_size"] != world:
        raise ValueError("Sharded resume requires the saved world size; consolidate before changing topology")
    # Inference reads only the selected weights; avoid eagerly reading the
    # optimizer and queued training data from the same shard file.
    state = torch.load(path / f"rank{rank:05d}.pt", map_location="cpu", weights_only=False,
                       mmap=optimizer is None and ema is None)
    if manifest["recipe"] != state["recipe"]:
        raise ValueError("Checkpoint manifest and rank recipe differ")
    if manifest["recipe"] != recipe_digest(cfg):
        saved_ema = state.get("ema")
        previous_cfg = deepcopy(cfg)
        if saved_ema is not None:
            previous_cfg.ema_weight = saved_ema["decay"]
            if ema_schedule_change is not None:
                previous_cfg.ema_warmup = saved_ema["warmup"]
        compatible = (saved_ema is not None and cfg.ema_weight
                      and (ema_schedule_change is not None or
                           decay_change_preserves_history(saved_ema, cfg.ema_weight, cfg.get("ema_warmup", True)))
                      and manifest["recipe"] == recipe_digest(previous_cfg))
        if not compatible:
            raise ValueError("WorldViews settings differ from the saved checkpoint")
        if ema_schedule_change is not None:
            state["ema_schedule_change"] = dict(
                previous=dict(decay=saved_ema["decay"], warmup=saved_ema["warmup"]),
                current=dict(decay=float(cfg.ema_weight), warmup=bool(cfg.get("ema_warmup", True))),
                step=state["step"], reason=ema_schedule_change, saved_history_preserved=True,
                history_identical_to_new_schedule=False)
            if rank == 0:
                print(f"Explicit EMA schedule transition: {state['ema_schedule_change']}", flush=True)
        else:
            state["ema_decay_change"] = dict(previous=saved_ema["decay"], current=float(cfg.ema_weight),
                                             step=state["step"], history_identical=True)
            if rank == 0:
                print(f"EMA decay cap changed before warmup saturation: {state['ema_decay_change']}", flush=True)
    if state.get("flow_convention") != FLOW_CONVENTION:
        raise ValueError("Checkpoint has an incompatible flow convention")
    if state["fs_size"] != cfg.fs_size or state["sp_size"] != cfg.sp_size:
        raise ValueError("FSDP/SP topology differs from the saved checkpoint")
    if bool(cfg.ema_weight) != (state.get("ema") is not None):
        raise ValueError("EMA checkpoint state is missing or differs from the configured recipe")
    if weights == "ema" and state.get("ema") is None:
        raise ValueError("Requested EMA inference from a checkpoint without EMA")
    if state.get("ema") is not None and state["ema"]["num_updates"] != state["step"]:
        raise ValueError("EMA update count does not match the optimizer checkpoint")
    selected = state["ema"]["weights"] if weights == "ema" else state["weights"]
    params = {canonical_name(n): p for n, p in model.named_parameters() if p.requires_grad}
    if params.keys() != selected.keys():
        raise ValueError("Trainable original-layer scope changed")
    if any(selected[name].shape != p.shape for name, p in params.items()):
        raise ValueError("Checkpoint local shard shape differs")
    if ema is not None:
        ema.load_state_dict(state["ema"], allow_schedule_change="ema_schedule_change" in state)
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(selected[name])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_random:
        restore_rng(state["rng"])
    return state
