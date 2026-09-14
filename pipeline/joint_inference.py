"""Native full-sequence, video-only H3 i2v sampling, with optional wrapped cameras."""

from dataclasses import replace

import torch
from omegaconf import OmegaConf

from h3.packing import patchify
from h3.utils.scheduler import MiniMaxH3Scheduler
from model.diffusion import WorldViewsObjective, unpatchify
from utils.distributed import broadcast_scoped


def joint_inputs(document, current, conditions, sigma, cfg, device, camera=True):
    local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    local.chunk_size = 10**9
    local.h3.single_sequence = False
    if local.h3.get("chunk_group_range") is not None:
        local.h3.chunk_group_range = None
    if local.h3.get("chunk_size_range") is not None:
        local.h3.chunk_size_range = None
    views = []
    for view in document["views"]:
        joint_view = dict(view)
        joint_view.pop("generation_chunks", None)
        views.append(joint_view)
    document = {**document, "views": views}
    state = dict(chunk=0, sigma=sigma, current=current, conditions=conditions, cached=True)
    inputs, _, _, records, _ = WorldViewsObjective(local).pack(document, device, inference=state)
    inputs["attention_mask"] = replace(inputs["attention_mask"], joint=True, history_dropout=None)
    if not camera:
        for key in list(inputs):
            if key.startswith("camera_"):
                inputs.pop(key)
        inputs["scale_log"] = None
    return inputs, records


@torch.no_grad()
def generate(model, document, negative, cfg, device, steps=None, use_cache=None, camera=True, observer=None):
    if cfg.sampling_solver != "h3_euler" or cfg.guidance_scale != 1 or cfg.cfg_rescale_factor:
        raise ValueError("Native joint H3 uses h3_euler, distilled CFG=1 and no CFG rescale")
    if any(len(v.get("texts", [(-1, v["text"])])) != 1 for v in document["views"]):
        raise ValueError("Native joint sampling takes one prompt per view; use AR for chunk captions")
    if not document["views"][0]["condition"]:
        raise ValueError("i2v requires a conditioning image for view 0")
    training = model.training
    model.eval()
    try:
        conditions = WorldViewsObjective(cfg).condition_latents(document, device)
        current = [
            broadcast_scoped(torch.randn(v["latent"].shape, device=device, dtype=torch.float32), "sp")
            for v in document["views"]
        ]
        scheduler = MiniMaxH3Scheduler(shift=cfg.timestep_shift)
        scheduler.set_timesteps(steps or cfg.sampling_steps, device=device)
        for index, timestep in enumerate(scheduler.timesteps):
            inputs, records = joint_inputs(document, current, conditions, 1 - float(timestep), cfg, device, camera)
            velocity = model(**inputs).sample
            if observer is not None:
                observer(index, inputs, velocity)
            prediction = torch.cat([velocity[:, record["start"]:record["stop"]] for record in records], 1)
            samples = torch.cat([patchify(current[record["view"]]) for record in records], 1)
            updated = scheduler.step(prediction, timestep, samples, return_dict=False)[0]
            offset = 0
            for record in records:
                n = record["stop"] - record["start"]
                current[record["view"]] = unpatchify(updated[:, offset:offset + n], record["shape"])
                offset += n
        return [x.cpu() for x in current]
    finally:
        model.train(training)
