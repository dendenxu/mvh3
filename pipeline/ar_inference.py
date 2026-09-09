"""Synchronized chunk rollout with WorldViews CFG, UniPC and history offload."""

import torch

from h3.modules.kv_cache import make_caches
from model.diffusion import WorldViewsObjective, chunk_ids, unpatchify
from h3.packing import patchify
from utils.distributed import broadcast_scoped
from h3.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


@torch.no_grad()
def generate(model, document, negative, cfg, device, steps=None, use_cache=None):
    core = model.module if hasattr(model, "module") else model
    training = model.training
    model.eval()
    objective = WorldViewsObjective(cfg)
    if cfg.sampling_solver != "unipc" or cfg.inference_multiview_mode != "synchronized":
        raise ValueError("Use the resolved WorldViews synchronized UniPC inference recipe")
    if cfg.kv_recompute_prefix_latents:
        raise ValueError("H3's nonuniform temporal stride requires a physical-time recompute-prefix setting")
    enabled = cfg.use_kv_cache if use_cache is None else use_cache
    positive_cache = make_caches(core, cfg, enabled)
    negative_cache = make_caches(core, cfg, enabled and cfg.use_uncond_kvcache)
    uncond = {
        **document, "views": [{
            **v, "text": negative,
            "texts": [(-1, negative)],
            "prompt": cfg.negative_prompt
        } for v in document["views"]]
    }
    history = [torch.zeros_like(v["latent"], device=device) for v in document["views"]]
    generated = [x.clone() for x in history]
    chunks = [chunk_ids(v["frames"].to(device), cfg.chunk_size) for v in document["views"]]
    n_chunks = max(int(c[v["valid"].to(device)].max()) + 1 for c, v in zip(chunks, document["views"]))
    context_noise = cfg.inference_context_noise if cfg.inference_context_noise is not None else cfg.context_noise

    def recompute_history(doc, chunk):
        # H3's joint text AdaLN depends on time. Recomputing clean history in
        # the noisy-target forward changes its keys at every solver step.
        # Rebuild at the same clean timestep used by persistent KV writes.
        caches = make_caches(core, cfg, True)
        if chunk:
            state = dict(chunk=chunk - 1,
                         sigma=0. if cfg.clean_adaln else context_noise,
                         history=history,
                         current=history,
                         cached=False,
                         update_cache=True)
            inputs, _, _, _, _ = objective.pack(doc, device, inference=state)
            model(**inputs, kv_caches=caches, update_cache=True)
        return caches

    try:
        for chunk in range(n_chunks):
            current = [broadcast_scoped(torch.randn_like(x), "sp") for x in history]
            scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps,
                                                    shift=cfg.timestep_shift,
                                                    use_dynamic_shifting=False)
            scheduler.set_timesteps(steps or cfg.sampling_steps, device=device, shift=cfg.timestep_shift)
            for timestep in scheduler.timesteps:
                pos_step_cache = positive_cache if positive_cache is not None else recompute_history(document, chunk)
                state = dict(chunk=chunk,
                             sigma=float(timestep) / cfg.num_train_timesteps,
                             history=history,
                             current=current,
                             cached=True)
                inputs, _, _, records, _ = objective.pack(document, device, inference=state)
                positive = model(**inputs, kv_caches=pos_step_cache).sample
                if cfg.guidance_scale != 1:
                    neg_step_cache = negative_cache if negative_cache is not None else recompute_history(uncond, chunk)
                    un_inputs, _, _, _, _ = objective.pack(uncond, device, inference=state)
                    unconditional = model(**un_inputs, kv_caches=neg_step_cache).sample
                    velocity = unconditional + cfg.guidance_scale * (positive - unconditional)
                    if cfg.cfg_rescale_factor:
                        std = positive.std(dim=(1, 2), keepdim=True)
                        scaled = velocity * std / velocity.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                        velocity = torch.lerp(velocity, scaled, cfg.cfg_rescale_factor)
                else:
                    velocity = positive
                if positive_cache is None:
                    for cache in pos_step_cache:
                        cache.clear()
                if negative_cache is None and cfg.guidance_scale != 1:
                    for cache in neg_step_cache:
                        cache.clear()
                predictions, samples = [], []
                for r in records:
                    predictions.append(velocity[:, r["start"]:r["stop"]])
                    samples.append(patchify(r["noisy"]))
                packed = torch.cat(samples, 1)
                # The original UniPC scheduler consumes noise-data velocity;
                # native H3 produces data-noise. Convert only at this boundary.
                updated = scheduler.step(-torch.cat(predictions, 1), timestep, packed, return_dict=False)[0]
                offset = 0
                for r, part in zip(records, samples):
                    n = part.shape[1]
                    current[r["view"]][:, :, r["selected"]] = unpatchify(updated[:, offset:offset + n], r["shape"])
                    offset += n
            for i, c in enumerate(chunks):
                select = c == chunk
                generated[i][:, :, select] = current[i][:, :, select]
                noise = broadcast_scoped(torch.randn_like(current[i]), "sp")
                damped = (1 - context_noise) * current[i] + context_noise * noise
                history[i][:, :, select] = damped[:, :, select]
            if positive_cache is not None:
                state = dict(chunk=chunk,
                             sigma=0. if cfg.clean_adaln else context_noise,
                             history=history,
                             current=history,
                             cached=True,
                             update_cache=True)
                inputs, _, _, _, _ = objective.pack(document, device, inference=state)
                model(**inputs, kv_caches=positive_cache, update_cache=True)
                if negative_cache is not None and cfg.guidance_scale != 1:
                    inputs, _, _, _, _ = objective.pack(uncond, device, inference=state)
                    model(**inputs, kv_caches=negative_cache, update_cache=True)
        return [x.cpu() for x in generated]
    finally:
        for caches in (positive_cache, negative_cache):
            if caches:
                for cache in caches:
                    cache.clear()
        model.train(training)
