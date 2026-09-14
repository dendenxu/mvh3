"""Synchronized chunk rollout with native H3 Euler or WorldViews UniPC."""

import torch

from h3.modules.kv_cache import make_caches
from model.diffusion import WorldViewsObjective
from model.chunks import view_chunk_ids
from h3.packing import patchify, unpatchify
from utils.distributed import broadcast_scoped
from h3.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from h3.utils.scheduler import MiniMaxH3Scheduler


@torch.no_grad()
def generate(model, document, negative, cfg, device, steps=None, use_cache=None):
    core = model.module if hasattr(model, "module") else model
    training = model.training
    model.eval()
    objective = WorldViewsObjective(cfg)
    document = objective.prepare_document(document, device, training=False)
    if cfg.sampling_solver not in ("unipc", "h3_euler") or cfg.inference_multiview_mode != "synchronized":
        raise ValueError("Synchronized inference supports unipc or h3_euler")
    native = cfg.sampling_solver == "h3_euler"
    if native and (cfg.guidance_scale != 1 or cfg.cfg_rescale_factor):
        raise ValueError("Released H3 inference uses a single distilled forward (guidance_scale=1)")
    if cfg.kv_recompute_prefix_latents:
        raise ValueError("H3's nonuniform temporal stride requires a physical-time recompute-prefix setting")
    enabled = cfg.use_kv_cache if use_cache is None else use_cache
    positive_cache = make_caches(core, cfg, enabled, document)
    negative_cache = make_caches(core, cfg, enabled and cfg.use_uncond_kvcache and cfg.guidance_scale != 1, document)
    uncond = {
        **document, "views": [{
            **v, "text": negative,
            "texts": [(-1, negative)],
            "prompt": cfg.negative_prompt
        } for v in document["views"]]
    }
    history = [torch.zeros_like(v["latent"], device=device) for v in document["views"]]
    generated = [x.clone() for x in history]
    chunks = [view_chunk_ids(v, cfg.chunk_size, device) for v in document["views"]]
    chunk_count = max(
        int(frame_chunks[v["valid"].to(device)].max()) + 1 for frame_chunks, v in zip(chunks, document["views"]))
    context_noise = cfg.inference_context_noise if cfg.inference_context_noise is not None else cfg.context_noise
    # Native keyframe augmentation is drawn once per request, before target
    # noise, and held fixed across solver steps, CFG branches and cache writes.
    conditions = objective.condition_latents(document, device)

    def recompute_history(doc, chunk):
        # H3's joint text AdaLN depends on time. Recomputing clean history in
        # the noisy-target forward changes its keys at every solver step.
        # Rebuild at the same clean timestep used by persistent KV writes.
        caches = make_caches(core, cfg, True, document)
        # Replay the same chunk writes as the persistent cache. Besides keeping
        # sink/window eviction identical, this preserves native BF16 rounding:
        # one large prefix forward is not numerically equivalent on the 33B model.
        for previous in range(chunk):
            state = dict(chunk=previous, sigma=0. if cfg.clean_adaln else context_noise, history=history,
                         current=history, conditions=conditions, cached=True, update_cache=True)
            inputs, _, _, _, _ = objective.pack(doc, device, inference=state)
            model(**inputs, kv_caches=caches, update_cache=True)
        return caches

    try:
        for chunk in range(chunk_count):
            current = [broadcast_scoped(torch.randn_like(x), "sp") for x in history]
            if native:
                scheduler = MiniMaxH3Scheduler(shift=cfg.timestep_shift)
                scheduler.set_timesteps(steps or cfg.sampling_steps, device=device)
            else:
                scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps,
                                                        shift=cfg.timestep_shift, use_dynamic_shifting=False)
                scheduler.set_timesteps(steps or cfg.sampling_steps, device=device, shift=cfg.timestep_shift)
            for timestep in scheduler.timesteps:
                pos_step_cache = positive_cache if positive_cache is not None else recompute_history(document, chunk)
                state = dict(chunk=chunk,
                             sigma=1 - float(timestep) if native else float(timestep) / cfg.num_train_timesteps,
                             history=history, current=current, conditions=conditions, cached=True)
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
                for record in records:
                    predictions.append(velocity[:, record["start"]:record["stop"]])
                    samples.append(patchify(record["noisy"]))
                packed = torch.cat(samples, 1)
                # The original UniPC scheduler consumes noise-data velocity;
                # native H3 produces data-noise. Convert only at this boundary.
                prediction = torch.cat(predictions, 1)
                updated = scheduler.step(prediction if native else -prediction, timestep, packed, return_dict=False)[0]
                offset = 0
                for record, part in zip(records, samples):
                    n = part.shape[1]
                    current[record["view"]][:, :, record["selected"]] = unpatchify(updated[:, offset:offset + n],
                                                                                   record["shape"])
                    offset += n
            for i, frame_chunks in enumerate(chunks):
                selected = frame_chunks == chunk
                generated[i][:, :, selected] = current[i][:, :, selected]
                noise = broadcast_scoped(torch.randn_like(current[i]), "sp")
                damped = (1 - context_noise) * current[i] + context_noise * noise
                history[i][:, :, selected] = damped[:, :, selected]
            if positive_cache is not None:
                state = dict(chunk=chunk, sigma=0. if cfg.clean_adaln else context_noise, history=history,
                             current=history, conditions=conditions, cached=True, update_cache=True)
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
