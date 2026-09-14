"""Chunk flow matching in H3's native data-ward convention."""

import math
from dataclasses import replace

import torch

from h3.modules.masking import CLEAN, CONDITION, NOISY, TokenLayout
from h3.data import spatial_rotary_grid, temporal_rotary_clock
from h3.packing import patchify
from h3.compile_shapes import pad_rows
from utils.scheduler import FlowMatchScheduler
from utils.distributed import broadcast_scoped, get_sp_size
from utils.camera import prepare_camera_geometry
from model.chunks import prepare_chunk_plan, prepare_clean_prefix, caption_chunk
from utils.captions import bind_caption_features


def chunk_ids(frames, chunk_size):
    # The first Wan chunk has 4*k-3 source frames; later chunks have 4*k.
    return ((frames + 3) / (4 * chunk_size)).floor().long()


def view_chunk_ids(view, chunk_size, device=None):
    """Keep decoder-support latents in the view's final requested chunk."""
    chunks = (view["generation_chunks"].to(device) if "generation_chunks" in view
              else chunk_ids(view["frames"].to(device), chunk_size))
    return chunks.clamp_max(chunks[view["valid"].to(device)].max())


def unpatchify(tokens, shape):
    _, channels, frames, height, width = shape
    return tokens.reshape(frames, height // 2, width // 2, channels, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(shape)


def same_text_conditioning(views, selected_chunk=None):
    reference = views[0]
    reference_specs = reference.get("texts", [(-1, reference["text"])])
    if selected_chunk is not None:
        reference_specs = [spec for spec in reference_specs if spec[0] == selected_chunk]
        if len(reference_specs) != 1:
            return False
    for view in views[1:]:
        specs = view.get("texts", [(-1, view["text"])])
        if selected_chunk is not None:
            specs = [spec for spec in specs if spec[0] == selected_chunk]
        if len(specs) != len(reference_specs):
            return False
        for (chunk, embedding), (ref_chunk, ref_embedding) in zip(specs, reference_specs):
            if chunk != ref_chunk or not torch.equal(embedding, ref_embedding):
                return False
            tags = view.get("text_tag_specs", {}).get(chunk, view.get("text_tags"))
            ref_tags = reference.get("text_tag_specs", {}).get(ref_chunk, reference.get("text_tags"))
            if (tags is None) != (ref_tags is None) or (tags is not None and not torch.equal(tags, ref_tags)):
                return False
    return True


class WorldViewsObjective:

    def __init__(self, cfg):
        self.cfg = cfg
        self.scheduler = FlowMatchScheduler(shift=cfg.model.timestep_shift,
                                            sigma_min=0,
                                            extra_one_step=True,
                                            num_train_timesteps=cfg.model.num_train_timesteps)
        self.scheduler.set_timesteps(cfg.model.num_train_timesteps, training=True)
        self.boundary_index = int(
            torch.searchsorted(-self.scheduler.timesteps_cpu,
                               -int(cfg.model.boundary * cfg.model.num_train_timesteps)))

    def prepare_document(self, document, device, training=True):
        document = prepare_chunk_plan(document, self.cfg, device)
        document = bind_caption_features(document, self.cfg)
        if training and self.cfg.h3.get("single_sequence", False):
            document = prepare_clean_prefix(document, self.cfg, device)
        return document

    def resample_document(self, document):
        if not self.cfg.h3.get("single_sequence", False):
            return document
        # RF promotes one predicted block to history. Reusing the same cut
        # would only recycle the already-clean prefix and be a no-op.
        counts = [int(v["generation_chunks"].max()) + 1 for v in document["views"]]
        limit = min(counts) - 1
        return {**document, "views": [
            {**v, "clean_prefix_chunks": min(v["clean_prefix_chunks"] + 1,
                                            n - 1 if document["isolated"] else limit)}
            for v, n in zip(document["views"], counts)]}

    def sample_sigmas(self, count, device):
        if self.cfg.h3.get("noise_sampling", "worldviews_ranges") == "shifted_uniform":
            u = broadcast_scoped(torch.rand((count,), device=device), "sp")
            shift = float(self.cfg.model.timestep_shift)
            sigma = shift * u / (1 + (shift - 1) * u)
            return sigma, torch.ones_like(sigma), False
        high = broadcast_scoped(torch.rand((), device=device), "global").item() < .5
        lower, upper = (0, self.boundary_index) if high else (self.boundary_index, self.cfg.model.num_train_timesteps)
        indices = broadcast_scoped(torch.randint(lower, upper, (count,), device=device), "sp")
        return (self.scheduler.sigmas.to(device)[indices],
                self.scheduler.linear_timesteps_weights.to(device)[indices], high)

    def condition_latents(self, document, device):
        level = float(self.cfg.h3.get("condition_noise", 0.))
        conditions = []
        for view in document["views"]:
            if view["condition"] is None:
                conditions.append(None)
                continue
            latent = view["condition"]["latent"].to(device)
            if level:
                noise = broadcast_scoped(torch.randn_like(latent, dtype=torch.float32), "sp")
                latent = (1 - level) * latent + level * noise
            conditions.append(latent)
        return conditions

    def pack(self, document, device, step=0, override=None, inference=None, evaluation_sigma=None):
        document = self.prepare_document(document, device, training=inference is None)
        document = prepare_camera_geometry(document, self.cfg)
        cfg, views = self.cfg, document["views"]
        single = cfg.h3.get("single_sequence", False)
        max_chunks = max(int(view_chunk_ids(v, cfg.chunk_size).max()) + 1 for v in views)
        counts = [int(view_chunk_ids(v, cfg.chunk_size).max()) + 1 for v in views]
        noise_count = sum(counts) if single and document["isolated"] else max_chunks
        if inference is None and evaluation_sigma is None:
            sigmas, weights, high = self.sample_sigmas(noise_count, device)
        else:
            high = False
            sigma = evaluation_sigma if inference is None else inference["sigma"]
            sigmas = torch.full((noise_count, ), float(sigma), device=device)
            weights = torch.ones_like(sigmas)
        view_sigmas = list(sigmas.split(counts)) if single and document["isolated"] else [sigmas] * len(views)
        view_weights = list(weights.split(counts)) if single and document["isolated"] else [weights] * len(views)
        if inference is None:
            dropout = broadcast_scoped(torch.rand((max_chunks, max_chunks), device=device),
                                       "sp") < cfg.history_dropout_ratio
            levels = broadcast_scoped(torch.randn(len(views), device=device), "sp")
            levels = (cfg.context_noise + cfg.context_noise_std * levels).clamp(0, 1)
        else:
            dropout = torch.zeros((max_chunks, max_chunks), device=device, dtype=torch.bool)
            levels = torch.full(
                (len(views), ),
                cfg.inference_context_noise if cfg.inference_context_noise is not None else cfg.context_noise,
                device=device)

        videos, poses, projections, inverses = [], [], [], []
        separate_references = document["isolated"] and len(views) > 1 and not cfg.model.prope_unwrapped
        reference_poses, reference_projections, reference_inverses = [], [], []
        kinds, chunks, scopes, active, positions, camera_ids = [], [], [], [], [], []
        token_sigmas, scales, targets, loss_weights, records = [], [], [], [], []
        text, text_kinds, text_scopes, text_chunks, text_positions, text_sigmas = [], [], [], [], [], []
        text_tags = []
        text_active = []
        text_time_origins = [0] * len(views)
        # Decide sharing per chunk so future edits cannot change earlier prefixes.
        shared_chunks = {chunk for chunk, _ in views[0].get("texts", [(-1, views[0]["text"])])
                         if not document["isolated"] and same_text_conditioning(views, chunk)
                         and all(caption_chunk(v, chunk, cfg.chunk_size) ==
                                 caption_chunk(views[0], chunk, cfg.chunk_size) for v in views)}
        for view_index, view in enumerate(views):
            specs = view.get("texts", [(-1, view["text"])])
            for chunk, embedding in specs:
                generation_chunk = caption_chunk(view, chunk, cfg.chunk_size)
                if single and inference is not None and generation_chunk > inference["chunk"]:
                    continue
                shared_text = chunk in shared_chunks
                if shared_text and view_index:
                    continue
                embedding = embedding.to(device)
                n = embedding.shape[1]
                if chunk <= 0:
                    text_time_origins[view_index] += n
                text.append(embedding)
                tags = view.get("text_tag_specs", {}).get(chunk, view.get("text_tags"))
                text_tags.append(torch.ones(n, device=device, dtype=torch.long) if tags is None else tags.to(device))
                text_kinds.append(torch.full((n, ), CONDITION, device=device, dtype=torch.long))
                text_scopes.append(
                    torch.full((n, ), -1 if shared_text else view_index, device=device, dtype=torch.long))
                text_chunks.append(torch.full((n, ), generation_chunk, device=device, dtype=torch.long))
                past_cached = (single and inference is not None and inference.get("cached", False)
                               and 0 <= generation_chunk < inference["chunk"])
                text_active.append(torch.full((n,), not past_cached, device=device, dtype=torch.bool))
                p = torch.zeros((n, 3), device=device, dtype=torch.float64)
                p[:, 0] = torch.arange(n, device=device)
                text_positions.append(p)
                text_sigma = view_sigmas[view_index][max(0, min(generation_chunk, counts[view_index] - 1))]
                if single:
                    cut = view["clean_prefix_chunks"] if inference is None else inference["chunk"]
                    if generation_chunk < cut:
                        text_sigma = levels[view_index] * (0 if cfg.clean_adaln else 1)
                text_sigmas.append(text_sigma.expand(n))
        text_count = sum(t.shape[1] for t in text)
        pose_offset = video_offset = 0

        def append(view, view_index, latent, selected, kind, noise_level, target=None, weight=None, geometry=None):
            nonlocal pose_offset, video_offset
            source = view if geometry is None else geometry
            latent = latent[:, :, selected]
            frame_indices = torch.arange(len(source["frames"]), device=device)[selected]
            frames = source["frames"].to(device)[selected]
            f, h, w = latent.shape[2:]
            if not f:
                return
            n = f * h * w // 4
            tokens = patchify(latent)
            videos.append(tokens)
            pose = source["pose"].to(device)[selected]
            poses.append(pose)
            projections.append(source["projection"].to(device)[selected])
            inverses.append(source["inverse"].to(device)[selected])
            if separate_references:
                # Independent videos must not inherit another video's camera gauge.
                reference_poses.append(view["pose"][:1].to(device).expand(f, -1))
                for key, destination in (("projection", reference_projections), ("inverse", reference_inverses)):
                    reference = view[key][0].to(device)
                    if reference.ndim == 3:
                        reference = reference[:1]
                    destination.append(reference[None].expand(f, *reference.shape))
            camera_ids.append(torch.arange(pose_offset, pose_offset + f, device=device).repeat_interleave(h * w // 4))
            pose_offset += f
            token_chunk = view_chunk_ids(source, cfg.chunk_size, device)[selected].repeat_interleave(h * w // 4)
            frame_kind = torch.as_tensor(kind, device=device).expand(len(source["frames"]))[selected]
            token_kind = frame_kind.repeat_interleave(h * w // 4)
            chunks.append(torch.where(token_kind == CONDITION, -1, token_chunk))
            kinds.append(token_kind)
            scopes.append(torch.full_like(token_chunk, view_index))
            # Temporal padding carries VAE reconstruction support. Masking it
            # corrupts real tail frames through the non-causal decoder.
            valid = (view["spatial_weights"].to(device)[None] > 0).expand(f, -1, -1)
            active.append(valid.flatten())
            spatial = spatial_rotary_grid(h, w).to(device).repeat(f, 1)
            clock = temporal_rotary_clock(len(source["frames"]), view["fps"]).to(device)[selected]
            # Unrelated or future captions cannot shift this video's RoPE clock.
            origin = text_time_origins[view_index] if document["isolated"] else sum(text_time_origins)
            times = origin + clock.repeat_interleave(h * w // 4)
            positions.append(torch.cat((times[:, None], spatial), -1))
            level = torch.as_tensor(noise_level, device=device).expand(len(source["frames"]))[selected]
            token_sigmas.append(level.repeat_interleave(h * w // 4))
            scales.append(torch.full((n, ), math.log(view["scale"]), device=device))
            targets.append(torch.zeros_like(tokens) if target is None else patchify(target[:, :, selected]))
            if weight is None:
                loss_weights.append(torch.zeros(n, device=device))
            else:
                loss_weights.append(
                    (weight[selected, None, None] * valid * view["spatial_weights"].to(device)[None]).flatten())
                records.append(
                    dict(view=view_index,
                         start=video_offset,
                         stop=video_offset + n,
                         selected=frame_indices,
                         shape=latent.shape,
                         noisy=latent.detach(),
                         sigmas=level))
            video_offset += n

        conditions = (inference.get("conditions") if inference is not None else None)
        if conditions is None:
            conditions = self.condition_latents(document, device)
        for i, view in enumerate(views):
            clean = view["latent"].to(device)
            frame_chunks = view_chunk_ids(view, cfg.chunk_size, device)
            sigma = view_sigmas[i][frame_chunks]
            valid = view["valid"].to(device)
            context_sel = frame_chunks < int(frame_chunks[valid].max())
            if view["condition"] is not None:
                cond = view["condition"]
                cond_chunks = view_chunk_ids(cond, cfg.chunk_size, device).clamp_max(max_chunks - 1)
                cond_sigma = view_sigmas[i][cond_chunks.clamp_max(counts[i] - 1)].clamp_max(float(cfg.h3.get("condition_noise", 0.)))
                append(view, i, conditions[i], slice(None), CONDITION, cond_sigma, geometry=cond)
            if inference is None:
                context = clean.clone() if override is None else override[i].to(device).clone()
                if override is not None:
                    keep = int(cfg.resampling_forcing_clean_chunks)
                    if cfg.resampling_forcing_clean_chunks_mv_only and len(views) == 1:
                        keep = 0
                    context[:, :, frame_chunks < keep] = clean[:, :, frame_chunks < keep]
                if not single or levels[i].item():
                    random = broadcast_scoped(torch.randn_like(context, dtype=torch.float32), "sp")
                    context = (1 - levels[i]) * context + levels[i] * random
                context = context * cfg.context_scale
                if not single:
                    append(view, i, context, context_sel, CLEAN, levels[i])
                noise = broadcast_scoped(torch.randn_like(clean, dtype=torch.float32), "sp")
                noisy = (1 - sigma[None, None, :, None, None]) * clean + sigma[None, None, :, None, None] * noise
                frame_weights = view_weights[i][frame_chunks]
                kind = NOISY
                if single:
                    prefix = frame_chunks < view["clean_prefix_chunks"]
                    noisy[:, :, prefix] = context[:, :, prefix]
                    sigma = torch.where(prefix, levels[i] * (0 if cfg.clean_adaln else 1), sigma)
                    frame_weights = frame_weights * ~prefix
                    kind = torch.where(prefix, CLEAN, NOISY)
                append(view, i, noisy, slice(None), kind, sigma, target=clean - noise, weight=frame_weights)
            else:
                current = frame_chunks == inference["chunk"]
                if not inference.get("cached", False):
                    history_sel = frame_chunks < inference["chunk"]
                    history = inference["history"][i].to(device)
                    append(view, i, history, history_sel, CLEAN, 0. if cfg.clean_adaln else levels[i])
                kind = CLEAN if inference.get("update_cache", False) else NOISY
                append(view,
                       i,
                       inference["current"][i].to(device),
                       current,
                       kind,
                       sigma,
                       weight=torch.ones_like(sigma))

        media_count = sum(v.shape[1] for v in videos)
        length = text_count + media_count
        buckets = cfg.h3.get("training_shape_buckets", {}) if inference is None else {}
        token_multiple = math.lcm(get_sp_size(), buckets.get("tokens", 1))
        pad = (-length) % token_multiple
        media = torch.cat(videos, 1)
        target = torch.cat(targets, 1)
        loss_weight = torch.cat(loss_weights)
        tags = torch.cat((*text_tags, torch.zeros(media_count, device=device, dtype=torch.long)))
        kind = torch.cat((*text_kinds, *kinds))
        chunk = torch.cat((*text_chunks, *chunks))
        scope = torch.cat((*text_scopes, *scopes))
        enabled = torch.cat((*text_active, *active))
        ids = torch.cat((torch.full((text_count, ), -1, device=device, dtype=torch.long), *camera_ids))
        pos = torch.cat((*text_positions, *positions))
        sigma = torch.cat((*text_sigmas, *token_sigmas))
        scale = torch.cat((torch.zeros(text_count, device=device), *scales))
        if pad:
            # Empty audio rows give SP a divisible sequence without inventing
            # video/camera observations. They are isolated in attention and loss.
            tags = torch.cat((tags, torch.full((pad, ), 2, device=device, dtype=torch.long)))
            kind = torch.cat((kind, torch.zeros(pad, device=device, dtype=torch.long)))
            chunk = torch.cat((chunk, torch.full((pad, ), -1, device=device, dtype=torch.long)))
            scope = torch.cat((scope, torch.full((pad, ), -1, device=device, dtype=torch.long)))
            enabled = torch.cat((enabled, torch.zeros(pad, device=device, dtype=torch.bool)))
            ids = torch.cat((ids, torch.full((pad, ), -1, device=device, dtype=torch.long)))
            pos = torch.cat((pos, torch.zeros((pad, 3), device=device)))
            sigma = torch.cat((sigma, torch.zeros(pad, device=device)))
            scale = torch.cat((scale, torch.zeros(pad, device=device)))
        # Share AdaLN rows across identical (sigma, scale), keeping conditioning
        # constant per view while noise remains independently sampled per chunk.
        table, time_ids = torch.unique(torch.stack((sigma, scale), -1), dim=0, return_inverse=True)
        table = pad_rows(table, 0, buckets.get("timesteps", 0), repeat=True)
        for dimension in (0, 1):
            dropout = pad_rows(dropout, dimension, buckets.get("chunks", 0))
        layout = TokenLayout(kind, chunk, scope, not document["isolated"], enabled, dropout, single_sequence=single)
        inputs = dict(hidden_states=media,
                      audio_hidden_states=media.new_zeros((1, pad, 32)),
                      encoder_hidden_states=torch.cat(text, 1),
                      timestep=1 - table[:, 0],
                      timestep_indices=time_ids,
                      token_tags=tags,
                      position_ids=pos,
                      video_indices=torch.arange(text_count, length, device=device),
                      audio_indices=torch.arange(length, length + pad, device=device),
                      text_indices=torch.arange(text_count, device=device),
                      camera_pose=torch.cat(poses)[None],
                      camera_indices=ids,
                      camera_projections=(torch.cat(projections)[None], torch.cat(inverses)[None]),
                      attention_mask=layout,
                      scale_log=table[:, 1] if cfg.model.scale_cond and cfg.h3.scale_conditioning == "spatial_rotary" else None)
        if single and inference is not None and inference.get("cached", False):
            # The refiner recomputes caption prefixes, while main attention
            # reads their stored clean KV exactly once from the history cache.
            refiner = replace(layout, active=None)
            indices = inputs["text_indices"]
            inputs["text_attention_mask"] = (refiner.dense(indices) if text_count <= 4096
                                               else refiner.block_mask(indices))
        if not cfg.model.prope_unwrapped:
            if separate_references:
                inputs["camera_reference"] = torch.cat(reference_poses)[None]
                inputs["camera_projection_reference"] = (torch.cat(reference_projections)[None],
                                                          torch.cat(reference_inverses)[None])
            else:
                ref = views[0]
                inputs["camera_reference"] = ref["pose"][:1].to(device)[None]
                p, pi = ref["projection"][0].to(device), ref["inverse"][0].to(device)
                if p.ndim == 3:
                    p, pi = p[:1], pi[:1]
                inputs["camera_projection_reference"] = (p[None, None], pi[None, None])
        return inputs, target, loss_weight, records, high

    def __call__(self, model, document, device, step, override=None):
        document = self.prepare_document(document, device)
        inputs, target, weights, records, high = self.pack(document, device, step, override)
        prediction = model(**inputs).sample
        loss = (
            (prediction.float() - target.float()).square().mean(-1)[0] * weights).sum() / weights.sum().clamp_min(1)
        rf = (cfg := self.cfg).resampling_forcing and step >= cfg.resampling_forcing_warmup_steps and not high
        rf = rf and any(int(view_chunk_ids(v, cfg.chunk_size).max()) > 0 for v in document["views"])
        if cfg.h3.get("single_sequence", False):
            next_document = self.resample_document(document)
            rf = rf and any(new["clean_prefix_chunks"] > old["clean_prefix_chunks"]
                            for old, new in zip(document["views"], next_document["views"]))
        x0 = None
        if rf:
            x0 = []
            for record in records:
                velocity = unpatchify(prediction[:, record["start"]:record["stop"]].detach(), record["shape"])
                denoised = record["noisy"] + record["sigmas"][None, None, :, None, None] * velocity
                x0.append(denoised.cpu())
        # Conditions/history have separate AdaLN rows. Log the generated
        # frames' noise levels, rather than averaging those clean rows into t.
        target_sigmas = torch.cat([
            record["sigmas"][document["views"][record["view"]]["valid"].to(device)[record["selected"]]]
            for record in records
        ])
        if cfg.h3.get("single_sequence", False):
            target_sigmas = torch.cat([record["sigmas"][
                document["views"][record["view"]]["valid"].to(device) &
                (view_chunk_ids(document["views"][record["view"]], cfg.chunk_size, device)
                 >= document["views"][record["view"]]["clean_prefix_chunks"])] for record in records])
        log = dict(high=high,
                          tokens=len(inputs["token_tags"]),
                          rf=rf,
                          x0=x0,
                          sigma=float(target_sigmas.mean()),
                          sigma_min=float(target_sigmas.min()),
                          sigma_max=float(target_sigmas.max()),
                          views=len(document["views"]))
        if cfg.h3.get("single_sequence", False):
            sizes = torch.cat([torch.bincount(view["generation_chunks"].cpu()) for view in document["views"]])
            media_layout = inputs["attention_mask"]
            log.update(chunk_size_min=int(sizes.min()), chunk_size_max=int(sizes.max()),
                       chunk_count=len(sizes),
                       clean_prefix_chunks=sum(v["clean_prefix_chunks"] for v in document["views"]),
                       clean_video_tokens=int((media_layout.kind[inputs["video_indices"]] == CLEAN).sum()),
                       supervised_video_tokens=int(weights.gt(0).sum()))
        return loss, log
