"""WorldViews' weighted chunk teacher forcing in H3's native flow convention."""

import math

import torch

from h3.modules.masking import CLEAN, CONDITION, NOISY, TokenLayout
from h3.packing import patchify
from utils.scheduler import FlowMatchScheduler
from utils.distributed import broadcast_scoped, get_sp_size


def chunk_ids(frames, chunk_size):
    # The first Wan chunk has 4*k-3 source frames; later chunks have 4*k.
    return ((frames + 3) / (4 * chunk_size)).floor().long()


def unpatchify(tokens, shape):
    _, channels, frames, height, width = shape
    return tokens.reshape(frames, height // 2, width // 2, channels, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(shape)


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

    def pack(self, document, device, step=0, override=None, inference=None):
        cfg, views = self.cfg, document["views"]
        max_chunks = max(int(chunk_ids(v["frames"][v["valid"]], cfg.chunk_size).max()) + 1 for v in views)
        if inference is None:
            high = broadcast_scoped(torch.rand((), device=device), "global").item() < .5
            lower, upper = (0, self.boundary_index) if high else (self.boundary_index, cfg.model.num_train_timesteps)
            indices = broadcast_scoped(torch.randint(lower, upper, (max_chunks, ), device=device), "sp")
            sigmas = self.scheduler.sigmas.to(device)[indices]
            weights = self.scheduler.linear_timesteps_weights.to(device)[indices]
        else:
            high = False
            sigmas = torch.full((max_chunks, ), float(inference["sigma"]), device=device)
            weights = torch.ones_like(sigmas)
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

        videos, poses, poses_f0, projections, inverses = [], [], [], [], []
        kinds, chunks, scopes, active, positions, camera_ids = [], [], [], [], [], []
        token_sigmas, scales, targets, loss_weights, records = [], [], [], [], []
        text, text_kinds, text_scopes, text_chunks, text_positions, text_sigmas = [], [], [], [], [], []
        shared_text = not document["isolated"] and all(v["prompt"] == views[0]["prompt"] for v in views)
        for view_index, view in enumerate(views):
            if shared_text and view_index:
                continue
            specs = view.get("texts", [(-1, view["text"])])
            for chunk, embedding in specs:
                embedding = embedding.to(device)
                n = embedding.shape[1]
                text.append(embedding)
                text_kinds.append(torch.full((n, ), CONDITION, device=device, dtype=torch.long))
                text_scopes.append(
                    torch.full((n, ), -1 if shared_text else view_index, device=device, dtype=torch.long))
                text_chunks.append(torch.full((n, ), chunk, device=device, dtype=torch.long))
                p = torch.zeros((n, 3), device=device, dtype=torch.float64)
                p[:, 0] = torch.arange(n, device=device)
                text_positions.append(p)
                text_sigmas.append(sigmas[max(0, min(chunk, len(sigmas) - 1))].expand(n))
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
            poses_f0.append(view["pose"][0].to(device).expand_as(pose))
            projections.append(source["projection"].to(device)[selected])
            inverses.append(source["inverse"].to(device)[selected])
            camera_ids.append(torch.arange(pose_offset, pose_offset + f, device=device).repeat_interleave(h * w // 4))
            pose_offset += f
            token_chunk = chunk_ids(frames, cfg.chunk_size).repeat_interleave(h * w // 4)
            chunks.append(torch.full_like(token_chunk, -1) if kind == CONDITION else token_chunk)
            kinds.append(torch.full_like(token_chunk, kind))
            scopes.append(torch.full_like(token_chunk, view_index))
            valid = source["valid"].to(device)[selected, None, None] & (view["spatial_weights"].to(device)[None] > 0)
            active.append(valid.flatten())
            area = math.sqrt(h * w)
            yy = ((1 - h / area) / 2 + torch.arange(h // 2, device=device) * 2 / area) * 32
            xx = ((1 - w / area) / 2 + torch.arange(w // 2, device=device) * 2 / area) * 32
            spatial = torch.stack(torch.meshgrid(yy, xx, indexing="ij"), -1).reshape(-1, 2).repeat(f, 1)
            times = text_count + frames.repeat_interleave(h * w // 4) * (40 / view["fps"])
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

        for i, view in enumerate(views):
            clean = view["latent"].to(device)
            frame_chunks = chunk_ids(view["frames"].to(device), cfg.chunk_size).clamp_max(max_chunks - 1)
            sigma = sigmas[frame_chunks]
            valid = view["valid"].to(device)
            context_sel = frame_chunks < int(frame_chunks[valid].max())
            if view["condition"] is not None:
                cond = view["condition"]
                append(view, i, cond["latent"].to(device), slice(None), CONDITION, 0., geometry=cond)
            if inference is None:
                context = clean.clone() if override is None else override[i].to(device).clone()
                if override is not None:
                    keep = int(cfg.resampling_forcing_clean_chunks)
                    if cfg.resampling_forcing_clean_chunks_mv_only and len(views) == 1:
                        keep = 0
                    context[:, :, frame_chunks < keep] = clean[:, :, frame_chunks < keep]
                random = broadcast_scoped(torch.randn_like(context, dtype=torch.float32), "sp")
                context = ((1 - levels[i]) * context + levels[i] * random) * cfg.context_scale
                append(view, i, context, context_sel, CLEAN, levels[i])
                noise = broadcast_scoped(torch.randn_like(clean, dtype=torch.float32), "sp")
                noisy = (1 - sigma[None, None, :, None, None]) * clean + sigma[None, None, :, None, None] * noise
                append(view, i, noisy, slice(None), NOISY, sigma, target=clean - noise, weight=weights[frame_chunks])
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
        pad = (-length) % get_sp_size()
        media = torch.cat(videos, 1)
        target = torch.cat(targets, 1)
        loss_weight = torch.cat(loss_weights)
        tags = torch.cat((torch.ones(text_count, device=device,
                                     dtype=torch.long), torch.zeros(media_count, device=device, dtype=torch.long)))
        kind = torch.cat((*text_kinds, *kinds))
        chunk = torch.cat((*text_chunks, *chunks))
        scope = torch.cat((*text_scopes, *scopes))
        enabled = torch.cat((torch.ones(text_count, device=device, dtype=torch.bool), *active))
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
        layout = TokenLayout(kind, chunk, scope, not document["isolated"], enabled, dropout)
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
                      camera_pose_f0=torch.cat(poses_f0)[None],
                      camera_indices=ids,
                      camera_projections=(torch.cat(projections)[None], torch.cat(inverses)[None]),
                      attention_mask=layout,
                      scale_log=table[:, 1] if cfg.model.scale_cond else None)
        return inputs, target, loss_weight, records, high

    def __call__(self, model, document, device, step, override=None):
        inputs, target, weights, records, high = self.pack(document, device, step, override)
        prediction = model(**inputs).sample
        loss = (
            (prediction.float() - target.float()).square().mean(-1)[0] * weights).sum() / weights.sum().clamp_min(1)
        x0 = []
        for record in records:
            velocity = unpatchify(prediction[:, record["start"]:record["stop"]].detach(), record["shape"])
            denoised = record["noisy"] + record["sigmas"][None, None, :, None, None] * velocity
            x0.append(denoised.cpu())
        rf = (cfg := self.cfg).resampling_forcing and step >= cfg.resampling_forcing_warmup_steps and not high
        rf = rf and any(int(chunk_ids(v["frames"][v["valid"]], cfg.chunk_size).max()) > 0 for v in document["views"])
        return loss, dict(high=high,
                          tokens=len(inputs["token_tags"]),
                          rf=rf,
                          x0=x0,
                          sigma=float((1 - inputs["timestep"]).mean()),
                          views=len(document["views"]))
