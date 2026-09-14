"""Chunk flow matching in H3's native data-ward convention."""

import torch

from h3.modules.masking import CLEAN, CONDITION, NOISY
from h3.packing import unpatchify
from utils.scheduler import FlowMatchScheduler
from utils.distributed import broadcast_scoped
from utils.camera import prepare_camera_geometry
from model.chunks import prepare_chunk_plan, prepare_clean_prefix, view_chunk_ids
from model.packing import SequencePacker
from utils.captions import bind_caption_features


class WorldViewsObjective:
    """Sample chunk noise, build training inputs and compute the flow loss."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.scheduler = FlowMatchScheduler(shift=cfg.model.timestep_shift, sigma_min=0, extra_one_step=True,
                                            num_train_timesteps=cfg.model.num_train_timesteps)
        self.scheduler.set_timesteps(cfg.model.num_train_timesteps, training=True)
        self.boundary_index = int(
            torch.searchsorted(-self.scheduler.timesteps_cpu,
                               -int(cfg.model.boundary * cfg.model.num_train_timesteps)))

    def prepare_document(self, document, device, training=True):
        # Captions belong to the sampled blocks, regardless of the later clean cut.
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
        chunk_counts = [int(v["generation_chunks"].max()) + 1 for v in document["views"]]
        limit = min(chunk_counts) - 1
        views = []
        for view, count in zip(document["views"], chunk_counts):
            last_cut = count - 1 if document["isolated"] else limit
            cut = min(view["clean_prefix_chunks"] + 1, last_cut)
            views.append({**view, "clean_prefix_chunks": cut})
        return {**document, "views": views}

    def sample_sigmas(self, count, device):
        if self.cfg.h3.get("noise_sampling", "worldviews_ranges") == "shifted_uniform":
            u = broadcast_scoped(torch.rand((count, ), device=device), "sp")
            shift = float(self.cfg.model.timestep_shift)
            sigma = shift * u / (1 + (shift - 1) * u)
            return sigma, torch.ones_like(sigma), False
        high = broadcast_scoped(torch.rand((), device=device), "global").item() < .5
        lower, upper = (0, self.boundary_index) if high else (self.boundary_index, self.cfg.model.num_train_timesteps)
        indices = broadcast_scoped(torch.randint(lower, upper, (count, ), device=device), "sp")
        return (self.scheduler.sigmas.to(device)[indices], self.scheduler.linear_timesteps_weights.to(device)[indices],
                high)

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
        """Return model inputs, flow targets, loss weights and rollout records.

        Inference supplies the current noisy block and generated history.
        Training draws noise in order: chunk sigma, history dropout, context
        level, image condition, then context/target noise for each view.
        """
        document = self.prepare_document(document, device, training=inference is None)
        document = prepare_camera_geometry(document, self.cfg)
        cfg, views = self.cfg, document["views"]
        single_sequence = cfg.h3.get("single_sequence", False)
        chunk_counts = [int(view_chunk_ids(v, cfg.chunk_size).max()) + 1 for v in views]
        max_chunks = max(chunk_counts)
        noise_count = sum(chunk_counts) if single_sequence and document["isolated"] else max_chunks
        if inference is None and evaluation_sigma is None:
            sigmas, weights, high = self.sample_sigmas(noise_count, device)
        else:
            high = False
            sigma = evaluation_sigma if inference is None else inference["sigma"]
            sigmas = torch.full((noise_count, ), float(sigma), device=device)
            weights = torch.ones_like(sigmas)
        if single_sequence and document["isolated"]:
            view_sigmas = list(sigmas.split(chunk_counts))
            view_weights = list(weights.split(chunk_counts))
        else:
            view_sigmas = [sigmas] * len(views)
            view_weights = [weights] * len(views)
        if inference is None:
            history_dropout = broadcast_scoped(torch.rand((max_chunks, max_chunks), device=device),
                                               "sp") < cfg.history_dropout_ratio
            context_levels = broadcast_scoped(torch.randn(len(views), device=device), "sp")
            context_levels = (cfg.context_noise + cfg.context_noise_std * context_levels).clamp(0, 1)
        else:
            history_dropout = torch.zeros((max_chunks, max_chunks), device=device, dtype=torch.bool)
            context_levels = torch.full(
                (len(views), ),
                cfg.inference_context_noise if cfg.inference_context_noise is not None else cfg.context_noise,
                device=device)

        packer = SequencePacker(document, cfg, device)
        packer.add_text(view_sigmas, chunk_counts, context_levels, inference)

        conditions = (inference.get("conditions") if inference is not None else None)
        if conditions is None:
            conditions = self.condition_latents(document, device)
        for i, view in enumerate(views):
            clean = view["latent"].to(device)
            frame_chunks = view_chunk_ids(view, cfg.chunk_size, device)
            sigma = view_sigmas[i][frame_chunks]
            valid = view["valid"].to(device)
            context_selected = frame_chunks < int(frame_chunks[valid].max())
            if view["condition"] is not None:
                cond = view["condition"]
                cond_chunks = view_chunk_ids(cond, cfg.chunk_size, device).clamp_max(max_chunks - 1)
                cond_sigma = view_sigmas[i][cond_chunks.clamp_max(chunk_counts[i] - 1)]
                cond_sigma = cond_sigma.clamp_max(float(cfg.h3.get("condition_noise", 0.)))
                packer.add_video(view, i, conditions[i], slice(None), CONDITION, cond_sigma, geometry=cond)
            if inference is None:
                context = clean.clone() if override is None else override[i].to(device).clone()
                if override is not None:
                    keep = int(cfg.resampling_forcing_clean_chunks)
                    if cfg.resampling_forcing_clean_chunks_mv_only and len(views) == 1:
                        keep = 0
                    context[:, :, frame_chunks < keep] = clean[:, :, frame_chunks < keep]
                if not single_sequence or context_levels[i].item():
                    random = broadcast_scoped(torch.randn_like(context, dtype=torch.float32), "sp")
                    context = (1 - context_levels[i]) * context + context_levels[i] * random
                context = context * cfg.context_scale
                if not single_sequence:
                    packer.add_video(view, i, context, context_selected, CLEAN, context_levels[i])
                noise = broadcast_scoped(torch.randn_like(clean, dtype=torch.float32), "sp")
                noisy = (1 - sigma[None, None, :, None, None]) * clean + sigma[None, None, :, None, None] * noise
                frame_weights = view_weights[i][frame_chunks]
                kind = NOISY
                if single_sequence:
                    # One sequence: replace its prefix with history and supervise
                    # only the noisy suffix. Decoder-support frames stay included.
                    prefix = frame_chunks < view["clean_prefix_chunks"]
                    noisy[:, :, prefix] = context[:, :, prefix]
                    sigma = torch.where(prefix, context_levels[i] * (0 if cfg.clean_adaln else 1), sigma)
                    frame_weights = frame_weights * ~prefix
                    kind = torch.where(prefix, CLEAN, NOISY)
                packer.add_video(view, i, noisy, slice(None), kind, sigma, target=clean - noise, weight=frame_weights)
            else:
                current = frame_chunks == inference["chunk"]
                if not inference.get("cached", False):
                    history_selected = frame_chunks < inference["chunk"]
                    history = inference["history"][i].to(device)
                    packer.add_video(view, i, history, history_selected, CLEAN,
                                     0. if cfg.clean_adaln else context_levels[i])
                kind = CLEAN if inference.get("update_cache", False) else NOISY
                packer.add_video(view, i, inference["current"][i].to(device), current, kind, sigma,
                                 weight=torch.ones_like(sigma))

        inputs, target, loss_weight, records = packer.finish(history_dropout, inference)
        return inputs, target, loss_weight, records, high

    def __call__(self, model, document, device, step, override=None):
        cfg = self.cfg
        document = self.prepare_document(document, device)
        inputs, target, weights, records, high = self.pack(document, device, step, override)
        prediction = model(**inputs).sample
        token_error = (prediction.float() - target.float()).square().mean(-1)[0]
        loss = (token_error * weights).sum() / weights.sum().clamp_min(1)
        rf = cfg.resampling_forcing and step >= cfg.resampling_forcing_warmup_steps and not high
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
        target_sigmas = []
        for record in records:
            view = document["views"][record["view"]]
            selected = view["valid"].to(device)
            if cfg.h3.get("single_sequence", False):
                frame_chunks = view_chunk_ids(view, cfg.chunk_size, device)
                selected = selected & (frame_chunks >= view["clean_prefix_chunks"])
            else:
                selected = selected[record["selected"]]
            target_sigmas.append(record["sigmas"][selected])
        target_sigmas = torch.cat(target_sigmas)
        log = dict(high=high, tokens=len(inputs["token_tags"]), rf=rf, x0=x0, sigma=float(target_sigmas.mean()),
                   sigma_min=float(target_sigmas.min()), sigma_max=float(target_sigmas.max()),
                   views=len(document["views"]))
        if cfg.h3.get("single_sequence", False):
            sizes = torch.cat([torch.bincount(view["generation_chunks"].cpu()) for view in document["views"]])
            media_layout = inputs["attention_mask"]
            log.update(chunk_size_min=int(sizes.min()), chunk_size_max=int(sizes.max()), chunk_count=len(sizes),
                       clean_prefix_chunks=sum(v["clean_prefix_chunks"] for v in document["views"]),
                       clean_video_tokens=int((media_layout.kind[inputs["video_indices"]] == CLEAN).sum()),
                       supervised_video_tokens=int(weights.gt(0).sum()))
        return loss, log
