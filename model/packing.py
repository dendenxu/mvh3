"""Pack caption and video tokens into H3's joint sequence.

SequencePacker only arranges tensors and metadata. The objective supplies the
already-noised latents, noise levels and loss weights; packing draws no noise.
"""

import math
from dataclasses import replace

import torch

from h3.compile_shapes import pad_rows
from h3.data import spatial_rotary_grid, temporal_rotary_clock
from h3.modules.masking import CONDITION, TokenLayout
from h3.packing import patchify
from model.chunks import caption_chunk, view_chunk_ids
from utils.distributed import get_sp_size


def same_text_conditioning(views, selected_chunk=None):
    """Share a caption only when its features, chunk and image tags all match."""
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


class SequencePacker:
    """Collect text, conditioning images and video blocks in their native order.

    Each buffer has one entry per segment and is concatenated in finish().
    Records map predicted video tokens back to their view and latent frames.
    """

    def __init__(self, document, cfg, device):
        self.document = document
        self.cfg = cfg
        self.device = device
        self.single_sequence = cfg.h3.get("single_sequence", False)
        views = document["views"]
        self.videos, self.targets, self.loss_weights, self.records = [], [], [], []
        self.kinds, self.chunks, self.scopes = [], [], []
        self.active, self.positions, self.camera_ids = [], [], []
        self.token_sigmas, self.scales = [], []
        self.poses, self.projections, self.inverses = [], [], []
        self.separate_references = document["isolated"] and len(views) > 1 and not cfg.model.prope_unwrapped
        self.reference_poses, self.reference_projections, self.reference_inverses = [], [], []
        self.text, self.text_tags, self.text_active = [], [], []
        self.text_kinds, self.text_scopes, self.text_chunks = [], [], []
        self.text_positions, self.text_sigmas = [], []
        self.text_time_origins = [0] * len(views)
        self.pose_offset = 0
        self.video_offset = 0

    def add_text(self, view_sigmas, chunk_counts, context_levels, inference=None):
        """Keep per-block captions causal and share identical joint-view captions."""
        document, cfg, device = self.document, self.cfg, self.device
        views = document["views"]
        single_sequence = self.single_sequence
        # Decide sharing per chunk so future edits cannot change earlier prefixes.
        shared_chunks = set()
        for chunk, _ in views[0].get("texts", [(-1, views[0]["text"])]):
            if document["isolated"] or not same_text_conditioning(views, chunk):
                continue
            generation_chunk = caption_chunk(views[0], chunk, cfg.chunk_size)
            if all(caption_chunk(view, chunk, cfg.chunk_size) == generation_chunk for view in views):
                shared_chunks.add(chunk)
        for view_index, view in enumerate(views):
            specs = view.get("texts", [(-1, view["text"])])
            for chunk, embedding in specs:
                generation_chunk = caption_chunk(view, chunk, cfg.chunk_size)
                if single_sequence and inference is not None and generation_chunk > inference["chunk"]:
                    continue
                shared_text = chunk in shared_chunks
                if shared_text and view_index:
                    continue
                embedding = embedding.to(device)
                token_count = embedding.shape[1]
                if chunk <= 0:
                    self.text_time_origins[view_index] += token_count
                self.text.append(embedding)
                tags = view.get("text_tag_specs", {}).get(chunk, view.get("text_tags"))
                self.text_tags.append(
                    torch.ones(token_count, device=device, dtype=torch.long) if tags is None else tags.to(device))
                self.text_kinds.append(torch.full((token_count, ), CONDITION, device=device, dtype=torch.long))
                self.text_scopes.append(
                    torch.full((token_count, ), -1 if shared_text else view_index, device=device, dtype=torch.long))
                self.text_chunks.append(torch.full((token_count, ), generation_chunk, device=device, dtype=torch.long))
                past_cached = (single_sequence and inference is not None and inference.get("cached", False)
                               and 0 <= generation_chunk < inference["chunk"])
                self.text_active.append(torch.full((token_count, ), not past_cached, device=device, dtype=torch.bool))
                position = torch.zeros((token_count, 3), device=device, dtype=torch.float64)
                position[:, 0] = torch.arange(token_count, device=device)
                self.text_positions.append(position)
                text_sigma = view_sigmas[view_index][max(0, min(generation_chunk, chunk_counts[view_index] - 1))]
                if single_sequence:
                    cut = view["clean_prefix_chunks"] if inference is None else inference["chunk"]
                    if generation_chunk < cut:
                        text_sigma = context_levels[view_index] * (0 if cfg.clean_adaln else 1)
                self.text_sigmas.append(text_sigma.expand(token_count))
        self.text_count = sum(t.shape[1] for t in self.text)

    def add_video(self, view, view_index, latent, selected, kind, noise_level, target=None, weight=None,
                  geometry=None):
        """Append one image/history/target segment, retaining decoder support."""
        document, cfg, device = self.document, self.cfg, self.device
        source = view if geometry is None else geometry
        latent = latent[:, :, selected]
        frame_indices = torch.arange(len(source["frames"]), device=device)[selected]
        frame_count, h, w = latent.shape[2:]
        if not frame_count:
            return
        token_count = frame_count * h * w // 4
        tokens = patchify(latent)
        self.videos.append(tokens)
        pose = source["pose"].to(device)[selected]
        self.poses.append(pose)
        self.projections.append(source["projection"].to(device)[selected])
        self.inverses.append(source["inverse"].to(device)[selected])
        if self.separate_references:
            # Independent videos must not inherit another video's camera gauge.
            self.reference_poses.append(view["pose"][:1].to(device).expand(frame_count, -1))
            for key, destination in (("projection", self.reference_projections), ("inverse", self.reference_inverses)):
                reference = view[key][0].to(device)
                if reference.ndim == 3:
                    reference = reference[:1]
                destination.append(reference[None].expand(frame_count, *reference.shape))
        self.camera_ids.append(
            torch.arange(self.pose_offset, self.pose_offset + frame_count,
                         device=device).repeat_interleave(h * w // 4))
        self.pose_offset += frame_count
        token_chunk = view_chunk_ids(source, cfg.chunk_size, device)[selected].repeat_interleave(h * w // 4)
        frame_kind = torch.as_tensor(kind, device=device).expand(len(source["frames"]))[selected]
        token_kind = frame_kind.repeat_interleave(h * w // 4)
        self.chunks.append(torch.where(token_kind == CONDITION, -1, token_chunk))
        self.kinds.append(token_kind)
        self.scopes.append(torch.full_like(token_chunk, view_index))
        # Temporal padding carries VAE reconstruction support. Masking it
        # corrupts real tail frames through the non-causal decoder.
        valid = (view["spatial_weights"].to(device)[None] > 0).expand(frame_count, -1, -1)
        self.active.append(valid.flatten())
        spatial = spatial_rotary_grid(h, w).to(device).repeat(frame_count, 1)
        clock = temporal_rotary_clock(len(source["frames"]), view["fps"]).to(device)[selected]
        # Unrelated or future captions cannot shift this video's RoPE clock.
        origin = self.text_time_origins[view_index] if document["isolated"] else sum(self.text_time_origins)
        times = origin + clock.repeat_interleave(h * w // 4)
        self.positions.append(torch.cat((times[:, None], spatial), -1))
        level = torch.as_tensor(noise_level, device=device).expand(len(source["frames"]))[selected]
        self.token_sigmas.append(level.repeat_interleave(h * w // 4))
        self.scales.append(torch.full((token_count, ), math.log(view["scale"]), device=device))
        self.targets.append(torch.zeros_like(tokens) if target is None else patchify(target[:, :, selected]))
        if weight is None:
            self.loss_weights.append(torch.zeros(token_count, device=device))
        else:
            self.loss_weights.append(
                (weight[selected, None, None] * valid * view["spatial_weights"].to(device)[None]).flatten())
            self.records.append(
                dict(view=view_index, start=self.video_offset, stop=self.video_offset + token_count,
                     selected=frame_indices, shape=latent.shape, noisy=latent.detach(), sigmas=level))
        self.video_offset += token_count

    def finish(self, history_dropout, inference=None):
        """Build joint masks, camera/time tables, targets and loss weights."""
        document, cfg, device = self.document, self.cfg, self.device
        views = document["views"]
        single_sequence = self.single_sequence
        media_count = sum(v.shape[1] for v in self.videos)
        length = self.text_count + media_count
        buckets = cfg.h3.get("training_shape_buckets", {}) if inference is None else {}
        token_multiple = math.lcm(get_sp_size(), buckets.get("tokens", 1))
        pad = (-length) % token_multiple
        media = torch.cat(self.videos, 1)
        target = torch.cat(self.targets, 1)
        loss_weight = torch.cat(self.loss_weights)
        tags = torch.cat((*self.text_tags, torch.zeros(media_count, device=device, dtype=torch.long)))
        kind = torch.cat((*self.text_kinds, *self.kinds))
        chunk = torch.cat((*self.text_chunks, *self.chunks))
        scope = torch.cat((*self.text_scopes, *self.scopes))
        enabled = torch.cat((*self.text_active, *self.active))
        ids = torch.cat((torch.full((self.text_count, ), -1, device=device, dtype=torch.long), *self.camera_ids))
        pos = torch.cat((*self.text_positions, *self.positions))
        sigma = torch.cat((*self.text_sigmas, *self.token_sigmas))
        scale = torch.cat((torch.zeros(self.text_count, device=device), *self.scales))
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
            history_dropout = pad_rows(history_dropout, dimension, buckets.get("chunks", 0))
        layout = TokenLayout(kind, chunk, scope, not document["isolated"], enabled, history_dropout,
                             single_sequence=single_sequence)
        inputs = dict(
            hidden_states=media, audio_hidden_states=media.new_zeros(
                (1, pad, 32)), encoder_hidden_states=torch.cat(self.text,
                                                               1), timestep=1 - table[:, 0], timestep_indices=time_ids,
            token_tags=tags, position_ids=pos, video_indices=torch.arange(self.text_count, length, device=device),
            audio_indices=torch.arange(length, length + pad, device=device),
            text_indices=torch.arange(self.text_count, device=device), camera_pose=torch.cat(self.poses)[None],
            camera_indices=ids, camera_projections=(torch.cat(self.projections)[None],
                                                    torch.cat(self.inverses)[None]), attention_mask=layout,
            scale_log=table[:, 1] if cfg.model.scale_cond and cfg.h3.scale_conditioning == "spatial_rotary" else None)
        if single_sequence and inference is not None and inference.get("cached", False):
            # The refiner recomputes caption prefixes, while main attention
            # reads their stored clean KV exactly once from the history cache.
            refiner = replace(layout, active=None)
            indices = inputs["text_indices"]
            inputs["text_attention_mask"] = (refiner.dense(indices)
                                             if self.text_count <= 4096 else refiner.block_mask(indices))
        if not cfg.model.prope_unwrapped:
            if self.separate_references:
                inputs["camera_reference"] = torch.cat(self.reference_poses)[None]
                inputs["camera_projection_reference"] = (torch.cat(self.reference_projections)[None],
                                                         torch.cat(self.reference_inverses)[None])
            else:
                reference = views[0]
                inputs["camera_reference"] = reference["pose"][:1].to(device)[None]
                projection = reference["projection"][0].to(device)
                inverse = reference["inverse"][0].to(device)
                if projection.ndim == 3:
                    projection, inverse = projection[:1], inverse[:1]
                inputs["camera_projection_reference"] = (projection[None, None], inverse[None, None])
        return inputs, target, loss_weight, self.records
