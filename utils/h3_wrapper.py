"""WorldViews samples, independent per-view H3 VAEs, and versioned Qwen caches."""

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from h3.data import temporal_layout, pad_video


def raw_collate(samples):
    if len(samples) != 1:
        raise ValueError("Mixed source shapes require one source sample per rank")
    return samples[0]


def extract_views(sample):
    """Undo the original spatial packing without changing pixels or geometry."""
    cpu, frames = sample["cpu"], sample["frames"]
    pack = cpu["pack"]
    as_batch = bool(cpu.get("view_as_batch", False))
    count = int(cpu["orig_mv"]) if as_batch else int(sample["mv"])
    height, width = int(pack["height"]), int(pack["width"])
    result = []
    for view in range(count):
        if as_batch:
            pixels = frames[view]
            geometry = {k: sample[k][view] for k in ("Ks", "Rs", "Ts", "projs", "projs_inv")}
        else:
            ratio = float(pack["rs"][view])
            h, w = round(height * ratio), round(width * ratio)
            x, y = round(width * float(pack["xs"][view])), round(height * float(pack["ys"][view]))
            pixels = frames[..., y:y + h, x:x + w]
            geometry = {
                k: sample[k].reshape(frames.shape[0], count, *sample[k].shape[1:])[:, view]
                for k in ("Ks", "Rs", "Ts", "projs", "projs_inv")
            }
        k, r, t = geometry["Ks"].float(), geometry["Rs"].float(), geometry["Ts"].float()
        # WorldViews builds normalized intrinsics against the nominal canvas,
        # including for reduced-resolution tiles. Preserve that convention.
        pose = torch.empty((len(pixels), 10), dtype=torch.float32)
        pose[:, 0], pose[:, 1] = k[:, 0, 0] / width, k[:, 1, 1] / height
        pose[:, 2], pose[:, 3] = k[:, 0, 2] / width - .5, k[:, 1, 2] / height - .5
        pose[:, 4:7] = torch.from_numpy(Rotation.from_matrix(r.mT.numpy()).as_rotvec().astype(np.float32))
        pose[:, 7:10] = -(r.mT @ t).squeeze(-1)
        prompts = cpu["prompts"]
        prompt = prompts[view] if isinstance(prompts, (tuple, list)) else prompts
        fps = sample.get("fps", 16)
        if isinstance(fps, (list, tuple, np.ndarray, torch.Tensor)):
            values = torch.as_tensor(fps).flatten()
            fps = values[view if values.numel() > 1 else 0]
        scale = cpu.get("pose_stable_factor", 1.0)
        if isinstance(scale, (list, tuple, np.ndarray, torch.Tensor)):
            values = torch.as_tensor(scale).flatten()
            scale = values[view if values.numel() > 1 else 0]
        result.append(
            dict(pixels=pixels,
                 pose=pose,
                 projection=geometry["projs"].float(),
                 inverse=geometry["projs_inv"].float(),
                 prompt=str(prompt),
                 fps=float(fps),
                 scale=float(scale),
                 source_view=view,
                 source_start=0,
                 chunk_prompts=(cpu["chunk_prompts"][view] if as_batch else cpu["chunk_prompts"])
                 if cpu.get("chunk_prompts") else None))
    return result


def source_documents(sample, stage, short_frames=77):
    views = extract_views(sample)
    if stage == 2:
        return [
            dict(views=views,
                 isolated=bool(sample["cpu"].get("view_isolated", False) or sample["cpu"].get("view_as_batch", False)),
                 source=sample["cpu"].get("parquet", ""))
        ]
    documents = []
    for view in views:
        for start in range(0, len(view["pixels"]), short_frames):
            stop = min(start + short_frames, len(view["pixels"]))
            clip = {
                **view,
                **{
                    key: view[key][start:stop]
                    for key in ("pixels", "pose", "projection", "inverse")
                }, "source_start": start
            }
            documents.append(dict(views=[clip], isolated=True, source=sample["cpu"].get("parquet", "")))
    return documents


def sample_condition_lengths(views, cfg, validation=False):
    prefix = "val_" if validation else ""
    isolated = len(views) == 1
    max_lat = int(cfg.get(prefix + "cond_image_max_frames", 1))
    one_ratio = float(cfg.get(prefix + "cond_image_one_ratio", 0))
    k = 1 if max_lat <= 1 or torch.rand(()).item() < one_ratio else int(torch.randint(2, max_lat + 1, ()).item())
    if validation and max_lat <= 1:
        k = int(cfg.val_cond_image_frames)
    # These settings count Wan latents. Convert the prefix to physical frames.
    frames = 4 * (min(k, int(cfg.max_cond_latent)) - 1) + 1
    other = torch.rand(()).item() < float(cfg.get(prefix + "other_view_cond_ratio", 0))
    selected = [0]
    if not isolated and other:
        selected += [
            i for i in range(1, len(views))
            if torch.rand(()).item() < float(cfg.get(prefix + "other_view_cond_per_view_ratio", .5))
        ]
    if not validation and torch.rand(()).item() < float(cfg.cond_image_dropout_ratio):
        selected = []
    return [min(frames, len(view["pixels"])) if i in selected else 0 for i, view in enumerate(views)]


class VideoEncoder:

    def __init__(self, path, device="cuda", compile=False):
        from h3.modules.vae import AutoencoderKLMiniMaxH3
        self.device = torch.device(device)
        self.model = AutoencoderKLMiniMaxH3.from_pretrained(
            path, torch_dtype=torch.float32, local_files_only=True).eval().requires_grad_(False).to(device)
        if compile:
            self.model.encode = torch.compile(self.model.encode)
            self.model.decode = torch.compile(self.model.decode)
        self.mean = torch.tensor(self.model.config.latents_mean, device=device).reshape(1, 24, 1, 1, 1)
        self.std = torch.tensor(self.model.config.latents_std, device=device).reshape(1, 24, 1, 1, 1)
        self.pixel_mean = torch.tensor([.485, .456, .406], device=device).reshape(1, 3, 1, 1, 1)
        self.pixel_std = torch.tensor([.229, .224, .225], device=device).reshape(1, 3, 1, 1, 1)

    @torch.no_grad()
    def encode(self, pixels):
        layout = temporal_layout(len(pixels))
        h, w = pixels.shape[-2:]
        video = pixels.permute(1, 0, 2, 3)[None].to(self.device)
        # Pad each view independently to VAE stride 16 * patch 2. Never resize
        # or encode a spatial seam between unrelated views.
        video = F.pad(video, (0, (-w) % 32, 0, (-h) % 32, 0, 0), mode="replicate")
        video = (pad_video(video, layout) - self.pixel_mean) / self.pixel_std
        latent = self.model.encode(video).latent_dist.sample().half().float()
        latent = (latent - self.mean) / self.std
        if latent.shape[2] != len(layout.valid):
            raise ValueError("H3 VAE temporal geometry changed")
        # Fractional edge-patch weights preserve the original pixel-area loss.
        weights = torch.ones((h, w), device=self.device)
        weights = F.pad(weights, (0, (-w) % 32, 0, (-h) % 32))
        weights = F.avg_pool2d(weights[None, None], 32, 32)[0, 0]
        return latent.cpu(), layout, weights.cpu()

    @torch.no_grad()
    def decode(self, latent, height, width, frames):
        latent = latent.to(self.device) * self.std + self.mean
        pixels = self.model.decode(latent).sample * self.pixel_std + self.pixel_mean
        return pixels[0, :, :frames, :height, :width].permute(1, 0, 2, 3).clamp(0, 1).cpu()

    def prepare(self, document, cfg, validation=False):
        lengths = sample_condition_lengths(document["views"], cfg, validation)
        if document["isolated"] and len(lengths) > 1:
            # Every independent video keeps its own condition image.
            lengths = [sample_condition_lengths([v], cfg, validation)[0] for v in document["views"]]
        encoded = []

        def projections(view, layout, key):
            starts = layout.rotary_frames.long()
            indices = (starts[:, None] + torch.arange(4)).minimum(layout.camera_frames[:, None])
            indices = indices.clamp(0, len(view[key]) - 1)
            return view[key][indices]

        for view, cond_frames in zip(document["views"], lengths):
            latent, layout, weights = self.encode(view["pixels"])
            condition = None
            if cond_frames:
                cond, cond_layout, _ = self.encode(view["pixels"][:cond_frames])
                condition = dict(latent=cond,
                                 frames=cond_layout.rotary_frames,
                                 valid=cond_layout.valid,
                                 pose=view["pose"][cond_layout.camera_frames],
                                 projection=projections(view, cond_layout, "projection"),
                                 inverse=projections(view, cond_layout, "inverse"))
            encoded.append(
                dict(latent=latent,
                     pose=view["pose"][layout.camera_frames],
                     projection=projections(view, layout, "projection"),
                     inverse=projections(view, layout, "inverse"),
                     condition=condition,
                     frames=layout.rotary_frames,
                     valid=layout.valid,
                     spatial_weights=weights,
                     fps=view["fps"],
                     scale=view["scale"],
                     prompt=view["prompt"],
                     height=view["pixels"].shape[-2],
                     width=view["pixels"].shape[-1],
                     source_frames=len(view["pixels"]),
                     source_view=view["source_view"],
                     source_start=view["source_start"],
                     chunk_prompts=view.get("chunk_prompts")))
        return {**document, "views": encoded}


class TextEncoder:
    """Cache only complete Qwen3-VL layer-50 features; synchronize FSDP misses."""

    def __init__(self, cfg, wrap=None, device="cuda"):
        from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration
        checkpoint = Path(cfg.h3.checkpoint)
        self.device, self.cfg, self.encoder = device, cfg, None
        self.cache = Path(cfg.h3.text_cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
        self.tokenizer.padding_side = "right"
        self.wrap, self.model_class = wrap, Qwen3VLForConditionalGeneration
        fingerprint = hashlib.sha256(b"Qwen3VL.hidden_states[50].no_special_tokens.v2")
        for directory in (checkpoint / "tokenizer", checkpoint / "text_encoder"):
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    fingerprint.update(path.name.encode())
                    if path.suffix == ".safetensors":
                        stat = path.stat()
                        fingerprint.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
                    else:
                        fingerprint.update(path.read_bytes())
        self.identity = fingerprint.hexdigest()

    def cache_path(self, text):
        key = hashlib.sha256((self.identity + str(self.cfg.h3.text_max_length) + "\0" + text).encode()).hexdigest()
        return self.cache / (key + ".pt")

    @torch.no_grad()
    def __call__(self, texts):
        import torch.distributed as dist
        paths = [self.cache_path(text) for text in texts]
        missing = torch.tensor(int(any(not p.is_file() for p in paths)), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(missing, op=dist.ReduceOp.MAX)
        if missing.item():
            if self.encoder is None:
                encoder = self.model_class.from_pretrained(Path(self.cfg.h3.checkpoint) / "text_encoder",
                                                           torch_dtype=torch.bfloat16,
                                                           local_files_only=True,
                                                           attn_implementation="sdpa").model
                encoder.eval().requires_grad_(False)
                self.encoder = self.wrap(encoder) if self.wrap is not None else encoder.to(self.device)
                if self.cfg.text_encoder_compile:
                    self.encoder = torch.compile(self.encoder)
            tokens = self.tokenizer(texts,
                                    add_special_tokens=False,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.cfg.h3.text_max_length,
                                    return_tensors="pt").to(self.device)
            if tokens.input_ids.shape[1] == 0:
                raise ValueError("An empty caption requires an explicit nonempty prompt override")
            output = self.encoder(**tokens,
                                  mm_token_type_ids=torch.zeros_like(tokens.input_ids),
                                  use_cache=False,
                                  output_hidden_states=True)
            features = output.hidden_states[50].detach().cpu()
            for i, path in enumerate(paths):
                value = features[i:i + 1, tokens.attention_mask[i].bool().cpu()].contiguous()
                temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
                torch.save(dict(identity=self.identity, features=value), temp)
                os.replace(temp, path)
        result = []
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=True)
            if data["identity"] != self.identity or data["features"].shape[-1] != 5120:
                raise ValueError("Incompatible text cache")
            result.append(data["features"])
        return result
