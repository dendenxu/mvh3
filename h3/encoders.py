"""H3 video VAE and image/text encoders used by training and inference."""

import os
import uuid
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from h3.data import pad_video, temporal_layout
from utils.camera import prepare_camera_geometry


def sample_condition_lengths(views, cfg, validation=False):
    prefix = "val_" if validation else ""
    isolated = len(views) == 1
    max_lat = int(cfg.get(prefix + "cond_image_max_frames", 1))
    one_ratio = float(cfg.get(prefix + "cond_image_one_ratio", 0))
    k = (
        1
        if max_lat <= 1 or torch.rand(()).item() < one_ratio
        else int(torch.randint(2, max_lat + 1, ()).item())
    )
    if validation and max_lat <= 1:
        k = int(cfg.val_cond_image_frames)

    # These settings count Wan latents. Convert the prefix to physical frames.
    frames = 4 * (min(k, int(cfg.max_cond_latent)) - 1) + 1
    other = torch.rand(()).item() < float(cfg.get(prefix + "other_view_cond_ratio", 0))
    selected = [0]
    if not isolated and other:
        selected += [
            i
            for i in range(1, len(views))
            if torch.rand(()).item() < float(cfg.get(prefix + "other_view_cond_per_view_ratio", 0.5))
        ]
    if not validation and torch.rand(()).item() < float(cfg.cond_image_dropout_ratio):
        selected = []
    return [min(frames, len(view["pixels"])) if i in selected else 0 for i, view in enumerate(views)]


class VideoEncoder:
    """Convert per-view [frames, RGB, H, W] pixels to native H3 VAE latents.

    Encoded views keep camera/temporal metadata beside [1, 24, T, H/16, W/16]
    latents. Encoding views independently prevents seams between cameras.
    """

    def __init__(self, path, device="cuda", compile=False):
        from h3.modules.vae import AutoencoderKLMiniMaxH3

        self.device = torch.device(device)
        self.model = (
            AutoencoderKLMiniMaxH3.from_pretrained(path, torch_dtype=torch.float32, local_files_only=True)
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        if compile:
            self.model.encode = torch.compile(self.model.encode)
            self.model.decode = torch.compile(self.model.decode)
        self.mean = torch.tensor(self.model.config.latents_mean, device=device).reshape(1, 24, 1, 1, 1)
        self.std = torch.tensor(self.model.config.latents_std, device=device).reshape(1, 24, 1, 1, 1)
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1, 1)
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1, 1)

    @torch.no_grad()
    def encode(self, pixels, generator=None):
        layout = temporal_layout(len(pixels))
        h, w = pixels.shape[-2:]
        video = pixels.permute(1, 0, 2, 3)[None].to(self.device)

        # Pad each view independently to VAE stride 16 * patch 2. Never resize
        # or encode a spatial seam between unrelated views.
        video = F.pad(video, (0, (-w) % 32, 0, (-h) % 32, 0, 0), mode="replicate")
        video = (pad_video(video, layout) - self.pixel_mean) / self.pixel_std
        latent = self.model.encode(video).latent_dist.sample(generator=generator).half().float()
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
        """Encode video and image prefixes, then align cameras to native latents."""
        document = prepare_camera_geometry(document, cfg)
        lengths = sample_condition_lengths(document["views"], cfg, validation)
        if document["isolated"] and len(lengths) > 1:
            # Every independent video keeps its own condition image.
            lengths = [sample_condition_lengths([v], cfg, validation)[0] for v in document["views"]]
        encoded = []

        def projections(view, layout, key):
            # A latent's projection tracks its four supporting pixel frames.
            # Tail support repeats the last real camera instead of extrapolating.
            starts = layout.rotary_frames.long()
            indices = (starts[:, None] + torch.arange(4)).minimum(layout.camera_frames[:, None])
            indices = indices.clamp(0, len(view[key]) - 1)
            return view[key][indices]

        for view, cond_frames in zip(document["views"], lengths):
            latent, layout, weights = self.encode(view["pixels"])
            condition = None
            if cond_frames:
                seed = cfg.h3.get("condition_encode_seed")

                # Native H3 draws posterior noise on CPU, even for a CUDA VAE.
                generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
                cond, cond_layout, _ = self.encode(view["pixels"][:cond_frames], generator=generator)
                condition = dict(
                    latent=cond,
                    frames=cond_layout.rotary_frames,
                    valid=cond_layout.valid,
                    pose=view["pose"][cond_layout.camera_frames],
                    projection=projections(view, cond_layout, "projection"),
                    inverse=projections(view, cond_layout, "inverse"),
                )
            encoded.append(
                dict(
                    latent=latent,
                    pose=view["pose"][layout.camera_frames],
                    projection=projections(view, layout, "projection"),
                    inverse=projections(view, layout, "inverse"),
                    condition=condition,
                    condition_image=view["pixels"][0].mul(255).round().byte().cpu() if cond_frames else None,
                    frames=layout.rotary_frames,
                    valid=layout.valid,
                    spatial_weights=weights,
                    fps=view["fps"],
                    scale=view["scale"],
                    source_pose_stable_factor=view.get("source_pose_stable_factor", view["scale"]),
                    prompt=view["prompt"],
                    height=view["pixels"].shape[-2],
                    width=view["pixels"].shape[-1],
                    source_frames=len(view["pixels"]),
                    source_view=view["source_view"],
                    source_start=view["source_start"],
                    caption_scene=view.get("caption_scene"),
                    caption_motions=view.get("caption_motions"),
                    caption_source_frames=view.get("caption_source_frames", len(view["pixels"])),
                    chunk_prompts=view.get("chunk_prompts"),
                )
            )
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

    def ensure_encoder(self):
        if self.encoder is None:
            encoder = self.model_class.from_pretrained(
                Path(self.cfg.h3.checkpoint) / "text_encoder",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                attn_implementation="sdpa",
            ).model
            encoder.eval().requires_grad_(False)
            self.encoder = self.wrap(encoder) if self.wrap is not None else encoder.to(self.device)
            if self.cfg.text_encoder_compile:
                self.encoder = torch.compile(self.encoder)

    @torch.no_grad()
    def i2v(self, requests):
        """Native FL2VA picture labels/vision tags and complete Qwen layer-50 features.

        Deduplicate shared image/caption conditions before the Qwen forward.
        All ranks still participate when any rank misses its local cache.
        """
        from PIL import Image
        import torch.distributed as dist
        from transformers import Qwen3VLProcessor

        if not hasattr(self, "processor"):
            self.processor = Qwen3VLProcessor.from_pretrained(
                Path(self.cfg.h3.checkpoint) / "processor", local_files_only=True
            )
        paths = []
        for caption, images in requests:
            digest = hashlib.sha256((self.identity + "FL2VA-pictures-v1\0" + caption).encode())
            for img in images:
                digest.update(str(tuple(img.shape)).encode())
                digest.update(img.contiguous().numpy().tobytes())
            paths.append(self.cache / (digest.hexdigest() + ".pt"))
        if not paths:
            raise ValueError("Distributed image/text encoding needs at least one local request")
        unique = {}
        for index, path in enumerate(paths):
            unique.setdefault(path, index)
        cached = {}
        for path in unique:
            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
                if value["identity"] != self.identity or value.get("format") != "fl2va-i2v-v1":
                    raise ValueError("Incompatible native image/text cache")
                cached[path] = value
            except (FileNotFoundError, OSError):
                cached[path] = None
        needed = [path for path, value in cached.items() if value is None]
        missing = torch.tensor(int(bool(needed)), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(missing, op=dist.ReduceOp.MAX)
        if missing.item():
            # A cache-hit rank performs one matching collective sequence while
            # another rank encodes; its existing cache record stays unchanged.
            active_paths = needed or [next(iter(unique))]
            active_requests = [requests[unique[path]] for path in active_paths]
            images = [
                Image.fromarray(img.permute(1, 2, 0).numpy()) for _, imgs in active_requests for img in imgs
            ]
            has_images = torch.tensor(int(bool(images)), device=self.device)
            if dist.is_initialized():
                dist.all_reduce(has_images, op=dist.ReduceOp.MIN)
            if not has_images.item():
                raise ValueError(
                    "Native distributed i2v requires an image on every rank; use text-only encoding for t2v"
                )
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            sequences, tags, offset = [], [], 0
            for caption, imgs in active_requests:
                ids, types = [], []
                for index in range(len(imgs)):
                    count = (
                        int(vision["image_grid_thw"][offset].prod())
                        // self.processor.image_processor.merge_size**2
                    )
                    label = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                    visual = (
                        [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                        + [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * count
                        + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                    )
                    ids += label + visual
                    types += [1] * len(label) + [0] * len(visual)
                    offset += 1
                prompt_ids = self.tokenizer(caption, add_special_tokens=False)["input_ids"]
                ids += prompt_ids
                types += [1] * len(prompt_ids)
                sequences.append(ids)
                tags.append(torch.tensor(types, dtype=torch.long))
            tokens = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt").to(
                self.device
            )
            mm = torch.tensor(
                self.processor.create_mm_token_type_ids(tokens.input_ids.tolist()), device=self.device
            )
            self.ensure_encoder()
            output = self.encoder(
                **tokens,
                mm_token_type_ids=mm,
                use_cache=False,
                output_hidden_states=True,
                pixel_values=vision["pixel_values"].to(self.device, torch.bfloat16),
                image_grid_thw=vision["image_grid_thw"].to(self.device),
            )
            owners = [list(map(str, needed))]
            rank = dist.get_rank() if dist.is_initialized() else 0
            if dist.is_initialized():
                owners = [None] * dist.get_world_size()
                dist.all_gather_object(owners, list(map(str, needed)))
            for index, path in enumerate(active_paths):
                if cached[path] is not None:
                    continue
                value = (
                    output.hidden_states[50][index : index + 1, : len(sequences[index])]
                    .detach()
                    .cpu()
                    .contiguous()
                )
                record = dict(identity=self.identity, features=value, tags=tags[index], format="fl2va-i2v-v1")
                cached[path] = record

                # One publisher per shared key. Return computed values directly:
                # shared storage can lag immediately after an atomic rename.
                if not any(str(path) in names for names in owners[:rank]):
                    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
                    torch.save(record, temporary)
                    os.replace(temporary, path)
        self.last_i2v_stats = dict(
            requests=len(paths),
            unique_requests=len(unique),
            cache_misses=len(needed),
            encoded_requests=len(active_paths) if missing.item() else 0,
        )
        return [cached[path] for path in paths]

    def cache_path(self, text):
        key = hashlib.sha256(
            (self.identity + str(self.cfg.h3.text_max_length) + "\0" + text).encode()
        ).hexdigest()
        return self.cache / (key + ".pt")

    @torch.no_grad()
    def __call__(self, texts):
        import torch.distributed as dist

        paths = [self.cache_path(text) for text in texts]
        missing = torch.tensor(int(any(not p.is_file() for p in paths)), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(missing, op=dist.ReduceOp.MAX)
        if missing.item():
            self.ensure_encoder()
            tokens = self.tokenizer(
                texts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=self.cfg.h3.text_max_length,
                return_tensors="pt",
            ).to(self.device)
            if tokens.input_ids.shape[1] == 0:
                raise ValueError("An empty caption requires an explicit nonempty prompt override")
            output = self.encoder(
                **tokens,
                mm_token_type_ids=torch.zeros_like(tokens.input_ids),
                use_cache=False,
                output_hidden_states=True,
            )
            features = output.hidden_states[50].detach().cpu()
            for i, path in enumerate(paths):
                value = features[i : i + 1, tokens.attention_mask[i].bool().cpu()].contiguous()
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
