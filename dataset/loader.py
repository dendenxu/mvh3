"""Load, encode and distribute the original WorldViews source batches."""

import time
from collections import deque
from functools import partial

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from dataset import create_dataset
from utils import distributed as groups
from utils.captions import caption_specs
from model.chunks import prepare_chunk_plan
from utils.config import stage_dataset_config
from dataset.mvgame import cycle, worker_init_fn


class BatchLoader:
    """Decode source samples, encode their views, and share them within SP.

    `pending` holds raw clips split from a source sample. `mixed` holds encoded
    documents gathered from all SP ranks; one document feeds one model update.
    A document keeps all its views together, preserving the source batch size.
    """

    def __init__(self, cfg, video, text, step=0, validation=False):
        self.cfg = cfg
        self.video = video
        self.text = text
        self.validation = validation
        self.stage = cfg.h3.stage
        self.pending = deque()
        self.mixed = deque()
        self.negative = None
        self.loader_step = step
        seed = cfg.seed if validation else cfg.seed + groups.get_rank() + step
        self.sampler_generator = torch.Generator().manual_seed(seed)
        self.build_loader(self.stage)

    def build_loader(self, stage):
        cfg = self.cfg
        data_cfg = stage_dataset_config(cfg, stage, self.validation)
        self.dataset = create_dataset(data_cfg, cfg)
        seed = cfg.seed if self.validation else cfg.seed + groups.get_rank() + self.loader_step
        if self.validation and cfg.get("inference_sequential_val", False):
            sampler = torch.utils.data.SequentialSampler(self.dataset)
        else:
            sampler = torch.utils.data.RandomSampler(
                self.dataset, generator=self.sampler_generator, replacement=True
            )
        loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=data_cfg.batch_size,
            pin_memory=True,
            num_workers=data_cfg.num_workers,
            persistent_workers=bool(data_cfg.num_workers),
            prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers else None,
            timeout=data_cfg.timeout if data_cfg.num_workers else 0,
            collate_fn=raw_collate,
            worker_init_fn=partial(worker_init_fn, seed=seed, dataset=self.dataset),
        )
        self.loader = cycle(loader)
        self.dataset_stage = stage

    def set_stage(self, stage):
        # Keep queued SHORT clips when switching the loader to FULL data.
        self.stage = stage
        if hasattr(self, "dataset") and self.dataset_stage != stage:
            self.build_loader(stage)

    def next(self):
        """Return one encoded document, replenishing the two queues as needed."""
        self.last_timings = dict(decode_seconds=0.0, vae_seconds=0.0, text_seconds=0.0, sp_gather_seconds=0.0)
        if self.mixed:
            return self.mixed.popleft()

        # Step 1: Decode the next source and retain every view/time window.
        started = time.monotonic()
        if not self.pending:
            self.pending.extend(source_documents(next(self.loader), self.stage, self.cfg.h3.short_frames))
        self.last_timings["decode_seconds"] = time.monotonic() - started

        # Step 2: Encode pixels and draw the latent blocks used by captions.
        started = time.monotonic()
        document = self.video.prepare(self.pending.popleft(), self.cfg, self.validation)

        # Each rank owns different raw samples until gather_mixed_batch. Plan
        # locally before encoding captions; broadcast the whole document later.
        document = prepare_chunk_plan(document, self.cfg, self.video.device, synchronize=False)
        self.last_timings["vae_seconds"] = time.monotonic() - started

        # Step 3: Encode each block's caption with its conditioning image.
        started = time.monotonic()
        self.encode_text(document)
        self.last_timings["text_seconds"] = time.monotonic() - started
        if self.cfg.h3.get("text_conditioning", "text_only") == "fl2va":
            self.last_timings.update(
                {"text_" + key: value for key, value in self.text.last_i2v_stats.items()}
            )

        # Step 4: Share complete documents, preserving each rank's source sample.
        # One independently decoded sample per rank; all SP ranks consume
        # every sample. Do not collapse the source batch by broadcasting rank 0.
        started = time.monotonic()
        self.mixed.extend(groups.gather_mixed_batch(document))
        self.last_timings["sp_gather_seconds"] = time.monotonic() - started
        return self.mixed.popleft()

    def encode_text(self, document):
        """Bind each planned caption to its image/text features before SP gather."""
        texts, specs, pictures = [], [], []
        shared_images = [
            v["condition_image"] for v in document["views"] if v.get("condition_image") is not None
        ]
        for view in document["views"]:
            pairs = caption_specs(view, self.cfg)
            if self.cfg.h3.get("single_sequence", False):
                view["texts_by_diffusion_chunk"] = True
                view["caption_specs"] = pairs
            view_specs = []
            for chunk, caption in pairs:
                view_specs.append((chunk, len(texts)))
                texts.append(caption)
                if document["isolated"] and view.get("condition_image") is not None:
                    pictures.append([view["condition_image"]])
                else:
                    pictures.append(shared_images)
            specs.append(view_specs)
        native_image = self.cfg.h3.get("text_conditioning", "text_only") == "fl2va"
        if native_image:
            image_encoded = self.text.i2v(list(zip(texts, pictures)))
            encoded = [value["features"] for value in image_encoded]

            # Text dropout retains the image semantics of i2v.
            dropped = None
            if self.cfg.cond_text_dropout_ratio and not self.validation:
                dropped = self.text.i2v([(self.cfg.negative_prompt, images) for images in pictures])
            self.negative = None
        else:
            encoded = self.text([*texts, self.cfg.negative_prompt])
            self.negative = encoded[-1]
        for view, view_specs in zip(document["views"], specs):
            values = [(chunk, encoded[index]) for chunk, index in view_specs]
            tag_values = (
                {chunk: image_encoded[index]["tags"] for chunk, index in view_specs} if native_image else {}
            )
            if not self.validation and torch.rand(()).item() < self.cfg.cond_text_dropout_ratio:
                if native_image:
                    values = [(chunk, dropped[index]["features"]) for chunk, index in view_specs]
                    tag_values = {chunk: dropped[index]["tags"] for chunk, index in view_specs}
                else:
                    values = [(chunk, self.negative) for chunk, _ in values]
            view["texts"] = values
            view["text"] = values[0][1]
            view["text_tag_specs"] = tag_values
            if native_image:
                view["text_tags"] = tag_values[values[0][0]]

    def state_dict(self):
        # Queues preserve already-decoded samples and their chunk/caption plan.
        # Worker prefetch cursors are not serialized by PyTorch's DataLoader.
        return dict(
            pending=list(self.pending),
            mixed=list(self.mixed),
            stage=self.stage,
            sampler=self.sampler_generator.get_state(),
        )

    def load_state_dict(self, state):
        self.pending, self.mixed = deque(state["pending"]), deque(state["mixed"])
        self.set_stage(state["stage"])
        self.sampler_generator.set_state(state["sampler"])


def raw_collate(samples):
    if len(samples) != 1:
        raise ValueError("Mixed source shapes require one source sample per rank")
    return samples[0]


def extract_views(sample):
    """Undo WorldViews spatial packing into per-view pixels and camera tracks.

    Input Rs/Ts are world-to-camera. Output pose rows are normalized intrinsics,
    camera-to-world rotation vectors, and world-space camera centers. Preserve
    the independently supplied projection/inverse matrices for matrix PRoPE.
    """
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
            pixels = frames[..., y : y + h, x : x + w]
            geometry = {
                k: sample[k].reshape(frames.shape[0], count, *sample[k].shape[1:])[:, view]
                for k in ("Ks", "Rs", "Ts", "projs", "projs_inv")
            }
        k, r, t = geometry["Ks"].float(), geometry["Rs"].float(), geometry["Ts"].float()

        # WorldViews builds normalized intrinsics against the nominal canvas,
        # including for reduced-resolution tiles. Preserve that convention.
        pose = torch.empty((len(pixels), 10), dtype=torch.float32)
        pose[:, 0], pose[:, 1] = k[:, 0, 0] / width, k[:, 1, 1] / height
        pose[:, 2], pose[:, 3] = k[:, 0, 2] / width - 0.5, k[:, 1, 2] / height - 0.5
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
            dict(
                pixels=pixels,
                pose=pose,
                projection=geometry["projs"].float(),
                inverse=geometry["projs_inv"].float(),
                prompt=str(prompt),
                fps=float(fps),
                scale=float(scale),
                source_view=view,
                source_start=0,
                caption_scene=(
                    (cpu["caption_scene"][view] if as_batch else cpu["caption_scene"])
                    if "caption_scene" in cpu
                    else None
                ),
                caption_motions=(
                    (cpu["caption_motions"][view] if as_batch else cpu["caption_motions"])
                    if "caption_motions" in cpu
                    else None
                ),
                caption_source_frames=cpu.get("caption_source_frames", len(pixels)),
                chunk_prompts=(
                    (cpu["chunk_prompts"][view] if as_batch else cpu["chunk_prompts"])
                    if cpu.get("chunk_prompts")
                    else None
                ),
            )
        )
    return result


def source_documents(sample, stage, short_frames=77):
    """Split Stage 1 time windows; keep each source's views in the same update."""
    views = extract_views(sample)
    if stage == 2:
        return [
            dict(
                views=views,
                isolated=bool(
                    sample["cpu"].get("view_isolated", False) or sample["cpu"].get("view_as_batch", False)
                ),
                source=sample["cpu"].get("parquet", ""),
            )
        ]
    documents = []

    # Shorten time without turning a source's view batch into separate updates.
    # The packed mask keeps each view independent, including its own image/text.
    for start in range(0, max(len(view["pixels"]) for view in views), short_frames):
        clips = []
        for view in views:
            if start >= len(view["pixels"]):
                continue
            stop = min(start + short_frames, len(view["pixels"]))
            clip = {
                **view,
                **{key: view[key][start:stop] for key in ("pixels", "pose", "projection", "inverse")},
                "source_start": start,
            }
            clips.append(clip)
        documents.append(dict(views=clips, isolated=True, source=sample["cpu"].get("parquet", "")))
    return documents
