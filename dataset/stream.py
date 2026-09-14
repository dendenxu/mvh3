"""Load, encode and distribute the original WorldViews source batches."""

from collections import deque
from functools import partial
import time

import torch

from dataset import create_dataset, cycle, worker_init_fn
from model.chunks import prepare_chunk_plan
from utils import distributed as groups
from utils.captions import caption_specs
from utils.config import stage_dataset_config
from utils.h3_wrapper import raw_collate, source_documents


class SourceStream:
    """Keep raw and encoded queues so every SP rank consumes every source sample."""

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
            sampler = torch.utils.data.RandomSampler(self.dataset, generator=self.sampler_generator, replacement=True)
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
        self.last_timings = dict(decode_seconds=0., vae_seconds=0., text_seconds=0., sp_gather_seconds=0.)
        if self.mixed:
            return self.mixed.popleft()
        started = time.monotonic()
        if not self.pending:
            self.pending.extend(source_documents(next(self.loader), self.stage, self.cfg.h3.short_frames))
        self.last_timings["decode_seconds"] = time.monotonic() - started
        started = time.monotonic()
        document = self.video.prepare(self.pending.popleft(), self.cfg, self.validation)
        # Each rank owns different raw samples until gather_mixed_batch. Plan
        # locally before encoding captions; broadcast the whole document later.
        document = prepare_chunk_plan(document, self.cfg, self.video.device, synchronize=False)
        self.last_timings["vae_seconds"] = time.monotonic() - started
        started = time.monotonic()
        self.encode_text(document)
        self.last_timings["text_seconds"] = time.monotonic() - started
        if self.cfg.h3.get("text_conditioning", "text_only") == "fl2va":
            self.last_timings.update({"text_" + key: value for key, value in self.text.last_i2v_stats.items()})
        # One independently decoded sample per rank; all SP ranks consume
        # every sample. Do not collapse the source batch by broadcasting rank 0.
        started = time.monotonic()
        self.mixed.extend(groups.gather_mixed_batch(document))
        self.last_timings["sp_gather_seconds"] = time.monotonic() - started
        return self.mixed.popleft()

    def encode_text(self, document):
        """Bind each planned caption to its image/text features before SP gather."""
        texts, specs, pictures = [], [], []
        shared_images = [v["condition_image"] for v in document["views"] if v.get("condition_image") is not None]
        for view in document["views"]:
            pairs = caption_specs(view, self.cfg)
            if self.cfg.h3.get("single_sequence", False):
                view["texts_by_bd"] = True
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
            tag_values = {chunk: image_encoded[index]["tags"] for chunk, index in view_specs} if native_image else {}
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
        return dict(pending=list(self.pending), mixed=list(self.mixed), stage=self.stage,
                    sampler=self.sampler_generator.get_state())

    def load_state_dict(self, state):
        self.pending, self.mixed = deque(state["pending"]), deque(state["mixed"])
        self.set_stage(state["stage"])
        self.sampler_generator.set_state(state["sampler"])
