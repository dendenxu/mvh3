"""Full-source two-stage WorldViews training on the existing H3 layers."""

from collections import deque
from functools import partial
import gc
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from h3.checkpoint import load_original_transformer
from utils.config import validate_config
from h3.distributed.fsdp import configure_model, wrap_model, wrap_text, compile_blocks, parameter_groups, load_compile_cache, save_compile_cache
from utils.h3_wrapper import raw_collate, source_documents, VideoEncoder, TextEncoder
from model.diffusion import WorldViewsObjective
from utils.checkpoint import load_checkpoint, save_checkpoint
from dataset import create_dataset, cycle, worker_init_fn
from utils import distributed as groups
from utils.misc import set_seed


class SourceStream:

    def __init__(self, cfg, video, text, step=0, validation=False):
        self.cfg, self.video, self.text, self.validation = cfg, video, text, validation
        dc = cfg.val_dataset if validation else cfg.dataset
        self.dataset = create_dataset(dc, cfg)
        seed = cfg.seed if validation else cfg.seed + groups.get_rank() + step
        self.sampler_generator = torch.Generator().manual_seed(seed)
        if validation and cfg.get("inference_sequential_val", False):
            sampler = torch.utils.data.SequentialSampler(self.dataset)
        else:
            sampler = torch.utils.data.RandomSampler(self.dataset, generator=self.sampler_generator, replacement=True)
        loader = torch.utils.data.DataLoader(self.dataset,
                                             sampler=sampler,
                                             batch_size=dc.batch_size,
                                             pin_memory=True,
                                             num_workers=dc.num_workers,
                                             persistent_workers=bool(dc.num_workers),
                                             prefetch_factor=dc.prefetch_factor if dc.num_workers else None,
                                             timeout=dc.timeout if dc.num_workers else 0,
                                             collate_fn=raw_collate,
                                             worker_init_fn=partial(worker_init_fn, seed=seed, dataset=self.dataset))
        self.loader, self.pending, self.mixed = cycle(loader), deque(), deque()
        self.stage = 2 if validation else cfg.h3.stage
        self.negative = None

    def set_stage(self, stage):
        if stage != self.stage:
            # Finish no partially consumed short source silently: the queued
            # clips remain available as independent examples in stage 2.
            self.stage = stage

    def next(self):
        if self.mixed:
            return self.mixed.popleft()
        if not self.pending:
            self.pending.extend(source_documents(next(self.loader), self.stage, self.cfg.h3.short_frames))
        document = self.video.prepare(self.pending.popleft(), self.cfg, self.validation)
        texts, specs = [], []
        for view in document["views"]:
            prompt = self.cfg.get("prompt_override", "") or view["prompt"] or self.cfg.negative_prompt
            if view.get("chunk_prompts") and not self.cfg.get("prompt_override", ""):
                from model.diffusion import chunk_ids
                count = int(chunk_ids(view["frames"][view["valid"]], self.cfg.chunk_size).max()) + 1
                start = int((view["source_start"] + 3) // (4 * self.cfg.chunk_size))
                captions = view["chunk_prompts"]
                pairs = [(chunk, captions[min(start + chunk, len(captions) - 1)]) for chunk in range(count)]
            else:
                pairs = [(-1, prompt)]
            view_specs = []
            for chunk, caption in pairs:
                view_specs.append((chunk, len(texts)))
                texts.append(caption)
            specs.append(view_specs)
        texts.append(self.cfg.negative_prompt)
        encoded = self.text(texts)
        self.negative = encoded[-1]
        for view, view_specs in zip(document["views"], specs):
            values = [(chunk, encoded[index]) for chunk, index in view_specs]
            if not self.validation and torch.rand(()).item() < self.cfg.cond_text_dropout_ratio:
                values = [(chunk, self.negative) for chunk, _ in values]
            view["texts"], view["text"] = values, values[0][1]
        # One independently decoded sample per rank; all SP ranks consume
        # every sample. Do not collapse the source batch by broadcasting rank 0.
        self.mixed.extend(groups.gather_mixed_batch(document))
        return self.mixed.popleft()

    def state_dict(self):
        return dict(pending=list(self.pending),
                    mixed=list(self.mixed),
                    stage=self.stage,
                    sampler=self.sampler_generator.get_state())

    def load_state_dict(self, state):
        self.pending, self.mixed = deque(state["pending"]), deque(state["mixed"])
        self.stage = state["stage"]
        self.sampler_generator.set_state(state["sampler"])


class Trainer:

    def __init__(self, cfg):
        self.cfg = validate_config(cfg)
        groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
        if groups.get_sp_size() != cfg.sp_size or groups.fs_size != cfg.fs_size:
            raise ValueError("Launch enough ranks for the requested FSDP/SP sizes; no silent clamping")
        self.device = torch.device("cuda", torch.cuda.current_device())
        set_seed(cfg.seed)
        self.logdir = Path(cfg.h3.logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        if groups.get_rank() == 0:
            OmegaConf.save(cfg, self.logdir / "resolved.yaml")
        load_compile_cache(cfg.h3.compile_cache)
        model = load_original_transformer(cfg.h3.checkpoint, progress=print if groups.get_rank() == 0 else None)
        self.signature = configure_model(model, cfg)
        self.model = wrap_model(model, cfg)
        compile_blocks(self.model.module, cfg)
        self.optimizer = torch.optim.AdamW(parameter_groups(self.model, cfg),
                                           betas=(cfg.beta1, cfg.beta2),
                                           weight_decay=cfg.weight_decay,
                                           fused=True)
        if cfg.optim_compile:
            self.optimizer.step = torch.compile(self.optimizer.step)
        self.objective = WorldViewsObjective(cfg)
        self.step, self.stage, self.pending_rf, self.depth = 0, cfg.h3.stage, None, 0
        resume = cfg.resume_ckpt
        if not resume and cfg.auto_resume and (self.logdir / "ckpt/latest.json").is_file():
            resume = str(self.logdir / "ckpt/latest.json")
        restored = load_checkpoint(self.model, self.optimizer, cfg, resume) if resume else None
        if self.stage == 2 and restored is None:
            raise ValueError("Stage 2 continues a stage-1 checkpoint; set resume_ckpt or use auto_resume")
        if restored:
            self.step, self.stage = restored["step"], max(cfg.h3.stage, restored["stage"])
            self.pending_rf, self.depth = restored["runtime"]["pending_rf"], restored["runtime"]["depth"]
        set_seed(cfg.seed + groups.get_rank() + self.step)
        self.video = VideoEncoder(cfg.h3.vae, self.device, cfg.vae_compile)
        self.text = TextEncoder(cfg, wrap=partial(wrap_text, cfg=cfg), device=self.device)
        self.stream = SourceStream(cfg, self.video, self.text, self.step)
        if restored:
            self.stream.load_state_dict(restored["runtime"]["stream"])
            from utils.checkpoint import restore_rng
            restore_rng(restored["rng"])
        self.stream.set_stage(self.stage)
        self.val_stream = None
        self.seen_shapes = set()
        self.wandb = None
        if groups.get_rank() == 0 and cfg.get("wandb_project"):
            import wandb
            self.wandb = wandb.init(project=cfg.wandb_project,
                                    entity=cfg.wandb_entity,
                                    name=self.logdir.name,
                                    dir=str(self.logdir),
                                    config=OmegaConf.to_container(cfg, resolve=True),
                                    resume="allow",
                                    id=self.logdir.name)

    def save(self):
        runtime = dict(stream=self.stream.state_dict(), pending_rf=self.pending_rf, depth=self.depth)
        path = save_checkpoint(self.model, self.optimizer, self.cfg, self.step, self.stage, runtime,
                               self.logdir / "ckpt")
        if groups.get_rank() == 0:
            import shutil
            complete = sorted(p.parent for p in (self.logdir / "ckpt").glob("step_*/manifest.json"))
            for old in complete[:-int(self.cfg.max_checkpoints)]:
                if old != path:
                    shutil.rmtree(old)
        return path

    def visualize(self, document, index=0):
        from pipeline.ar_inference import generate
        from utils.visualization import write_visualization
        negative = self.text([self.cfg.negative_prompt])[0]
        outputs = generate(self.model,
                           document,
                           negative,
                           self.cfg,
                           self.device,
                           steps=self.cfg.vis_sampling_steps if self.cfg.task == "train" else self.cfg.sampling_steps,
                           use_cache=self.cfg.vis_use_kv_cache if self.cfg.task == "train" else self.cfg.use_kv_cache)
        failed = torch.zeros((), dtype=torch.int32, device=self.device)
        if groups.get_sp_rank() == 0:
            try:
                write_visualization(outputs, document, self.video, self.cfg, self.step, groups.get_rank(), index,
                                    self.stage)
            except Exception as error:
                print(f"Visualization output failed on rank {groups.get_rank()}: {error}", flush=True)
                failed.fill_(1)
        # Output failures happen only on SP leaders. Agree before any rank
        # enters the next text/model collective, including when errors are fatal.
        if dist.is_initialized():
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item() and self.cfg.raise_vis_error:
            raise RuntimeError("Visualization output failed; see the SP leader log")

    def validate(self, count=None):
        from utils.checkpoint import rng_state, restore_rng
        state = rng_state()
        try:
            if self.val_stream is None:
                self.val_stream = SourceStream(self.cfg, self.video, self.text, validation=True)
            count = self.cfg.vis_num_samples if count is None else count
            for index in range(count):
                self.visualize(self.val_stream.next(), index)
        finally:
            restore_rng(state)

    def train(self):
        cfg = self.cfg
        self.model.train()
        if cfg.vis_init:
            self.validate()
        while self.step < cfg.max_iters:
            started = time.monotonic()
            if self.stage == 1 and self.step >= cfg.h3.stage1_steps:
                self.save()
                self.stage = 2
                self.stream.set_stage(2)
            if self.pending_rf is None:
                document, override, self.depth = self.stream.next(), None, 0
            else:
                document, override = self.pending_rf
            shape = tuple(
                (tuple(v["latent"].shape), v["text"].shape[1], v["condition"] is not None) for v in document["views"])
            if shape not in self.seen_shapes:
                torch.cuda.empty_cache()
                self.seen_shapes.add(shape)
            self.optimizer.zero_grad(set_to_none=True)
            loss, log = self.objective(self.model, document, self.device, self.step, override)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {self.step}")
            loss.backward()
            norm = self.model.clip_grad_norm_(cfg.clip_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient at step {self.step}")
            if cfg.warmup_steps:
                for group in self.optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * min(1., (self.step + 1) / cfg.warmup_steps)
            self.optimizer.step()
            rf = bool(log.pop("rf")) and (not cfg.resampling_forcing_max_depth
                                          or self.depth < cfg.resampling_forcing_max_depth)
            eligible = torch.tensor(int(rf), device=self.device)
            if cfg.resampling_forcing_global_sync:
                dist.all_reduce(eligible, op=dist.ReduceOp.MIN)
            x0 = log.pop("x0")
            self.pending_rf = (document, x0) if eligible.item() else None
            self.depth = self.depth + 1 if self.pending_rf else 0
            self.step += 1
            log.update(step=self.step,
                       stage=self.stage,
                       loss=float(loss),
                       grad_norm=float(norm),
                       sf_depth=self.depth,
                       seconds=time.monotonic() - started)
            if groups.get_rank() == 0:
                print(json.dumps(log), flush=True)
                if self.step >= cfg.warn_loss_start_step and float(loss) > cfg.warn_loss:
                    print(f"Loss warning at step {self.step}: {float(loss):.6f} > {cfg.warn_loss}", flush=True)
                with (self.logdir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(log) + "\n")
                if self.wandb:
                    self.wandb.log(log, step=self.step)
            if cfg.save_interval and self.step % cfg.save_interval == 0:
                self.save()
            if (cfg.vis_interval > 0 and self.step % cfg.vis_interval == 0) or self.step in cfg.vis_list:
                self.validate()
            if cfg.save_artifacts_interval and self.step % cfg.save_artifacts_interval == 0:
                save_compile_cache(cfg.h3.compile_cache)
            if cfg.gc_interval and self.step % cfg.gc_interval == 0:
                gc.collect()
            if self.step <= cfg.empty_cache_warmup_steps or (cfg.empty_cache_interval
                                                             and self.step % cfg.empty_cache_interval == 0):
                torch.cuda.empty_cache()
        self.save()
