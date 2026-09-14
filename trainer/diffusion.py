"""Full-source two-stage WorldViews training on the existing H3 layers."""

from contextlib import nullcontext
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
from h3.distributed.fsdp import (
    compile_blocks,
    configure_model,
    load_compile_cache,
    parameter_groups,
    save_compile_cache,
    wrap_model,
    wrap_text,
)
from utils.h3_wrapper import VideoEncoder, TextEncoder
from model.diffusion import WorldViewsObjective
from utils.checkpoint import load_checkpoint, save_checkpoint
from dataset.stream import SourceStream
from utils import distributed as groups
from utils.misc import set_seed
from utils.tracking import Tracker
from utils.control import Requests
from utils.ema import ShardedEMA, inference_weight_kind


class Trainer:
    """Own distributed training, checkpoint state, validation and run tracking."""

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

        # FSDP determines parameter views and shard storage before AdamW is built.
        load_compile_cache(cfg.h3.compile_cache)
        model = load_original_transformer(cfg.h3.checkpoint, progress=print if groups.get_rank() == 0 else None)
        self.signature = configure_model(model, cfg)
        self.model = wrap_model(model, cfg)
        compile_blocks(self.model.module, cfg)
        self.ema = ShardedEMA.from_config(self.model, cfg)
        self.optimizer = torch.optim.AdamW(parameter_groups(self.model, cfg), betas=(cfg.beta1, cfg.beta2),
                                           weight_decay=cfg.weight_decay, fused=True)
        if cfg.optim_compile:
            self.optimizer.step = torch.compile(self.optimizer.step)
        self.objective = WorldViewsObjective(cfg)
        self.step = 0
        self.stage = cfg.h3.stage
        self.pending_rf = None
        self.depth = 0
        resume = cfg.resume_ckpt
        if not resume and cfg.auto_resume and (self.logdir / "ckpt/latest.json").is_file():
            resume = str(self.logdir / "ckpt/latest.json")
        restored = load_checkpoint(self.model, self.optimizer, cfg, resume, ema=self.ema) if resume else None
        if self.stage == 2 and restored is None:
            raise ValueError("Stage 2 continues a stage-1 checkpoint; set resume_ckpt or use auto_resume")
        if restored:
            self.step, self.stage = restored["step"], max(cfg.h3.stage, restored["stage"])
            self.pending_rf, self.depth = restored["runtime"]["pending_rf"], restored["runtime"]["depth"]

        # Encoder/loader setup can consume randomness. Restore the saved RNG
        # after construction so the next training step starts at its saved state.
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
        self.tracker = Tracker(cfg, self.logdir, self.step)
        self.requests = Requests()

    def save(self):
        runtime = dict(stream=self.stream.state_dict(), pending_rf=self.pending_rf, depth=self.depth)
        path = save_checkpoint(self.model, self.optimizer, self.cfg, self.step, self.stage, runtime,
                               self.logdir / "ckpt", ema=self.ema)
        self.tracker.checkpoint(path, self.step)
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
        negative = self.text([self.cfg.negative_prompt])[0] if self.cfg.guidance_scale != 1 else None
        outputs = generate(self.model, document, negative, self.cfg, self.device,
                           steps=self.cfg.vis_sampling_steps if self.cfg.task == "train" else self.cfg.sampling_steps,
                           use_cache=self.cfg.vis_use_kv_cache if self.cfg.task == "train" else self.cfg.use_kv_cache)
        failed = torch.zeros((), dtype=torch.int32, device=self.device)
        if groups.get_sp_rank() == 0:
            try:
                write_visualization(outputs, document, self.video, self.cfg, self.step, groups.get_rank(), index,
                                    self.stage)
                if getattr(self, "tracker", None):
                    root = Path(self.cfg.vis_dir or (self.logdir / "vis"))
                    directory = root / f"step_{self.step:09d}" / f"rank{groups.get_rank()}" / f"sample{index:03d}"
                    self.tracker.media(directory, self.step, f"validation/sample{index}")
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
            self.val_stream.set_stage(self.stage)
            count = self.cfg.vis_num_samples if count is None else count
            use_ema = inference_weight_kind(self.cfg, validation=True) == "ema"
            context = self.ema.average_parameters(self.model) if use_ema else nullcontext()
            with context:
                for index in range(count):
                    self.visualize(self.val_stream.next(), index)
        finally:
            restore_rng(state)

    def train(self):
        try:
            self.train_loop()
        except BaseException:
            self.tracker.finish(success=False)
            raise
        else:
            self.tracker.finish()

    def train_step(self, document, override=None):
        """Update raw weights, then EMA, then decide whether to reuse the sample."""
        cfg = self.cfg
        timings = {}
        self.optimizer.zero_grad(set_to_none=True)
        phase_started = time.monotonic()
        loss, log = self.objective(self.model, document, self.device, self.step, override)
        log["fwd_mem"] = torch.cuda.memory_allocated() // 1024**2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {self.step}")
        timings["forward_seconds"] = time.monotonic() - phase_started
        phase_started = time.monotonic()
        loss.backward()
        norm = self.model.clip_grad_norm_(cfg.clip_grad_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at step {self.step}")
        timings["backward_seconds"] = time.monotonic() - phase_started
        phase_started = time.monotonic()
        if cfg.warmup_steps:
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * min(1., (self.step + 1) / cfg.warmup_steps)
        self.optimizer.step()
        ema_started = time.monotonic()
        if self.ema is not None:
            self.ema.update(self.model)
            log.update(ema_updates=self.ema.num_updates, ema_decay=self.ema.current_decay)
        timings["ema_seconds"] = time.monotonic() - ema_started
        rf = bool(log.pop("rf")) and (not cfg.resampling_forcing_max_depth
                                      or self.depth < cfg.resampling_forcing_max_depth)
        # FSDP replicas must agree on reusing the sample before the next forward.
        eligible = torch.tensor(int(rf), device=self.device)
        if cfg.resampling_forcing_global_sync:
            dist.all_reduce(eligible, op=dist.ReduceOp.MIN)
        x0 = log.pop("x0")
        self.pending_rf = (self.objective.resample_document(document), x0) if eligible.item() else None
        self.depth = self.depth + 1 if self.pending_rf else 0
        timings["optimizer_and_rf_seconds"] = time.monotonic() - phase_started
        log.update(timings)
        log.update(loss=float(loss), grad_norm=float(norm))
        return log

    def train_loop(self):
        cfg = self.cfg
        self.model.train()
        if cfg.vis_init:
            self.validate()
        while self.step < cfg.max_iters:
            started = time.monotonic()
            previous_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            if self.stage == 1 and self.step >= cfg.h3.stage1_steps:
                self.save()
                self.stage = 2
                self.stream.set_stage(2)
            if self.pending_rf is None:
                document = self.stream.next()
                override = None
                self.depth = 0
                timings = dict(self.stream.last_timings)
                document = self.objective.prepare_document(document, self.device)
            else:
                document, override = self.pending_rf
                timings = dict(decode_seconds=0., vae_seconds=0., text_seconds=0., sp_gather_seconds=0.)
            timings["data_seconds"] = time.monotonic() - started
            shape = tuple(
                (tuple(v["latent"].shape), v["text"].shape[1], v["condition"] is not None) for v in document["views"])
            if shape not in self.seen_shapes:
                torch.cuda.empty_cache()
                self.seen_shapes.add(shape)
            log = self.train_step(document, override)
            # A peer can keep compiling while rank zero reuses a warm graph.
            total_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            compile_counts = torch.tensor([total_graphs - previous_graphs, total_graphs], device=self.device)
            dist.all_reduce(compile_counts, op=dist.ReduceOp.MAX)
            self.step += 1
            log.update(timings)
            log.update(step=self.step, stage=self.stage, sf_depth=self.depth,
                       new_compile_graphs=int(compile_counts[0]), compile_graphs_total=int(compile_counts[1]),
                       seconds=time.monotonic() - started, self=int(override is not None))
            self.tracker.training(log, self.optimizer, document)
            if groups.get_rank() == 0:
                print(json.dumps(log), flush=True)
                if self.step >= cfg.warn_loss_start_step and log["loss"] > cfg.warn_loss:
                    print(f"Loss warning at step {self.step}: {log['loss']:.6f} > {cfg.warn_loss}", flush=True)
                with (self.logdir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(log) + "\n")
            if cfg.save_interval and self.step % cfg.save_interval == 0 and self.step < cfg.max_iters:
                self.save()
            if (cfg.vis_interval > 0 and self.step % cfg.vis_interval == 0) or self.step in cfg.vis_list:
                self.validate()
            for kind, token in self.requests.poll():
                self.requests.acknowledge(kind, token, self.step, "started")
                if kind == "save":
                    path = self.save()
                else:
                    self.validate()
                    path = ""
                self.requests.acknowledge(kind, token, self.step, "ok", path)
            if cfg.save_artifacts_interval and self.step % cfg.save_artifacts_interval == 0:
                save_compile_cache(cfg.h3.compile_cache)
            if cfg.gc_interval and self.step % cfg.gc_interval == 0:
                gc.collect()
            if self.step <= cfg.empty_cache_warmup_steps or (cfg.empty_cache_interval
                                                             and self.step % cfg.empty_cache_interval == 0):
                torch.cuda.empty_cache()
        self.save()
