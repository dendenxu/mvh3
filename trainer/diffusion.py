"""Full-source two-stage WorldViews training on the existing H3 layers."""

import gc
import json
import time
from pathlib import Path
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from utils.random import set_seed
from utils.control import Requests
from utils.tracking import Tracker
from dataset.loader import BatchLoader
from utils import distributed as groups
from utils.config import validate_config
from utils.distributed import canonical_name
from model.diffusion import DiffusionObjective
from h3.encoders import TextEncoder, VideoEncoder
from utils.ema import ShardedEMA, inference_weight_kind
from h3.modules.model import MiniMaxH3Transformer3DModel
from utils.checkpoint import rng_state, restore_rng, load_checkpoint, save_checkpoint
from h3.distributed.fsdp import wrap_text, wrap_model, compile_blocks, load_compile_cache, save_compile_cache


class DiffusionTrainer:
    """Own distributed training, checkpoint state, validation and run tracking."""

    def __init__(self, cfg):
        # Step 1: Establish the distributed groups and this run's configuration.
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

        # Step 2: Wrap the model before constructing AdamW and EMA on its shards.
        load_compile_cache(cfg.h3.compile_cache)
        model = MiniMaxH3Transformer3DModel.from_pretrained(
            cfg.h3.checkpoint,
            progress=print if groups.get_rank() == 0 else None,
        )
        model.configure_attention(cfg)
        self.model = wrap_model(model, cfg)
        compile_blocks(self.model.module, cfg)
        self.ema = ShardedEMA.from_config(self.model, cfg)
        self.optimizer = torch.optim.AdamW(
            parameter_groups(self.model, cfg),
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
            fused=True,
        )
        if cfg.optim_compile:
            self.optimizer.step = torch.compile(self.optimizer.step)
        self.objective = DiffusionObjective(cfg)
        self.step = 0
        self.stage = cfg.h3.stage
        self.pending_resampling_forcing = None
        self.resampling_forcing_depth = 0

        # Step 3: Restore weights and optimizer state before constructing encoders.
        resume = cfg.resume_ckpt
        if not resume and cfg.auto_resume and (self.logdir / "ckpt/latest.json").is_file():
            resume = str(self.logdir / "ckpt/latest.json")
        restored = load_checkpoint(self.model, self.optimizer, cfg, resume, ema=self.ema) if resume else None
        if self.stage == 2 and restored is None:
            raise ValueError("Stage 2 continues a stage-1 checkpoint; set resume_ckpt or use auto_resume")
        if restored:
            self.step, self.stage = restored["step"], max(cfg.h3.stage, restored["stage"])
            self.pending_resampling_forcing, self.resampling_forcing_depth = (
                restored["runtime"]["pending_resampling_forcing"],
                restored["runtime"]["resampling_forcing_depth"],
            )

        # Step 4: Construct encoders and queues, then restore RNG consumed by setup.
        set_seed(cfg.seed + groups.get_rank() + self.step)
        self.video = VideoEncoder(cfg.h3.vae, self.device, cfg.vae_compile)
        self.text = TextEncoder(cfg, wrap=partial(wrap_text, cfg=cfg), device=self.device)
        self.data_loader = BatchLoader(cfg, self.video, self.text)
        if restored:
            self.data_loader.load_state_dict(restored["runtime"]["stream"])
            restore_rng(restored["rng"])
        self.data_loader.set_stage(self.stage)
        self.validation_loader = None
        self.seen_shapes = set()
        self.tracker = Tracker(cfg, self.logdir, self.step)
        self.requests = Requests()

    def train(self):
        try:
            self.train_loop()
        except BaseException:
            self.tracker.finish(success=False)
            raise
        else:
            self.tracker.finish()

    def train_loop(self):
        """Fetch or reuse one document, update weights, then handle run outputs."""
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
                self.data_loader.set_stage(2)
            if self.pending_resampling_forcing is None:
                # Resampling forcing can reuse a planned sequence. Fresh
                # samples already carry their chunk captions from the loader.
                document = self.data_loader.next()
                override = None
                self.resampling_forcing_depth = 0
                timings = dict(self.data_loader.last_timings)
                document = self.objective.prepare_document(document, self.device)
            else:
                document, override = self.pending_resampling_forcing
                timings = dict(decode_seconds=0.0, vae_seconds=0.0, text_seconds=0.0, sp_gather_seconds=0.0)
            timings["data_seconds"] = time.monotonic() - started
            shape = tuple(
                (tuple(v["latent"].shape), v["text"].shape[1], v["condition"] is not None)
                for v in document["views"]
            )
            if shape not in self.seen_shapes:
                torch.cuda.empty_cache()
                self.seen_shapes.add(shape)
            log = self.train_step(document, override)

            # A peer may compile, decode or process a larger sample while rank
            # zero is already waiting. Record replica bounds in the same reduce.
            total_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            replica_metrics = dict(
                new_compile_graphs=total_graphs - previous_graphs,
                compile_graphs_total=total_graphs,
                max_tokens=log["tokens"],
                negative_min_tokens=-log["tokens"],
                max_data_seconds=timings["data_seconds"],
                max_forward_seconds=log["forward_seconds"],
                max_backward_compute_seconds=log["backward_compute_seconds"],
                max_gradient_clip_seconds=log["gradient_clip_seconds"],
            )
            values = torch.tensor(list(replica_metrics.values()), device=self.device, dtype=torch.float64)
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            replica_metrics = dict(zip(replica_metrics, values.tolist()))
            replica_metrics["min_tokens"] = -replica_metrics.pop("negative_min_tokens")
            for name in ("new_compile_graphs", "compile_graphs_total", "min_tokens", "max_tokens"):
                replica_metrics[name] = int(replica_metrics[name])
            self.step += 1
            log.update(timings)
            log.update(replica_metrics)
            log.update(
                step=self.step,
                stage=self.stage,
                resampling_forcing_depth=self.resampling_forcing_depth,
                seconds=time.monotonic() - started,
                self=int(override is not None),
            )
            self.tracker.training(log, self.optimizer, document)
            if groups.get_rank() == 0:
                print(json.dumps(log), flush=True)
                if self.step >= cfg.warn_loss_start_step and log["loss"] > cfg.warn_loss:
                    print(
                        f"Loss warning at step {self.step}: {log['loss']:.6f} > {cfg.warn_loss}", flush=True
                    )
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
            if self.step <= cfg.empty_cache_warmup_steps or (
                cfg.empty_cache_interval and self.step % cfg.empty_cache_interval == 0
            ):
                torch.cuda.empty_cache()
        self.save()

    def train_step(self, document, override=None):
        """Update raw weights, then EMA, then decide whether to reuse the sample."""
        cfg = self.cfg
        timings = {}

        # Step 1: Pack the planned sequence and predict its flow target.
        self.optimizer.zero_grad(set_to_none=True)
        phase_started = time.monotonic()
        loss, log = self.objective.compute_loss(self.model, document, self.device, self.step, override)
        log["fwd_mem"] = torch.cuda.memory_allocated() // 1024**2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {self.step}")
        timings["forward_seconds"] = time.monotonic() - phase_started

        # Step 2: Backpropagate and clip the global sharded gradient.
        phase_started = time.monotonic()
        loss.backward()
        clipping_started = time.monotonic()
        timings["backward_compute_seconds"] = clipping_started - phase_started

        # FSDP computes the norm across parameter shards. A local torch norm
        # would clip each shard differently and change the global update.
        norm = self.model.clip_grad_norm_(cfg.clip_grad_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at step {self.step}")
        clipping_finished = time.monotonic()

        # Retain the combined interval for older runs. These wall times include
        # existing waits; measuring them adds no CUDA synchronization.
        timings["gradient_clip_seconds"] = clipping_finished - clipping_started
        timings["backward_seconds"] = clipping_finished - phase_started

        # Step 3: Update the FP32 raw weights, then their EMA once.
        phase_started = time.monotonic()
        if cfg.warmup_steps:
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] * min(1.0, (self.step + 1) / cfg.warmup_steps)
        self.optimizer.step()
        ema_started = time.monotonic()
        if self.ema is not None:
            self.ema.update(self.model)
            log.update(ema_updates=self.ema.num_updates, ema_decay=self.ema.current_decay)
        timings["ema_seconds"] = time.monotonic() - ema_started

        # Step 4: Agree whether to reuse this sequence for resampling forcing.
        resampling_forcing = bool(log.pop("resampling_forcing")) and (
            not cfg.resampling_forcing_max_depth
            or self.resampling_forcing_depth < cfg.resampling_forcing_max_depth
        )

        # FSDP replicas must agree on reusing the sample before the next forward.
        eligible = torch.tensor(int(resampling_forcing), device=self.device)
        if cfg.resampling_forcing_global_sync:
            dist.all_reduce(eligible, op=dist.ReduceOp.MIN)
        x0 = log.pop("x0")
        self.pending_resampling_forcing = (
            (self.objective.resample_document(document), x0) if eligible.item() else None
        )
        self.resampling_forcing_depth = (
            self.resampling_forcing_depth + 1 if self.pending_resampling_forcing else 0
        )
        timings["optimizer_and_resampling_forcing_seconds"] = time.monotonic() - phase_started
        log.update(timings)
        log.update(loss=float(loss), grad_norm=float(norm))
        return log

    def validate(self, count=None):
        """Render with selected raw/EMA weights without advancing training RNG."""
        state = rng_state()
        try:
            if self.validation_loader is None:
                self.validation_loader = BatchLoader(self.cfg, self.video, self.text, validation=True)
            self.validation_loader.set_stage(self.stage)
            count = self.cfg.vis_num_samples if count is None else count
            use_ema = inference_weight_kind(self.cfg, validation=True) == "ema"
            context = self.ema.average_parameters(self.model) if use_ema else nullcontext()
            with context:
                for index in range(count):
                    self.visualize(self.validation_loader.next(), index)
        finally:
            restore_rng(state)

    def visualize(self, document, index=0):
        from pipeline.chunked_inference import generate
        from utils.visualization import write_visualization

        negative = self.text([self.cfg.negative_prompt])[0] if self.cfg.guidance_scale != 1 else None
        outputs = generate(
            self.model,
            document,
            negative,
            self.cfg,
            self.device,
            steps=self.cfg.vis_sampling_steps if self.cfg.task == "train" else self.cfg.sampling_steps,
            use_cache=self.cfg.vis_use_kv_cache if self.cfg.task == "train" else self.cfg.use_kv_cache,
        )
        failed = torch.zeros((), dtype=torch.int32, device=self.device)
        if groups.get_sp_rank() == 0:
            try:
                write_visualization(
                    outputs, document, self.video, self.cfg, self.step, groups.get_rank(), index, self.stage
                )
                if getattr(self, "tracker", None):
                    root = Path(self.cfg.vis_dir or (self.logdir / "vis"))
                    directory = (
                        root / f"step_{self.step:09d}" / f"rank{groups.get_rank()}" / f"sample{index:03d}"
                    )
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

    def save(self):
        # Keep the serialized 'stream' key compatible with existing checkpoints.
        # Runtime queues carry the exact caption plan and pending resampling forcing clean cut.
        runtime = dict(
            stream=self.data_loader.state_dict(),
            pending_resampling_forcing=self.pending_resampling_forcing,
            resampling_forcing_depth=self.resampling_forcing_depth,
        )
        path = save_checkpoint(
            self.model,
            self.optimizer,
            self.cfg,
            self.step,
            self.stage,
            runtime,
            self.logdir / "ckpt",
            ema=self.ema,
        )
        self.tracker.checkpoint(path, self.step)
        if groups.get_rank() == 0:
            import shutil

            complete = sorted(p.parent for p in (self.logdir / "ckpt").glob("step_*/manifest.json"))
            for old in complete[: -int(self.cfg.max_checkpoints)]:
                if old != path:
                    shutil.rmtree(old)
        return path


def parameter_groups(model, cfg):
    """Group the selected original attention weights by their configured learning rate."""
    buckets = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        name = canonical_name(name)
        index = int(name.split("transformer_blocks.")[1].split(".")[0])
        lr = float(cfg.ar_lr if index % cfg.model.ar_interval == 0 else cfg.sa_lr)
        buckets.setdefault(lr, []).append(parameter)
    return [dict(params=params, lr=lr, initial_lr=lr) for lr, params in buckets.items()]
