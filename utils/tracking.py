"""Online experiment metrics, media, provenance and durable WandB run identity."""

import hashlib
import json
import subprocess
import tarfile
from importlib import metadata
from numbers import Real
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from utils import distributed as groups
from utils.config import recipe_digest


class Tracker:

    def __init__(self, cfg, directory, step=0):
        self.run = None
        self.peak_allocated = 0
        self.step_offset = 0
        self.directory = Path(directory)
        error = [None]
        if groups.get_rank() == 0 and cfg.get("wandb_project"):
            try:
                import wandb

                metadata.version("byted-wandb")
                if not getattr(wandb, "_IS_TRACKING", False):
                    raise RuntimeError(
                        "WorldViews requires byted-wandb internal Tracking; check WANDB_OFFICIAL"
                    )
                self.wandb = wandb
                identity = self.directory / "wandb_run.json"
                previous = json.loads(identity.read_text()) if identity.is_file() else None
                run_id = previous["id"] if previous else wandb.util.generate_id()
                self.run = wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    name=self.directory.name,
                    id=run_id,
                    resume="must" if previous else "never",
                    mode="online",
                    dir=str(self.directory),
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
                if previous:
                    self.run.config.update(
                        {
                            "ema_weight": float(cfg.ema_weight),
                            "ema_warmup": bool(cfg.get("ema_warmup", True)),
                        },
                        allow_val_change=True,
                    )
                # Tracking can resume at SDK step 1 after a failed init that
                # never trained. Keep global optimizer steps in the `step`
                # metric while respecting the SDK's monotonic history cursor.
                self.step_offset = max(0, int(self.run.step) - int(step))
                self.run.define_metric("step")
                self.run.define_metric("global_step")
                self.run.define_metric("*", step_metric="global_step")
                self.run.define_metric("eval/mean_loss", summary="min")
                identity.write_text(
                    json.dumps(
                        dict(id=run_id, url=self.run.url, project=cfg.wandb_project, entity=cfg.wandb_entity),
                        indent=2,
                    )
                    + "\n"
                )
                root = Path(__file__).resolve().parents[1]
                files = sorted(
                    p
                    for folder in (
                        "h3",
                        "model",
                        "pipeline",
                        "trainer",
                        "dataset",
                        "utils",
                        "scripts",
                        "configs",
                    )
                    for p in (root / folder).rglob("*")
                    if p.suffix in (".py", ".yaml", ".json")
                )
                files.append(root / "main.py")
                provenance = dict(
                    recipe=recipe_digest(cfg),
                    resume_step=step,
                    git_head=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root, text=True
                    ).strip(),
                    hashes={
                        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
                    },
                )
                (self.directory / "code_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
                artifact = wandb.Artifact(
                    f"{run_id}-provenance", type="provenance", metadata={"recipe": provenance["recipe"]}
                )
                artifact.add_file(str(self.directory / "code_provenance.json"))
                artifact.add_file(str(self.directory / "resolved.yaml"))
                snapshot = self.directory / "source.tar.gz"
                with tarfile.open(snapshot, "w:gz") as archive:
                    for path in files:
                        archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
                artifact.add_file(str(snapshot))
                self.run.log_artifact(artifact)
                self.run.summary.update(
                    {
                        "resume_step": step,
                        "modality": "video-only",
                        "guidance_scale": cfg.guidance_scale,
                        "camera_wrapped": not cfg.model.prope_unwrapped,
                        "ema/enabled": bool(cfg.ema_weight),
                        "ema/decay": float(cfg.ema_weight),
                        "ema/warmup": bool(cfg.get("ema_warmup", True)),
                        "ema/cpu_offload": bool(cfg.get("ema_cpu_offload", True)),
                    }
                )
            except Exception as exc:
                error[0] = f"Online WandB initialization failed: {type(exc).__name__}"
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])

    def log(self, values, step):
        if self.run:
            from wandb.sdk.data_types.base_types.wb_value import WBValue

            history, metadata_values = {}, {}
            for key, value in values.items():
                if isinstance(value, Real):
                    # The internal Scalar schema rejects JSON booleans too.
                    history[key] = float(value)
                elif isinstance(value, WBValue):
                    history[key] = value
                else:
                    metadata_values[key] = value
            # byted-wandb serializes bare strings as Scalar.Value, which the
            # server rejects for the entire metrics batch. Keep paths/text in
            # summary/provenance and only numeric/media values in history.
            if metadata_values:
                self.run.summary.update(metadata_values)
            self.run.log({"step": step, "global_step": step, **history}, step=int(step) + self.step_offset)

    def training(self, row, optimizer, document):
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
            loss = torch.tensor(row["loss"], device=device)
            peak = torch.tensor(torch.cuda.max_memory_allocated(), device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            row["loss"] = float(loss)
            row["vram_max"] = int(peak) // 1024**2
            self.peak_allocated = max(self.peak_allocated, int(peak))
            row["vram"] = torch.cuda.max_memory_allocated() // 1024**2
            row["mem_now"] = torch.cuda.memory_allocated() // 1024**2
        batch_size = len(document["views"]) if document["isolated"] else 1
        row.update(
            gnorm=row["grad_norm"],
            mv=len(document["views"]),
            bs=batch_size,
            source_batch_size=1,
            lat=max(int(v["valid"].sum()) for v in document["views"]),
            iso=int(document["isolated"]),
            cond=sum(
                int(v["condition"]["valid"].sum()) for v in document["views"] if v["condition"] is not None
            ),
            time=row["seconds"],
            t=1 - row["sigma"],
        )
        values = {"train/" + k: v for k, v in row.items() if isinstance(v, (float, int)) and k != "step"}
        values.update(row)
        values.update({f"train/lr_{i}": group["lr"] for i, group in enumerate(optimizer.param_groups)})
        values.update(
            {
                "data/views": len(document["views"]),
                "data/source": str(document["source"]),
                "data/frames": sum(v["source_frames"] for v in document["views"]),
                "data/conditioned_views": sum(v["condition"] is not None for v in document["views"]),
            }
        )
        if torch.cuda.is_available():
            values.update(
                {
                    "system/allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                    "system/reserved_gib": torch.cuda.memory_reserved() / 1024**3,
                    "system/peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                }
            )
        self.log(values, row["step"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def evaluation(self, value):
        metrics = {"eval/mean_loss": value["mean_loss"]}
        for row in value["entries"]:
            metrics[f"eval/sample{row['sample']}/sigma{row['sigma']}"] = row["loss"]
        if "ema_mean_loss" in value:
            metrics["eval/ema_mean_loss"] = value["ema_mean_loss"]
            for row in value["ema_entries"]:
                metrics[f"eval/ema/sample{row['sample']}/sigma{row['sigma']}"] = row["loss"]
        self.log(metrics, value["step"])

    def checkpoint(self, path, step):
        if self.run:
            manifest = json.loads((Path(path) / "manifest.json").read_text())
            self.run.summary.update(
                {
                    "checkpoint/path": str(Path(path).resolve()),
                    "checkpoint/step": step,
                    "checkpoint/ema": bool(manifest.get("ema", False)),
                    "checkpoint/ema_updates": manifest.get("ema_updates", 0),
                }
            )
            artifact = self.wandb.Artifact(f"{self.run.id}-checkpoint-manifest", type="checkpoint-manifest")
            artifact.add_file(str(Path(path) / "manifest.json"))
            self.run.log_artifact(artifact, aliases=["latest", f"step-{step}"])

    def media(self, directory, step, prefix="validation"):
        if self.run:
            directory = Path(directory)
            values = {
                f"{prefix}/video/{p.stem}": self.wandb.Video(str(p), format="mp4")
                for p in sorted(directory.rglob("*.mp4"))
            }
            values.update(
                {
                    f"{prefix}/image/{p.stem}": self.wandb.Image(str(p))
                    for p in sorted(directory.rglob("*"))
                    if p.suffix in (".jpg", ".png")
                }
            )
            self.log(values, step)

    def finish(self, success=True):
        if self.run:
            self.run.finish(exit_code=0 if success else 1)
