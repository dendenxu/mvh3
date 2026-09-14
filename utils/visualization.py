"""Validation video, caption/pose metadata, and asynchronous HDFS mirroring."""

import hashlib
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from h3.encoders import VideoEncoder
from utils.video import write_video

mirror_executor = None


def write_visualization(outputs, document, video, cfg, step, rank, index=0, stage=None):
    global mirror_executor
    directory = (
        Path(cfg.vis_dir or (Path(cfg.h3.logdir) / "vis"))
        / f"step_{step:09d}"
        / f"rank{rank}"
        / f"sample{index:03d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    for view_index, (latent, view) in enumerate(zip(outputs, document["views"])):
        generated = video.decode(latent, view["height"], view["width"], view["source_frames"])
        panels = [generated]
        if cfg.vis_gt:
            panels.append(video.decode(view["latent"], view["height"], view["width"], view["source_frames"]))
        if cfg.vis_pose:
            h, w = generated.shape[-2:]
            panel = np.zeros((h, w, 3), dtype=np.uint8)
            points = view["pose"][:, 7:10].numpy()[:, [0, 2]]
            points = (points - points.min(0)) / (np.ptp(points, axis=0).max() + 1e-6)
            points = (points * np.array([w * 0.8, h * 0.8]) + np.array([w * 0.1, h * 0.1])).astype(np.int32)
            cv2.polylines(panel, [points], False, (70, 210, 210), 2)
            panels.append(
                torch.from_numpy(panel)
                .permute(2, 0, 1)
                .float()
                .div(255)[None]
                .expand(len(generated), -1, -1, -1)
            )
        frames = torch.cat(panels, -1).permute(0, 2, 3, 1).mul(255).byte().numpy()
        write_video(str(directory / f"view{view_index:03d}.mp4"), frames, fps=view["fps"])
        meta = {key: value for key, value in view.items() if isinstance(value, (str, float, int))}
        if "generation_chunks" in view:
            meta["generation_chunks"] = view["generation_chunks"].tolist()
        if "caption_specs" in view:
            meta["caption_specs"] = view["caption_specs"]
        meta.update(step=step, source=document["source"], stage=cfg.h3.stage if stage is None else stage)
        (directory / f"view{view_index:03d}.json").write_text(json.dumps(meta, indent=2) + "\n")
        if cfg.decode_offload:
            torch.cuda.empty_cache()
    if cfg.vis_copy_to_hdfs:

        def mirror():
            destination = f"{cfg.vis_hdfs_dir.rstrip('/')}/{Path(cfg.h3.logdir).name}/vis/step_{step:09d}/rank{rank}/sample{index:03d}"
            for attempt in range(int(cfg.vis_hdfs_copy_retries) + 1):
                try:
                    subprocess.run(
                        [cfg.vis_hdfs_cli, "dfs", "-mkdir", "-p", destination],
                        timeout=cfg.vis_hdfs_copy_timeout,
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        [cfg.vis_hdfs_cli, "dfs", "-put", "-f", *map(str, directory.iterdir()), destination],
                        timeout=cfg.vis_hdfs_copy_timeout,
                        check=True,
                        capture_output=True,
                    )
                    break
                except (OSError, subprocess.SubprocessError) as error:
                    print(
                        f"Visualization mirror attempt {attempt+1} failed: {type(error).__name__}", flush=True
                    )
                    if attempt < int(cfg.vis_hdfs_copy_retries):
                        time.sleep(min(2**attempt, 30))

        if mirror_executor is None:
            mirror_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-vis-copy")
        mirror_executor.submit(mirror)


def pixel_metrics(prediction, reference):
    mse = (prediction - reference).square().mean().item()
    return dict(
        mse=mse, psnr_db=-10 * math.log10(max(mse, 1e-12)), mae=(prediction - reference).abs().mean().item()
    )


def uint8_frames(pixels):
    # OpenCV's in-place annotations require interleaved, contiguous RGB pixels.
    return pixels.permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).byte().contiguous().numpy()


def write_overfit_generation(cfg, documents, run, label, device="cuda:0"):
    """Publish each scheduled generation while the training run is still active."""
    run = Path(run)
    directory = run / "generations" / label
    directory.mkdir(parents=True, exist_ok=True)
    decoder = VideoEncoder(cfg.h3.vae, device)
    for index, document in enumerate(documents):
        view = document["views"][0]
        latents = torch.load(run / f"{label}_{index}.pt", map_location="cpu", weights_only=True)
        pixels = decoder.decode(latents[0], view["height"], view["width"], view["source_frames"])
        if not torch.isfinite(pixels).all():
            raise FloatingPointError("Nonfinite pixels in a scheduled generation")
        frames = uint8_frames(pixels)
        write_video(str(directory / f"sample{index}.mp4"), frames, fps=view["fps"], crf=18, preset="faster")
        selected = np.linspace(0, len(frames) - 1, 4, dtype=int)
        width = 448
        height = round(width * view["height"] / view["width"])
        sheet = np.concatenate([cv2.resize(frames[i], (width, height)) for i in selected], axis=1)
        if not cv2.imwrite(str(directory / f"sample{index}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
            raise OSError("Could not write a scheduled contact sheet")
    return directory


def write_overfit_comparison(cfg, features, run, device="cuda:0"):
    """Save source/raw/EMA comparisons and the overfit loss curves."""
    features = Path(features)
    run = Path(run)

    # Step 1: Verify that generated latents still refer to the fixed training clip.
    report = json.loads((run / "training_report.json").read_text())
    if report["status"] != "training_and_generation_complete":
        raise ValueError("Training and both fixed-seed generations must finish before review")
    if hashlib.sha256((features / "documents.pt").read_bytes()).hexdigest() != report["feature_sha256"]:
        raise ValueError("Source features changed after training")
    documents = torch.load(features / "documents.pt", map_location="cpu", weights_only=True)
    decoder = VideoEncoder(cfg.h3.vae, device)
    directory = run / "review"
    directory.mkdir(parents=True, exist_ok=True)
    # Step 2: Decode matching frames and label the side-by-side videos.
    rows = []
    for index, document in enumerate(documents):
        view = document["views"][0]
        reference = (
            torch.load(features / f"pixels_{index}.pt", map_location="cpu", weights_only=True).float() / 255
        )
        panels = [uint8_frames(reference)]
        row = dict(sample=index, source_view=report["source"]["views"][index])
        labels = ["before", "after"] + (["after_ema"] if report.get("ema_enabled") else [])
        for label in labels:
            latents = torch.load(run / f"{label}_{index}.pt", map_location="cpu", weights_only=True)
            if len(latents) != 1 or not torch.isfinite(latents[0]).all():
                raise ValueError("Expected one finite monocular generation per source document")
            pixels = decoder.decode(latents[0], view["height"], view["width"], view["source_frames"])
            if pixels.shape != reference.shape or not torch.isfinite(pixels).all():
                raise ValueError("Decoded generation differs from the source geometry")
            row[label] = pixel_metrics(pixels, reference)
            frames = uint8_frames(pixels)
            panels.append(frames)
            write_video(
                str(directory / f"{label}_{index}.mp4"), frames, fps=view["fps"], crf=18, preset="faster"
            )
            del pixels
        write_video(
            str(directory / f"reference_{index}.mp4"), panels[0], fps=view["fps"], crf=18, preset="faster"
        )
        comparison = np.concatenate(panels, axis=2)
        width = view["width"]
        titles = ["Source", "Before training", f"Raw step {report['training'][-1]['step']}"]
        if report.get("ema_enabled"):
            titles.append(f"EMA step {report['training'][-1]['step']}")
        for frame in comparison:
            for column, label in enumerate(titles):
                x = column * width
                cv2.rectangle(frame, (x, 0), (x + width, 48), (16, 16, 16), -1)
                cv2.putText(
                    frame, label, (x + 18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA
                )
        write_video(
            str(directory / f"comparison_{index}.mp4"), comparison, fps=view["fps"], crf=18, preset="faster"
        )
        chosen = np.linspace(0, len(comparison) - 1, 4, dtype=int)
        sheet = np.concatenate([cv2.resize(comparison[i], (1344, 256)) for i in chosen], axis=0)
        if not cv2.imwrite(
            str(directory / f"comparison_{index}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
        ):
            raise OSError("Could not write the comparison contact sheet")
        rows.append(row)
        print(json.dumps(row), flush=True)
        del panels, comparison, reference
    # Step 3: Record reconstruction metrics and fixed-noise training curves.
    result = dict(
        status="complete",
        feature_sha256=report["feature_sha256"],
        fixed_noise_relative_improvement=report["relative_improvement"],
        loss_criterion_passed=report["loss_criterion_passed"],
        samples=rows,
        metric_scope=f"Pixel similarity on {len(documents)} training clips; not held-out scene quality",
    )
    temporary = directory / "review_report.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(directory / "review_report.json")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    training = report["training"]
    axes[0].plot([r["step"] for r in training], [r["loss"] for r in training], alpha=0.2, color="#426778")
    window = min(64, len(training))
    smooth = np.convolve([r["loss"] for r in training], np.ones(window) / window, mode="valid")
    axes[0].plot(
        [r["step"] for r in training][window - 1 :], smooth, color="#155968", label=f"{window}-step mean"
    )
    evaluations = report["evaluations"]
    axes[1].plot(
        [r["step"] for r in evaluations],
        [r["mean_loss"] for r in evaluations],
        "o-",
        color="#b05728",
        label="Fixed evaluation noise",
    )
    if report.get("ema_enabled"):
        axes[1].plot(
            [r["step"] for r in evaluations],
            [r["ema_mean_loss"] for r in evaluations],
            "o-",
            color="#155968",
            label="EMA, same fixed noise",
        )
    for axis, title in zip(axes, ("Training MSE", "Fixed-noise MSE")):
        axis.set(xlabel="Optimizer step", ylabel="MSE", title=title)
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(directory / "loss_curve.png", dpi=160)
    figure.savefig(directory / "loss_curve.svg")
    plt.close(figure)
    return result
