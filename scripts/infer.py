#!/usr/bin/env python3
"""Standalone i2v: torchrun scripts/infer.py --request request.json --config training.yaml."""

import argparse
from functools import partial
import json
from pathlib import Path
import time

import runtime_env
import torch
from omegaconf import OmegaConf

from h3.checkpoint import load_original_transformer
from h3.distributed.fsdp import configure_model, wrap_model, wrap_text, compile_blocks
from pipeline.i2v_input import prepare_request
from utils import distributed as groups
from utils.checkpoint import load_checkpoint
from utils.config import load_config, validate_config, recipe_digest
from utils.h3_wrapper import VideoEncoder, TextEncoder
from utils.video import write_video
from utils.tracking import Tracker
from utils.ema import inference_weight_kind


def run(cfg, request_path, output, checkpoint=None, protocol="auto", seed=81000, steps=None, camera=True,
        verify_init=False):
    validate_config(cfg)
    if protocol == "auto":
        protocol = "ar" if checkpoint else "joint"
    if verify_init and (checkpoint or protocol != "joint" or not camera):
        raise ValueError("Initialization parity requires joint sampling, cameras and base weights")
    paths = [Path(path) for path in request_path] if isinstance(request_path, (list, tuple)) else [Path(request_path)]
    output = Path(output)
    groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    if groups.get_rank() == 0:
        OmegaConf.save(cfg, output / "resolved.yaml")
    tracker = Tracker(cfg, output)
    if tracker.run:
        tracker.run.summary.update({"operation": "inference", "camera_enabled": camera, "protocol": protocol})
    try:
        text = TextEncoder(cfg, wrap=partial(wrap_text, cfg=cfg), device=device)
        video = VideoEncoder(cfg.h3.vae, device, cfg.vae_compile)
        prepared = []
        for request_path in paths:
            started = time.monotonic()
            request = json.loads(request_path.read_text())
            document = prepare_request(request, request_path.parent, video, text, cfg, chunked=protocol == "ar")
            if verify_init:
                for view in document["views"]:
                    if not torch.equal(view["pose"], view["pose"][:1].expand_as(view["pose"])):
                        raise ValueError("Initialization parity requires an unchanged camera sequence")
            prepared.append((request_path, request, document, time.monotonic() - started))
        del text
        video.model.to("cpu")
        torch.cuda.empty_cache()
        model = load_original_transformer(cfg.h3.checkpoint, progress=print if groups.get_rank() == 0 else None)
        configure_model(model, cfg)
        model = wrap_model(model, cfg)
        if checkpoint:
            state = load_checkpoint(model, None, cfg, checkpoint, restore_random=False,
                                    weights=inference_weight_kind(cfg))
            checkpoint_step = state["step"]
            if tracker.run:
                tracker.run.summary["inference/weights"] = inference_weight_kind(cfg)
            del state
        else:
            checkpoint_step = 0
        compile_blocks(model.module, cfg)
        for case, (request_path, request, document, encode_seconds) in enumerate(prepared):
            torch.manual_seed(seed)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            parity = []

            def compare_init(index, inputs, prediction):
                base = {key: value for key, value in inputs.items() if not key.startswith("camera_")}
                base["scale_log"] = None
                with torch.no_grad():
                    expected = model(**base).sample
                error = float((prediction.float() - expected.float()).abs().max())
                parity.append(dict(evaluation=index, max_abs_error=error))
                if error != 0:
                    raise AssertionError(f"Static-camera initialization differs from base at evaluation {index}: {error}")

            if protocol == "joint":
                from pipeline.joint_inference import generate
                outputs = generate(model, document, None, cfg, device, steps=steps, camera=camera,
                                   observer=compare_init if verify_init else None)
            else:
                if not camera:
                    raise ValueError("--no-camera is a native joint parity control")
                from pipeline.ar_inference import generate
                outputs = generate(model, document, None, cfg, device, steps=steps)
            torch.cuda.synchronize()
            denoise_seconds = time.monotonic() - started
            if groups.get_rank() == 0:
                directory = output if len(prepared) == 1 else output / f"{case:02d}_{request_path.parent.name}_{request_path.stem}"
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(outputs, directory / "latents.pt")
                torch.save(document, directory / "request_features.pt")
                video.model.to(device)
                for index, (latent, view) in enumerate(zip(outputs, document["views"])):
                    pixels = video.decode(latent, view["height"], view["width"], view["source_frames"])
                    write_video(str(directory / f"view{index:03d}.mp4"), pixels.permute(0, 2, 3, 1).mul(255).byte().numpy(), fps=view["fps"])
                video.model.to("cpu")
                report = dict(status="complete", request=request, request_path=str(request_path.resolve()),
                              checkpoint=str(checkpoint), checkpoint_step=checkpoint_step,
                              seed=seed, recipe=recipe_digest(cfg), protocol=protocol, camera=camera,
                              sigma_points=steps or cfg.sampling_steps, guidance_scale=cfg.guidance_scale,
                              audio=False, ground_truth_loaded=False, encoding_seconds=encode_seconds,
                              denoise_seconds=denoise_seconds, total_seconds=time.monotonic() - started,
                              peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
                if verify_init:
                    report["initialization_parity"] = dict(status="passed", checks=parity,
                                                          all_solver_predictions_exact=True)
                (directory / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
                tracker.log({"inference/denoise_seconds": denoise_seconds, "inference/encoding_seconds": encode_seconds,
                             "inference/request": str(request_path), "inference/checkpoint_step": checkpoint_step}, case)
                tracker.media(directory, case, f"inference/case{case}")
                print(json.dumps(dict(case=case, output=str(directory), denoise_seconds=denoise_seconds)), flush=True)
            groups.barrier()
    except BaseException:
        tracker.finish(success=False)
        raise
    else:
        tracker.finish()
    groups.shutdown_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Exact training resolved.yaml when loading a checkpoint")
    parser.add_argument("--request", required=True, action="append", help="Repeat to reuse the encoders and backbone across cases")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--weights", choices=["auto", "raw", "ema"], default="auto")
    parser.add_argument("--protocol", choices=["auto", "ar", "joint"], default="auto",
                        help="Native joint for base weights; AR after loading a training checkpoint")
    parser.add_argument("--seed", type=int, default=81000)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--verify-init", action="store_true", help="Compare every static-camera prediction with base weights")
    args = parser.parse_args()
    if args.weights == "ema" and not args.checkpoint:
        parser.error("--weights ema requires a training checkpoint containing EMA")
    cfg = load_config(args.config)
    cfg.inference_weights = args.weights
    run(cfg, args.request, args.output, args.checkpoint, args.protocol, args.seed, args.steps,
        not args.no_camera, args.verify_init)


if __name__ == "__main__":
    main()
