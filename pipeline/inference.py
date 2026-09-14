"""Run image/camera requests with the native or chunked H3 sampler."""

import json
import time
from pathlib import Path
from functools import partial

import torch
from omegaconf import OmegaConf

from utils.random import set_seed
from utils.tracking import Tracker
from utils.video import write_video
from utils import distributed as groups
from utils.ema import inference_weight_kind
from utils.checkpoint import load_checkpoint
from pipeline.i2v_input import prepare_request
from h3.encoders import TextEncoder, VideoEncoder
from utils.config import recipe_digest, validate_config
from h3.modules.model import MiniMaxH3Transformer3DModel
from pipeline.chunked_inference import generate as generate_chunks
from h3.distributed.fsdp import wrap_text, wrap_model, compile_blocks
from pipeline.full_sequence_inference import generate as generate_joint


def run_inference(
    cfg,
    request_path,
    output,
    checkpoint=None,
    protocol="auto",
    seed=81000,
    steps=None,
    camera=True,
    verify_init=False,
):
    """Encode requests, load raw/EMA weights, sample videos and save tracked outputs.

    Both main.py and the standalone CLI call this function. The joint protocol
    preserves native initialization; the ar protocol rolls out the trained chunks.
    """
    validate_config(cfg)
    if protocol == "auto":
        protocol = "ar" if checkpoint else "joint"
    if verify_init and (checkpoint or protocol != "joint" or not camera):
        raise ValueError("Initialization parity requires joint sampling, cameras and base weights")
    if isinstance(request_path, (list, tuple)):
        paths = [Path(path) for path in request_path]
    else:
        paths = [Path(request_path)]
    output = Path(output)
    groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.set_num_threads(1)
    set_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    if groups.get_rank() == 0:
        OmegaConf.save(cfg, output / "resolved.yaml")
    tracker = Tracker(cfg, output)
    if tracker.run:
        tracker.run.summary.update({"operation": "inference", "camera_enabled": camera, "protocol": protocol})
    try:
        # Step 1: Encode image/caption inputs and build the requested camera path.
        text_encoder = TextEncoder(cfg, wrap=partial(wrap_text, cfg=cfg), device=device)
        video_encoder = VideoEncoder(cfg.h3.vae, device, cfg.vae_compile)
        prepared = []
        for request_path in paths:
            started = time.monotonic()
            request = json.loads(request_path.read_text())
            document = prepare_request(
                request, request_path.parent, video_encoder, text_encoder, cfg, chunked=protocol == "ar"
            )
            if verify_init:
                for view in document["views"]:
                    if not torch.equal(view["pose"], view["pose"][:1].expand_as(view["pose"])):
                        raise ValueError("Initialization parity requires an unchanged camera sequence")
            prepared.append((request_path, request, document, time.monotonic() - started))
        del text_encoder

        # Release encoder memory before loading the full denoiser. Reuse the
        # prepared requests across cases without constructing a training dataset.
        video_encoder.model.to("cpu")
        torch.cuda.empty_cache()

        # Step 2: Load base weights, then the requested raw/EMA training weights.
        model = MiniMaxH3Transformer3DModel.from_pretrained(
            cfg.h3.checkpoint,
            progress=print if groups.get_rank() == 0 else None,
        )
        model.configure_attention(cfg)
        model = wrap_model(model, cfg)
        if checkpoint:
            state = load_checkpoint(
                model, None, cfg, checkpoint, restore_random=False, weights=inference_weight_kind(cfg)
            )
            checkpoint_step = state["step"]
            if tracker.run:
                tracker.run.summary["inference/weights"] = inference_weight_kind(cfg)
            del state
        else:
            checkpoint_step = 0
        compile_blocks(model.module, cfg)
        for case, (request_path, request, document, encode_seconds) in enumerate(prepared):
            set_seed(seed)
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
                    raise AssertionError(
                        f"Static-camera initialization differs from base at evaluation {index}: {error}"
                    )

            if protocol == "joint":
                outputs = generate_joint(
                    model,
                    document,
                    None,
                    cfg,
                    device,
                    steps=steps,
                    camera=camera,
                    observer=compare_init if verify_init else None,
                )
            else:
                if not camera:
                    raise ValueError("--no-camera is a native joint parity control")
                outputs = generate_chunks(model, document, None, cfg, device, steps=steps)
            torch.cuda.synchronize()
            denoise_seconds = time.monotonic() - started
            if groups.get_rank() == 0:
                directory = output
                if len(prepared) > 1:
                    directory = output / f"{case:02d}_{request_path.parent.name}_{request_path.stem}"
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(outputs, directory / "latents.pt")
                torch.save(document, directory / "request_features.pt")
                video_encoder.model.to(device)
                for index, (latent, view) in enumerate(zip(outputs, document["views"])):
                    pixels = video_encoder.decode(
                        latent, view["height"], view["width"], view["source_frames"]
                    )
                    write_video(
                        str(directory / f"view{index:03d}.mp4"),
                        pixels.permute(0, 2, 3, 1).mul(255).byte().numpy(),
                        fps=view["fps"],
                    )
                video_encoder.model.to("cpu")
                report = dict(
                    status="complete",
                    request=request,
                    request_path=str(request_path.resolve()),
                    checkpoint=str(checkpoint),
                    checkpoint_step=checkpoint_step,
                    seed=seed,
                    recipe=recipe_digest(cfg),
                    protocol=protocol,
                    camera=camera,
                    sigma_points=steps or cfg.sampling_steps,
                    guidance_scale=cfg.guidance_scale,
                    audio=False,
                    ground_truth_loaded=False,
                    encoding_seconds=encode_seconds,
                    denoise_seconds=denoise_seconds,
                    total_seconds=time.monotonic() - started,
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                )
                if verify_init:
                    report["initialization_parity"] = dict(
                        status="passed", checks=parity, all_solver_predictions_exact=True
                    )
                (directory / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
                tracker.log(
                    {
                        "inference/denoise_seconds": denoise_seconds,
                        "inference/encoding_seconds": encode_seconds,
                        "inference/request": str(request_path),
                        "inference/checkpoint_step": checkpoint_step,
                    },
                    case,
                )
                tracker.media(directory, case, f"inference/case{case}")
                print(
                    json.dumps(dict(case=case, output=str(directory), denoise_seconds=denoise_seconds)),
                    flush=True,
                )
            groups.barrier()
    except BaseException:
        tracker.finish(success=False)
        raise
    else:
        tracker.finish()
    groups.shutdown_distributed()
