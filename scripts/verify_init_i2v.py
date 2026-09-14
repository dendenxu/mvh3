#!/usr/bin/env python3
"""Full-weight initialized H3 parity at every denoising step of a static-camera i2v request."""

import argparse
import copy
from functools import partial
import json
from pathlib import Path
import time

import runtime_env
import torch

from h3.checkpoint import load_original_transformer
from h3.distributed.fsdp import configure_model, wrap_model, wrap_text, compile_blocks
from pipeline.joint_inference import generate
from utils import distributed as groups
from utils.config import load_config, validate_config


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", type=Path)
    source.add_argument("--request", type=Path, help="Raw first image and static camera request")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/init_wrapped.yaml")
    parser.add_argument("--diagnose", action="store_true", help="Record the first differing block and its input")
    parser.add_argument("--native-repeat", action="store_true", help="Control: compare two identical no-camera forwards")
    parser.add_argument("--eager-blocks", action="store_true", help="Diagnose block fusion; sparse attention stays compiled")
    parser.add_argument("--no-fusion", action="store_true", help="Keep compiled blocks but isolate pointwise operations")
    parser.add_argument("--no-patterns", action="store_true", help="Isolate compiler pattern rewriting while keeping fusion")
    parser.add_argument("--no-epilogue", action="store_true", help="Isolate compiler epilogue fusion")
    args = parser.parse_args()
    cfg = validate_config(load_config(args.config, ["generator_cpu_offload=false"]))
    if args.eager_blocks:
        cfg.attn_block_compile = False
    if args.no_fusion:
        torch._inductor.config.max_fusion_size = 1
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.pattern_matcher = False
    if args.no_patterns:
        torch._inductor.config.pattern_matcher = False
    if args.no_epilogue:
        torch._inductor.config.epilogue_fusion = False
    torch.set_num_threads(1)
    groups.launch_distributed_job(sp_size_arg=cfg.sp_size, fs_size_arg=cfg.fs_size)
    device = torch.device("cuda", torch.cuda.current_device())
    video = None
    if args.request:
        from pipeline.i2v_input import prepare_request
        from utils.h3_wrapper import VideoEncoder, TextEncoder
        video = VideoEncoder(cfg.h3.vae, device, cfg.vae_compile)
        text = TextEncoder(cfg, wrap=partial(wrap_text, cfg=cfg), device=device)
        doc = prepare_request(json.loads(args.request.read_text()), args.request.parent, video, text, cfg)
        del text
        video.model.to("cpu")
        torch.cuda.empty_cache()
        for view in doc["views"]:
            if not torch.equal(view["pose"], view["pose"][:1].expand_as(view["pose"])):
                raise ValueError("Initialization parity needs an unchanged camera sequence")
    else:
        doc = copy.deepcopy(torch.load(args.features / "documents.pt", map_location="cpu", weights_only=True)[0])
        for view in doc["views"]:
            for part in (view, view["condition"]):
                for key in ("pose", "projection", "inverse"):
                    part[key] = view[key][:1].expand_as(part[key]).clone()
            view["latent"].zero_()
    model = load_original_transformer(cfg.h3.checkpoint, progress=print if groups.get_rank() == 0 else None)
    signature = configure_model(model, cfg)
    model = wrap_model(model, cfg)
    compile_blocks(model.module, cfg)
    checks = []
    comparing, activations, layer_checks = False, {}, []
    if args.diagnose:
        def make_hook(index, before=False):
            def hook(module, arguments, output=None):
                value = arguments[0] if before else output
                key = (index, before)
                if not comparing:
                    activations[key] = value.detach().cpu()
                    if before and groups.get_rank() == 0:
                        camera = arguments[5]
                        identity = torch.eye(4, device=device)
                        print(json.dumps(dict(camera_wrapped=not cfg.model.prope_unwrapped,
                                              neutral_camera_bypass=camera is None,
                                              projection_identity_error=0. if camera is None else float((camera.matrix.projection - identity).abs().max()),
                                              inverse_identity_error=0. if camera is None else float((camera.matrix.inverse - identity).abs().max()),
                                              hidden_absmax=float(value.abs().max()))), flush=True)
                        args.output.mkdir(parents=True, exist_ok=True)
                        torch.save(value.detach().cpu(), args.output / "first_block_input_rank0.pt")
                else:
                    error = (value.detach().float().cpu() - activations.pop(key).float()).abs().max().item()
                    record = dict(block=index, input=before, max_abs_error=error)
                    layer_checks.append(record)
                    if groups.get_rank() == 0:
                        print(json.dumps(record), flush=True)
            return hook
        for index, block in enumerate(model.module.transformer_blocks):
            if index == 0:
                block.register_forward_pre_hook(make_hook(index, before=True))
            block.register_forward_hook(make_hook(index))

    def compare(index, inputs, actual):
        nonlocal comparing
        base = {k: v for k, v in inputs.items() if not k.startswith("camera_")}
        base["scale_log"] = None
        comparing = True
        expected = model(**base).sample
        comparing = False
        error = float((actual.float() - expected.float()).abs().max())
        if not torch.isfinite(actual).all() or error != 0:
            if args.diagnose and groups.get_rank() == 0:
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / "diagnosis.json").write_text(json.dumps(dict(max_abs_error=error, layers=layer_checks), indent=2) + "\n")
            raise AssertionError(f"Static-camera init parity failed at evaluation {index}: max error {error}")
        checks.append(dict(evaluation=index, max_abs_error=error))
        if groups.get_rank() == 0:
            print(json.dumps(checks[-1]), flush=True)

    torch.manual_seed(81000)
    started = time.monotonic()
    outputs = generate(model, doc, None, cfg, device, camera=not args.native_repeat, observer=compare)
    torch.cuda.synchronize()
    denoise_seconds = time.monotonic() - started
    if groups.get_rank() == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, args.output / "static_latents.pt")
        torch.save(doc, args.output / "request_features.pt")
        if video is not None:
            from utils.video import write_video
            video.model.to(device)
            for index, (output, view) in enumerate(zip(outputs, doc["views"])):
                pixels = video.decode(output, view["height"], view["width"], view["source_frames"])
                write_video(str(args.output / f"static{index}.mp4"), pixels.permute(0, 2, 3, 1).mul(255).byte().numpy(), fps=view["fps"])
    groups.shutdown_distributed()
    if groups.get_rank() == 0:
        report = dict(status="passed", scope="initialized native joint video-only i2v; same inputs and weights",
                      input_source="raw_image_camera" if args.request else "cached_features",
                      ground_truth_loaded=not bool(args.request), ground_truth_used=False,
                      camera_wrapped=not cfg.model.prope_unwrapped,
                      camera_modes=[cfg.model.prope_mode, cfg.model.mv_prope_mode],
                      packing_version=cfg.h3.packing_version,
                      native_repeat_control=args.native_repeat,
                      block_compile=cfg.attn_block_compile, no_fusion=args.no_fusion,
                      no_patterns=args.no_patterns, no_epilogue=args.no_epilogue,
                      denoise_and_comparison_seconds=denoise_seconds,
                      total_parameters=sum(torch.tensor(s).prod().item() for s in signature.values()),
                      added_parameters=0, guidance_scale=1, sigma_points=50, checks=checks,
                      all_solver_predictions_exact=True, ground_truth_loaded_for_shape_only=not bool(args.request))
        (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
