#!/usr/bin/env python3
"""Render paired-caption generations with their exact source video windows."""

import argparse
import hashlib
import json
from pathlib import Path

import runtime_env
import cv2
import numpy as np
import torch

from utils.video import write_video


def read_video(path, frames, fps):
    reader = cv2.VideoCapture(str(path))
    actual_fps = reader.get(cv2.CAP_PROP_FPS)
    decoded = []
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        reader.release()
    if len(decoded) != frames or abs(actual_fps - fps) > 1e-6:
        raise ValueError(f"Incomplete video {path}: {len(decoded)} frames at {actual_fps} FPS")
    return np.stack(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=int, action="append", help="Render a completed case while other cases run")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    assert manifest["status"] == "passed"
    reports = {}
    for directory in args.inference:
        for path in directory.rglob("inference.json"):
            report = json.loads(path.read_text())
            request = Path(report["request_path"])
            key = (request.parent.name, request.stem)
            if key in reports:
                raise ValueError(f"Ambiguous generation for {key}")
            assert report["status"] == "complete" and not report["ground_truth_loaded"]
            reports[key] = (path, report)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for case in manifest["cases"]:
        if args.row and case["row"] not in args.row:
            continue
        name = f"row{case['row']:06d}"
        paths, pair = zip(*(reports[(name, policy)] for policy in ("source", "scene_motion")))
        for key in ("seed", "recipe", "protocol", "camera", "sigma_points", "guidance_scale", "checkpoint_step"):
            assert pair[0][key] == pair[1][key], (name, key)
        assert pair[0]["protocol"] == "joint" and pair[0]["checkpoint_step"] == 0
        requests = [entry["request"] for entry in pair]
        for key in ("views", "frames", "fps", "height", "width"):
            assert requests[0].get(key) == requests[1].get(key), (name, key)
        request = requests[0]
        raw = torch.load(args.inputs / name / "raw.pt", map_location="cpu", weights_only=True)
        reference = raw["views"][0]["pixels"].permute(0, 2, 3, 1).numpy()
        camera = np.load(args.inputs / name / request["views"][0]["camera"], allow_pickle=False)
        fps, frames = request["fps"], len(camera)
        indices = np.rint(np.linspace(0, len(reference) - 1, frames)).astype(int)
        reference = reference[indices]
        panels = [reference] + [read_video(path.parent / "view000.mp4", frames, fps) for path in paths]
        assert all(value.shape == reference.shape for value in panels)
        assert np.array_equal(camera, np.broadcast_to(camera[:1], camera.shape))
        image = args.inputs / name / request["views"][0]["image"]
        width = reference.shape[2]
        body = np.concatenate(panels, axis=2)
        header = np.full((frames, 40, body.shape[2], 3), 20, dtype=np.uint8)
        comparison = np.concatenate((header, body), axis=1)
        labels = ("Source (recorded camera)", "Old source caption", "New scene + motion caption")
        for index, frame in enumerate(comparison):
            for column, label in enumerate(labels):
                cv2.putText(frame, label, (column * width + 12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{index / fps:.2f}s", (width - 85, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (245, 245, 245), 1, cv2.LINE_AA)
        video_path = args.output / f"{name}_comparison.mp4"
        write_video(str(video_path), comparison, fps=fps, crf=18, preset="faster")
        selected = np.linspace(0, frames - 1, 4, dtype=int)
        tile_width = 1494
        tile_height = round(comparison.shape[1] * tile_width / comparison.shape[2])
        sheet = np.concatenate([cv2.resize(comparison[i], (tile_width, tile_height)) for i in selected])
        sheet_path = args.output / f"{name}_comparison.jpg"
        if not cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Cannot write {sheet_path}")
        read_video(video_path, frames, fps)
        results.append(dict(row=case["row"], frames=frames, fps=fps, seed=pair[0]["seed"],
                            source=case, image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                            static_camera=True, old_caption=requests[0]["prompt"],
                            new_caption=requests[1]["prompt"], inference_reports=[str(p.resolve()) for p in paths],
                            video=str(video_path.resolve()), contact_sheet=str(sheet_path.resolve())))
    assert results and (not args.row or {case["row"] for case in results} == set(args.row))
    report = dict(status="complete", cases=results,
                  scope="Base H3 global-caption A/B. Same image, static camera, seed and sampler. "
                        "Source reference retains recorded camera motion. No chunk-caption or trained-control conclusion.")
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="complete", cases=len(results), output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
