#!/usr/bin/env python3
"""Exercise the repository's byted-wandb run resume and media publication."""

import argparse
import json
from pathlib import Path
import time

import runtime_env
from omegaconf import OmegaConf

from utils.config import load_config
from utils.tracking import Tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overfit_i2v.yaml")
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Read back an already finished integration run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg.h3.logdir = str(args.output)
    if not args.verify_only:
        OmegaConf.save(cfg, args.output / "resolved.yaml")
        first = Tracker(cfg, args.output)
        identity = first.run.id
        first.log({"checks/resume_index": 1}, 0)
        first.run.summary["scope"] = "byted-wandb integration; attached media are historical probe artifacts"
        first.finish()
        second = Tracker(cfg, args.output, 1)
        assert second.run.id == identity
        second.log({"checks/resume_index": 2}, 1)
        second.media(args.media, 1, "historical_probe")
        second.finish()
    else:
        identity = json.loads((args.output / "wandb_run.json").read_text())["id"]
    import wandb
    from wandb.sdk.internal.tracking_cloud import cloud
    remote = wandb.TrackingPublicApi().run(project=cfg.wandb_project, run_id=identity)
    deadline = time.monotonic() + 60
    history = remote.history(name=["checks/resume_index"])
    while set(history["checks/resume_index"]) != {1, 2} and time.monotonic() < deadline:
        time.sleep(2)
        history = remote.history(name=["checks/resume_index"])
    assert set(history["checks/resume_index"]) == {1, 2}
    expected = {f"historical_probe/{'video' if p.suffix == '.mp4' else 'image'}/{p.stem}"
                for p in args.media.rglob("*") if p.suffix in (".jpg", ".png", ".mp4")}
    actual, deadline = set(), time.monotonic() + 60
    while actual != expected and time.monotonic() < deadline:
        response = cloud.request("/inner/ListTrackingRunEntities", json={"RunIds": [identity], "ProjectId": remote.project_id,
                                 "Types": ["image-file", "video-file"]})
        actual = {row["Name"] for row in response.json()["Result"]}
        if actual != expected:
            time.sleep(2)
    assert actual == expected, (actual, expected)
    report = dict(status="passed", run=json.loads((args.output / "wandb_run.json").read_text()),
                  resume_history=history, media_entities=sorted(actual), remote_status=remote.status,
                  scope="remote scalar history, identical run ID across resume, and image/video entity readback")
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
