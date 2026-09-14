#!/usr/bin/env python3
"""Verify each scheduled generation against its remotely stored Tracking bytes."""

import argparse
import hashlib
import json
from pathlib import Path

import runtime_env
import requests
import wandb
from wandb.sdk.internal.tracking_cloud import cloud


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--through-step", type=int, required=True)
    parser.add_argument("--include-final-review", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert getattr(wandb, "_IS_TRACKING", False), "Use the existing byted-wandb runtime"
    identity = json.loads((args.run_dir / "wandb_run.json").read_text())
    remote = wandb.TrackingPublicApi().run(project=identity["project"], run_id=identity["id"])
    media = []
    for directory in sorted((args.run_dir / "generations").iterdir()):
        label = directory.name
        if label == "before":
            step = 0
        elif label.startswith("step") and label[4:].isdigit():
            step = int(label[4:])
        else:
            # Final raw/EMA have distinct names in the complete review directory.
            continue
        if step <= args.through_step:
            media.extend((path, step, "overfit/generation") for path in directory.iterdir()
                         if path.suffix in (".mp4", ".png", ".jpg"))
    if args.include_final_review:
        completion = json.loads((args.run_dir / "completion.json").read_text())
        assert completion["status"] == "complete"
        media.extend((path, args.through_step, "overfit") for path in (args.run_dir / "review").iterdir()
                     if path.suffix in (".mp4", ".png", ".jpg"))
    assert media, "No local generated media"
    histories, rows = {}, []
    for path, step, prefix in media:
        kind = "video" if path.suffix == ".mp4" else "image"
        name = f"{prefix}/{kind}/{path.stem}"
        if name not in histories:
            histories[name] = remote.scan_history(name=["global_step", name])
        history = histories[name]
        candidates = [history["step"][index] for index, global_step in enumerate(history["global_step"])
                      if global_step == step and history[name][index] is not None]
        expected_hash, expected_size = file_hash(path), path.stat().st_size
        matched = None
        for sdk_step in candidates:
            result = cloud.request("/inner/ListTrackingRunsEntityMediaSteps", json={
                "RunIds": [identity["id"]], "ProjectId": remote.project_id, "Name": name,
                "Step": sdk_step}).json()["Result"]
            for entry in result:
                item = entry["Item"]
                if (entry["RunId"] != identity["id"] or entry["Step"] != sdk_step
                        or item.get("sha256") != expected_hash or item.get("size") != expected_size):
                    continue
                digest, size = hashlib.sha256(), 0
                with requests.get(item["link"], stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                assert digest.hexdigest() == expected_hash and size == expected_size, path
                matched = dict(local_path=str(path), optimizer_step=step, sdk_step=sdk_step,
                               entity=name, sha256=expected_hash, bytes=size, downloaded_bytes_exact=True)
                break
            if matched:
                break
        assert matched is not None, f"No matching remote payload for {name} at optimizer step {step}"
        rows.append(matched)
        print(json.dumps(matched), flush=True)
    report = dict(status="passed", run=identity, through_step=args.through_step,
                  include_final_review=args.include_final_review, media=rows,
                  scope="per-generation optimizer/SDK step mapping, stored metadata and downloaded byte hashes")
    output = args.output or args.run_dir / "tracking_media_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
