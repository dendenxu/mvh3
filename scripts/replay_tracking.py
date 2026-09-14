#!/usr/bin/env python3
"""Recover numeric training history from durable local metrics and verify it remotely."""

import argparse
import json
from pathlib import Path
import time

import runtime_env
import wandb

from utils.config import load_config
from utils.tracking import Tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.directory / "resolved.yaml")
    identity = json.loads((args.directory / "wandb_run.json").read_text())
    rows = [json.loads(line) for line in (args.directory / "metrics.jsonl").read_text().splitlines()]
    expected = {int(row["step"]): row for row in rows}
    assert expected and getattr(wandb, "_IS_TRACKING", False)
    remote = wandb.TrackingPublicApi().run(project=cfg.wandb_project, run_id=identity["id"])
    if not args.verify_only:
        if remote.status == "running":
            raise RuntimeError("Finish the active writer before replaying its history")
        # Replaying logs must not replace the training run's source snapshot.
        tracker = Tracker.__new__(Tracker)
        tracker.run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                                 id=identity["id"], resume="must", mode="online", dir=str(args.directory))
        tracker.step_offset = max(0, int(tracker.run.step) - min(expected))
        tracker.run.define_metric("global_step")
        tracker.run.define_metric("*", step_metric="global_step")
        success = False
        try:
            tracker.run.summary.update({"history_repair": "Replayed durable metrics with numeric scalar types and explicit global_step",
                                        "history_repair_source": str(args.directory / "metrics.jsonl")})
            for step, row in sorted(expected.items()):
                values = {"train/" + key: value for key, value in row.items()
                          if isinstance(value, (int, float)) and key != "step"}
                values.update(row)
                tracker.log(values, step)
            success = True
        finally:
            tracker.finish(success=success)
    deadline = time.monotonic() + 120
    actual = {}
    while True:
        history = remote.history(name=["global_step", "train/loss"])
        actual = {int(step): loss for step, loss in zip(history["global_step"], history["train/loss"])
                  if step is not None and loss is not None}
        if expected.keys() <= actual.keys() or time.monotonic() >= deadline:
            break
        time.sleep(2)
    assert expected.keys() <= actual.keys(), sorted(expected.keys() - actual.keys())
    assert all(abs(actual[step] - row["loss"]) <= 1e-8 for step, row in expected.items())
    report = dict(status="passed", run_id=identity["id"], verified_steps=sorted(expected),
                  scope="remote global_step and loss match the durable training records")
    (args.directory / "tracking_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="passed", run_id=identity["id"], verified_count=len(expected))), flush=True)


if __name__ == "__main__":
    main()
