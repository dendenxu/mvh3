#!/usr/bin/env python3
"""Read back complete optimizer-step history and final media from byted-wandb."""

import argparse
import json
import math
from numbers import Real
from pathlib import Path
import time

import runtime_env


def check_history(history, rows, fields):
    actual = {}
    for index, step in enumerate(history.get("global_step", [])):
        if step is None:
            continue
        values = actual.setdefault(int(step), {})
        for field in fields:
            value = history.get(field, [None] * len(history["global_step"]))[index]
            if value is not None:
                values[field] = value
    errors = []
    for row in rows:
        for field in fields:
            if not isinstance(row.get(field), Real):
                continue
            value = actual.get(row["step"], {}).get(field)
            if value is None or not math.isclose(value, float(row[field]), rel_tol=1e-6, abs_tol=1e-8):
                errors.append(dict(step=row["step"], field=field, local=row[field], remote=value))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--through-step", type=int,
                        help="Verify a fixed completed prefix while training continues")
    parser.add_argument("--evaluations", type=Path,
                        help="Overfit progress/training report containing raw and EMA evaluations")
    parser.add_argument("--media", type=Path)
    parser.add_argument("--media-prefix", default="overfit")
    parser.add_argument("--settle-seconds", type=float, default=300,
                        help="Allow pending same-step history and remote ingestion to settle")
    parser.add_argument("--history-batch-size", type=int, default=8,
                        help="Numeric fields per history query; reduce for server rate limits")
    parser.add_argument("--history-query-interval", type=float, default=0,
                        help="Seconds between history queries when the server throttles reads")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.history_batch_size < 1 or args.history_query_interval < 0:
        parser.error("History batch size must be positive and query interval nonnegative")
    identity = json.loads((args.run_dir / "wandb_run.json").read_text())
    rows = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines()]
    if args.through_step is not None:
        rows = [row for row in rows if row["step"] <= args.through_step]
        assert rows and rows[-1]["step"] == args.through_step, "Requested step is not locally complete"
    assert rows and len({row["step"] for row in rows}) == len(rows), "Missing or repeated local optimizer steps"
    if args.expected_steps is not None:
        assert len(rows) == args.expected_steps and rows[-1]["step"] == args.expected_steps
    fields = sorted({key for row in rows for key, value in row.items()
                     if key != "step" and isinstance(value, Real)})
    assert {"loss", "grad_norm", "seconds"} <= set(fields)
    evaluations = []
    if args.evaluations:
        for value in json.loads(args.evaluations.read_text())["evaluations"]:
            if value["step"] > rows[-1]["step"]:
                continue
            row = {"step": value["step"], "eval/mean_loss": value["mean_loss"]}
            for entry in value["entries"]:
                row[f"eval/sample{entry['sample']}/sigma{entry['sigma']}"] = entry["loss"]
            if "ema_mean_loss" in value:
                row["eval/ema_mean_loss"] = value["ema_mean_loss"]
                for entry in value["ema_entries"]:
                    row[f"eval/ema/sample{entry['sample']}/sigma{entry['sigma']}"] = entry["loss"]
            evaluations.append(row)
        assert evaluations, "No completed local evaluations"
    evaluation_fields = sorted({key for row in evaluations for key in row if key != "step"})
    expected_media = set()
    if args.media:
        expected_media = {f"{args.media_prefix}/{'video' if path.suffix == '.mp4' else 'image'}/{path.stem}"
                          for path in args.media.rglob("*") if path.suffix in (".mp4", ".png", ".jpg")}
        assert expected_media, "No local review media"
    import wandb
    from wandb.sdk.internal.tracking_cloud import cloud
    assert getattr(wandb, "_IS_TRACKING", False), "Use the repository's internal byted-wandb SDK"
    remote = wandb.TrackingPublicApi().run(project=identity["project"], run_id=identity["id"])
    deadline = time.monotonic() + args.settle_seconds
    while True:
        errors = []
        global_history = remote.scan_history(name=["global_step"])
        global_steps = dict(zip(global_history.get("step", []), global_history.get("global_step", [])))
        if args.history_query_interval:
            time.sleep(args.history_query_interval)

        def history_for(subset):
            # Join by the SDK history index. Resumes may offset that index
            # from optimizer steps; read its mapping once per verification pass.
            history = remote.scan_history(name=subset)
            history["global_step"] = [global_steps.get(step) for step in history.get("step", [])]
            if args.history_query_interval:
                time.sleep(args.history_query_interval)
            return history

        for offset in range(0, len(fields), args.history_batch_size):
            subset = fields[offset:offset + args.history_batch_size]
            errors.extend(check_history(history_for(subset), rows, subset))
        for offset in range(0, len(evaluation_fields), args.history_batch_size):
            subset = evaluation_fields[offset:offset + args.history_batch_size]
            errors.extend(check_history(history_for(subset), evaluations, subset))
        actual_media = set()
        if expected_media:
            response = cloud.request("/inner/ListTrackingRunEntities", json={
                "RunIds": [identity["id"]], "ProjectId": remote.project_id,
                "Types": ["image-file", "video-file"]})
            actual_media = {entity["Name"] for entity in response.json()["Result"]}
        missing_media = sorted(expected_media - actual_media)
        if not errors and not missing_media or time.monotonic() >= deadline:
            break
        time.sleep(3)
    report = dict(status="failed" if errors or missing_media else "passed", run=identity,
                  optimizer_steps=len(rows), first_step=rows[0]["step"], last_step=rows[-1]["step"],
                  through_step=args.through_step,
                  evaluations=len(evaluations), evaluation_fields=evaluation_fields,
                  checked_fields=fields, scalar_errors=errors, expected_media=sorted(expected_media),
                  missing_media=missing_media, remote_status=remote.status,
                  scope="numeric optimizer/evaluation fields in unaggregated history; media entity presence")
    output = args.output or args.run_dir / "tracking_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
