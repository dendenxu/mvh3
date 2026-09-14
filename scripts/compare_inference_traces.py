#!/usr/bin/env python3
"""Locate the first packed-input or prediction divergence in cache traces."""

import argparse
import json
from pathlib import Path

import runtime_env
import torch


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [name for key in sorted(left.keys() | right.keys())
                for name in differences(left.get(key), right.get(key), f"{path}.{key}".lstrip("."))]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [name for i, (a, b) in enumerate(zip(left, right)) for name in differences(a, b, f"{path}[{i}]")]
    return [] if left == right else [path]


def compare_call(directory, reference, actual):
    row = dict(chunk=reference["chunk"], cached_call=reference["call"], recomputed_call=actual["call"],
               update=reference["update"], input_differences=differences(reference["inputs"], actual["inputs"]),
               cache_lengths_equal=reference["cache_lengths"] == actual["cache_lengths"],
               predictions_exact=reference["prediction"] == actual["prediction"])
    if not row["predictions_exact"]:
        before = torch.load(directory / "cached" / f"call_{reference['call']:04d}.pt", weights_only=True).float()
        after = torch.load(directory / "recomputed" / f"call_{actual['call']:04d}.pt", weights_only=True).float()
        if before.shape == after.shape:
            error = after - before
            row.update(max_abs_error=float(error.abs().max()),
                       relative_l2=float(error.square().sum().sqrt() / before.square().sum().sqrt().clamp_min(1e-30)))
        else:
            row["shape_difference"] = [list(before.shape), list(after.shape)]
    return row


def compare(directory):
    traces = {name: [json.loads(line) for line in (directory / name / "calls.jsonl").read_text().splitlines()]
              for name in ("cached", "recomputed")}
    denoising = {name: [row for row in values if not row["update"]] for name, values in traces.items()}
    if len(denoising["cached"]) != len(denoising["recomputed"]):
        raise ValueError("Traces must contain the same complete denoising rollout")
    steps = []
    for a, b in zip(denoising["cached"], denoising["recomputed"]):
        if a["chunk"] != b["chunk"]:
            raise ValueError("Denoising chunk order differs")
        steps.append(compare_call(directory, a, b))
    original_writes = {row["chunk"]: row for row in traces["cached"] if row["update"]}
    history = [compare_call(directory, original_writes[row["chunk"]], row)
               for row in traces["recomputed"] if row["update"]]
    first = next((row for row in steps if row["input_differences"] or not row["predictions_exact"]), None)
    return dict(status="equal" if first is None else "different", first_denoising_difference=first,
                denoising=steps, history_writes=history,
                scope="Trace diagnostics; tensor retention or immediate CPU copies can affect runtime scheduling")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing cached/ and recomputed/ traces")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.directory)
    output = args.output or args.directory / "comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "first_difference": report["first_denoising_difference"],
                      "denoising_calls": len(report["denoising"]), "history_writes": len(report["history_writes"])}))


if __name__ == "__main__":
    main()
