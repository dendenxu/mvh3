#!/usr/bin/env python3
"""Audit both stage recipes against the complete source mixture and live Parquets."""

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=Path(__file__).resolve().parents[1] / "configs")
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--reference-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = yaml.safe_load((args.configs / "stage1_short_mono.yaml").read_text())
    second = yaml.safe_load((args.configs / "stage2_long_multiview.yaml").read_text())
    references = [(args.configs / stage["data"]["reference"]).resolve() for stage in (first, second)]
    if references[0] != references[1]:
        raise ValueError("The two stages must use the identical source mixture")
    expanded = os.path.expandvars(references[0].read_text())
    if "${" in expanded:
        raise ValueError("Set MVH3_DATA_ROOT and MVH3_DATA_ROOT3 before auditing the data")
    recipe = json.loads(expanded)
    original = yaml.safe_load(args.reference_config.read_text())["dataset"]
    if recipe != original:
        raise ValueError("The dataset recipe differs from the frozen WorldGen reference")
    inventory = json.loads(args.reference_inventory.read_text())
    if len(recipe["datasets"]) != inventory["source_count"]:
        raise ValueError("Source count differs from the full inventory")
    sources = []
    for source, expected in zip(recipe["datasets"], inventory["sources"], strict=True):
        path = Path(source["data_path"])
        if str(path) != expected["path"] or source["type"] != expected["type"]:
            raise ValueError(f"Source identity changed: {path}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected["rows"]:
            raise ValueError(f"Source row count changed: {path}")
        # Augmented game rows keep geometry in video_meta, not a top-level pose column.
        geometry_column = "video_meta" if source["type"] == "mvgame" else "pose"
        required = {"video_path", "caption", geometry_column}
        if not required.issubset(parquet.schema_arrow.names):
            raise ValueError(f"Missing source-family columns in {path}: {required - set(parquet.schema_arrow.names)}")
        row = next(parquet.iter_batches(batch_size=1, columns=sorted(required))).to_pylist()[0]
        if any(row[column] is None for column in required):
            raise ValueError(f"Missing first-row video/text/geometry in {path}")
        result = {"source": path.name, "type": source["type"], "rows": parquet.metadata.num_rows,
                  "geometry_column": geometry_column, "columns": parquet.schema_arrow.names}
        sources.append(result)
        print(f"PASS {source['type']}: {parquet.metadata.num_rows:,} rows; {geometry_column}; {path.name}", flush=True)
    total = sum(source["rows"] for source in sources)
    if total != inventory["source_rows_sum"]:
        raise ValueError("Summed source rows differ from the frozen inventory")
    report = {"status": "passed", "identical_stage_sources": True, "exact_reference_dataset_settings": True,
              "source_count": len(sources), "source_rows_sum": total, "sources": sources,
              "scope": "Parquet metadata and first-row schema; does not validate full-family sampling or every video"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"All {len(sources)} source entries / {total:,} rows match both stages and the full reference.")


if __name__ == "__main__":
    main()
