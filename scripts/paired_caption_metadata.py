"""Attach exact paired captions to previously decoded acceptance inputs."""

import json

import pyarrow.parquet as pq


def attach_captions(document, case, directory):
    source = pq.ParquetFile(directory / f"stage{case['stage']}.parquet")
    ranges = json.loads(source.schema_arrow.metadata[b"presampled_shape_ranges"])
    index = ranges[case["original_bucket"]][0]
    columns = ["row", "gen", "scene", "chunks", "view_isolated"]
    offset = 0
    for batch in source.iter_batches(batch_size=1, columns=columns):
        if offset == index:
            row = batch.to_pylist()[0]
            break
        offset += 1
    else:
        raise IndexError(index)
    if row["row"] != case["source_rows"]:
        raise ValueError("Caption row differs from the decoded acceptance input")
    views = []
    for i, view in enumerate(document["views"]):
        text_view = i if row["view_isolated"] else 0
        scene = str(row["scene"][text_view] or "")
        views.append({**view, "caption_scene": scene,
                      "caption_motions": list(row["chunks"][text_view] or []),
                      "caption_source_frames": row["gen"] * 4 - 3, "prompt": scene})
    return {**document, "views": views}
