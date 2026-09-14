"""Select matched source captions without inserting numbering or time labels."""

from model.chunks import source_chunk_ids, chunk_intervals


def merge_chunk_captions(captions):
    """Keep one shared scene and the selected actions in their original order."""
    if not captions:
        raise ValueError("Cannot build a clip caption from an empty selection")
    parts = [caption.partition("\n[CHUNK]") for caption in captions]
    if all(separator and scene == parts[0][0] for scene, separator, _ in parts):
        return "\n".join(value for value in [parts[0][0].strip(), *(part[2].strip() for part in parts)] if value)
    return "\n".join(caption.strip() for caption in captions if caption.strip())


def bind_caption_features(document, cfg):
    """Select pre-encoded strings for repeated fixed-video convergence probes."""
    if not cfg.h3.get("single_sequence", False) or not any("caption_feature_bank" in v for v in document["views"]):
        return document
    views = []
    for view in document["views"]:
        if "caption_feature_bank" not in view:
            views.append(view)
            continue
        specs = caption_specs(view, cfg)
        bank = view["caption_feature_bank"]
        values = [(chunk, bank[caption]) for chunk, caption in specs]
        views.append({**view, "texts": [(chunk, value["features"]) for chunk, value in values],
                      "text": values[0][1]["features"], "texts_by_bd": True, "caption_specs": specs,
                      "text_tag_specs": {chunk: value["tags"] for chunk, value in values},
                      "text_tags": values[0][1]["tags"]})
    return {**document, "views": views}


def caption_specs(view, cfg):
    override = cfg.get("prompt_override", "")
    if cfg.h3.get("single_sequence", False):
        intervals = chunk_intervals(view, cfg.chunk_size, cfg.h3.get("chunk_size_range") is not None)
        if override:
            return [(i, override) for i in range(len(intervals))]
        scene = view.get("caption_scene")
        motions = view.get("caption_motions")
        if motions is None:
            parts = [value.partition("\n[CHUNK]") for value in (view.get("chunk_prompts") or [])]
            if parts:
                scene, motions = parts[0][0], [p[2] if p[1] else p[0] for p in parts]
        if motions is None:
            return [(i, view.get("prompt") or cfg.negative_prompt) for i in range(len(intervals))]
        offset = view.get("source_start", 0)
        duration = view.get("caption_source_frames", offset + view["source_frames"])
        threshold = cfg.h3.get("caption_overlap_threshold", .5)
        result = []
        for i, (start, stop) in enumerate(intervals.tolist()):
            selected = []
            for j, motion in enumerate(motions):
                left = max(0, 4 * cfg.chunk_size * j - 3)
                right = min(duration, 4 * cfg.chunk_size * (j + 1) - 3)
                overlap = max(0, min(stop + offset, right) - max(start + offset, left))
                if right > left and overlap > threshold * (right - left):
                    selected.append(str(motion).strip())
            # A small block can cover no majority window. Retain only the scene.
            result.append((i, "\n".join(value for value in [str(scene or "").strip(), *selected] if value)
                           or cfg.negative_prompt))
        return result
    if override or not view.get("chunk_prompts"):
        return [(-1, override or view["prompt"] or cfg.negative_prompt)]
    count = int(source_chunk_ids(view, cfg.chunk_size).max()) + 1
    start = int((view.get("source_start", 0) + 3) // (4 * cfg.chunk_size))
    captions = view["chunk_prompts"]
    selected = [captions[min(start + chunk, len(captions) - 1)] for chunk in range(count)]
    return list(enumerate(selected))
