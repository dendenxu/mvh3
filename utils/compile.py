"""Read and write complete Torch compile-cache snapshots without losing older entries."""

import os
from collections import defaultdict
from os.path import dirname

import torch
import torch.utils.checkpoint
from torch.compiler._cache import CacheArtifactManager, _deserialize_single_cache
from torch.utils._appending_byte_serializer import AppendingByteSerializer


def atomic_write(data, path):
    """Write bytes via a per-process tmp + atomic rename, so a reader never sees a partial file."""
    os.makedirs(dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def union_fold(blobs):
    """Union-fold raw serialized bytes (each possibly multi-chunk) into one COMPLETE {type: [CacheArtifact]}.
    Recovers everything physically present, undoing torch's dict-fold (last-chunk-per-type) shadowing."""
    CacheArtifactManager._ensure_cache_artifacts_registered()
    union = defaultdict(list)
    seen = set()
    for data in blobs:
        if not data:
            continue
        for atype, arts in AppendingByteSerializer.to_list(data, deserialize_fn=_deserialize_single_cache):
            for a in arts:
                k = (a.type(), a.key)
                if k not in seen:
                    seen.add(k)
                    union[atype].append(a)
    return union


def snapshot_bytes(union):
    """Serialize a complete union as ONE chunk per type (fresh serializer + _new = union) so torch's stock
    deserialize reads it back in full. Returns (bytes, count). Mutates the manager's serializer/_new -- fine,
    since every snapshot rebuilds from the full union."""
    if not union:
        return b"", 0
    count = sum(len(v) for v in union.values())  # TRUE union size -- the count that matches the written bytes
    M = CacheArtifactManager
    M._serializer.clear()
    # Do NOT report torch's CacheInfo count: serialize() does `_cache_info.add(a)` for every _new artifact and
    # NEVER clears _cache_info, then returns a deepcopy of the whole thing. Since each snapshot sets
    # `_new = the full union`, that process-cumulative count grows by |union| on every call (the logs showed
    # 7269 -> 14543 -> ... -> 320323) and the per-save deepcopy of it grows unbounded too. It is a
    # scary-but-cosmetic OVER-COUNT -- the bytes actually WRITTEN are the clean union (verified: a 3.4 GB
    # per-node bin deserializes to ~7.6k unique artifacts, 0 dup keys). Reset _cache_info so the reported
    # count is real and the deepcopy stays bounded (only the count metadata is affected; the written data
    # comes from _serializer, not _cache_info).
    try:
        M._cache_info = M._cache_info.__class__()
    except Exception:
        pass
    M._new_cache_artifacts = union
    res = torch.compiler.save_cache_artifacts()
    if not res:
        return b"", 0
    data, _ = res
    return data, count


def current_snapshot():
    """COMPLETE single-snapshot of everything THIS process holds: flush the accumulated (multi-chunk) bytes,
    union-fold them, re-serialize as one chunk per type. Returns (bytes, count)."""
    res = torch.compiler.save_cache_artifacts()  # cumulative multi-chunk bytes; also flushes _new
    raw = res[0] if res else b""
    return snapshot_bytes(union_fold([raw]))


def reregister(union):
    """Re-register a loaded/merged union so the next save includes it (the manager only serializes what its
    _new holds). Returns the count of NEW ones."""
    n = 0
    for atype, arts in union.items():
        for a in arts:
            if a not in CacheArtifactManager._seen_artifacts:
                CacheArtifactManager._new_cache_artifacts[atype].append(a)
                CacheArtifactManager._seen_artifacts.add(a)
                n += 1
    return n
