"""torch.compile helpers — cache-artifact management + the checkpoint-under-compile wrapper
(run_checkpointed) — orthogonal to training, kept out of the trainer.

  - load_compile_artifacts(): at startup, load the cross-node MERGED bake ($BN/misc/artifact.bin first --
    hr-restarts skip entry.sh's cp, so the local ~/.cache copy can be stale -- then local fallback).
  - save_node_artifact(): on a NEW shape, write this node-leader's COMPLETE bake to local (read fallback) +
    the per-node $BN/misc/artifacts/<host>.bin (no clobber across nodes).
  - merge_node_artifacts(): global rank 0 folds the per-node bins into the single $BN/misc/artifact.bin.

CRITICAL -- why every write is a single COMPLETE snapshot, not an incremental delta:
torch's deserialize is `dict(AppendingByteSerializer.to_list(...))`, which keeps only the LAST appended
(type, list) chunk per type. And `save_cache_artifacts()` clears `_new_cache_artifacts` after each save, so a
repeated (incremental) save appends only the delta-since-last-save as a fresh chunk. A file written that way
therefore reads back as ONLY the last delta -- e.g. a 4.6 GB file with 24 chunks deserialized to 991 while its
true union was 4666 (every chunk is physically present, just shadowed). The earlier "byte size keeps growing =
healthy" signal was an illusion: the bytes grow but the readable set collapses to the last delta.

So: snapshot_bytes() re-serializes the FULL union as ONE chunk per type (torch's stock deserialize then reads
it whole), and load UNION-folds every chunk so any existing multi-chunk file is recovered in full. Writes are
ATOMIC (tmp + os.replace). All $BN I/O is best-effort (a bytenas hiccup is logged, never fatal).
"""
import os
import glob
import socket
from os.path import exists, dirname, expanduser, getmtime
from collections import defaultdict

import torch
import torch.utils.checkpoint
from torch.utils._appending_byte_serializer import AppendingByteSerializer
from torch.compiler._cache import CacheArtifactManager, _deserialize_single_cache

from utils.console import log
from utils.console import yellow
from utils.console import magenta
from utils.console import blue
from utils.distributed import is_main_process
from utils.distributed import is_node_main
from utils.distributed import get_local_rank


def run_checkpointed(forward, *args, **kwargs):
    """Self-checkpoint `forward` when grad is enabled (passthrough at inference). compile_blocks binds
    this onto each block's forward (functools.partial) and THEN compiles the block, so the checkpoint
    becomes an in-graph `tag_activation_checkpoint` HOP — i.e. compile(checkpoint(block)). That order
    lets the compiler partition save/recompute once inside the joint graph: forward and its recompute
    share one partition (no check_recomputed_tensors_match mismatch) and the recompute runs compiled,
    not eager. FSDP-agnostic: `forward` is the block's own forward (FSDP off) or FSDP.forward incl. the
    all-gather (FSDP on, so the recompute re-gathers)."""
    if torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(forward, *args, use_reentrant=False, **kwargs)
    return forward(*args, **kwargs)


def artifact_paths():
    """(local_bin, read_candidates). Prefer the cross-node merged bake on $BN, fall back to the local copy;
    an explicit WORLDGEN_TORCHCOMPILE_ARTIFACT_BIN override wins."""
    home = expanduser("~")
    bn = os.environ.get('BN')
    local = f'{home}/.cache/worldgen_torchcompile/artifact.bin'
    override = os.environ.get('WORLDGEN_TORCHCOMPILE_ARTIFACT_BIN')
    local_bin = override or local
    if override:
        candidates = [override]
    elif bn:
        candidates = [f'{bn}/misc/artifact.bin', local]
    else:
        candidates = [local]
    return local_bin, candidates


def atomic_write(data, path):
    """Write bytes via a per-process tmp + atomic rename, so a reader never sees a partial file."""
    os.makedirs(dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'wb') as f:
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
        return b'', 0
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
        return b'', 0
    data, _ = res
    return data, count


def current_snapshot():
    """COMPLETE single-snapshot of everything THIS process holds: flush the accumulated (multi-chunk) bytes,
    union-fold them, re-serialize as one chunk per type. Returns (bytes, count)."""
    res = torch.compiler.save_cache_artifacts()  # cumulative multi-chunk bytes; also flushes _new
    raw = res[0] if res else b''
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


def load_compile_artifacts():
    """Load the bake ($BN merged first, local fallback), UNION-folding every chunk (so an existing multi-chunk
    file is recovered in full) + populate + re-register. Best-effort. Returns (path|None, count)."""
    _, candidates = artifact_paths()
    for cand in candidates:
        try:
            with open(cand, 'rb') as f:
                data = f.read()
            union = union_fold([data])
            count = sum(len(v) for v in union.values())
            if not count:
                raise RuntimeError('no artifacts deserialized')
            CacheArtifactManager.populate_caches(union)
            reregister(union)
            if is_node_main():
                log(yellow(f'Loaded torch.compile artifacts from: {blue(cand)} ({count} entries)'))
            return cand, count
        except FileNotFoundError:
            if is_node_main():
                log(yellow(f'torch.compile artifacts not found: {blue(cand)}'))
        except Exception as e:
            if is_node_main():
                log(yellow(f'torch.compile artifacts load failed from {blue(cand)} ({type(e).__name__}: {e})'))
    if is_node_main():
        log(yellow('no torch.compile artifacts loaded; will compile and save later'))
    return None, 0


def save_node_artifact(stage=''):
    """Write this node-leader's COMPLETE snapshot: local (read fallback) + per-node $BN/misc/artifacts/<host>.bin
    (no clobber). Atomic, best-effort. Callers gate on new-shape (not called every step)."""
    local_bin, _ = artifact_paths()
    data, count = current_snapshot()
    if not data:
        return
    try:
        atomic_write(data, local_bin)
    except Exception as e:
        log(yellow(f"  local artifact write failed ({type(e).__name__}: {e}); ignored"))

    bn = os.environ.get('BN')
    if bn and get_local_rank() == 0:
        dst = f'{bn}/misc/artifacts/{socket.gethostname()}.bin'
        try:
            atomic_write(data, dst)
            log(magenta(f"  per-node artifacts -> {blue(dst)} ({count} entries, {len(data)} bytes, stage {stage})"))
        except Exception as e:
            log(yellow(f"  per-node artifact write to {dst} failed ({type(e).__name__}: {e}); ignored"))


def merge_node_artifacts():
    """Global rank 0: fold the per-node bins (those newer than the merged file) into rank 0's accumulated union,
    then write the single COMPLETE $BN/misc/artifact.bin. Atomic, best-effort. Incremental read (only changed
    bins), but the written snapshot is always the full union rank 0 holds."""
    bn = os.environ.get('BN')
    if not (is_main_process() and bn):
        return
    node_dir = f'{bn}/misc/artifacts'
    merged = f'{bn}/misc/artifact.bin'
    try:
        node_bins = sorted(glob.glob(f'{node_dir}/*.bin'))
    except Exception as e:
        log(yellow(f"  merge: list {node_dir} failed ({type(e).__name__}: {e}); ignored"))
        return
    if not node_bins:
        return
    try:
        merged_mtime = getmtime(merged) if exists(merged) else -1.0
    except Exception:
        merged_mtime = -1.0
    try:
        sources = [b for b in node_bins if getmtime(b) > merged_mtime]
    except Exception:
        sources = node_bins
    if not sources:
        return
    n_new = 0
    for b in sources:
        try:
            with open(b, 'rb') as f:
                n_new += reregister(union_fold([f.read()]))
        except Exception as e:
            log(yellow(f"  merge: skip {b} ({type(e).__name__}: {e})"))
    try:
        data, count = current_snapshot()
        if not data:
            return
        atomic_write(data, merged)
        log(magenta(f"  merged {len(sources)} node bins (+{n_new} new) -> {blue(merged)} ({count} entries, {len(data)} bytes)"))
    except Exception as e:
        log(yellow(f"  merge: write {merged} failed ({type(e).__name__}: {e}); ignored"))
