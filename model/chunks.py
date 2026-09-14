"""Plan BD blocks from source-caption windows or native H3 latent counts."""

import torch

from utils.distributed import broadcast_scoped


def source_chunk_ids(view, chunk_size):
    """Use the original Wan time windows, including the first shorter window."""
    chunks = ((view["frames"].cpu() + 3) / (4 * chunk_size)).floor().long()
    valid = view["valid"].cpu()
    requested = int(valid.sum())
    if not requested or not torch.equal(valid, torch.arange(len(valid)) < requested):
        raise ValueError("Chunk grouping requires a nonempty requested latent prefix")
    return chunks.clamp_max(chunks[valid].max())


def group_partition(count, minimum, maximum):
    """Draw a group size per block; a short terminal remainder stays intact."""
    sizes, remaining = [], count
    while remaining:
        lower, upper = min(minimum, remaining), min(maximum, remaining)
        size = int(torch.randint(lower, upper + 1, ())) if lower < upper else lower
        sizes.append(size)
        remaining -= size
    return torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))


def native_partition(length, requested, minimum, maximum):
    """Cover every latent, keeping decoder support in the last real BD block."""
    if length < minimum:
        return torch.zeros(length, dtype=torch.long)
    final_minimum = max(minimum, length - requested + 1)
    possible = [False] * (length + 1)
    for remaining in range(1, length + 1):
        possible[remaining] = (final_minimum <= remaining <= maximum or any(
            possible[remaining - size] for size in range(minimum, min(maximum, remaining - 1) + 1)))
    if not possible[length]:
        raise ValueError("Chunk range cannot cover this video and its decoder support")
    sizes, remaining = [], length
    while remaining:
        choices = [size for size in range(minimum, min(maximum, remaining) + 1)
                   if (size == remaining and size >= final_minimum)
                   or (size < remaining and possible[remaining - size])]
        size = choices[int(torch.randint(len(choices), ()))] if len(choices) > 1 else choices[0]
        sizes.append(size)
        remaining -= size
    return torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))


def joint_native_partition(views, minimum, maximum):
    """Keep shared boundaries valid at every view's requested/support tail."""
    tails = sorted({(len(v["valid"]), int(v["valid"].sum())) for v in views})
    if len(tails) == 1:
        return native_partition(*tails[0], minimum, maximum)
    length = tails[-1][0]

    def allowed(start, stop):
        for end, requested in tails:
            if start >= end:
                continue
            if min(stop, end) - start < min(minimum, end):
                return False
            if requested <= stop < end:
                return False
        return True

    choices = [[] for _ in range(length + 1)]
    reachable = [False] * length + [True]
    for start in range(length - 1, -1, -1):
        choices[start] = [stop for stop in range(start + min(minimum, length), min(start + maximum, length) + 1)
                          if reachable[stop] and allowed(start, stop)]
        reachable[start] = bool(choices[start])
    if not reachable[0]:
        raise ValueError("Chunk range cannot cover the joint videos and their decoder support")
    sizes, start = [], 0
    while start < length:
        ends = choices[start]
        stop = ends[int(torch.randint(len(ends), ()))] if len(ends) > 1 else ends[0]
        sizes.append(stop - start)
        start = stop
    return torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))


def prepare_chunk_plan(document, cfg, device, synchronize=True):
    native = cfg.h3.get("chunk_size_range") is not None
    bounds = cfg.h3.get("chunk_size_range") if native else cfg.h3.get("chunk_group_range")
    if bounds is None and not cfg.h3.get("single_sequence", False):
        return document
    if bounds is None:
        bounds = (1, 1)
    minimum, maximum = map(int, bounds)
    views = document["views"]
    sources = [source_chunk_ids(view, cfg.chunk_size) for view in views]
    if all("generation_chunks" in view for view in document["views"]):
        mappings = []
        for view, source in zip(views, sources):
            plan, valid = view["generation_chunks"].cpu(), view["valid"].cpu()
            if (plan.dtype != torch.long or plan.shape != valid.shape or not valid.any()
                    or int(plan[0]) != 0 or ((plan[1:] - plan[:-1] < 0)
                                           | (plan[1:] - plan[:-1] > 1)).any()
                    or not plan[~valid].eq(plan[valid][-1]).all()):
                raise ValueError("Invalid saved chunk grouping")
            if native:
                mapping = plan
            else:
                starts = torch.nonzero(torch.cat((torch.ones(1, dtype=torch.bool), source[1:] != source[:-1]))).flatten()
                mapping = plan[starts]
                if not torch.equal(mapping[source], plan):
                    raise ValueError("A BD block cannot split an original caption chunk")
            counts = torch.bincount(mapping)
            if ((counts > maximum).any()
                    or (counts[:-1] < minimum).any()
                    or (native and counts[-1] < minimum and len(plan) >= minimum)):
                raise ValueError("Saved chunk grouping is outside the configured chunk range")
            mappings.append(mapping)
        if not document["isolated"]:
            longest = max(mappings, key=len)
            if any(not torch.equal(mapping[view["valid"].cpu()] if native else mapping,
                                   longest[:len(mapping)][view["valid"].cpu()] if native else longest[:len(mapping)])
                   for mapping, view in zip(mappings, views)):
                raise ValueError("Joint views must share the same chunk grouping")
        return document
    if any("generation_chunks" in view for view in document["views"]):
        raise ValueError("A document cannot mix planned and unplanned views")
    def draw(view, count):
        plan = (native_partition(len(view["valid"]), int(view["valid"].sum()), minimum, maximum)
                if native else group_partition(count, minimum, maximum)).to(device)
        return (broadcast_scoped(plan, "sp") if synchronize else plan).cpu()

    counts = [int(source.max()) + 1 for source in sources]
    longest = max(range(len(views)), key=lambda i: len(views[i]["valid"]))
    shared = None
    if not document["isolated"]:
        if native:
            shared = joint_native_partition(views, minimum, maximum).to(device)
            shared = (broadcast_scoped(shared, "sp") if synchronize else shared).cpu()
        else:
            shared = draw(views[longest], max(counts))
    planned = []
    for view, source, count in zip(views, sources, counts):
        mapping = draw(view, count) if shared is None else shared
        plan = mapping[:len(source)].clone() if native else mapping[source]
        plan.clamp_max_(int(plan[view["valid"].cpu()].max()))
        planned.append({**view, "generation_chunks": plan})
    return {**document, "views": planned}


def prepare_clean_prefix(document, cfg, device):
    """Draw one cut per sequence, retaining at least one supervised BD block."""
    views = document["views"]
    counts = [int(view["generation_chunks"].cpu()[view["valid"].cpu()].max()) + 1 for view in views]
    present = ["clean_prefix_chunks" in view for view in views]
    if any(present):
        if not all(present):
            raise ValueError("A document cannot mix planned and unplanned clean prefixes")
        for view, count in zip(views, counts):
            cut = view["clean_prefix_chunks"]
            if type(cut) is not int or not 0 <= cut < count:
                raise ValueError("A clean prefix must leave a supervised chunk")
        if not document["isolated"] and len({v["clean_prefix_chunks"] for v in views}) != 1:
            raise ValueError("Joint views must share one clean-prefix cut")
        return document

    def draw(count):
        cut = 0
        if count > 1 and torch.rand(()).item() < cfg.h3.get("clean_prefix_probability", .5):
            cut = int(torch.randint(1, count, ()))
        return int(broadcast_scoped(torch.tensor(cut, device=device), "sp"))

    shared = None if document["isolated"] else draw(min(counts))
    return {**document, "views": [{**view, "clean_prefix_chunks": draw(count) if shared is None else shared}
                                  for view, count in zip(views, counts)]}


def caption_chunk(view, caption_id, source_chunk_size):
    """Retain each source caption and assign it to its enclosing BD block."""
    if caption_id < 0 or "generation_chunks" not in view or view.get("texts_by_bd", False):
        return caption_id
    indices = torch.nonzero(source_chunk_ids(view, source_chunk_size) == caption_id).flatten()
    if not len(indices):
        raise ValueError("Caption chunk is outside the requested video")
    return int(view["generation_chunks"][int(indices[0])])


def chunk_intervals(view, source_chunk_size, native=False):
    plan = view["generation_chunks"].cpu()
    starts = torch.nonzero(torch.cat((torch.ones(1, dtype=torch.bool), plan[1:] != plan[:-1]))).flatten()
    if native:
        times = view["frames"].cpu()[starts].float()
    else:
        source = source_chunk_ids(view, source_chunk_size)[starts]
        times = (4 * source_chunk_size * source - 3).clamp_min(0).float()
    ends = torch.cat((times[1:], times.new_tensor([view["source_frames"]])))
    return torch.stack((times, ends), dim=-1)
