import os
from datetime import timedelta

import torch
import torch.distributed as dist


def gather_mixed_batch(local_batch):
    """
    Gather batches from all SP ranks using per-rank broadcast.

    Each rank broadcasts its own tensors to all other ranks. This allows
    different ranks to have different batch schemas (different tensor counts,
    shapes, dtypes) — e.g. when co-training datasets with different mv_size.

    Returns a list of sp_size batches, one from each rank, each with that
    rank's original structure and shapes.

    Communication: 1 all_gather_object (skeletons + tensor metadata, small) +
    sum(n_tensors_per_rank) broadcasts. Total bytes = same as the old
    all_gather_into_tensor approach; slightly more collectives but negligible
    since this runs once per grad-accum batch, not per step.
    """
    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    sp_global_ranks = dist.get_process_group_ranks(sp_group)

    local_tensors = []

    # 1. Flatten: extract tensors, replace with {__t__, dev} placeholders
    def flatten(obj):
        if isinstance(obj, torch.Tensor):
            local_tensors.append(obj)
            return {"__t__": len(local_tensors) - 1, "dev": str(obj.device)}
        elif type(obj) is dict:
            return {k: flatten(v) for k, v in obj.items()}
        elif type(obj) in (list, tuple):
            return type(obj)(flatten(v) for v in obj)
        return obj

    local_skeleton = flatten(local_batch)
    local_meta = [(tuple(t.shape), str(t.dtype)) for t in local_tensors]

    # 2. Exchange skeletons + tensor metadata (variable-size, pickle-based)
    gathered = [None] * sp_size
    dist.all_gather_object(gathered, (local_skeleton, local_meta), group=sp_group)
    skeletons = [g[0] for g in gathered]
    metas = [g[1] for g in gathered]

    # 3. For each source rank, broadcast its tensors one by one.
    #    Tensor count, shapes, dtypes can differ per source — that's the point.
    _dtype_map = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.int16": torch.int16,
        "torch.int8": torch.int8,
        "torch.bool": torch.bool,
        "torch.float64": torch.float64,
        "torch.uint8": torch.uint8,
    }
    all_tensors = [[] for _ in range(sp_size)]
    for src in range(sp_size):
        src_global = sp_global_ranks[src]
        for i, (shape, dtype_str) in enumerate(metas[src]):
            dtype = _dtype_map[dtype_str]
            if sp_rank == src:
                # VAE latents and matrix inverses may retain nonstandard
                # strides; NCCL broadcast requires contiguous source storage.
                t = local_tensors[i].contiguous()
                t = t.cuda() if t.device.type == "cpu" else t
            else:
                t = torch.empty(shape, dtype=dtype, device="cuda")
            dist.broadcast(t, src=src_global, group=sp_group)
            all_tensors[src].append(t)

    # 4. Unflatten each rank's skeleton with that rank's tensor list
    def unflatten(obj, tensors):
        if type(obj) is dict and "__t__" in obj:
            device = obj["dev"] if "cuda" not in obj["dev"] else "cuda"
            return tensors[obj["__t__"]].to(device)
        elif type(obj) is dict:
            return {k: unflatten(v, tensors) for k, v in obj.items()}
        elif type(obj) in (list, tuple):
            return type(obj)(unflatten(v, tensors) for v in obj)
        return obj

    return [unflatten(skeletons[i], all_tensors[i]) for i in range(sp_size)]


def get_sp_size():
    return sp_size


def shutdown_distributed():
    """Finish all work before communicator teardown; support the tested host workaround."""
    if not dist.is_initialized():
        return
    dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()
    if os.environ.get("MVH3_NCCL_ABORT_ON_EXIT") == "1":
        # This host's NCCL 2.29 shutdown stalls even after successful barriers.
        # Abort is used only after every rank completed and synchronized all work.
        from torch.distributed.distributed_c10d import _abort_process_group

        _abort_process_group()
    else:
        dist.destroy_process_group()


def get_sp_group():
    return sp_group


def get_sp_rank():
    rank = get_rank()
    sp_size = get_sp_size()
    return rank % sp_size  # the rank inside the sp group


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return int(os.environ.get("WORLD_SIZE", 1))
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return int(os.environ.get("RANK", 0))
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def is_node_main() -> bool:
    """Whether this process is the main (local rank 0) process on its own node.

    Unlike ``is_main_process()`` (True only on global rank 0), this is True on
    every node's local rank 0. Use it to gate *logging* that should surface once
    per node — e.g. per-node debugging where each machine may be doing something
    different (OOM, data-loading stalls). Do NOT use it for operations that must
    run exactly once globally (checkpoint save, wandb, symlinks, or any cross-rank
    collective) — those must stay behind ``is_main_process()``.
    """
    return get_local_rank() == 0


def get_local_rank() -> int:
    """
    Returns the local rank (node-local GPU index) of the current process.

    Prefers the ``LOCAL_RANK`` environment variable, which torchrun sets
    authoritatively for every worker and which ``launch_distributed_job`` already
    trusts for ``torch.cuda.set_device``. This is exact regardless of how many GPUs
    are visible per process — unlike the ``global_rank % device_count`` fallback
    below, which silently misfires when ``torch.cuda.device_count() != nproc_per_node``
    (e.g. ``CUDA_VISIBLE_DEVICES`` restricts visibility, or fewer procs are launched
    than the node has GPUs). A wrong value here breaks every ``is_node_main()``
    gate — per-node logging vanishes and per-node-leader ops (artifact/compile-cache
    save) skip non-zero nodes.

    NOTE: The modulo fallback assumes a homogeneous cluster with node-contiguous
    rank assignment where every node has the same number of GPUs.
    """
    # Preferred: authoritative launcher-provided local rank.
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    # Fallback if distributed training is not initialized (and no LOCAL_RANK env).
    if not dist.is_available() or not dist.is_initialized():
        return 0

    # Last resort: derive from global rank assuming node-contiguous assignment.
    # Example: Global Rank 9 on an 8-GPU node -> 9 % 8 = 1 (Local Rank 1)
    global_rank = dist.get_rank()
    num_gpus_per_node = torch.cuda.device_count()
    if num_gpus_per_node == 0:  # avoid division by zero on CPU-only runs
        return 0
    return global_rank % num_gpus_per_node


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


# ===================== Custom ops for torch.compile =====================
# Using torch.library.custom_op instead of autograd.Function avoids graph
# breaks in torch.compile. The register_fake (FakeTensor meta) tells the
# compiler the output shapes directly, so it can continue tracing without
# breaking the graph — keeping all shape information intact.


def all_to_all_impl(x: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    """Actual distributed all_to_all implementation."""
    sp_size = get_sp_size()
    if sp_size > 1:
        inputs = [u.contiguous() for u in x.chunk(sp_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=get_sp_group())
        return torch.cat(outputs, dim=gather_dim).contiguous()
    # sp_size=1: custom_op forbids output aliasing input, so clone.
    return x.clone()


@torch.library.custom_op("worldviews::all_to_all", mutates_args=())
def all_to_all_op(x: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    return all_to_all_impl(x, scatter_dim, gather_dim)


@all_to_all_op.register_fake
def all_to_all_fake(x: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    """Tell torch.compile the output shape without running the actual op.

    Uses int() on input dims to produce concrete (non-symbolic) output shapes.
    This creates guards that specialize on the current sizes — if sizes change,
    torch.compile recompiles (which is fine since SP topology is fixed).
    Without this, scatter_dim // sp_size produces a symbolic floor() expression
    that flex_attention's inductor lowering can't handle (bitwise & on sympy floor).
    """
    sp_size = get_sp_size()
    shape = list(x.shape)
    shape[scatter_dim] = int(shape[scatter_dim]) // sp_size
    shape[gather_dim] = int(shape[gather_dim]) * sp_size
    return x.new_empty(shape)


def all_to_all_backward(ctx, grad_output):
    scatter_dim, gather_dim = ctx.scatter_dim, ctx.gather_dim
    # backward of all_to_all is all_to_all with swapped dims
    return all_to_all_op(grad_output, gather_dim, scatter_dim), None, None


def all_to_all_setup_context(ctx, inputs, output):
    x, scatter_dim, gather_dim = inputs
    ctx.scatter_dim = scatter_dim
    ctx.gather_dim = gather_dim


all_to_all_op.register_autograd(all_to_all_backward, setup_context=all_to_all_setup_context)


@torch.library.custom_op("worldviews::all_gather", mutates_args=())
def all_gather_op(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    sp_size = get_sp_size()
    if sp_size > 1:
        tensor_list = [torch.empty_like(tensor) for _ in range(sp_size)]
        dist.all_gather(tensor_list, tensor, group=get_sp_group())
        return torch.cat(tensor_list, dim=dim).contiguous()
    # sp_size=1: custom_op forbids output aliasing input, so clone.
    return tensor.clone()


@all_gather_op.register_fake
def all_gather_fake(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    sp_size = get_sp_size()
    shape = list(tensor.shape)
    shape[dim] = tensor.shape[dim] * sp_size
    return tensor.new_empty(shape)


def all_gather_backward(ctx, grad_output):
    dim = ctx.dim
    sp_size = get_sp_size()
    # backward of all_gather is reduce_scatter
    grad_outputs_list = list(grad_output.chunk(sp_size, dim=dim))
    grad_input = torch.empty_like(grad_outputs_list[0])
    dist.reduce_scatter(grad_input, grad_outputs_list, op=dist.ReduceOp.SUM, group=get_sp_group())
    return grad_input, None


def all_gather_setup_context(ctx, inputs, output):
    tensor, dim = inputs
    ctx.dim = dim


all_gather_op.register_autograd(all_gather_backward, setup_context=all_gather_setup_context)

# ===================== Public API =====================


def all_to_all(x, scatter_dim, gather_dim, **kwargs):
    """
    Scatter along one dimension and gather along another.
    Supports gradient propagation. No graph break in torch.compile.
    """
    if get_sp_size() == 1:
        return x
    return all_to_all_op(x, scatter_dim, gather_dim)


def gather_forward(input, dim):
    """
    Gather sequence and concatenate along the specified dimension.
    Supports gradient propagation. No graph break in torch.compile.
    """
    sp_size = get_sp_size()
    if sp_size == 1:
        return input
    return all_gather_op(input, dim)


def scatter_forward(input, dim):
    """
    Scatter input tensor along the specified dimension.
    No graph break: narrow() produces simplifiable shape expressions.
    """
    sp_size = get_sp_size()
    sp_rank = get_sp_rank()
    if sp_size == 1:
        return input

    # narrow() gives torch.compile size = input.shape[dim] // sp_size (simplifiable),
    # unlike chunk() which produces opaque ceil-division expressions.
    chunk_size = input.shape[dim] // sp_size
    assert (
        input.shape[dim] % sp_size == 0
    ), f"scatter_forward: dim {dim} size {input.shape[dim]} not divisible by sp_size {sp_size}"
    return input.narrow(dim, sp_rank * chunk_size, chunk_size).contiguous()


def broadcast_scoped(tensor: torch.Tensor, scope: str = "sp") -> torch.Tensor:
    """Broadcast a tensor from the source rank of the chosen scope in-place.

    scope='sp':     source = first rank of this rank's SP group. SP groups
                    are independent → different groups can end up with
                    different values. Use for per-step decisions that must
                    be consistent within an SP group (to keep SP collective
                    shapes aligned) but can diverge across nodes.
    scope='global': source = global rank 0. All ranks end up with the same
                    value. Use when every rank must agree (e.g. dual-model
                    selection where different nodes picking different models
                    would cause inter-node all_reduce to pair gradients from
                    different FlatParams, silently corrupting both).

    No-op when distributed is not initialized (single-rank runs) or the SP
    group wasn't set up. Returns the same tensor for call-site chaining.
    """
    if not dist.is_initialized():
        return tensor
    if scope == "sp":
        group = get_sp_group()
        if group is None:
            return tensor
        src = dist.get_process_group_ranks(group)[0]
        dist.broadcast(tensor, src=src, group=group)
    elif scope == "global":
        dist.broadcast(tensor, src=0)
    else:
        raise ValueError(f"Unknown broadcast scope: {scope!r}")
    return tensor


def barrier():
    if dist.is_initialized():
        dist.barrier()


sp_group = None
sp_size = 1
device_mesh = None
fs_size = 8


def launch_distributed_job(backend: str = "nccl", sp_size_arg=1, fs_size_arg=1, timeout: int = 600):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    host = os.environ.get("MASTER_ADDR", "localhost")
    port = int(os.environ.get("MASTER_PORT", "0"))

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    timeout = timedelta(minutes=timeout)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timeout,
        device_id=torch.device("cuda", local_rank) if backend == "nccl" else None,
    )

    # Init sp management group
    global sp_group, sp_size
    sp_size = min(sp_size_arg, world_size)
    num_sp_groups = world_size // sp_size

    for i in range(num_sp_groups):
        ranks = list(range(i * sp_size, (i + 1) * sp_size))
        group = dist.new_group(
            ranks,
            backend=backend,
            timeout=timeout,
            device_id=torch.device("cuda", local_rank) if backend == "nccl" else None,
        )
        if rank in ranks:
            sp_group = group

    # Init fsdp device mesh
    global device_mesh, fs_size
    fs_size = min(fs_size_arg, world_size)
    # A single shard group has no replication axis. Avoid creating unused
    # singleton NCCL communicators. The one-group FULL_SHARD reduction is
    # numerically equivalent to HYBRID_SHARD with a replication dimension of 1.
    shape = (fs_size,) if world_size == fs_size else (world_size // fs_size, fs_size)
    device_mesh = dist.device_mesh.init_device_mesh("cuda", shape)

    return sp_size, fs_size


def canonical_name(name):
    """Keep parameter names stable across FSDP and torch.compile wrappers."""
    return name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
