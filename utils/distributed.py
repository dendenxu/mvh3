import os
from typing import Optional
from functools import partial
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flat_param import FlatParameter, FlatParamHandle
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy, lambda_auto_wrap_policy


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
        'torch.float32': torch.float32, 'torch.float16': torch.float16,
        'torch.bfloat16': torch.bfloat16, 'torch.int64': torch.int64,
        'torch.int32': torch.int32, 'torch.int16': torch.int16,
        'torch.int8': torch.int8, 'torch.bool': torch.bool,
        'torch.float64': torch.float64, 'torch.uint8': torch.uint8,
    }
    all_tensors = [[] for _ in range(sp_size)]
    for src in range(sp_size):
        src_global = sp_global_ranks[src]
        for i, (shape, dtype_str) in enumerate(metas[src]):
            dtype = _dtype_map[dtype_str]
            if sp_rank == src:
                t = local_tensors[i]
                t = t.cuda() if t.device.type == 'cpu' else t
            else:
                t = torch.empty(shape, dtype=dtype, device='cuda')
            dist.broadcast(t, src=src_global, group=sp_group)
            all_tensors[src].append(t)

    # 4. Unflatten each rank's skeleton with that rank's tensor list
    def unflatten(obj, tensors):
        if type(obj) is dict and "__t__" in obj:
            device = obj["dev"] if 'cuda' not in obj["dev"] else 'cuda'
            return tensors[obj["__t__"]].to(device)
        elif type(obj) is dict:
            return {k: unflatten(v, tensors) for k, v in obj.items()}
        elif type(obj) in (list, tuple):
            return type(obj)(unflatten(v, tensors) for v in obj)
        return obj

    return [unflatten(skeletons[i], all_tensors[i]) for i in range(sp_size)]


def fsdp_move_device(model: FSDP, optim: torch.optim.AdamW = None, device: str = 'cuda'):
    """
    Directly swaps FSDP sharded storage and optimizer states between devices.
    """

    def fast_move(tensor: torch.Tensor):
        if str(tensor.device) == str(device):
            return tensor
        else:
            return tensor.to(device, non_blocking=True)

    # 0. Special handling for cpu offloading
    is_cpu = str(device) == 'cpu'

    # 1. Handle non-sharded buffers (BN stats, etc.)
    for buffer in model.buffers():
        buffer.data = fast_move(buffer.data)

    # 2. Handle sharded data
    for module in model.modules():
        if isinstance(module, FSDP):
            handle: FlatParamHandle = module._handle
            if handle is None:  # skip if not used
                continue

            # Doing data copy alone doesn't work, will also have to update the view
            # the flat_param_to api does these two things
            # module._handle.flat_param.data = module._handle.flat_param.data.to(device)
            # module._handle._use_sharded_views()
            # we will have to manually clean the lingering _local_shard and grad ref

            # View refresh
            fp_data = handle.flat_param.data
            handle.flat_param.data = fast_move(fp_data)

            # Update view
            if is_cpu:
                size_0_empty_tensor = torch.empty(0, device=handle.flat_param.device, dtype=handle.flat_param.dtype)
                for param in handle.flat_param._params + handle.flat_param._shared_params + handle.flat_param._tensors:
                    if param is not None:
                        param.data = size_0_empty_tensor  # cleanup after moving
            else:
                handle._use_sharded_views()

            # Local shard pointer update
            handle.flat_param._local_shard = handle.flat_param.data

            # Set gradient to none
            handle.flat_param.grad = None  # empty grad

            # FOR CPU OFFLOADING PARAMS
            # Haven't implemented the state management logic for pure gpu case + no fsdp cpu offloading
            if module.cpu_offload.offload_params:
                handle._offload_params = is_cpu
                if is_cpu:
                    handle.flat_param._cpu_grad = torch.zeros_like(handle.flat_param._local_shard).pin_memory()
                else:
                    if hasattr(handle.flat_param, '_cpu_grad'):
                        del handle.flat_param._cpu_grad

    # 3. Handle optimizer states for THIS specific model (Generator)
    if optim is not None:
        model_params = set(model.parameters())
        for param, state in optim.state.items():
            if param in model_params:
                for k, v in state.items():
                    if k == 'step':
                        continue  # no need to move step
                    if isinstance(v, torch.Tensor):
                        state[k] = fast_move(v)

    # Synchronize to ensure all memory operations are complete before the next step
    torch.cuda.synchronize()
    dist.barrier()  # make sure all gpus completed this move operation


def fsdp_set_grad_to_none(model: FSDP):
    """
    Directly swaps FSDP sharded storage and optimizer states between devices.
    """
    # 1. Handle sharded data
    with model._deregister_orig_params_ctx():  # unregister the hook
        for param in model.parameters():
            param.grad = None


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


def get_sp_group_rank():
    rank = get_rank()
    sp_size = get_sp_size()
    return rank // sp_size  # the rank inside the sp group


def get_distributed():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


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


def get_local_size() -> int:
    """
    NOTE: This implementation assumes a homogeneous cluster where
    every node has the same number of GPUs.
    """
    if not dist.is_available() or not dist.is_initialized():
        return int(os.environ.get("NPROC_PER_NODE", 8))
    return torch.cuda.device_count()


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

def _all_to_all_impl(x: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
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
def _all_to_all_op(x: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
    return _all_to_all_impl(x, scatter_dim, gather_dim)


@_all_to_all_op.register_fake
def _all_to_all_fake(x: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
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


def _all_to_all_backward(ctx, grad_output):
    scatter_dim, gather_dim = ctx.scatter_dim, ctx.gather_dim
    # backward of all_to_all is all_to_all with swapped dims
    return _all_to_all_op(grad_output, gather_dim, scatter_dim), None, None


def _all_to_all_setup_context(ctx, inputs, output):
    x, scatter_dim, gather_dim = inputs
    ctx.scatter_dim = scatter_dim
    ctx.gather_dim = gather_dim


_all_to_all_op.register_autograd(_all_to_all_backward, setup_context=_all_to_all_setup_context)


def _all_gather_impl(tensor: torch.Tensor) -> torch.Tensor:
    """Actual distributed all_gather implementation, returns concatenated result."""
    sp_size = get_sp_size()
    if sp_size > 1:
        tensor_list = [torch.empty_like(tensor) for _ in range(sp_size)]
        dist.all_gather(tensor_list, tensor, group=get_sp_group())
        return torch.cat(tensor_list, dim=0).contiguous()
    return tensor.contiguous()


@torch.library.custom_op("worldviews::all_gather", mutates_args=())
def _all_gather_op(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    sp_size = get_sp_size()
    if sp_size > 1:
        tensor_list = [torch.empty_like(tensor) for _ in range(sp_size)]
        dist.all_gather(tensor_list, tensor, group=get_sp_group())
        return torch.cat(tensor_list, dim=dim).contiguous()
    # sp_size=1: custom_op forbids output aliasing input, so clone.
    return tensor.clone()


@_all_gather_op.register_fake
def _all_gather_fake(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    sp_size = get_sp_size()
    shape = list(tensor.shape)
    shape[dim] = tensor.shape[dim] * sp_size
    return tensor.new_empty(shape)


def _all_gather_backward(ctx, grad_output):
    dim = ctx.dim
    sp_size = get_sp_size()
    # backward of all_gather is reduce_scatter
    grad_outputs_list = list(grad_output.chunk(sp_size, dim=dim))
    grad_input = torch.empty_like(grad_outputs_list[0])
    dist.reduce_scatter(grad_input, grad_outputs_list, op=dist.ReduceOp.SUM, group=get_sp_group())
    return grad_input, None


def _all_gather_setup_context(ctx, inputs, output):
    tensor, dim = inputs
    ctx.dim = dim


_all_gather_op.register_autograd(_all_gather_backward, setup_context=_all_gather_setup_context)


# ===================== Public API =====================

def all_to_all(x, scatter_dim, gather_dim, **kwargs):
    """
    Scatter along one dimension and gather along another.
    Supports gradient propagation. No graph break in torch.compile.
    """
    if get_sp_size() == 1:
        return x
    return _all_to_all_op(x, scatter_dim, gather_dim)


def gather_forward(input, dim):
    """
    Gather sequence and concatenate along the specified dimension.
    Supports gradient propagation. No graph break in torch.compile.
    """
    sp_size = get_sp_size()
    if sp_size == 1:
        return input
    return _all_gather_op(input, dim)


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
    assert input.shape[dim] % sp_size == 0, \
        f"scatter_forward: dim {dim} size {input.shape[dim]} not divisible by sp_size {sp_size}"
    return input.narrow(dim, sp_rank * chunk_size, chunk_size).contiguous()


def broadcast_scoped(tensor: torch.Tensor, scope: str = 'sp') -> torch.Tensor:
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
    if scope == 'sp':
        group = get_sp_group()
        if group is None:
            return tensor
        src = dist.get_process_group_ranks(group)[0]
        dist.broadcast(tensor, src=src, group=group)
    elif scope == 'global':
        dist.broadcast(tensor, src=0)
    else:
        raise ValueError(f"Unknown broadcast scope: {scope!r}")
    return tensor


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(model,
              sharding_strategy="full",
              mixed_precision=False,
              wrap_strategy="size",
              min_num_params=int(5e7),
              transformer_module=None,
              cpu_offload=False,
              offload_params=True,
              forward_prefetch=False,
              lambda_fn=None
              ):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,  # forward backward, not storage dtype
            reduce_dtype=torch.float32,  # gradient reduction dtype, reduce nccl bw
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,  # <--- Disables input conversion
            cast_root_forward_inputs=False,  # <--- Disables input conversion
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    elif wrap_strategy == "lambda":
        auto_wrap_policy = partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda_fn
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    if cpu_offload:
        cpu_offload = CPUOffload(offload_params=offload_params)
    else:
        cpu_offload = None

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=cpu_offload,
        sync_module_states=False,  # Load ckpt on rank 0 and sync to other ranks
        device_mesh=device_mesh,
        forward_prefetch=forward_prefetch,
    )

    # Allow any gradient dtype to prevent torch.compile FakeTensor error
    # when FSDP mixed precision uses reduce_dtype=float32 on bfloat16 params
    for param in model.parameters():
        param.grad_dtype = None

    return model


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
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=init_method, timeout=timeout,
                            device_id=torch.device("cuda", local_rank) if backend == "nccl" else None)

    # Init sp management group
    global sp_group, sp_size
    sp_size = min(sp_size_arg, world_size)
    num_sp_groups = world_size // sp_size

    for i in range(num_sp_groups):
        ranks = list(range(i * sp_size, (i + 1) * sp_size))
        group = dist.new_group(ranks, backend=backend, timeout=timeout,
                               device_id=torch.device("cuda", local_rank) if backend == "nccl" else None)
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


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
