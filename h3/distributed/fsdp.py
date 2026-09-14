"""FSDP/Ulysses wrapping, activation checkpointing and compilation."""

from functools import partial
from pathlib import Path

import torch
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.checkpoint import checkpoint

from utils import distributed as groups


def fsdp_options(cfg, mixed=True, text=False):
    strategy = cfg.sharding_strategy
    if strategy == "hybrid_full" and groups.device_mesh.ndim == 1:
        strategy = "full"
    return dict(
        sharding_strategy={
            "hybrid_full": ShardingStrategy.HYBRID_SHARD,
            "full": ShardingStrategy.FULL_SHARD,
            "no_shard": ShardingStrategy.NO_SHARD,
        }[strategy],
        device_mesh=groups.device_mesh,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        limit_all_gathers=True,
        forward_prefetch=cfg.forward_prefetch,
        cpu_offload=CPUOffload(
            offload_params=cfg.text_encoder_cpu_offload if text else cfg.generator_cpu_offload
        ),
        mixed_precision=(
            MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
                cast_forward_inputs=False,
                cast_root_forward_inputs=False,
            )
            if mixed and cfg.mixed_precision
            else None
        ),
    )


def wrap_model(model, cfg):
    if model.config.num_attention_heads % groups.get_sp_size():
        raise ValueError("H3 attention heads must be divisible by SP size")
    mp, full = fsdp_options(cfg), fsdp_options(cfg, mixed=False)
    # Keep mixed checkpoint dtypes in separate FlatParameters. Wrapping the
    # whole native model at once would flatten BF16 and FP32 storage together.
    for i, block in enumerate(model.transformer_blocks):
        block.attn.processor.sequence_parallel = True
        block.attn = FSDP(block.attn, **mp)
        model.transformer_blocks[i] = FSDP(block, **mp)
    for i, block in enumerate(model.token_refiner.refiner_blocks):
        model.token_refiner.refiner_blocks[i] = FSDP(block, **mp)
    for name in ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out", "time_embedder"):
        setattr(model, name, FSDP(getattr(model, name), **full))
    model.norm_out.linear = FSDP(model.norm_out.linear, **full)
    model.sequence_parallel = True
    model = FSDP(model, **mp)
    for p in model.parameters():
        p.grad_dtype = None
    return model


def wrap_text(encoder, cfg):
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    return FSDP(
        encoder,
        auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=50_000_000),
        **fsdp_options(cfg, text=True),
    )


def compile_blocks(model, cfg):
    # H3's shape pool changes sequence lengths, text lengths and device scopes.
    # An exhausted guard cache must never cause eager quadratic flex attention.
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)
    torch._dynamo.config.accumulated_recompile_limit = max(
        torch._dynamo.config.accumulated_recompile_limit, 4096
    )
    torch._dynamo.config.automatic_dynamic_shapes = False
    # Preserve native BF16 rounding at casts even when a camera overlay makes
    # a different fusion graph. Otherwise an identity camera changes the output.
    if cfg.h3.get("exact_init", False):
        torch._inductor.config.emulate_precision_casts = True
    model.gradient_checkpointing = False
    model.blocks_checkpointed = bool(cfg.gradient_checkpointing)
    for block in model.transformer_blocks:
        outside = bool(cfg.h3.get("checkpoint_outside_compile", False))
        if outside and cfg.attn_block_compile:
            # FSDP mutates its runtime state. Keep it outside checkpoint HOP
            # tracing, as in WorldViews, and compile the original block math.
            target = block.module if isinstance(block, FSDP) else block
            target.forward = torch.compile(target.forward, dynamic=False)
        original = block.forward

        def make_forward(function):

            def forward(*args, **kwargs):
                if torch.is_grad_enabled() and cfg.gradient_checkpointing:
                    return checkpoint(function, *args, use_reentrant=False, **kwargs)
                return function(*args, **kwargs)

            return forward

        block.forward = make_forward(original)
        if cfg.attn_block_compile and not outside:
            block.forward = torch.compile(block.forward, dynamic=False)


def save_compile_cache(directory):
    from utils.compile import atomic_write, current_snapshot

    data, _ = current_snapshot()
    if data:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        atomic_write(data, str(path / f"rank{groups.get_rank()}.bin"))


def load_compile_cache(directory):
    from utils.compile import CacheArtifactManager, reregister, union_fold

    path = Path(directory) / f"rank{groups.get_rank()}.bin"
    if path.is_file():
        union = union_fold([path.read_bytes()])
        CacheArtifactManager.populate_caches(union)
        reregister(union)
