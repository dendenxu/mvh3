"""Resolve the complete WorldViews recipe and its explicit H3 adaptation."""

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def load_config(path, overrides=()):
    path = Path(path)
    cfg = OmegaConf.load(path)
    base = OmegaConf.create({})
    for parent in cfg.pop("parents", []):
        base = OmegaConf.merge(base, load_config(path.parent / parent))
    return OmegaConf.merge(base, cfg, OmegaConf.from_dotlist(list(overrides)))


def stage_dataset_config(cfg, stage, validation=False):
    data = OmegaConf.create(OmegaConf.to_container(cfg.val_dataset if validation else cfg.dataset, resolve=True))
    key = "stage1_val_dataset" if validation else "stage1_dataset"
    if stage == 1 and cfg.h3.get(key):
        override = cfg.h3[key]
        data = OmegaConf.merge(data, override)
        # An empty SHORT remap clears FULL's caps, rather than merging with them.
        if "shape_remap" in override:
            data.shape_remap = OmegaConf.to_container(override.shape_remap, resolve=True)
    return data


def recipe_digest(cfg):
    """Allow stage/log paths to change, but reject accidental recipe drift on resume."""
    from h3.modules.camera import TRANSLATION_FREQUENCIES

    data = OmegaConf.to_container(cfg, resolve=True)
    # Analytic camera constants change checkpoint semantics without adding weights.
    data["camera_encoding"] = {"translation_frequencies": list(TRANSLATION_FREQUENCIES)}
    # CPU and CUDA generators produce different posterior samples at the same seed.
    data["condition_encoding"] = {"posterior_generator_device": "cpu"}
    h3 = data.get("h3", {})
    for key in ("checkpoint", "vae", "text_cache", "compile_cache", "logdir", "stage"):
        h3.pop(key, None)
    for key in ("resume_ckpt", "auto_resume", "max_iters", "task", "inference_request", "inference_protocol",
                "inference_weights", "validation_weights"):
        data.pop(key, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def validate_config(cfg):
    if not isinstance(cfg.h3.get("checkpoint_outside_compile", False), bool):
        raise ValueError("H3 checkpoint_outside_compile must be a boolean")
    buckets = cfg.h3.get("training_shape_buckets", {})
    if not isinstance(buckets, dict) and not OmegaConf.is_dict(buckets):
        raise ValueError("H3 training_shape_buckets must be a mapping")
    if set(buckets) - {"tokens", "timesteps", "cameras", "chunks"}:
        raise ValueError("Unknown H3 training shape bucket")
    if any(type(value) is not int or value <= 0 for value in buckets.values()):
        raise ValueError("H3 training shape buckets must be positive integer multiples")
    if cfg.h3.get("caption_mode") is not None:
        raise ValueError("Caption modes were replaced by per-BD temporal-overlap selection")
    if not isinstance(cfg.h3.get("single_sequence", False), bool):
        raise ValueError("H3 single_sequence must be a boolean")
    for name in ("clean_prefix_probability", "caption_overlap_threshold"):
        value = cfg.h3.get(name, .5)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"H3 {name} must be in [0, 1]")
    for name in ("chunk_group_range", "chunk_size_range"):
        bounds = cfg.h3.get(name)
        if bounds is not None and (not isinstance(bounds, (list, tuple)) and not OmegaConf.is_list(bounds)
                                  or len(bounds) != 2 or any(type(value) is not int for value in bounds)
                                  or not 1 <= bounds[0] <= bounds[1]):
            raise ValueError(f"H3 {name} must be a positive integer [minimum, maximum] pair")
    if cfg.h3.get("chunk_size_range") is not None:
        if not cfg.h3.get("single_sequence", False) or cfg.h3.get("chunk_group_range") is not None:
            raise ValueError("Free native latent chunks require single_sequence and no chunk_group_range")
    if not 0 <= cfg.ema_weight < 1:
        raise ValueError("EMA weight must be zero (disabled) or a decay below one")
    for name in ("inference_weights", "validation_weights"):
        selection = cfg.get(name, "auto")
        if selection not in ("auto", "raw", "ema") or (selection == "ema" and not cfg.ema_weight):
            raise ValueError(f"Invalid {name} selection for the configured EMA")
    if not isinstance(cfg.h3.get("grouped_attention_backward", False), bool):
        raise ValueError("H3 grouped attention backward must be a boolean")
    block_size = cfg.h3.get("training_attention_block_size", (128, 128))
    if (not isinstance(block_size, (list, tuple)) and not OmegaConf.is_list(block_size)) or len(block_size) != 2:
        raise ValueError("H3 training attention block size must be a pair")
    if any(not isinstance(value, int) or value <= 0 or value % 128 for value in block_size):
        raise ValueError("H3 training attention block dimensions must be positive multiples of 128")
    noise_sampling = cfg.h3.get("noise_sampling", "worldviews_ranges")
    if noise_sampling not in ("worldviews_ranges", "shifted_uniform"):
        raise ValueError("Unknown H3 training noise distribution")
    if cfg.model.timestep_shift <= 0 or cfg.timestep_shift <= 0:
        raise ValueError("Flow shifts must be positive")
    if not 0 <= cfg.h3.get("condition_noise", 0.) <= 1:
        raise ValueError("Condition noise must be in [0, 1]")
    if cfg.sampling_solver == "h3_euler" and (cfg.guidance_scale != 1 or cfg.cfg_rescale_factor):
        raise ValueError("H3 Euler inference uses the released CFG-distilled single-forward recipe")
    # The baseline Wan geometry is intentionally retained for source sampling.
    # Only the H3 encoder/transformer interpret the actual H3 latent geometry.
    if list(cfg.model.vae_stride) != [4, 8, 8]:
        raise ValueError("Dataset durations must remain in WorldViews/Wan reference units")
    if any(cfg.model[key] not in ("matrix", "decomposed") for key in ("prope_mode", "mv_prope_mode")):
        raise ValueError("H3 camera encoding supports decomposed or matrix on either original-layer stream")
    if not cfg.model.prope_pure_t or cfg.model.prope_dim != 40:
        raise ValueError("Expected the pure-T, 40-channel matrix recipe")
    if cfg.model.decomposed_rot_mode != "c2w" or cfg.model.decomposed_trans_mode != "C":
        raise ValueError("Expected canonical c2w/C camera encoding")
    if cfg.h3.stage not in (1, 2) or cfg.h3.short_frames < 1:
        raise ValueError("Invalid two-stage curriculum")
    if cfg.model.scale_cond and cfg.h3.scale_conditioning not in ("spatial_rotary", "camera_geometry"):
        raise ValueError("Scale conditioning must use spatial RoPE or scaled camera geometry")
    if not cfg.model.prope_unwrapped and cfg.h3.scale_conditioning != "camera_geometry":
        raise ValueError("Wrapped init parity requires scale to stay in camera geometry")
    if cfg.gradient_accumulation_steps != 1:
        raise ValueError("Use the reference gradient_accumulation_steps=1")
    if cfg.h3.stage1_steps < 1 or cfg.h3.camera_policy != "replace_every_ar_interval":
        raise ValueError("Use a positive short-stage duration and original-layer replacement")
    if cfg.tf_df_mix or cfg.diffusion_forcing or cfg.clean_context_ar:
        raise ValueError("Use the reference teacher-forcing recipe")
    if cfg.model.selective_ckpt or cfg.model.ckpt_group_size != 1:
        raise ValueError("Use the reference checkpoint settings")
    if cfg.dataset.batch_size != 1:
        raise ValueError("Heterogeneous source documents require source batch_size=1")
    if any(cfg[key] for key in ("lr", "qk_lr", "ca_lr", "ca_qk_lr")):
        raise ValueError("The port selects existing attention through ar_lr/sa_lr; other parameter scopes need an explicit mapping")
    if cfg.camera_cfg or cfg.cond_pose_dropout_ratio or cfg.context_image_fill or cfg.context_mask_fill:
        raise ValueError("Use the reference conditioning recipe; H3 joint attention needs explicit mappings for these options")
    if not all(cfg.model[key] for key in ("base_model_history", "base_model_multiview", "ar_model_history", "ar_model_multiview")):
        raise ValueError("The current H3 recipe uses history/multiview attention in both original-layer streams")
    if cfg.resampling_forcing_staircase or cfg.force_clean_history:
        raise ValueError("Use the reference RF/history settings")
    return cfg
