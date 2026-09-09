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


def recipe_digest(cfg):
    """Allow stage/log paths to change, but reject accidental recipe drift on resume."""
    data = OmegaConf.to_container(cfg, resolve=True)
    h3 = data.get("h3", {})
    for key in ("checkpoint", "vae", "text_cache", "compile_cache", "logdir", "stage"):
        h3.pop(key, None)
    for key in ("resume_ckpt", "auto_resume", "max_iters", "task"):
        data.pop(key, None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def validate_config(cfg):
    # The baseline Wan geometry is intentionally retained for source sampling.
    # Only the H3 encoder/transformer interpret the actual H3 latent geometry.
    if list(cfg.model.vae_stride) != [4, 8, 8]:
        raise ValueError("Dataset durations must remain in WorldViews/Wan reference units")
    if cfg.model.prope_mode != "matrix" or cfg.model.mv_prope_mode != "decomposed":
        raise ValueError("This recipe requires WorldViews matrix + decomposed PRoPE")
    if not cfg.model.prope_pure_t or not cfg.model.prope_unwrapped or cfg.model.prope_dim != 40:
        raise ValueError("Expected the current pure-T, unwrapped 40-channel matrix recipe")
    if cfg.model.decomposed_rot_mode != "c2w" or cfg.model.decomposed_trans_mode != "C":
        raise ValueError("Expected canonical c2w/C camera encoding")
    if cfg.h3.stage not in (1, 2) or cfg.h3.short_frames < 1:
        raise ValueError("Invalid two-stage curriculum")
    if cfg.model.scale_cond and cfg.h3.scale_conditioning != "spatial_rotary":
        raise ValueError("Scale conditioning must use existing spatial RoPE channels")
    if cfg.gradient_accumulation_steps != 1:
        raise ValueError("Use the reference gradient_accumulation_steps=1")
    if cfg.h3.stage1_steps < 1 or cfg.h3.camera_policy != "replace_every_ar_interval":
        raise ValueError("Use a positive short-stage duration and original-layer replacement")
    if cfg.tf_df_mix or cfg.diffusion_forcing or cfg.clean_context_ar:
        raise ValueError("Use the reference teacher-forcing recipe")
    if cfg.ema_weight or cfg.model.selective_ckpt or cfg.model.ckpt_group_size != 1:
        raise ValueError("Use the reference EMA/checkpoint settings")
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
