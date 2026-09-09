# MVH3 agent guide

Read `docs/MVH3.md` before changing the training port.

- Work in this H3 checkout. `origin` is `https://github.com/dendenxu/mvh3`; `upstream` is the official MiniMax repository.
- Camera conditioning must add zero trainable parameters. Change analytic encoding on the existing attention Q/K and train existing H3 weights. Do not add AR branches, LoRA, camera MLPs, adapters, gates, or expanded projections.
- Use two stages on the same full WorldGen source mixture: short monocular clips first, then full-duration/multiview continuation. In stage 1 every view is an independent batch item, never a spatially packed joint-attention view. Preserve source views and partial tails.
- Use the frozen Wigner bases in `mvh3/wigner_bases.json`; never regenerate them.
- Native H3 RoPE has split-half T/H/W pairs. Camera transforms must preserve T and the unrotated tail. Camera schema is normalized-intrinsic c2w `[fx, fy, cx, cy, rotvec(R_c2w), C_world]`.
- Native H3 joint attention includes text/audio. Causal/view masks must cover all modalities and the text refiner. Validate future-perturbation invariance across multiple layers.
- H3 uses clean timestep `t=1` and data-ward target `clean-noise`. Use `t=1-sigma`, including noisy context; text inherits the generated-video timestep. Check against the pinned H3 scheduler rather than copying Wan conventions.
- GPU block masks require compiled flex attention. Never allow eager flex fallback on long sequences; bounded validation must accommodate device/shape compile variants and actually test the stage transition.
- Keep source-data paths in local configuration/environment variables. Never commit weights, data, caches, credentials, or local runtime reports.
- Run `python scripts/test_cpu.py` for model/mask/curriculum changes. Tiny CPU tests do not establish GPU memory, pretrained quality, or distributed correctness.
- Do not launch or stop cluster training without an explicit user instruction to do so. In particular, no `mlx job submit/stop`, `hr set/restart/stop/nuke`, or broad process kills.
- Keep code, comments, and durable notes in English. User-facing times use UTC+8.
