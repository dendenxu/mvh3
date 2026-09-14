# Single-sequence convergence validation

The reusable runner is `scripts/overfit.py`, configured by
`configs/overfit_diffusion_forcing.yaml`. It trains one complete paired Ours
77-frame sequence at 16 FPS and 448x832 with the native H3 VAE and Qwen image/text
features. A fresh latent partition is drawn before each clean-prefix cut.

`scripts/prepare_overfit.py` builds the contiguous caption feature bank from the
exact source row and previously encoded video. Temporal-overlap caption selection
then uses the same function as live data loading; there is no numbered caption
format or duplicated clean video.

```bash
bash scripts/run_overfit.sh local/overfit_diffusion_forcing \
  local/overfit_diffusion_forcing/features configs/overfit_diffusion_forcing.yaml
```

The launcher runs 256 updates, saves, then starts a fresh process and continues
to 2048. The resume check compares raw weights, AdamW, EMA and fixed-noise loss.
Raw/EMA evaluations run every 64 updates and generations every 256. EMA decay is
fixed at 0.995. Use the full Stage 1 context-noise/compile overrides when checking
that production recipe, and save the resolved config beside the results.

Read the run's `completion.json`, loss curve, before/raw/EMA videos and final
frames. Convergence on the training sequence does not establish held-out camera
control. A successful exit and completed remote byted-wandb readback are required
before treating an experiment as complete.

Full-batch capacity, native initialization parity and 64-GPU throughput are
separate checks. Temporary scripts for those checks and their outputs belong in
ignored `local/`, not in the source tree. Preserve the exact input/RNG and recipe
when comparing implementations; do not relax tolerances to accept a failed run.
