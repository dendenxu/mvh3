# H3 paired convergence experiment

## New DF convergence recipe

`configs/overfit_diffusion_forcing.yaml` preserves the full paired 77-frame video
and native image/video features, but re-draws 3-20-latent blocks before choosing
each clean-prefix cut. `scripts/prepare_df_overfit.py` encodes all contiguous
motion-caption combinations from the exact source row with native Qwen i2v.
Training and inference bind those cached strings after drawing the partition.
There is one video sequence and no clean copy; the noise convention, fixed EMA
0.995, SP8/FSDP8/CPU offload and original attention scope remain unchanged.
This is a new convergence experiment, not a continuation of the TF loss curve.
The new local run is `local/overfit_diffusion_forcing`, using the explicit
native deterministic-backward/dynamic-metadata candidate. It starts EMA at fixed
0.995 immediately, and retains the same config across its step-256 restart.
The bounded diagnostic finishes all 2048 updates and both configured loss
criteria pass. Cross-process reproducibility and production acceptance remain
unresolved; the final convergence result does not override those failed checks.

The initial DF phase completed fixed evaluations through step 256. Raw mean loss
changes from 0.31970188 to 0.21599586 (32.44% lower); fixed-0.995 EMA changes from
0.31943483 to 0.25085571 (21.47% lower). All 256 optimizer updates, 30 numeric
fields, five raw/EMA evaluations with 10 fields, and the scheduled generation's
two media entities pass remote readback in
`local/overfit_diffusion_forcing/tracking_step256.json`.
The Chinese curve, full 77-frame source/before/raw comparison, last-eight-frame
sheet and frozen source data are in `review_step256_zh` under that run. Source,
seed 81000 and partition [16, 6, 5] match the before generation. Decoded compressed
video MSE against original source pixels falls from 0.09778197 to 0.07955490;
last-four-frame MSE falls from 0.13476828 to 0.08051471. No terminal block collapse
appears in the inspected tail, but motion, faces and text still differ. There is
no EMA video at this intermediate milestone. The first process saved checkpoint
256 and exited naturally. The fresh process then failed its EMA evaluation gate:
relative error 0.0011188091 exceeds the unchanged 0.001 tolerance; raw evaluation
passes, and no step-257 update occurs. See `resume_000000256_failed.json`.
`local/df_resume_eval_probe_v1` then passes exact raw/AdamW/EMA restoration on all
eight ranks. Four raw and four EMA evaluations are identical within that process
before/after a backward without an optimizer step, and all pass the original
0.001 gate. Raw shards remain exact and EMA count remains 256. All eight diagnostic
rows pass remote tracking readback.

A normal retry nevertheless fails the unchanged EMA gate again: relative error
0.0010497402, while raw passes at 0.0001080527. See `resume_000000256.json` and
`continuation_supervisor.json`; neither normal retry performs update 257. The
independent diagnostic does not explain these failures. Completed full-model
traces first differ at block zero, with identical packed inputs and preceding
encoders. A separate full-topology, truncated-computation control reproduces
compiled block-zero drift with exact camera/mask/block inputs, whereas its eager
block output matches across processes. Precision-cast emulation also drifts and
has not been adopted. CPU copies can affect scheduling; the cause is unproved.

An explicit bounded convergence continuation starts on September 12 at 02:55
UTC+8 under `diagnostic_supervisor.json`. It uses
`--diagnostic-resume-after-eval-drift REASON --stop-after 2048` with the identical
candidate recipe and feature digest. Every raw weight, AdamW state and EMA value
is checked exactly on all eight ranks before evaluation; these state checks pass.
The original 0.001 evaluation tolerance and both failed normal receipts are
preserved. A diagnostic can continue finite evaluations after an exact restore,
but its report/checkpoints/Tracking retain `diagnostic_only` and failed prior
acceptance. Nonfinite evaluations or any rank's state mismatch still stop it.
The original step-256 checkpoint is protected from retention cleanup. This does
not accept the multi-step trajectory or 64-GPU production.

The diagnostic continuation completes all 2048 updates at 06:16 UTC+8 on
September 12, exit 0. Its fresh step-256 raw/EMA evaluation
passes the original gate at relative errors 0.00023013/0.00031016. This isolated
pass does not resolve the earlier failures. Final raw fixed loss is 0.15085705,
52.81% below initial; EMA is 0.15315709, 52.05% below initial. All 2048 training
losses and gradient norms are finite; all EMA updates use 0.995. The last 256
updates have a 4.528-second median, scoped to this fixed-feature single video.
All 2048 updates, 30 numeric fields, 33 raw/EMA evaluations, ten evaluation fields
and seven final media entities pass remote readback in `tracking_complete2048.json`.
The byted-wandb run `8eqvw7ib` is finished.

`review_step2048_zh` contains the final Chinese comparison video (source, before,
raw 2048 and EMA 2048), loss curve, timeline, tail and frozen evidence. All 77
frames decode at 16 FPS, using the same source, seed 81000 and partition [16, 6, 5].
Compressed-MP4 pixel MSE against uncompressed source pixels is 0.09778197 before,
0.02595861 raw and 0.02890094 EMA; last-four-frame MSE is 0.13476828, 0.02038987 and
0.02216735 respectively. The whole-sequence frame sample and every last eight
frames show improved scene/person continuity with no terminal block collapse.
Hands, faces, motion details and text still differ; this is not exact memorization
or a generalization test. Raw is currently better than EMA on these metrics.

`completion_watch.json` records completed CPU rendering and remote readback.
`completion.json` and `diagnostic_supervisor.json` confirm successful process
completion, while retaining failed historical restoration acceptance. The
overfit script now refreshes terminal `progress.json` after successful review;
the completed run's stale running snapshot has been refreshed from its final
training report. This changes status reporting only.

Final Chinese video, curve, timeline and tail are published in the
[H3 experiment document](https://bytedance.larkoffice.com/docx/SXv0dqwVfojGXPxzZSvcb82ynoh)
and read back at revision 229. `review_step2048_zh/publish_receipt.json` verifies
the video preview, media order and exact preservation of 19 other blocks,
including 16 historical media blocks and the unresolved-restoration diagnosis.

The launcher's bounded first phase exits before model initialization when its
committed checkpoint already reaches `--stop-after`; a resumed 256-step run no
longer repeats full training inside that phase. All 199 CPU tests pass, including
negative controls for corrupted raw/AdamW/EMA and nonfinite diagnostic evaluation.

## Historical TF single-video acceptance run

`configs/overfit_sequence.yaml` selects one complete paired Ours window at
77 frames, 16 FPS and 448x832. It retains all-frame decomposed wrap, CPU-offloaded
SP8/FSDP8, checkpointing and compile, and uses the validated native FA4
grouped-backward path. `local/overfit_sequence_grouped/features` refreshes the first-frame
VAE posterior with the native CPU seed 42 while retaining the exact source
pixels, video latents, captions and camera geometry.

The latest recipe enables trainable-shard FP32 CPU EMA with constant decay 0.995
and no EMA warmup. This completed experiment started with update-count warmup and
a 0.999 cap; its 256-step restore changed the cap to 0.995 before either cap had
affected the EMA trajectory. The user's later constant-decay selection is
completed at checkpoint 768, retaining raw weights, AdamW, EMA and update count.
`fixed_ema_transition.json` and `resume_000000768.json` verify the transition;
do not claim the earlier steps used constant decay.
Both raw and EMA fixed-noise losses are measured every 64 updates.
The runner checks a fresh-process restore at 256 and runs 2048 updates, with
intermediate videos every 256 and raw/EMA final generations. Check actual
completion and review receipts before accepting convergence or launching the
64-GPU full-data run. Original batch capacity is tested separately; selecting a
single sequence for convergence is not a capacity reduction.

The fresh-process 256-step restore passes with cap 0.995: raw/EMA fixed-noise
losses differ by 0.0435%/0.0254% from the preceding process, within the 0.1%
tolerance. The historical TF run completed 2,048 updates and final raw/EMA
generation. Fixed-noise loss falls from 0.33376218 to 0.12852090 raw and 0.13126909
EMA. On this same training video, pixel MSE changes from 0.10805651 before training
to 0.01576258 raw and 0.01714134 EMA; this is not held-out scene quality.
All 2,048 optimizer steps, 24 numeric fields, 33 raw/EMA evaluations and seven
final media entities pass remote byted-wandb readback with remote status finished.
Reports are `completion.json`, `review/review_report.json` and
`tracking_complete2048.json` under the run directory. This establishes the old
TF experiment's convergence, not acceptance of the new DF recipe.

```bash
bash scripts/run_overfit.sh local/overfit_sequence_grouped \
  local/overfit_sequence_grouped/features configs/overfit_sequence.yaml

python scripts/verify_experiment_tracking.py --run-dir local/overfit_sequence_grouped \
  --through-step 256 --expected-steps 256 \
  --evaluations local/overfit_sequence_grouped/progress.json
```

## Current paired Ours recipe

Use `configs/overfit_paired.yaml` with the existing paired Ours feature documents:
two complete 77-frame windows at 16 FPS and 448x832, with their exact global/chunk
captions re-encoded by native H3 image/text conditioning. All original attention
layers use all-frame decomposed wrap, with five frequencies from 0.01 to 32 and
PSF/scale conditioning disabled. The original every-second-attention scope stays
3,853,523,200 parameters at 1e-5, with no added parameters.

```bash
bash scripts/run_overfit.sh local/overfit_paired local/overfit_paired/features configs/overfit_paired.yaml
```

This diagnostic runs 2048 updates, evaluates every 64, checkpoints every 256 and
generates intermediate videos every 512. It saves at 256 and starts a new process
to check the restored fixed-noise evaluation before continuing. Context noise and
history dropout inherit WorldViews; RF stays off for this diagnostic. This does
not choose the separate full-mixture stage boundary or RF warmup. The generator
shards use CPU offload. The separate original batch-envelope check also runs
the live Qwen/VAE encoders.

Old scaled image/text/latent feature caches remain usable: their recorded PSF is
undone for camera centers and independent projection/inverse translations before
packing. Existing training checkpoints with a different encoding or PSF recipe
cannot silently resume. Check `completion.json` and `review/review_report.json`
for actual completion, then read back every numeric training field and review
media from internal Tracking:

```bash
python scripts/verify_experiment_tracking.py --run-dir local/overfit_paired \
  --expected-steps 2048 --media local/overfit_paired/review
```

The source-row, 39-frame, 24-FPS experiments below are historical and do not
describe the current paired Ours recipe.

## Historical native probes

The completed 128-step result below is historical: its Qwen features were text only, although the first frame was encoded by the VAE. It is not a full native FL2VA i2v reproduction. The new `configs/overfit_i2v.yaml` uses both image and caption through full Qwen3-VL and the native first-frame VAE conditioner. Its planned 2048 steps retain inherited camera/history/RF settings, evaluate every 64 steps, checkpoint every 256, and generate intermediate samples every 512. The runner exits at step 256 and starts a fresh process to verify restore before continuing. The run is not complete merely because this config exists.

This is a bounded overfit check of the released full FL2VA model on two fixed real monocular documents. It retains the MVH3 analytic camera encoding, causal chunks and alternating original-attention training scope. It adds no model parameters. It is separate from the full-mixture two-stage `worldviews.yaml` recipe.

## Numerical settings

The original checkpoint's `model_index.json`, `_minimax_h3.sigma_shift_scales.video`, specifies shift 12. The pinned H3 scheduler at Diffusers revision `d30c748f5f5d0925a5af14dc0e6a6de983025e63` defines `t=1-sigma`, clean at `t=1`, and data-ward velocity `clean-noise`. Its Euler schedule uses 50 sigma points including terminal zero, hence 49 model evaluations per chunk. Released inference is CFG-distilled and uses guidance 1. Video timing is 24 FPS; the experiment uses a 768 x 1344 canvas. The pinned conditioning pipeline draws image augmentation noise once per request, at sigma 0.001, and uses posterior seed 42 when encoding the image.

These values are in `configs/h3_native.yaml`. H3's original training distribution and optimizer recipe are not published. Uniform `u` followed by `sigma=12*u/(1+11*u)`, plain MSE, the fixed sample selection, and the overfit optimizer are explicit experimental choices. They must not be described as a reproduction of the original H3 training recipe.

## Fixed documents and measurements

`configs/overfit.yaml` selects source row 0 and camera views 0/1 as two independent monocular documents. Each contains 39 physical sampled frames at 24 FPS. The 16-FPS source is explicitly resampled; repeated indices are recorded in `features.json`. Each document uses the complete H3 video VAE and the full Qwen3-VL layer-50 caption embedding. Feature preparation verifies the reused text caption and encoder checkpoint.

The run starts from original FL2VA weights and trains 3,853,523,200 original attention parameters, with `ar_lr=1e-5` and FP32 AdamW. All other original parameters remain frozen. It uses SP8/FSDP8, BF16 forward, compiled blocks and activation checkpointing. The fixed features allow parameter shards to remain on GPU. RF, history dropout and context noise are disabled for this diagnostic.

The first run has 128 optimizer steps with fresh training noise. Evaluation every 16 steps uses sigmas 0.25, 0.5, 0.75 and 0.95, with a separate fixed seed stream. A 50% mean evaluation-MSE decrease is the configured diagnostic criterion; the report records whether it passes. Before/after generation uses the same seed per document and the same native schedule. Decoded source/before/after videos and contact sheets are separate review artifacts. Pixel similarity on these two training clips is not evidence of held-out scene generalization.

## Execution and recovery

Set the checkpoint, converted VAE, dataset-root environment variables from `.env.example`, and optionally `MVH3_PYTHON` to the desired Python interpreter. Refresh `local/overfit_native/features` with `scripts/prepare_i2v_features.py --source local/overfit_native/features --output local/overfit_i2v/features`, then run:

```bash
bash scripts/run_overfit.sh
```

The runner takes optional output, feature-directory and config arguments, holds a filesystem lock against duplicate writers, records each attempt in a distinct log, and automatically resumes only its own latest committed checkpoint. Checkpoints contain all trainable shards, AdamW, RNG and the experiment report. Resume verifies the recipe, SP/FSDP topology and the exact feature SHA256. An interrupted shard generation never replaces the previous committed manifest. Historical text-only checkpoints are a different experiment and cannot initialize this continuation.

Training reports and generated latents use atomic writes. Rank 0 decodes source/before/after comparisons and plots the loss curves before closing the online Tracking run, so media and metrics use the same run ID. `latest.log` names the current attempt; its matching `.exit` file records the complete runner's exit code. A training report alone does not establish that decoding succeeded: check `completion.json` and `review/review_report.json` too.

The current host uses the explicit `MVH3_NCCL_ABORT_ON_EXIT=1` workaround documented in `MVH3.md`. The wrapper does not submit, stop or replace any scheduler allocation.

## Completed 128-step result

The first native overfit run completed training, both before/after generations and decoded reviews on 2026-09-10 (UTC+8), with runner exit 0. All 128 losses and gradient norms stayed finite; frozen-weight samples remained unchanged. The 40 existing CPU tests and four new native-noise/flow/condition/scheduler regressions passed before the run.

| Measurement | Before | Step 128 |
| --- | ---: | ---: |
| Mean fixed-noise MSE, 2 clips x 4 noise levels | 0.23764654 | 0.15136215 |
| Source-view 0 generated-video PSNR | 8.45218 dB | 17.59347 dB |
| Source-view 1 generated-video PSNR | 10.14482 dB | 19.17022 dB |

Fixed-noise MSE falls by 36.3079%, so the configured 50% reduction criterion is **not met**. Both decoded generations fit the source scene much more closely, but still contain blur and motion/detail errors. These are training-clip results, not a generalization claim. High-noise error improves more than low-noise error; extending the same diagnostic can distinguish additional learning from a plateau.

The completed checkpoint is `local/overfit_native/ckpt/step_000000128_ffa3d04d86ad`, with eight weight/AdamW shards totaling about 43.07 GiB. `ckpt/latest.json` selects it. Runtime reports, exact feature/code provenance, the loss curve and two source/before/after comparison videos live under `local/overfit_native`. These artifacts remain outside Git. Steady optimizer-step median is 2.24 seconds on eight H100s; this fixed-feature timing excludes loading, evaluation, saving, generation and live encoders. Rank-0 peak allocated memory is 17.80 GiB and is not the node-wide reserved-memory peak.
