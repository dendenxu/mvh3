# MVH3

Camera-conditioned MiniMax H3 training with the WorldViews data and training recipe.
The model changes analytic camera encoding inside existing H3 attention layers;
its parameter names, shapes and count stay unchanged. No additional AR branch,
adapter, LoRA, gate, or camera MLP is introduced.

The core model is ordinary PyTorch. Transformer, attention, embeddings, video
VAE, and schedulers live here and can be edited here. Diffusers is used only by
optional reference/conversion checks, not by the training or inference runtime.

## Structure

```text
main.py                         # same config entry pattern as WorldViews
configs/worldviews.yaml         # complete reference recipe + explicit H3 overlay
configs/diffusion_forcing.yaml  # single sequence, random BD blocks, clean-prefix cut
configs/worldviews_stage2.yaml  # continuation on full-duration/multiview samples
trainer/diffusion.py            # data loop, optimizer, RF, validation, resume
model/diffusion.py              # single-sequence DF and historical TF compatibility
pipeline/ar_inference.py        # chunk rollout, CFG, UniPC, KV history
h3/modules/model.py             # native packed video/text/audio transformer
h3/modules/attention.py         # SDPA and compiled flex kernels
h3/modules/camera.py            # matrix and decomposed PRoPE
h3/modules/vae.py               # CNN encoder, ViT decoder, tiling and stitching
h3/distributed/fsdp.py          # FSDP, Ulysses, checkpointing, compilation
h3/utils/                      # native scheduler and WorldViews UniPC/DPM
h3/checkpoint.py                # strict original-weight streaming conversion
utils/h3_wrapper.py             # pixels/cameras -> H3 latents and Qwen features
utils/checkpoint.py             # atomic sharded AdamW/weight/runtime checkpoints
dataset/                       # all WorldViews source-family samplers
utils/                         # original data/distributed/compile optimizations
scripts/                       # bounded verification and feature preparation
tests/                         # numerical and behavioral regression tests
```

## Recipe

`configs/worldviews_reference.yaml` is the fully resolved WorldViews YAML,
including the train/validation mixtures and all sampling/augmentation settings.
`configs/presampled_data.yaml` selects the exact paired Ours SHORT/FULL Parquets
used by the 1.3B runs, including their original batch groups and shape remaps.
The older 19-source loader remains available for historical experiments.

The default `configs/diffusion_forcing.yaml` first draws each BD block size in
3-20 native H3 latents, including decoder support in the final real block, then
chooses one clean-prefix cut at an existing boundary. The suffix receives
independent per-block diffusion noise and loss; the clean prefix has zero loss.
There is one video sequence, without a duplicated clean context. Caption prose
contains the scene and source motions with strictly greater than 50% temporal
overlap. A block covering no majority window gets scene text only. Video and
text use the same block-causal visibility, including the text refiner.

- Stage 1: every sampled source view becomes independent monocular clips of at
  most 77 actual frames. Every view and partial tail is retained.
- Stage 2: continue the model, optimizer and global step on the original full
  duration/view distribution. A stage-1 checkpoint is required.
- All original H3 attention layers default to wrapped decomposed PRoPE using
  each token's actual camera; the conditioning camera is only the relative
  reference. `camera_matrix.yaml` and `camera_alternating.yaml` preserve matrix
  and alternating alternatives, also wrapped and time-varying. Every
  `model.ar_interval=2` original attention still trains at `ar_lr=1e-5`, with
  other original weights frozen at `sa_lr=0`. Total parameters: 33,122,992,896;
  trainable original parameters: 3,853,523,200; added parameters: zero.
- Default decomposed translation uses five log-spaced frequencies from 0.01 to
  32 with PSF and scale conditioning disabled. Both training stages and validation
  use unscaled camera translation; matrix/alternating alternatives retain PSF.
- The default short-stage duration is an explicit **provisional 10,000 steps**.
  It is configurable and has not been chosen from H3 learning curves. RF
  warmup is a separate, undecided schedule; do not treat the inherited 10,000
  steps as an accepted H3 optimum.

H3 has different latent geometry, joint text attention and a single denoiser.
[The port guide](docs/MVH3.md) explains these unavoidable architecture mappings,
including camera units and optional zero-parameter scale conditioning. Equal YAML values do not imply
numerically identical Wan and H3 models.

## Setup and launch

Use the existing WorldViews environment and its prebuilt FA4 runtime:

```bash
export MVH3_CHECKPOINT=/path/to/MiniMax-H3/FL2VA
export MVH3_VAE=/path/to/converted/video/vae
export MVH3_DATA_ROOT=/path/to/first/dataset/root
export MVH3_DATA_ROOT3=/path/to/second/dataset/root
python main.py -c configs/diffusion_forcing.yaml --print-config
torchrun --nproc_per_node=8 main.py -c configs/diffusion_forcing.yaml
```

For the current host, export the variables listed in `.env.example` (including the
explicit NCCL exit workaround described in the port guide).

The downloaded local bundle can use sibling `python_deps` without upgrading the
original WorldViews environment. Original FL2VA weights load directly; the video
VAE uses the strict converted names produced by `scripts/prepare_real_probe.py`.
No model weights or dataset files are stored in this Git repository.

Stage 2 runs automatically at `h3.stage1_steps`, or explicitly:

```bash
torchrun --nproc_per_node=8 main.py -c configs/worldviews_stage2.yaml \
  resume_ckpt=/path/to/ckpt/latest.json
```

Resume requires the same FSDP/SP topology. It restores trainable shards, AdamW,
step, RNG and pending source/RF queues; it does not promise an identical future
DataLoader prefetch cursor across fresh processes.

## Validation

For the current paired Ours diagnostic (two 77-frame windows at 16 FPS, 448x832,
native Euler, shift 12 and CFG 1), see [the overfit guide](docs/OVERFIT.md).
`scripts/run_overfit.sh` runs the diagnostic, saves resumable AdamW checkpoints,
and decodes fixed-seed before/after comparisons. Its measured results and limits
are recorded separately from the full-mixture training recipe.

```bash
pip install -e '.[test]'
python scripts/test_cpu.py
python scripts/verify_worldviews_config.py --reference /path/to/worldviews/configs/worldviews.yaml
python scripts/verify_worldviews_data.py --output local/data_decode.json
torchrun --standalone --nproc_per_node=2 scripts/verify_worldviews_runtime.py \
  --compile --inference --output local/runtime_tiny
```

See [validation details and limitations](docs/MVH3.md) before interpreting these
checks as training readiness. Only a completed report and clean process exit
establish a successful run. Short diagnostic updates do not establish convergence
or camera-control quality.

## Image and camera inference

The image-camera request interface uses the released FL2VA conventions: image features
from both the video VAE and full Qwen3-VL layer 50, video only, distilled CFG 1,
shift 12, 24 FPS, and 50 Euler sigma points (49 evaluations). Training preserves
the paired data's actual 16 FPS; the temporal clock uses the real sampled FPS. The untouched
WorldViews snapshot is `worldviews_reference.yaml`; the config verifier checks
its exact identity and lists the native H3 adaptations separately.

`scripts/infer.py` takes a request containing only an input image, prompt and
camera path. It constructs target shapes without reading future video.

The default `--protocol auto` uses native full-sequence joint sampling with
base weights, preserving the initialization comparison. Loading a training
checkpoint selects the trained AR rollout. Explicit `--protocol joint` and
`--protocol ar` remain available for controlled comparisons.

```json
{
  "prompt": "A street viewed from a moving car.",
  "fps": 24,
  "views": [{"image": "input.png", "camera": "camera.npy", "scale": 1.0}]
}
```

Each camera file is float32 `[frames, 10]`, in the WorldViews runtime convention:
`[fx/width, fy/height, cx/width-0.5, cy/height-0.5, rotvec(R_c2w), C_world]`.
Translations are world-locked; the default uses unscaled dataset units and
`scale=1.0`. Old scaled camera files must record the applied pose-stable factor
in `scale`; the PSF-off recipe restores their translations automatically.
Additional views can specify another image or just `height`/`width`.
Paths are relative to the request JSON. Use the checkpoint's exact resolved
training config; loading weights does not construct an optimizer or dataset.

```bash
torchrun --standalone --nproc_per_node=8 scripts/infer.py \
  --config local/experiment/resolved.yaml --checkpoint local/experiment/ckpt/latest.json \
  --request request.json --output local/inference --protocol ar
```

`--protocol joint` selects the native whole-sequence sampler; `--no-camera` is
its initialized base-model control. AR rollout intentionally has chunk-causal
visibility. The canonical recipe wraps decomposed cameras in every original
attention layer, using all temporal poses. `--verify-init` compares every native
joint prediction with the no-camera base on a static-camera request. Temporal
padding remains active decoder support; output keeps the exact requested duration.

Tracking uses WorldGen's `byted-wandb` distribution imported as `wandb`, online
project `wgl`, entity `zhenxu.zx`. Run identity, source snapshot, scalar metrics,
fixed-noise evaluations, decoded media and checkpoint manifests are persisted.
Internal readback uses `wandb.TrackingPublicApi`, not the public `wandb.Api`.

## ByteDance commit hook

Like WorldViews, `post-commit` mirrors each committed tree into an independent
ByteDance snapshot history. The target is explicitly
`git@code.byted.org:zhenxu.zx/mvh3.git`, branch `main`, regardless of the checkout
directory name. The helper uses a locked bare cache at `~/.gitlab/mvh3.git` and
normal fast-forward pushes. Uncommitted files are excluded.

```bash
install -m 755 scripts/merlin/post-commit "$(git rev-parse --git-path hooks)/post-commit"
install -m 755 scripts/merlin/commit-msg "$(git rev-parse --git-path hooks)/commit-msg"
bash scripts/merlin/push2bd --dry-run
```

The commit-message hook applies WorldViews' GitHub-to-ByteDance link rewrite.
If a push fails after a successful local commit, retry with
`bash scripts/merlin/push2bd`; the local commit is already saved.

## Upstream

Based on [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3), snapshot
`d21241f0a4b3acbb34c97dae47fa417b7065e438`, and H3 implementations from Diffusers
`d30c748f5f5d0925a5af14dc0e6a6de983025e63`. Source copyrights and licenses remain.
[Original MiniMax README](README.upstream.md).

Official H3 supports SGLang, vLLM, Diffusers and ComfyUI; SGLang is its README
example. Diffusers is not a requirement of H3 training. The official repository
contains raw video/audio VAE code and checkpoint configs; it does not release a
standalone raw transformer training implementation. This work adapts the released
CFG-distilled H3-Base FL2VA model. H3-Context-IR and H3-Regenerate-2K are API-only.
