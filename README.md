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
configs/worldviews_stage2.yaml  # continuation on full-duration/multiview samples
trainer/diffusion.py            # data loop, optimizer, RF, validation, resume
model/diffusion.py              # weighted flow matching and teacher forcing
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
The only local-data changes are environment-variable substitutions. All 19
training sources remain enabled, including the largest dynamic sources.

- Stage 1: every sampled source view becomes independent monocular clips of at
  most 77 actual frames. Every view and partial tail is retained.
- Stage 2: continue the model, optimizer and global step on the original full
  duration/view distribution. A stage-1 checkpoint is required.
- Every `model.ar_interval=2` existing H3 main attention uses matrix PRoPE and
  `ar_lr=1e-5`; intervening original attention uses first-frame decomposed PRoPE
  and remains frozen with `sa_lr=0`. Total parameters: 33,122,992,896; trainable
  original parameters: 3,853,523,200; added parameters: zero.
- The default short-stage duration is an explicit **provisional 10,000 steps**.
  It is configurable and has not been chosen from H3 learning curves.

H3 has different latent geometry, joint text attention and a single denoiser.
[The port guide](docs/MVH3.md) explains these unavoidable architecture mappings,
including zero-parameter scale conditioning. Equal YAML values do not imply
numerically identical Wan and H3 models.

## Setup and launch

Use a dedicated environment with a suitable PyTorch/CUDA build:

```bash
pip install -e '.[train]'
export MVH3_CHECKPOINT=/path/to/MiniMax-H3/FL2VA
export MVH3_VAE=/path/to/converted/video/vae
export MVH3_DATA_ROOT=/path/to/first/dataset/root
export MVH3_DATA_ROOT3=/path/to/second/dataset/root
python main.py -c configs/worldviews.yaml --print-config
torchrun --nproc_per_node=8 main.py -c configs/worldviews.yaml
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
