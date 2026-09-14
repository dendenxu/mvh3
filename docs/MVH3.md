# MVH3 code guide

MVH3 trains the existing MiniMax H3 attention weights on WorldViews data. The
layout follows WorldViews: the trainer runs updates, the model defines the
objective, pipelines generate videos, and datasets supply examples. Native H3
layers and kernels live in `h3/`. There is one root `utils/` for shared helpers;
there are no nested utility or vendor trees. Imports are grouped and sorted by
length. Long workflows have a blank line before each phase's Step comment.
Business terms such as resampling forcing are written out in full.

## Start here

| Responsibility                                   | Code to read                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------ |
| Config loading and task selection                | `main.py`, `utils/config.py`                                             |
| Training loop and one optimizer update           | `trainer/diffusion.py`: `DiffusionTrainer.train_loop`, `train_step`      |
| Source loading, encoding and SP sample queue     | `dataset/loader.py`: `BatchLoader.next`                                  |
| Video VAE and native Qwen image/text features    | `h3/encoders.py`                                                         |
| Chunk boundaries, clean cut and caption overlap  | `model/chunks.py`, `utils/captions.py`                                   |
| Noise sampling, flow loss and resampling forcing | `model/diffusion.py`: `DiffusionObjective`                               |
| Joint token order, masks, camera/time tables     | `model/packing.py`: `SequencePacker`                                     |
| Image/camera request to saved videos             | `pipeline/inference.py`: `run_inference`                                 |
| Native joint sampling / trained chunk rollout    | `pipeline/full_sequence_inference.py`, `pipeline/chunked_inference.py`   |
| Native denoiser, camera encoding and attention   | `h3/modules/model.py`, `camera.py`, `masking.py`, `grouped_attention.py` |
| Camera modes and trainable attention             | `h3/modules/model.py`: `MiniMaxH3Transformer3DModel.configure_attention` |
| Optimizer parameter groups                       | `trainer/diffusion.py`: `parameter_groups`                               |
| FSDP, checkpointing and compilation              | `h3/distributed/fsdp.py`                                                 |
| Checkpoint state and averaged weights            | `utils/checkpoint.py`, `utils/ema.py`                                    |

`scripts/infer.py` is a command-line adapter to `pipeline.inference.run_inference`.
Launch it with `torchrun --module scripts.infer` from the repository root.
`main.py` calls that same pipeline for an `inference_request`; dataset validation
uses `DiffusionTrainer.validate`. Scripts contain reusable experiment commands,
not implementations imported by training. Temporary verification probes stay
under ignored `local/`.

The network is `MiniMaxH3Transformer3DModel` in `h3/modules/model.py`. Its
`__init__` defines input projections, the text refiner, Transformer blocks and
output heads; `from_pretrained` loads the original checkpoint, and `forward`
runs the layers. Call `configure_attention` on that model
before `wrap_model`. FSDP handles distribution, while the model owns its camera
modes and trainable layers.

## Configs

Parents merge left to right, then command-line dot-list overrides apply. The
current full-data Stage 1 recipe follows this chain:

```text
worldviews_reference.yaml + presampled_data.yaml
  -> worldviews.yaml               native H3, wrapped decomposed cameras, PSF off
  -> worldviews_grouped.yaml       grouped FA4 backward, fixed EMA 0.995
  -> diffusion_forcing.yaml        one sequence, random blocks, clean-prefix cut
  -> stage1_diffusion_forcing.yaml resampling forcing off, context noise on, stay in Stage 1
  -> stage1_compile_buckets.yaml   padding buckets and checkpoint/compile order
```

`main.py` defaults to `diffusion_forcing.yaml`. Select
`stage1_compile_buckets.yaml` explicitly for the full-data Stage 1 run.
Continue a diffusion forcing checkpoint with its saved `resolved.yaml` and `h3.stage=2`.
Stage transition, learning rate warmup and resampling forcing warmup are separate settings.

## Training data flow

1. `BatchLoader` samples the paired Ours Parquets using the existing WorldViews
   dataset classes. Stage 1 keeps SHORT views independent within the same update;
   Stage 2 uses FULL windows. Original batch groups and shape remaps are retained.
2. `VideoEncoder` encodes each view separately, including its conditioning image
   and actual camera timeline. The native VAE supplies 24-channel latents.
3. `prepare_chunk_plan` partitions the full latent sequence into blocks of 3-20
   latents. `caption_specs` selects ordinary scene/motion prose for those blocks.
   Qwen3-VL layer 50 encodes each caption with its conditioning image.
4. Each SP rank prepares its own source sample. `gather_mixed_batch` queues every
   sample on every SP rank. `DiffusionObjective.prepare_document` then chooses a
   clean-prefix cut at an existing block boundary.
5. The objective samples noise and supplies latents to `SequencePacker`. The
   packer lays out text, image-condition and video tokens, then inactive padding.
   Its records map output tokens back to the original view and latent frames.
6. The model predicts `clean - noise`. Loss covers the noisy suffix, including
   decoder-support latents. The trainer clips gradients, updates AdamW, updates
   EMA once, then decides whether resampling forcing will reuse the sample.

A document is a dictionary with `views`, `isolated` and `source`. Each encoded
view carries `latent`, `condition`, `text`/`texts`, camera geometry, temporal
coordinates and spatial loss weights. `generation_chunks` maps every latent to
a block; `clean_prefix_chunks` is the cut. These fields also travel with pending
resampling forcing/checkpoint state. Packing uses these concrete tensors directly.

Source captions retain their original Wan windows: the first has 17 source
frames, later windows have 20. Include a motion sentence only when a generated
block covers strictly more than half its source window; exact ties are excluded.
SHORT uses its already-sliced captions. Select captions before the clean cut,
without numbering or artificial markers.

Noise draws retain a fixed order: partition, clean cut, chunk sigmas, history
dropout, context levels, image-condition noise, then context/target noise per
view. Reordering these changes continuation even with the same seed.

## Native H3 contracts

- H3 uses clean timestep `t=1`, noise fraction `sigma=1-t`, and velocity
  `clean-noise`. The denoised estimate is `x_t + sigma * velocity`. The optional
  UniPC sampler changes the velocity sign only at its scheduler boundary.
- Released FL2VA inference uses distilled CFG 1, no CFG rescale, shift 12 and
  50 Euler sigma points (49 evaluations). Native joint initialization controls
  use 24 FPS; chunked training/inference use actual input FPS, including 16.
- Conditioning VAE posterior noise uses a CPU generator, even with a CUDA VAE.
  A CUDA generator with the same seed gives a different condition tensor.
- The native temporal layout is nonuniform: `17*n+5` padded frames produce
  `5*n+2` latents. `h3/data.py` owns the layout and its 40-unit/second rotary clock.
  `TemporalLayout.valid` marks requested intervals, not attention/loss validity.
  The non-causal decoder needs the aligned tail latents; crop only decoded pixels.
- Video and text share block causality, including the text refiner. Inactive
  audio rows only pad the sequence for SP/compile buckets; there are no audio
  observations or audio loss. Independent videos also isolate their camera
  references and initial-caption time origins.

## Cameras and trainable weights

All original attention layers use every token's actual camera. Wrapped PRoPE
changes Q/K analytically while preserving native RoPE; exactly neutral cameras
select the native computation. Initialization parity compares base weights,
not a fine-tuned model. Matrix and alternating recipes remain available in
`camera_matrix.yaml` and `camera_alternating.yaml`.

Poses are float32 normalized-intrinsic c2w
`[fx, fy, cx, cy, rotvec(R_c2w), C_world]`. Dataset projections and their inverses
are carried independently. A conditioning camera is the relative reference;
it does not replace the later cameras. Each independent video uses its own
reference. Decomposed translation uses five log-spaced frequencies from 0.01 to
32; default PSF and scale conditioning are off. `utils/camera.py` restores
recorded scale in older inputs, including their projection matrices.

Every `ar_interval=2` original attention trains at `ar_lr=1e-5`. Other weights
stay frozen. There are 33,122,992,896 original parameters, 3,853,523,200 trainable
parameters and no added parameters. `ar` in inherited setting names does not
denote an added H3 branch.

## Distributed state and optimization

The full recipe uses SP8/FSDP8, CPU parameter offload, activation checkpointing
and compiled blocks. FSDP stores trainable shards/gradients in FP32 and casts
gathered forward weights to BF16. Keep the existing WorldViews environment,
original FA4 runtime and FSDP boundaries. The grouped backward path belongs to
H3: queries with identical visible K/V share native varlen backward calls.
AdamW clears gradients in place, retaining their CPU storage so FSDP does not
reconstruct the flat gradient during backward.
The CUDA allocator also retains buffers across source samples. Caption length
changes do not flush it; explicit warmup and interval settings control cleanup.

Training, overfit and inference seed Python, NumPy and Torch and enable
deterministic library algorithms before model execution. This also fixes
Inductor reduction selection and repeated-KV gradient accumulation. cuBLAS
uses a reproducible workspace configuration; cuDNN benchmarking is disabled.
Text cache identities include this numerical policy. Equal seeds alone do not
establish reproducibility: compare fresh-process outputs and resumed updates.

Any decision affecting FSDP collective order must agree across replicas.
Sample metadata travels through SP gather; global noise-range and resampling forcing decisions
use their existing collectives. Check max-across-rank compile counts and all
node logs when diagnosing throughput.

Checkpoints contain raw trainable shards, AdamW, fixed EMA 0.995, RNG, the global
step, source queues and pending resampling forcing state. Frozen weights load from original H3.
The manifest becomes visible after all shards complete. Resume requires the same
recipe and SP/FSDP topology. The source sampler saves its unused random draws and
issued-but-unconsumed indices. New workers replay those indices; source datasets
seed augmentation from each index. Source decoding and worker initialization do
not consume the model's RNG. Legacy checkpoints without a source cursor restore
weights and queued documents, but report that future source replay is unavailable.

Validation defaults to EMA when enabled, then restores raw optimizer weights.
Standalone inference selects `--weights auto|raw|ema`. Byted-wandb records code,
resolved config, numeric history, media and checkpoint manifests.

## Checking changes

Run `PYTHONPATH=../python_deps:../diffusers/src python -m pytest tests -q` in the existing environment. The suite covers
causality, camera identity, caption overlap, decoder support, loss, EMA, resume
contracts and compile padding. Training-policy or packing refactors also need
fixed-input, restored-RNG comparisons against the previous implementation.
CPU checks do not establish full-model GPU throughput or convergence.

Commands and request examples are in the root README. Temporary validation
scripts and dated reports belong under ignored `local/`. Keep this guide focused
on current code, without old experiment recipes or failed-probe diaries.
