# MVH3 training port

Read the root README for the directory map and commands. The active entry is
`main.py -c configs/worldviews.yaml`; old stage filenames are compatibility aliases
for this same recipe, not alternate optimizer settings.

## Source and architecture

The official MiniMax repository was checked live at snapshot
`d21241f0a4b3acbb34c97dae47fa417b7065e438`. It supports multiple inference runtimes
and does not mandate Diffusers. It supplies original VAE Python code and original
and converted checkpoint configurations, but no standalone raw transformer
training loop. Local H3 implementations were expanded from Diffusers commit
`d30c748f5f5d0925a5af14dc0e6a6de983025e63`, retaining its computation and strict
state names. `h3/modules/model.py` and `h3/modules/vae.py` are plain `nn.Module`
classes; helper layers, attention kernels and numerical schedulers are local.
The optional full audio conversion command retains a lazy Diffusers import;
the video training/inference runtime does not import it.

Original transformer conversion reads every one of 535 source tensors across
13 shards, checks all expected shapes/dtypes, reorders per-head interleaved QKV,
and swaps the original SwiGLU gate/value halves. Nothing is partially initialized.
The full native model has 50 blocks, residual width 5376, 56 heads of width 128,
FFN width 14336, video latent width 24 and text feature width 5120.

## WorldViews mapping

| WorldViews behavior | H3 implementation |
| --- | --- |
| Complete resolved YAML | `configs/worldviews_reference.yaml`, with environment substitutions for dataset roots |
| Added AR branch every second layer | Existing main attention at those indices uses matrix PRoPE, original projections train at 1e-5 |
| Base first-frame decomposed attention | Intervening existing attention uses first-frame decomposed PRoPE, frozen at sa_lr=0 |
| Dual high/low noise experts | One released H3 denoiser receives both original 50/50 sampled noise ranges |
| Wan text cross-attention/T5 cache | Native joint H3 attention with full Qwen3-VL layer-50 text; no T5 cache reuse |
| Added scale MLP | `log(scale)` shifts existing video spatial RoPE coordinates relative to text; no added weights |
| Wan VAE geometry | Native H3 padding, temporal camera anchors, spatial stride 16 |
| Five-Wan-latent AR chunks | Same physical boundaries: first 17 source frames, then 20-frame chunks |
| TF / RF | Shifted weighted flow loss, noisy context, history dropout, synchronized low-range RF and warmup |
| Parallel optimization | Ulysses SP, mixed-storage FSDP, FP32 trainable shards/AdamW, CPU offload, activation checkpointing, block compile |
| Validation / inference | Separate original val_dataset, condition sampler, CFG5, 70-step UniPC, CPU-offloaded history KV |

The scale encoding and attention allocation are explicit adaptations, not a
claim of numerical equivalence to Wan. H3 head width cannot hold Wan's exact
channel allocation unchanged. The native packed joint attention also has no
separate text cross-attention cache to which `kv_offload_crossattn` could apply.
The high/low offload-switch knob has no second H3 expert to switch.

All remaining parameters stay frozen. Existing every-second-layer attention
contains 3,853,523,200 trainable parameters; all original model parameters total
33,122,992,896, with no added parameter or state-dict key. FP32 optimizer/storage
avoids rounding small updates away in BF16. Mixed precision casts gathered weights
for forward computation; checkpoint shards retain FP32 trainable weights.

## Camera and time

The pose schema is canonical c2w `[fx,fy,cx,cy,rotvec(R_c2w),C_world]`, normalized
intrinsics and already WorldViews-locked/scaled geometry. Dataset projection
matrices are carried independently and are never reconstructed from the pose in
the production path. Four subframe projections per H3 latent preserve per-head
camera sampling; SP head offsets use the original global head group.

Native H3 heads are `[T16,H16,W16,T16,H16,W16,tail32]`. Camera transformations preserve
all temporal pairs and the unrotated tail. Matrix PRoPE uses 20 existing slow H and
20 slow W channels; only those spatial RoPE pairs are removed. Q is transformed by
P-transpose, K/V by P-inverse, and output by P with inverse RoPE. Decomposed PRoPE
uses 28 H and 28 W channels with the frozen WorldViews Wigner bases. These are
activations, never new model parameters.

Scale conditioning adds `log(pose_stable_factor)` to existing H/W coordinates for
video rows, leaving text/audio coordinates and all temporal phases unchanged.
The resulting sine/cosine factors remain bounded. The native timestep MLP and
AdaLN inputs are unchanged. Feeding scale into the pretrained time MLP was rejected
after full-weight tests: scale 10 produced a delta RMS of 11.88 versus native
timestep embedding RMS 0.009-0.028, with losses above 9,500. Checkpoints from that
obsolete `existing_time_embedding` recipe must not be resumed; the recipe digest
rejects them.

H3 uses clean timestep `t=1`, noise fraction `sigma=1-t`, and velocity
`clean-noise`. The x0 estimate is `x_t + sigma*v`. The original UniPC solver accepts
`noise-clean`, so the sign flips only at its interface. Text receives the generated
video timestep; clean KV history is explicitly rebuilt at its clean timestep so
joint text modulation does not change previously cached media keys.

The VAE maps `17*n+5` source frames to `5*n+2` latents. Camera interval ends are
`[0,4,8,12,16]+17*n`; rotary interval starts are `[0,1,5,9,13]+17*n`. The native clock
is 40 units/second. Each source view is independently spatially padded to a multiple
of 32; edge patches retain fractional pixel-area loss weights. Temporal tails are
retained, padded intervals are masked out. The local VAE handles short decode tails
by repeating to a normal overlap chunk and cropping the real physical duration.

Masks cover video, text and inactive audio padding, including text refinement, to
prevent future captions or modality relays. Sparse flex attention has an explicit
compile-only guard: exceeding Dynamo's variant budget cannot silently allocate a
quadratic eager score matrix.

## Two stages and state

Stage 1 uses all 19 original sources. Every view and every at-most-77-frame window
is consumed, including partial tails. Stage 2 resumes the same weights, optimizer
and step with the reference duration/view distribution. Already queued short clips
are consumed instead of discarded. `h3.stage1_steps=10000` is a provisional,
configurable default aligned with the existing RF warmup, not a measured optimum.

Checkpoints are unique generations, each containing per-rank trainable shards,
AdamW, RNG, pending raw/encoded source queues and RF state. The manifest/latest
pointer is committed only after every rank finishes. Resume enforces identical
recipe and FSDP/SP topology; frozen original weights are loaded from the base again.
There is no resharding tool. Worker-local decoder/prefetch state is not serialized;
do not claim exact future-sample replay after restarting a DataLoader.

## Verification record

Reports are local, ignored artifacts. They must say `status=passed`, and the
corresponding processes must exit successfully. A forward, existing checkpoint,
or partially written report alone is insufficient.

- All 19 source entries / 1,372,262 summed source rows match both stages and the
  frozen source inventory. The decode audit uses 8 preload rows per source and
  actually samples/decodes one source document per entry; it is not an exhaustive
  media-integrity scan.
- All 40 local CPU tests pass. They compare full parameter topology and exact no-camera outputs
  against pinned Diffusers, verify causal/view isolation across three blocks,
  gradients, stage continuation, matrix inverse behavior, native flow direction,
  and local VAE/scheduler parity. Runtime imports are tested with Diffusers blocked.
  Scale conditioning preserves the original timestep embeddings and all temporal
  and text phases exactly; visualization output failures follow `raise_vis_error`.
- The complete 2,603,868,984-parameter video VAE was checked on real 448x832, 22-frame
  source video: encode posterior, normalized sampled latents and decoded pixels
  match exactly (maximum error 0); the single-frame tail and four subframe camera
  preparation paths pass. Peak allocated memory: 13.97 GiB on one H100.
- Two-rank tiny SP/FSDP with block compilation completed mono and multiview updates,
  matched the unsharded loss/gradient oracle, exactly restored trainable shards and
  AdamW moments, and matched cached/recomputed CFG rollout exactly. The final
  process exits succeeded with the host teardown workaround described below.

- Complete Qwen3-VL-32B FSDP encoding matches the saved full-encoder layer-50
  features exactly (123 x 5120); different ranks hitting/missing the text cache
  complete without collective mismatch, and cached features restore exactly.
- Every one of 130 top-level resolved config keys matches the live WorldViews
  YAML, including all 19 training and 19 validation source entries.
- The C++ random-move wrapper matches the compiled extension exactly on a seeded
  257-sample trajectory. Installation includes the tested FA4/Cutlass versions and
  all active video-decoding dependencies. Main/autograd threads use the original
  fixed-count setting, independently of the dataset worker thread count.

The full original 33,122,992,896-parameter model completed four real-feature updates
with SP=8, FSDP=8, BF16 forward, FA4, activation checkpointing, block compilation,
CPU-offloaded parameters and FP32 AdamW. The inputs are full-VAE/full-Qwen features
from real 448x832 video, with the reference camera centering and pose-stable factors.
These fixed documents test the model runtime; live decoding and text encoding were
validated separately as described above.

| Input | Packed tokens | Weighted loss | Gradient norm before clipping |
| --- | ---: | ---: | ---: |
| Stage 1: 1 x 77 frames | 16,872 | 0.282203 | 0.750333 |
| Stage 2: 2 x 90 frames after restore | 36,888 | 0.492176 | 1.281207 |
| 8 x 37 frames | 49,992 | 0.625744 | 0.834202 |
| 1 x 297 frames | 63,824 | 0.210931 | 0.177268 |

Every step changes trainable weights, retains frozen weight samples, and stays
finite. The stage-1 checkpoint exactly restores every trainable shard and the full
AdamW state. This process exited successfully; its rank-0 peak allocated GPU memory
was 9.90 GiB for the backbone check, excluding the separately tested VAE/text
encoder, dataset queues, CUDA allocator reservation and other processes. The timings
include first-shape compilation and do not establish production throughput.

A separate fresh eight-rank process reloads the stage-1 checkpoint, checks every
trainable shard and AdamW tensor exactly, then performs the two-view update. Its
loss matches the continuous run exactly at 0.4921759367. Gradient norm is
1.28149188 versus 1.28120673, relative error 0.00022256 (0.0223%), within the
explicit BF16 backward tolerance of 0.001. This process also exits successfully.
Restored state is exact; backward computation is not claimed to be bitwise
identical across fresh compiled processes.

Reproduce the bounded full-model check with the prepared real feature directories:

```bash
torchrun --standalone --nproc_per_node=8 scripts/verify_worldviews_runtime.py \
  --full --compile --checkpoint "$MVH3_CHECKPOINT" --features local/real_probe \
  --extra-features local/envelope_8x37/long_multiview.pt \
  --extra-features local/envelope_1x297/long_multiview.pt \
  --output local/worldviews_full8_final

torchrun --standalone --nproc_per_node=8 scripts/verify_worldviews_runtime.py \
  --full --compile --checkpoint "$MVH3_CHECKPOINT" --features local/real_probe \
  --resume local/worldviews_full8_final/ckpt/latest.json \
  --compare-to local/worldviews_full8_final/verification.json \
  --output local/worldviews_full8_fresh_verified
```

The FA4 BF16 dense/sparse check passes with maximum output error 0.023438 and
relative input/Q-gradient errors 0.000979/0.002961, including exact three-layer
future-perturbation invariance. FA4 indirect history-mask indices use explicit
Int32 casts; CPU or non-FA4 checks alone did not catch this kernel requirement.

### Local NCCL teardown

On the current PyTorch 2.12.1 / NCCL 2.29.7 host, `destroy_process_group` hangs in
communicator shutdown even after successful barriers. The issue reproduces in a
minimal two-rank test, with and without the network plugin and with eagerly bound
groups. Set `MVH3_NCCL_ABORT_ON_EXIT=1` on this host: the local cleanup helper first
synchronizes every rank and all CUDA work, then releases the communicators with
PyTorch's abort cleanup. Successful reports are written after this cleanup. This
is an explicit host workaround, not a claim that upstream graceful shutdown was
fixed. It does not change the training collectives. A single shard group uses the
numerically equivalent FULL_SHARD path; multi-group training retains HYBRID_SHARD.

Earlier full-weight single-process layer placement checks are separate from this
SP/FSDP verification. Short checks establish executable contracts, not
convergence, pretrained visual quality under CFG5, or robust long-horizon control.
The full source mixture has not undergone a long training run. Multi-node hybrid
replication and remote HDFS copying are implemented but not exercised by these
local checks; long-training compile-variant coverage is also unestablished.

## Provenance

Original FL2VA weights: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`;
84 files / 144,051,241,571 bytes were fully checksummed. Weights stay outside Git.
`docs/WORLDVIEWS_SOURCE.json` records imported source hashes and local adaptations.
Apache licensing for expanded Diffusers code is in `licenses/DIFFUSERS-APACHE-2.0.txt`.
