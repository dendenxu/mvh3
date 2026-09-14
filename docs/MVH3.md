# MVH3 training port

Read the root README for the directory map and commands. The active entry is
`main.py -c configs/diffusion_forcing.yaml`; old stage filenames are compatibility aliases
for this same recipe, not alternate optimizer settings. The default now uses the
released FL2VA image/text conditioner and distilled native sampler. Historical
verification records below are scoped to their recorded recipe; the completed
single-sequence DF convergence result is recorded separately below.

## Full-data Stage 1 restarted with compile buckets (2026-09-14)

The user explicitly authorizes updating HR302316153 and monitoring training.
At11:39UTC+8, all8 nodes start the repair payload in
`local/production_stage1_df64_cn2_buckets_20260914`, node0 run38. It selects
`configs/stage1_compile_buckets.yaml` and the unchanged complete Stage1 dataset,
RFoff/cn2/fixedEMA0.995/SP8/FSDP8/offload recipe. This is a fresh base-weight run
because the previous142-step run had no checkpoint. HR302889727 is untouched.
All151 core file hashes match the passed local check, and all8 node environment
receipts preserve original FA4. Tracking is `wgl/78rwfzih`; the first3 optimizer
updates and all numeric fields pass remote readback. Initial steps include cold
compilation. Read live metrics and all-node monitoring before reporting steady
performance. The monitor is read-only and GPU samples are nvidia-smi busy
percentages, distinct from the platform's two-hour SMA metric.

## Full-data Stage 1 through existing HR (2026-09-12)

The user paused the full-scale run on `sleep 1` at 15:03 UTC+8 after a platform
alert for 4.3% mean SM utilization. Run36 stopped at142 finite updates, before
the first checkpoint/validation at500. Six nodes exhausted256 Dynamo variants;
the rank0 log alone did not show these warnings. The142 rank0 updates contain
106 distinct sequence lengths. In a local three-layer H3 reproduction with
native sequence dimensions, eight new shapes average40.18 seconds while their
repetitions average0.102 seconds.

`configs/stage1_compile_buckets.yaml` is a repair candidate, not a resumed run.
It pads unused token/time/camera/dropout dimensions into a small set of static
shapes, preserving every observation, caption, chunk boundary and loss weight.
It also compiles the original block computation with checkpointing outside
compilation, avoiding checkpoint-HOP capture of FSDP state mutation. The
three-layer GPU check has no new graphs after the first update when captions
and partitions change (mean0.195 seconds); loss difference is at most1.01e-5
and gradient-norm relative difference is at most0.000662. Original FA4 kernels
and static-mask support remain unchanged. Full33B SP8 verification at
`local/compile_buckets_full8_v1` passes16 updates and exits0. Two token buckets
compile on steps1/5 (150.59/100.27 seconds); the14 warm updates average4.37
seconds with no new graphs after step5. Losses/gradients are finite, EMA remains
0.995 with16 updates, and frozen-weight samples are unchanged. All16 steps and
33 numeric fields pass remote byted-wandb readback. This uses one video's
existing native features and does not establish64-GPU/live-source throughput.
The final CPU suite passes208 tests. HR stays paused; a fresh-run payload is
prepared at `local/production_stage1_df64_cn2_buckets_20260912/hr_train.sh` but
has not been deployed. The existing configs keep their prior behavior.

The verified WorldViews July8 history retracts its earlier checkpoint-order
diagnosis: thread-count guard churn was fixed by `torch.set_num_threads(1)`,
then external checkpointing was restored. H3 already pins threads1. Current
WorldViews executable code uses checkpoint(compile(block)); its old inside-order
docstring is stale. The present H3 evidence is excessive static shape variants:
the142 previous updates'106 token lengths reduce to just4 padded lengths.
Keep FA4 static and bound shape/state variants; neither a larger Dynamo limit
nor frequent artifact writes alone resolves this. Current WorldViews disables
new-shape artifact writes and saves periodically at1000 steps. Trainer metrics
now report max-across-ranks new/total compile graphs, and the read-only HR
monitor checks all eight node logs and recent timings.

The paused full-scale run uses existing HR allocation 302316153, run 36,
with eight 8-H100 nodes. HR302889727 remains untouched. Use these existing HR
allocations for this project, not a new MLX allocation. The command is recorded
in `local/production_stage1_df64_cn2_20260912/hr_train.sh` and selects
`configs/stage1_diffusion_forcing.yaml`: Stage 1 for the full configured horizon,
RF disabled, WorldViews cn2 history noise 0.2 mean / 0.1 std / 0.05 inference.
All prior DF/camera/i2v/optimizer settings remain, including fixed EMA 0.995,
SP8/FSDP8 CPU offload, gradient checkpointing and attention compilation.

The original paired SHORT Parquet has 674,980 rows. Its existing malformed-path
filter, identical to WorldViews, excludes row 403559, leaving 674,979 usable rows
and all seven shape groups. Original batches and shapes are not reduced.
By 13:07 UTC+8, three optimizer updates are finite; losses are
0.27934617 / 0.25748068 / 0.30866158 with matching EMA counts. All three updates
and 46 numeric fields pass remote byted-wandb readback. This establishes
distributed startup, not convergence or throughput parity. The first update
includes 677.94 seconds in cold Qwen initialization/encoding; subsequent new
shapes still include compilation. All eight runtime receipts preserve the
original WorldViews environment and FA4 sources. Tracking is `wgl/1162akcy`.
Use the run's `metrics.jsonl`, `monitoring.json`, `live_config_verification.json`
and remote Tracking receipts for later progress. The CPU monitor observes HR
without automatic restarts. Historical no-production statements below predate
this explicit full-data launch.

## Single-sequence DF update (2026-09-11)

The latest recipe supersedes the historical TF/caption policy below. First draw
the complete BD partition (3-20 native H3 latents per block), then choose a clean
prefix cut only at one of those boundaries. Never duplicate the clean video.
The clean prefix has zero loss; suffix blocks independently sample sigmas, with
native `t=1-sigma` and velocity `clean-noise`. The configurable prefix probability
defaults to 0.5; a nonempty supervised suffix is mandatory. Decoder support stays
in the last real block and is generated/supervised rather than masked away.

Before choosing the cut, select captions using source-time overlap divided by
the original source-caption window duration. Include a motion sentence only for
strictly greater than 50% overlap; an exact tie is excluded. Use scene text alone
when no source window has a majority. SHORT rows use their already-sliced motions;
global parent motion fields are not reused. No caption modes or artificial
numbering/markers are introduced. The source-caption timeline remains in its
original five-Wan-latent units; BD block sizes use native H3 latents.

All modalities share block causality, bidirectional within each BD block. Text
refinement uses the same partition. Cached inference keeps past caption KV once,
at its clean history timestep, while refining the complete caption prefix and
omitting future captions. Image conditioning remains native i2v; there is no
invented audio target. Chunked requests retain their actual positive FPS (16 FPS
training data need no forced conversion); native joint/base parity stays at 24.

RF retains the partition and moves the cut one block forward, using the preceding
prediction as history. Reusing an unchanged cut would recycle the already-clean
prefix and make RF ineffective. Pending RF saves the updated cut and exact plan.
The DF recipe sets `resampling_forcing_clean_chunks=0`: the sampled prefix already
defines GT history, and the old TF default would overwrite the first RF prediction
with GT when the original cut is zero.
`configs/overfit_diffusion_forcing.yaml` uses a finite native Qwen caption-feature
bank so the same complete video can be repartitioned on every update without
freezing its first caption selection or re-encoding the video.

199 CPU tests pass, including unequal joint-view lengths and cached rollout with
shared/independent captions and CPU history storage. The full 33B
SP8/FSDP8/CPU-offload/checkpoint/compile canary at
`local/df_native_runtime_v1` has completed 24 finite updates, including RF and fixed
EMA 0.995. Its step-18 repeated backward has equal loss and relative gradient L2
0.005653787862692423, within the existing 1% gate. Its raw cached/recomputed
inference comparison failed (maximum error 0.54755735). Fresh-process restoration
of raw/AdamW/EMA is exact, but the continuation diverges at RF step 19. A separate
same-weight/input/restored-RNG check fails there with gradient relative L2
0.0485813. Deferred raw/EMA cache traces have equal inputs and predictions, but
retaining GPU tensors may change lifetime/timing. Plain inference replay from
the original step-24 weights is therefore checked separately. All 24 optimizer steps and
46 numeric fields pass remote byted-wandb readback. These runs predate disabling
the legacy forced-GT RF setting. The corrected original-batch envelope later
passes as recorded below; new-recipe continuation acceptance remains pending.
The 64-GPU run has not started. No FA4 source/kernel/cache
has been changed.

The optional `h3.grouped_attention_deterministic` setting calls the existing
native FA4 deterministic-backward API. The full 33B probe at
`local/df_rf_deterministic_v1` passes all eight repeated backward checks at
steps 17-24: maximum relative L2 0.00504019, with exact gradients at RF step 19.
Raw/AdamW/EMA restoration is exact and the process exits 0. This probe deliberately
retains the old forced-GT setting to isolate that option; its timings include
duplicate forwards/backwards and do not measure ordinary training throughput.
The option remains off by default while `local/df_optimized_runtime_v1` checks
the corrected single-sequence recipe together with dynamic backward metadata.
The latter changes only backward index/metadata specialization, preserving the
sparse forward. No kernel, synchronization patch or JIT-cache replacement is used.

`scripts/verify_inference_replay.py` replays saved encoded inputs from raw/EMA
checkpoints without call-level tracing or retained GPU tensors. It checks both
cached/recomputed equality and repeated rollout equality. The full-model replay
at `local/df_inference_plain_v1` passes raw and EMA, each with two repeated
cached/recomputed rollouts: all four views have exactly zero error and finite
latents. It uses the original failed run's step-24 weights, but a different saved
input document; the original failure's exact input was not retained. This result
does not establish that failure's root cause. Inference checkpoint
loads mmap the shard so unused optimizer and queued training-data tensors are
not eagerly read. Training/optimizer restoration retains its existing loader.

The corrected optimized DF canary and new single-video overfit use the native
deterministic-backward option and dynamic backward metadata as explicit candidate
settings. Main defaults remain unchanged pending full-model acceptance. The new
overfit is `local/overfit_diffusion_forcing`, with fixed EMA 0.995 from its first
update. Its initial phase completes 256 updates and a fixed-seed video; raw/EMA fixed losses fall
to 0.21599586/0.25085571 from 0.31970188/0.31943483. All 256 updates, 30 numeric
fields, five raw/EMA evaluations and two generation media entities pass remote
readback. The fresh process passes raw evaluation but fails the unchanged EMA
gate: relative difference 0.0011188091 versus tolerance 0.001. A normal retry
also fails at 0.0010497402; raw relative error is 0.0001080527. Neither process
performs update 257. The bounded diagnostic continuation is recorded below.

`local/df_resume_eval_probe_v1` separately confirms exact raw/AdamW/EMA restoration
on all eight ranks. Four raw evaluations and four EMA evaluations are identical
within that process, including before/after a backward without an optimizer step.
All eight pass the original 0.001 evaluation gate. Every raw shard remains exact
after backward and EMA count stays 256. All eight diagnostic rows pass remote
tracking readback. This does not explain the two normal resume failures.
`local/df_forward_trace_v1` completes 24 comparisons: all 16 within-process
repetitions match every recorded tensor hash, while all eight cross-process
comparisons first differ at block zero. Packed inputs and preceding encoders
match exactly. Tracing CPU copies can change scheduling and are diagnostic only.

`local/df_first_block_full_topology_v1` retains the complete native initialization
and FSDP structure, but truncates computation after block zero. Both processes
exit successfully. All 55 recorded inputs/outputs match within each process.
Across processes the compiled block output differs (sampled relative L2
9.647e-6), while the eager block matches exactly; camera, mask and block inputs
are exact. The precision-cast diagnostic also differs across processes
(1.592e-4), so it is not an accepted fix. The reduced-prefix control does not
reproduce the full topology's behavior. All four first-block control runs have
six remotely verified tracking rows and finished status. The internal cause is
still unproved; no FA4 or numerical runtime change follows from these controls.

The overfit script now permits an explicit bounded convergence diagnostic with
`--diagnostic-resume-after-eval-drift REASON --stop-after STEP`. It requires an
unchanged recipe, matching source features, the checkpoint's fixed evaluation,
and exact raw/AdamW/EMA state on every rank. The original 0.001 gate and failed
receipts remain unchanged; an allowed diagnostic retains failed acceptance in
its report and tracking summary. Nonfinite evaluations and state mismatches
still abort. The origin checkpoint is protected from retention cleanup.
This option is confined to the overfit experiment and cannot accept production
or the unresolved multi-step trajectory. The local continuation starts at 02:55
UTC+8 on September 12 and completes all 2048 updates at 06:16, exit 0. All eight
rank state checks pass. Its fresh evaluation passes the original gate (raw
0.00023013, EMA 0.00031016 relative error); the two earlier failures remain
recorded and unexplained. Raw fixed loss falls from 0.31970188 to 0.15085705
(52.81%); EMA falls from 0.31943483 to 0.15315709 (52.05%). Both configured loss
criteria pass. All losses and gradient norms are finite, with fixed EMA 0.995
on every update. The last 256 updates have a 4.528-second median; this fixed-feature
single-video timing does not establish full-data throughput equivalence.

`tracking_complete2048.json` verifies every update and 30 numeric fields, 33
raw/EMA evaluations and 10 fields, plus all seven final media entities. Run
`8eqvw7ib` is finished. `review_step2048_zh` contains the final Chinese four-column
source/before/raw/EMA video, curve, timeline and last-eight-frame sheet. The
77-frame 16-FPS comparison shares source, seed 81000 and partition [16, 6, 5].
Compressed-video pixel MSE against the source is 0.09778197 before training,
0.02595861 for raw and 0.02890094 for EMA; last-four-frame MSE is respectively
0.13476828, 0.02038987 and 0.02216735. Sampled full-sequence frames and every final
eight frames show no terminal block collapse, but hands, faces, motion details
and text still differ. This supports convergence on the one sequence, not exact
memorization or generalization. Completion, review and remote tracking receipts
are authoritative; the overfit script now refreshes terminal `progress.json`
after successful review rather than leaving its earlier running snapshot.

The corrected DF original-batch probe `local/df_batch_envelope_v1` passes all
44 cases and 48 updates with live native Qwen/VAE, fixed CPU EMA 0.995,
SP8/FSDP8 CPU offload, checkpointing and compile. It retains all original batches
and shapes, including 297 frames and 20 views. Maximum allocation is 29,258 MiB;
no extra batch/shape reduction is introduced. All 48 updates and 47 numeric
fields pass remote readback, with remote status finished. Different random
partitions and concurrent work confound warm-throughput comparisons.

`scripts/run/train_hr.sh` now selects this single-sequence DF entry. It no longer
silently limits training to 5,000 updates before the provisional 10,000-step
stage/RF boundaries. The horizon, stage boundary and RF warmup come from YAML or
explicit dot-list overrides; their final production values remain undecided.
No 64-GPU production job has been launched.

The corrected canary completed 24 finite updates with exact repeated gradients
at steps 18 and 19, then was interrupted during inference by a container CPU
memory OOM while three full-model experiments ran concurrently. The container
limit is 1,554,778,161,152 bytes; host MemAvailable is not its available budget.
This does not measure a single original batch's GPU capacity. The exact saved
input and step-24 checkpoint remain available. Subsequent native probes are
serialized alongside the fixed-feature overfit.

The fresh step-16 continuation restores all raw/AdamW/EMA tensors exactly and
completes eight finite updates. Its raw and EMA cached/recomputed inference both
have zero error in all four views. However, its multi-step trajectory fails the
unchanged 0.1% comparison: gradient norm differs by 8.56% at step 18 and 39.90%
at RF step 19, despite exact same-process repeated gradients at both steps.
`local/df_optimized_resume_v1/verification.json` remains failed; exact restoration
and inference equality must not be conflated with trajectory acceptance.
The reported pre-clip norms at steps 17-24 stay below 0.622 in both processes,
well below the configured threshold 10, so those steps do not trigger clipping.
A sampled endpoint comparison of 62,400 values across all eight shards measures
raw relative L2 1.313e-5, EMA 2.099e-7 and Adam first-moment relative L2 0.1333.
This quantifies drift without establishing its cause or accepting the trajectory.
An isolated duplicated-KV merge check with synthetic BF16 gradients is exactly
repeatable in both FP32 and final BF16; it does not justify replacing the existing
`index_add_` or establish full-model determinism.

The KV-cache mask path now reuses the training mask's split reduction/sort
builder, including rectangular query/key lengths. The former fused builder
produced 173,892 PTX lines and multi-minute ptxas work. All block metadata matches
the native CPU oracle on GPU, including a 3,907-by-10,621 case that builds in
1.89 seconds cold and 1.6-1.8 milliseconds warm. These are mask-builder timings,
not full-model throughput. FA4 kernels and caches are unchanged. The periodic
checkpoint save also skips the final iteration, where the unconditional final
save already writes the same state; this avoids duplicate final shard writes.

The builder also has an explicit non-recursive compiler boundary: otherwise an
outer compiled transformer block inlines the split functions and fuses the large
reduction with grid sorting again. The internal GPU reduction stays compiled.
Nested GPU checks at 37-by-533 and 3,907-by-10,621 match all eight native block
metadata tensors exactly; graph capture confirms the separation. This is recorded
in `local/rectangular_mask_nested_fixed.json`. Full-model replay with this final
boundary passes at `local/df_inference_saved_input_v2`: the interrupted corrected
canary's exact saved input and raw/EMA step-24 weights each run two repeated
cached/recomputed rollouts. All four views have exactly zero cache/recompute and
repeat error across eight rollouts. All eight rows and five numeric fields pass
remote tracking readback. Cold cached inference takes 394.65 seconds including
compile; warm cached inference takes 27.42-28.36 seconds versus 57.25-57.50 seconds
recomputed. These are bounded four-sigma-point inference timings, not training
throughput or an explanation of the historical failure.

The OOM canary's missing final optimizer row was recovered from durable local
metrics without changing its source/config snapshot. Complete remote readback
now passes all 24 updates and 46 numeric fields; its remote status correctly
remains failed. See `local/df_optimized_runtime_v1/tracking_repaired.json`.
Tracking verification reads the SDK-to-optimizer-step mapping once per pass and
paces individual field reads to avoid redundant requests during server throttling.

## Current camera and data decision (2026-09-10)

The latest user correction selects the existing complete paired Parquets and
configs used by `worldviews_1p3b_pre200_ours.yaml`: SHORT has 674,980 rows and
FULL has 161,673 rows. `configs/presampled_data.yaml` imports only the data
settings from WorldGen `pre200_short.yaml` / `pre200.yaml`; H3 keeps its own
attention scope, SP8/FSDP8 and distilled sampler. This supersedes the original
source-caption-only decision recorded below. The dataset retains actual 16 FPS
sampling, exact windows/views/crops, the original batch grouping and existing
shape remaps. No extra memory caps are introduced. Native 24 FPS comparisons
explicitly preserve physical duration when resampling the camera timeline.

The active caption policy keeps these exact paired rows and their scene/motion
fields, then selects motions by the greater-than-half overlap rule described
above. H3 re-encodes plain scene/motion prose with its native image/text
conditioner and does not load T5 embedding payloads. The historical 90% switch
and inserted `[CHUNK]` formatter are retained only for old TF controls.
Initialization equality remains a separate base-H3 check.
Paired base-model caption comparisons use FULL rows 97752/97753, identical
images/cameras/seeds and the Parquet's own simple-caption/global-caption fields;
artifacts live under `local/paired_caption_cases` and `local/paired_caption_inference`.

The user explicitly selected wrapped **decomposed** encoding in every original
attention layer, with each token's actual time-varying camera. There is no AR
bypass: never broadcast first-frame poses into the frozen layers. The conditioning
camera remains a fixed relative reference. Independent SHORT videos each use
their own reference; only jointly modeled views share the first view's reference.
Matrix-only and alternating matrix /
decomposed alternatives remain in `configs/camera_matrix.yaml` and
`configs/camera_alternating.yaml`. Both are wrapped and use all temporal cameras.
The original alternating trainable-weight scope remains unchanged.

The user subsequently disabled PSF for this default decomposed recipe. Both
training stages and both validation datasets set `pose_stable_factors=[1.0]`,
with `model.scale_cond=false`. Cached features and image-camera requests carrying
an older PSF restore camera centers and the independent projection/inverse
translation columns, including their conditioning cameras. Image, latent and
text features are unchanged. The matrix and alternating configurations explicitly
retain the historical PSF bank and optional scale handling.

The historical data verification used the full 19-source large-model mixture and
its exact source-row `caption` strings. That source-only recipe is superseded by
the paired Ours Parquets and their scene/chunk captions described above. The
19-source real decode and exact caption readback audit passed. Its sampling can yield
15/16/24/30 FPS; the H3 temporal clock uses the actual sampled FPS, never a label
changed without resampling. Native comparison requests use actual 24 FPS inputs.

The stage-1 and RF warmup boundaries are separate provisional configuration
values. Their shared historical value of 10,000 is not an accepted H3 schedule.
A bounded runtime probe uses a two-step warmup only to exercise the RF state
machine and checkpoint continuation. Do not carry that value into training.

The older verification records below describe their then-current camera modes.
They do not establish the new all-frame decomposed training or performance.

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
| Added AR branch every second layer | No added branch; selected existing main attention still trains at 1e-5 |
| Base first-frame decomposed attention | All existing layers use all-frame wrapped decomposed PRoPE; intervening weights stay frozen at sa_lr=0 |
| Dual high/low noise experts | One released H3 denoiser receives both original 50/50 sampled noise ranges |
| Wan text cross-attention/T5 cache | Native joint H3 attention with full Qwen3-VL layer-50 image/text features; no T5 cache reuse |
| Added scale MLP | Default decomposed uses unscaled camera translation and disables scale conditioning; no added weights |
| Wan VAE geometry | Native H3 padding, temporal camera anchors, spatial stride 16 |
| Five-Wan-latent AR chunks | Original caption windows remain first 17 then 20 source frames; BD blocks use random 3-20 native H3 latents |
| TF / RF | Single-sequence DF with a clean-prefix cut, no duplicated context; RF advances the cut without repartitioning |
| Parallel optimization | Ulysses SP, mixed-storage FSDP, FP32 trainable shards/AdamW, CPU offload, activation checkpointing, block compile |
| Validation / inference | Separate original val_dataset and condition sampler; H3-distilled CFG1, shift12, 50-point Euler, CPU-offloaded history KV |

The reference snapshot matches the live WorldViews config. The verifier lists
the native H3 sampler and FPS differences explicitly. `h3_native.yaml` additionally
selects the fixed-probe canvas and shifted-uniform diagnostic noise distribution;
the default full-mixture recipe retains WorldViews range sampling and weighting.

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
intrinsics and WorldViews-locked geometry, with PSF disabled by default. Dataset projection
matrices are carried independently and are never reconstructed from the pose in
the production path. Four subframe projections per H3 latent preserve per-head
camera sampling; SP head offsets use the original global head group.

Native H3 heads are `[T16,H16,W16,T16,H16,W16,tail32]`. Wrapped camera transforms
preserve all native spatial/temporal RoPE and the unrotated tail. Decomposed PRoPE
uses 28 H and 28 W channels with frozen Wigner bases. The wrapped matrix alternative
uses 20 existing H and 20 W channels: Q receives P-transpose and K receives
P-inverse. Both wrapped modes leave V/O unchanged and add no parameters.
Unwrapped compatibility mode removes slow H/W rotary pairs and transforms V/O;
it is not the default and does not satisfy neutral-camera base equality.

The five decomposed translation frequencies are log-spaced over 0.01 to 32:
approximately `(0.01, 0.0752121, 0.5656854, 4.2546367, 32)`.
This retains the endpoints of WorldGen's historical 12-frequency bank while
keeping the existing 30 translation channels and 56-channel camera overlay.
Individual phase periods span approximately 0.196 to 628.3 encoded coordinate
units. With PSF disabled these are the dataset's original translation units;
source-specific metric calibration is unchanged. Five frequencies sample
this range more sparsely than twelve. The frequency bank is included in the
checkpoint recipe digest, so old narrow-bank checkpoints cannot silently resume
with the changed encoding. Training and inference use the same camera module.

The live WorldViews `not.yaml` still uses matrix unwrap, but the user explicitly
selected wrap and all-frame decomposed encoding for H3 because there is no AR
bypass. Exactly neutral camera tensors select the native compiled graph. This
algebraic identity avoids unnecessary camera kernels and preserves native BF16
rounding without disabling fusion for moving cameras. Full-model GPU equality
remains an explicit acceptance check; tiny-model equality is insufficient.

The current full 33B raw-image check passes all 49 Euler predictions exactly
against the same initialized no-camera model, using normal block compilation,
fusion, patterns and epilogues with no diagnostic hooks. The report is
`local/init_default_normal/verification.json`; the process exits 0. It uses
all-frame decomposed wrapping, loads no future GT, and decodes 39 output frames.
After the temporal-support fix, `local/paired_caption_temporal_fixed/verification.json`
passes four complete 115-frame requests: 196/196 full-33B predictions exactly
match initialized base and all 460 output frames decode. The process exits 0;
all four final-12-frame contact sheets were inspected. Corrected caption A/B
media are in `local/paired_caption_comparison_fixed`, with byted-wandb remote
readback in `tracking_verification.json` (run `8jmsm15o`). The full CPU suite
passes 66 tests. These checks establish initialization and local contracts;
original-batch GPU capacity, the optimization combination and long training
remain separate acceptance checks. These comparisons share the port's encoded
inputs; they do not independently validate native condition-image encoding.

Training retains the original 128x128 attention block default. Optional
`h3.training_attention_block_size` exists for diagnostics and is included in the
checkpoint recipe digest. The attempted 640x128 default was withdrawn after
ordinary production replay failed gradient consistency with SP8/FSDP8/CPU
offload/checkpoint/compile enabled: identical predictions, finite gradient norms
0.3525939800 and 0.1125880869, but sampled relative gradient error 1.0881571688
and cosine 0.1872819497 over 1,649,600 elements. Neither pass is independently
accepted. The capacity preflight rejected this result; no capacity or long
training launched. See `local/attention_rectangular_production_v1` and
`local/paired_batch_gpu_rectangular_v1/preflight.log`.

Earlier 640x128 diagnostic passes in `local/attention_rectangular_fullopt_v1`
and `local/attention_native_sparse80_v1` were finite and repeatable within 1%,
but do not establish a reliable fix. All 85 CPU tests pass, including exact
rectangular-mask coverage; those tests cannot validate the full GPU backward.
No FA4 source, function, synchronization behavior, or JIT cache is replaced.
The integration root cause remains unresolved. The attempted local-window
control was invalid because FA4 does not compose local windows with `mask_mod`;
its NaNs are excluded from the evidence.

`h3.grouped_attention_backward=true` is a disabled-by-default
H3-side alternative. It retains the original sparse FA4 forward and groups
queries with identical visible keys for the original FA4 varlen backward.
Duplicated KV gradients accumulate in FP32. Every camera transform, model
parameter and no-gradient inference call retains its existing path. This does
not modify FA4, its PDL behavior, or its caches. Fixed-case diagnostic results
in `local/attention_grouped_v1` do not establish production or full-batch
acceptance. Normal-entry replay now passes in
`local/attention_grouped_production_v1/acceptance.json`: three bitwise-equal
predictions, finite gradients, repeat sample differences 0.4425-0.6983%, and
0.4585-0.7690% versus both valid dense controls over 1,649,600 samples. The 1%
gate is unchanged; the earlier fixed-case 1.1367% comparison remains recorded.
The explicit `configs/worldviews_grouped.yaml` path passes the original
44-bucket envelope in `local/paired_batch_gpu_grouped_v1`: all 48 updates are
finite with no batch or shape reduction, including 297 frames and 20 joint
views. Maximum allocated memory is 29,837 MiB. That run had EMA disabled;
its complete byted-wandb numeric history also passes remote readback. Repeated
warm shapes take 8.55-12.94 seconds for 62,976-101,528 packed tokens, while new
shapes can spend about 60 seconds compiling. This is not yet a full-data
throughput comparison against WorldViews. The grouped
adapter corrects the pinned Torch FLASH auxiliary-LSE conversion before native
varlen backward; its forward remains unchanged. Do not reuse that conversion
assumption after a Torch upgrade without the kernel comparison.

## Sharded EMA

The user requested EMA independently of the old WorldViews implementation.
The explicit grouped and single-sequence recipes enable `ema_weight=0.995`,
`ema_cpu_offload=true`, and `ema_warmup=false`: the latest user decision is a
constant decay from the first production update. `utils/ema.py` stores FP32 local
shards of trainable existing attention weights only; frozen weights are reused.
Every successful optimizer update is followed by one EMA update. Optional warmup
remains available for existing runs; when enabled,
the effective decay is `min(ema_weight, (1 + updates) / (10 + updates))`.
This EMA warmup is independent of LR, stage, and RF warmup.

Checkpoints store raw weights, optimizer, EMA weights, and the exact EMA update
count together. Resume requires matching shards and schedule; EMA is never
silently reset. A decay-cap change is accepted only before warmup has reached
either cap, when every historical EMA coefficient is identical; all other recipe
settings must still match. The requested 0.999-to-0.995 change therefore takes
effect at the first 256-step resume without resetting the averaged weights.
Changing an already-used schedule requires an explicit `--ema-schedule-change`
reason in the overfit runner. It retains the saved raw weights, AdamW, EMA and
update count, verifies every non-EMA recipe setting, and records the old/new
schedule and boundary. It does not relabel the old trajectory as constant decay.
The current run is scheduled to adopt constant 0.995 after its committed step-768
checkpoint; check `local/overfit_sequence_grouped/fixed_ema_transition.json` and
the resume receipt for actual completion. All 133 CPU tests pass, including
the exact state-preserving transition and rejection of unrelated recipe drift.
Validation uses EMA when enabled, with a reversible weight swap
that restores raw optimizer weights even after an inference error. Standalone
`scripts/infer.py --weights auto|raw|ema` selects the checkpoint weights; `auto`
uses EMA when configured. Raw and EMA checkpoints keep the same base model and
camera parameters. Topology changes still require explicit consolidation.

FSDP can still be reading pinned CPU shards through asynchronous H2D copies
after a forward returns. EMA waits for CUDA completion before overwriting those
shards at swap entry and exit. This is confined to evaluation/inference weight
transitions, outside FA4 and ordinary optimizer updates. A two-GPU nested FSDP
CPU-offload stress test passes 128 exact raw-prediction restoration cycles,
EMA checkpoint loading and decay-cap continuation. The isolated version without
these waits reproduces an incorrect restored prediction despite exact CPU shards.
`scripts/verify_ema_distributed.py` also passes on four GPUs with two FSDP replica
groups and unequal input batches: corresponding raw/EMA shards stay exactly equal
across replicas, optimizer restoration is exact, and 16 inference swaps restore
predictions exactly. This tiny-model check does not test multi-host transport.

The EMA-on follow-up in `local/paired_batch_gpu_ema_extremes_v1` passes eight
original capacity boundary cases, including 13 independent videos, 297-frame
sequences and 20 joint views. All eight updates and EMA shards are finite,
CPU offload/checkpoint/compile stay enabled, and no input is reduced. Maximum
allocated memory is 29,633 MiB; all recorded numeric fields pass remote readback.
This is an eight-case supplement to the earlier 44-case EMA-off envelope.

The single complete 77-frame paired Ours sequence uses corrected native CPU
posterior conditioning in `local/overfit_sequence_grouped/features`. Its run
tracks raw and EMA fixed-noise evaluation, EMA update time, fresh-process resume
at 256, and final raw/EMA decoded generations. The full suite passes 131 CPU
tests; all eight EMA tests also pass after adding the swap waits. The 33B run
successfully resumes at step 256 with EMA cap 0.995 and its existing update count.
Raw and EMA fixed-noise losses differ from the pre-resume values by 0.0435%
and 0.0254%, respectively, within the fixed 0.1% tolerance; predictions are not
claimed bitwise equal across fresh processes. The first 256 optimizer steps,
all 24 recorded numeric fields and generation media entities pass byted-wandb
readback. Step-256 raw/EMA losses are 0.190945/0.195191 versus 0.333762 at init.
The decoded 77-frame comparison improves source pixel MSE from 0.107799 to
0.062812 and has no observed collapse in the last eight frames. Motion, faces
and text still differ from the source. Training continues to 2048; final
overfit acceptance and the subsequent 64-GPU launch remain pending.

The 2026-09-11 fault-data audit found a separate native-input mismatch: the
released `encode_vae_condition` draws posterior noise with a CPU generator,
whereas the port selected a CUDA generator. Equal seed 42 gave a 3.8815% relative
condition-latent difference (maximum absolute difference 0.382286) on the fault
image. Training and raw-image inference now both use the native CPU generator.
The complete pretrained VAE produces exact native tensors through both entry
points (maximum error 0), and all 85 CPU tests pass. The runtime recipe digest
includes the generator device so old checkpoints cannot silently resume under
changed conditioning. Existing encoded feature artifacts and historical media
retain their original CUDA-seeded conditions; re-encode their condition images
before treating them as evidence of the corrected native recipe. See
`local/fault_data_audit_v1/condition_seed_verification.json`.

The same audit re-decoded SHORT row 0, checked its sampled raw source poses,
reconstructed the flow target, and re-ran all 64 Qwen layers using the pinned
native image/text helper. Pixels, selected source poses, captions, Qwen features
and tags match; Qwen's 21,120 activation is present in the native output too.
Six full33B input substitutions retain SP8/FSDP8/checkpoint/compile/offload.
Original FA4 produces invalid gradients even for random video/text tensors with
no camera, while SDPA gives finite norms 0.1024-0.9607 for all six cases.
Replacing isolated audio padding with video padding also retains the fault.
Conversely, standalone original FA4 with the exact H3 mask and ordinary random
Q/K/V passes eager and CUDA-graph backward comparisons (maximum relative error
0.237%). These results exclude several data-content causes, but do not establish
the remaining full-model failure's root cause or authorize changing FA4.
Reports are under `local/fault_data_audit_v1`; FA4 internals remain unchanged.
Corrected native conditioning still yields nonfinite gradients in two repeated
full-model backward passes. Disabling checkpointing also retains the failure
(2.158 billion nonfinite elements), while the same checkpoint-disabled SDPA
control gives a finite norm of 0.13509. This is not a reason to disable production
checkpointing. All these bounded diagnostics have exited, and long training
remains stopped pending a correct original-runtime backward path.

The default disables PSF and scale conditioning. Explicit matrix/alternating
recipes retain scaled camera geometry and its recorded factor. Historical
`spatial_rotary` mode adds `log(pose_stable_factor)` to existing video H/W
coordinates and remains available only as an explicit alternative. The native
timestep MLP and AdaLN inputs are unchanged. Feeding scale into the pretrained time MLP was rejected
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
retained. Every aligned temporal latent participates in attention, denoising and
loss: the non-causal VAE decoder reads the repeated-frame support beyond the
requested duration. `TemporalLayout.valid` identifies requested intervals only;
it must not mask this decoder support. Support latents belong to the final real
chunk, including in rollout writeback. Only empty SP/audio padding is isolated.
The current `native_grid_v7_independent_causal_positions` packing version rejects
pre-fix training recipes, including the older temporal-support-only version.
The local VAE handles short decode tails
by repeating to a normal overlap chunk and cropping the real physical duration.

Masks cover video, text and inactive audio padding, including text refinement, to
prevent future captions or modality relays. Sparse flex attention has an explicit
compile-only guard: exceeding Dynamo's variant budget cannot silently allocate a
quadratic eager score matrix.

Independent videos also isolate camera references and media time origins. The
media origin counts only initially visible caption tokens in that video's scope;
joint views share their initial prefix. Other independent captions and future
chunk captions cannot shift earlier media RoPE positions when their lengths
change. Shared captions are deduplicated per chunk, so a future caption edit
cannot change how earlier conditions are shared. A single native prompt retains
its original exact H3 token positions.

## Two stages and state

Stage 1 uses the existing SHORT Parquet, retaining its independent views within
one update. Stage 2 switches the loader to the existing FULL Parquet while
preserving weights, optimizer, step and already queued samples. Validation
switches to the corresponding paired validation Parquet. The current 10,000-step
stage duration and RF warmup remain separate provisional values, not measured
or user-approved H3 schedules.

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

### Historical FA4 synchronization diagnostic (withdrawn)

On 2026-09-11 the user rejected modifying FA4 and requested the original existing
WorldViews environment. The automatic runtime override and alternate backward
cache have been removed from the production entry and GPU regression script.
The diagnostic is retained only in `local/fa4_compat_historical.py`; do not enable
it in training. Further investigation must reproduce the issue with original FA4
and distinguish H3 input/mask/stream/compiler behavior from a kernel defect.
PDL permits overlapping independent work; a wait placed at the entire sparse
producer entry is earlier than dense backward's wait immediately before LSE.
No performance parity has been established for the withdrawn override.

The following results describe the historical experiment, not an accepted fix
for the current unmodified runtime.

The pinned FA4 runtime enables PDL between backward preprocessing and the main
SM90 kernel. Preprocessing releases dependent kernels before writing LSE,
dPsum, and dQaccum. Dense backward waits for those writes; the upstream sparse
producer omits that wait. This reproduced pre-clip NaNs on the original SHORT
1x17 input from base weights, with SP8/FSDP8/checkpoint/compile/CPU offload.
Changing the clipping threshold cannot repair these already invalid gradients.

The historical experiment added GPU `griddepcontrol_wait` without editing the
shared installed package or adding host synchronization. It used a separate
backward cache namespace. This still changed FA4 behavior inside the process;
the production path now uses the original function and cache unchanged.

The exact full-33B captured-input control fails without the dependency and passes
with it in `local/singleton_flash_fenced_v1` (exit 0, norm 0.09681236, no optimizer
update). An independent observed replay compares 400 real kernel calls to FP32
SDPA at identical Q/K/V/dO: aggregate Q/K/V gradient errors are 0.586%, 0.298%,
and 0.228%; maximum single-call Q error is 4.046%. These are BF16 numerical
comparisons, not bitwise gradient equality. The unobserved and observed forward
predictions are exact. Original-batch capacity and long-training completion
require their own successful reports. `scripts/verify_fa4_sparse_backward.py`
provides a bounded repeated sparse-kernel regression with allocator reuse.
All 84 CPU tests and 36 GPU repetitions at 512/3560/4096 tokens pass; the latter
have at most 0.241% gradient error against their identical-input FP32 reference.

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
