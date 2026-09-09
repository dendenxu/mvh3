# MVH3: camera encoding on the original H3 attention

This checkout starts from the official MiniMax H3 source snapshot and adds a camera-conditioned transformer with exactly the original parameter names and shapes. `origin` is `https://github.com/dendenxu/mvh3`. The original inference documentation is retained in `README.md` and the translated README files.

## Agreed experiment

Use the same largest WorldGen data mixture in two stages. Stage 1 splits every source view into independent short monocular clips. Stage 2 continues those weights and optimizer state on the full duration/view distribution, retaining the naturally monocular sources. Camera encoding is deterministic: no new attention branch, camera network, adapter, LoRA, projection width, or trainable parameter group.

The initial trainable scope is all original main-block Q/K/V/O projections and Q/K RMS norms, approximately 7.71B of H3's 33.12B parameters. `mvh3.training.attention_parameters` selects this scope. The proposed initial LR is `1e-6`; it has not been tuned on H3. The remaining transformer weights, text encoder/refiner, and VAEs stay frozen in this initial recipe. Training additional original weights later would also preserve the zero-new-parameter contract.

## Camera implementation

`mvh3/transformer.py` is based on Diffusers' pinned H3 implementation. The forward path accepts `camera_pose` and `camera_indices` explicitly, including during activation checkpoint recomputation; it does not hide per-batch camera state on a mutable attention processor. Poses are precomputed once per forward into rotations and trigonometric coefficients. They are activations, not model parameters or checkpoint buffers.

H3's 128-channel heads use the layout `[T16,H16,W16,T16,H16,W16,tail32]`. Paired channels are split across the two 48-channel halves. The implementation gathers these pairs, applies WorldGen's decomposed camera transform to 28 existing channels in H and 28 in W, and scatters back. All temporal pairs, four remaining channels per spatial axis, and the tail stay unchanged. The tail contains pretrained features; it is not spare or newly allocated capacity.

The camera representation is c2w `[fx,fy,cx,cy,rotvec(R_c2w),C_world]`, with normalized intrinsics and already locked/scaled geometry. H uses D1+D2, x translation, part of z translation, fx/cx. W uses D1+D3, y translation, the remaining z frequencies, fy/cy. Translation frequencies are `[1,2,4,8,16]`; intrinsics use frequency 4 on `[log(fx),log(fy),cx,cy]`. The frozen WorldGen Wigner basis JSON is copied byte-for-byte. Both Q and K receive the same orthogonal transform after native RoPE; V and output projections retain their original computation. This starts with the decomposed encoding, not WorldGen's separate matrix-PRoPE AR branch.

`camera_pose` has shape `[batch, num_poses, 10]`; `camera_indices` has one index per packed token. Every video token addresses its aligned pose; text/audio use `-1`. Omitted camera arguments reproduce upstream computation exactly. Identity poses reproduce it within Wigner basis floating-point tolerance.

## Two stages and data

Both `configs/stage1_short_mono.yaml` and `configs/stage2_long_multiview.yaml` reference `configs/worldviews_dataset.json`. This preserves the full resolved dataset settings from WorldGen, including all 19 source entries, source-specific sampling powers, and augmentation settings. The original inventory summed to 1,372,262 source Parquet rows on 2026-09-09; this is not a count of sampled training examples. `pre200` is a distinct subset and is not substituted here.

Set `MVH3_DATA_ROOT` and `MVH3_DATA_ROOT3` to the two local dataset roots when resolving the data reference. `scripts/verify_data_sources.py` compares the entire dataset section with the frozen WorldGen configuration, checks that both stages resolve the same mixture, and reads live metadata/first rows for all 19 Parquets. Augmented game geometry lives in `video_meta`; the other source families expose `pose`. The exact original reference and inventory are preserved under ignored `local/`. This audit establishes source/configuration identity, not full-family sampling or decoding of every video. The stage files remain reference recipes rather than a full-data distributed training launcher.

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Source mixture | All 19 sources | Same 19 sources |
| Video size/cadence | 448x832, 16 FPS | Same |
| Duration | At most 77 actual sampled frames per independent clip | Full reference duration distribution |
| View relationship | One view per batch item; all source views retained | Cross-view attention for synchronized multiview samples |
| Parameters | Original main attention only; zero added | Same topology, continued weights/optimizer/global step |
| Camera | Decomposed PRoPE on existing spatial Q/K channels | Same encoding |

The 77-frame maximum is the physical limit of WorldGen's 20-Wan-latent short stage. It is not 20 H3 latents. `short_mono_windows` is a source-frame split primitive and keeps all partial tails; it is not the old Wan presampled splitter. H3 needs its own VAE padding/pose mapping and regenerated caches. Stage lengths remain unset until a real pretrained memory/throughput check and learning curves inform the decision.

`mvh3.data.temporal_layout` pads before encoding to the H3 VAE's `17*n+5` geometry, retains the real partial tail, and excludes fully padded intervals from attention and loss. The causal camera anchors are `[0,4,8,12,16]+17*n`; native rotary interval starts are `[0,1,5,9,13]+17*n`. The rotary clock uses 40 units/second at the adapted 16 FPS. `mvh3.packing` assigns the old five-Wan-latent chunk a physical 20-frame duration rather than relabeling it as five H3 latents.

The bounded training path implements the reference linear context mixture with noise 0.2/std 0.1, sampled independently per view, and the first frame of view 0 as clean conditioning. H3's native convention is `t=1-sigma`, clean data at `t=1`, and velocity target `clean-noise`. Context receives `t=1-context_noise`; text inherits the generated-video timestep. A regression compares noise construction and `x0 = x_t + sigma*v` directly with the pinned H3 scheduler. The Wan velocity sign and clean-end timestep must not be copied into H3. History dropout (0.1), RF warmup (10,000), accumulation (1), and SP/FSDP sizes (8) remain recorded reference settings; RF/history-dropout and distributed integration are pending.

`mvh3.optim.MasterAdamW` maintains FP32 master weights and AdamW moments while updating the existing mixed-precision model parameters. This preserves small `1e-6` updates that BF16-only optimization can round away. Masters and moments are optimizer state, not added model parameters.

## Visibility and training boundaries

`mvh3.masking.TokenLayout` defines visibility for the whole packed sequence. CONDITION tokens read only available CONDITION tokens. CLEAN media read available conditions and clean chunks up to the current chunk. NOISY media read available conditions, earlier CLEAN chunks, and their own NOISY chunk. Noisy targets cannot see same-chunk clean answers. Global conditions cannot collect local conditions and relay them elsewhere. Stage-1 views belong on the batch axis; explicit scope masks also support isolation within a document.

The text refiner receives the text restriction of a dense mask or `TokenLayout`. For a raw `BlockMask`, callers must supply its text mask explicitly. This prevents future chunk captions from leaking through text refinement before the joint blocks. The module exposes dense reference masks with a size guard and PyTorch flex block-mask construction for larger layouts. The GPU path explicitly compiles flex attention with `fullgraph=True`: falling back to eager flex would materialize quadratic attention scores. The verification runner sets a bounded compile-variant allowance for its devices and shapes.

This port currently rejects camera/masked context parallel execution until the sequence-sharding adapter is implemented and validated. The inherited native H3 context-parallel hooks alone cannot shard camera indices or guarantee global mask semantics. Do not bypass this check by setting SP=8 only in a config.

The strict streaming loader now consumes every original transformer tensor, including the full AdaLN weights, with upstream QKV reordering and SwiGLU conversion. Real video/text feature generation uses the full released VAE and Qwen3-VL-32B layer 50. `read_parquet_clip` is a bounded native MP4 reader with matched image crop/intrinsics and c2w world locking; unsupported windowed/list-schema rows fail explicitly. It does not implement the full augmented-game/static-self-view/multicamera sampler, pose-scale augmentation, or dataset-wide caches. RF rollout/KV caching and distributed masked attention/save-resume remain required before a full-data cluster run. The existing Wan T5/16-channel latent caches and Wan AR weights cannot be loaded into this model. No cluster job is launched by the setup or tests.

## Dependencies and validation

The H3 transformer requires the pinned Diffusers commit in `pyproject.toml`. The downloaded bundle can use sibling `diffusers/src` and `python_deps` through `scripts/runtime_env.py` without upgrading the WorldGen environment. For a standalone checkout, install into a dedicated environment with `pip install -e '.[test,verification]'`; use an appropriate existing PyTorch/CUDA build. The local GPU validation environment uses PyTorch 2.12.1+cu129, Transformers 5.9.0, and H100 80 GiB devices.

```bash
python scripts/test_cpu.py
```

Tests use tiny random CPU models. They compare parameter topology/state keys and no-camera outputs against upstream, check camera response and preserved temporal/tail channels, exercise attention-only optimizer updates with and without activation checkpointing, round-trip model/optimizer state across a stage boundary, and perturb future clean video/audio and local captions across three layers. They also verify stage-1 batch independence and full source/view/tail preservation. Passing these tests does not establish pretrained generation quality, production GPU memory, or distributed readiness.

### Real pretrained verification

The executable verification path loads the complete 33,122,992,896-parameter model, compares the loaded first/middle/last blocks with pinned upstream computation, trains all 7,707,046,400 existing main-attention parameters, writes an actual 100.49 GiB stage checkpoint, and compares every restored trainable weight/master/AdamW state tensor before continuing. It uses one process with intact layers placed across eight local GPUs. This is not SP, FSDP, or a throughput benchmark for distributed training.

Real source features prepared with the complete released VAE and text encoder cover these shapes at 448x832 and 16 FPS. The 297-frame and 8-view shapes exercise duration/view boundaries from the reference; they do not exhaust the full sampler's shape distribution.

| Source frames | Views | Padded frames | H3 latents per view / valid | Packed teacher-forcing tokens |
| --- | --- | --- | --- | --- |
| 77 | 1 | 90 | 27 / 23 | 19,779 |
| 90 | 2 | 90 | 27 / 27 | 39,435 |
| 37 | 8 | 39 | 12 / 12 | 70,011 |
| 297 | 1 | 311 | 92 / 88 | 67,099 |

```bash
python scripts/verify_data_sources.py \
  --reference-config local/worldviews_reference.yaml \
  --reference-inventory local/worldviews_data_reference.json \
  --output local/full_source_verification.json
python scripts/probe_gpu_attention.py --device cuda:0 --output local/gpu_attention_parity.json
python scripts/prepare_real_probe.py --checkpoint /path/to/FL2VA \
  --parquet /path/to/native_multicamera.parquet --row 0 --views 0,1 \
  --frames 90 --short-frames 77 --device cuda:0 --output local/real_probe
python scripts/verify_pretrained.py --checkpoint /path/to/FL2VA \
  --features local/real_probe --output local/pretrained_final
python scripts/verify_pretrained.py --checkpoint /path/to/FL2VA \
  --features local/real_probe --output local/fresh_resume \
  --resume local/pretrained_final/stage1_resume.pt
```

Feature preparation can reuse the same source caption's already computed full-encoder features with `--reuse-text-from local/real_probe`, and the same converted VAE with `--vae-cache local/real_probe/vae`. To exercise additional real stage-2 shapes, pass their feature files with repeated `--extra-features PATH` arguments. The source audit does not need GPU execution. The other GPU commands are bounded diagnostic runs and should run on available local devices.

Only a completed `verification.json` with `status=passed` establishes success for a particular run. `progress.json` records individual successful updates; an existing checkpoint file or completed forward alone is not a successful two-stage verification. Earlier probes with the reversed Wan time/velocity convention are invalid for H3 training and their checkpoints are rejected by the corrected resume loader. The runner also checks sampled frozen-weight preservation and unchanged complete parameter topology. A few finite training steps do not establish convergence, camera-control quality, or production readiness.

### Recorded local results (2026-09-09, UTC+8)

The corrected full-pretrained run completed six actual forward/backward/AdamW updates, with `status=passed`. These use fixed real features and target noise fraction 0.5; they are execution/correctness checks, not a training curve. Timings include first-use compilation where applicable.

| Global step | Real shape (views x frames) | Loss | Gradient norm | Update seconds |
| --- | --- | --- | --- | --- |
| 1 | 1 x 77 | 0.24300867 | 0.285197 | 90.2 |
| 2 | 1 x 77 | 0.24322049 | 0.432458 | 12.8 |
| 3, after save/restore | 2 x 90 | 0.23259926 | 0.156642 | 106.6 |
| 4 | 2 x 90 | 0.23258391 | 0.155951 | 24.2 |
| 5 | 8 x 37 | 0.26231951 | 0.136742 | 166.7 |
| 6 | 1 x 297 | 0.19223946 | 0.152630 | 106.7 |

Every step changed both FP32 masters and the existing model-weight sample digest; total model parameters remained 33,122,992,896, with 7,707,046,400 trainable and zero added. All stage checkpoint trainable weights, masters, AdamW moments, hyperparameters and step counters matched after restore. First/middle/last original blocks matched upstream exactly without camera/mask. The highest per-device PyTorch allocated-memory peak over the training run was 48.30 GiB. Frozen-weight sampling and complete parameter topology checks passed. Full runtime reports/checkpoints remain ignored under `local/`.

A second, fresh process loaded the original base and the saved stage-1 checkpoint, restored global step 2, and performed the first two-view update again. It completed with `status=passed`. Its step-3 loss (0.23259925842285156), gradient norm (0.15664238807317138), and sampled model/master digests before and after the update matched the continuous run exactly. This establishes actual optimizer continuation beyond merely deserializing a checkpoint in the same process.

The separate BF16 CUDA kernel oracle compared sparse attention against dense SDPA: maximum output absolute error 0.015625, input-gradient relative error 0.000864, and Q-weight-gradient relative error 0.002863. Three-layer future-perturbation invariance was exact. All 22 CPU cases passed. The full-source audit matched all 19 live Parquet entries / 1,372,262 source rows and the complete resolved dataset section in both stages.

## Provenance

- Official source snapshot: `MiniMax-AI/MiniMax-H3@d21241f0a4b3acbb34c97dae47fa417b7065e438`. It was downloaded as an archive, so the local Git history begins from a snapshot and does not claim upstream ancestry.
- H3 transformer/Diffusers dependency: `huggingface/diffusers@d30c748f5f5d0925a5af14dc0e6a6de983025e63`; Apache license in `licenses/DIFFUSERS-APACHE-2.0.txt`, source header retained.
- Full original FL2VA weights: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`, stored outside this Git checkout in `../ckpts/MiniMax-H3/FL2VA/`. All 84 selected files / 144,051,241,571 bytes passed full checksums on 2026-09-09 at 16:59:32 UTC+8.
- WorldGen resolved full-data config SHA256: `a161359868d0412a8b307d5ce616ad3e8d1b949c5fd22c962cdc28116bf35fc5`.
