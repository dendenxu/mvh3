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

Set `MVH3_DATA_ROOT` and `MVH3_DATA_ROOT3` to the two local dataset roots when implementing/resolving the data adapter. The exact original reference and inventory are preserved under ignored `local/`. These files are reference recipes, not an executable full-data distributed training launcher.

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Source mixture | All 19 sources | Same 19 sources |
| Video size/cadence | 448x832, 16 FPS | Same |
| Duration | At most 77 actual sampled frames per independent clip | Full reference duration distribution |
| View relationship | One view per batch item; all source views retained | Cross-view attention for synchronized multiview samples |
| Parameters | Original main attention only; zero added | Same topology, continued weights/optimizer/global step |
| Camera | Decomposed PRoPE on existing spatial Q/K channels | Same encoding |

The 77-frame maximum is the physical limit of WorldGen's 20-Wan-latent short stage. It is not 20 H3 latents. `short_mono_windows` is a source-frame split primitive and keeps all partial tails; it is not the old Wan presampled splitter. H3 needs its own VAE padding/pose mapping and regenerated caches. Stage lengths remain unset until a real pretrained memory/throughput check and learning curves inform the decision.

The reference context noise (0.2, std 0.1), history dropout (0.1), RF warmup (10,000), accumulation (1), and SP/FSDP sizes (8) are recorded in both recipes. Their H3 training/runtime integration is still pending. The old 5-latent AR chunk is recorded explicitly as a Wan reference rather than mislabeled as an equivalent H3 chunk.

## Visibility and training boundaries

`mvh3.masking.TokenLayout` defines visibility for the whole packed sequence. CONDITION tokens read only available CONDITION tokens. CLEAN media read available conditions and clean chunks up to the current chunk. NOISY media read available conditions, earlier CLEAN chunks, and their own NOISY chunk. Noisy targets cannot see same-chunk clean answers. Global conditions cannot collect local conditions and relay them elsewhere. Stage-1 views belong on the batch axis; explicit scope masks also support isolation within a document.

The text refiner receives the text restriction of the full dense mask. For sparse masks, callers must supply its text mask explicitly. This prevents future chunk captions from leaking through text refinement before the joint blocks. The module exposes dense reference masks with a size guard and PyTorch flex block-mask construction for larger layouts.

This port currently rejects camera/masked context parallel execution until the sequence-sharding adapter is implemented and validated. The inherited native H3 context-parallel hooks alone cannot shard camera indices or guarantee global mask semantics. Do not bypass this check by setting SP=8 only in a config.

Remaining work before a full-data cluster run: original-to-Diffusers checkpoint conversion, H3 Qwen3-VL layer-50 text and 24-channel VAE caches, exact physical-time/pose/loss-padding alignment, raw-data adapter, RF rollout/KV caching, and distributed masked attention plus save/resume. The existing Wan T5/16-channel latent caches and Wan AR weights cannot be loaded into this model. No cluster job is launched by the setup or tests.

## Dependencies and validation

The H3 transformer requires the pinned Diffusers commit in `pyproject.toml`. The downloaded bundle can use sibling `diffusers/src` and `python_deps` without upgrading the WorldGen environment. For a standalone checkout, install into a dedicated environment with `pip install -e '.[test]'`; use an appropriate existing PyTorch/CUDA build.

```bash
python scripts/test_cpu.py
```

Tests use tiny random CPU models. They compare parameter topology/state keys and no-camera outputs against upstream, check camera response and preserved temporal/tail channels, exercise attention-only optimizer updates with and without activation checkpointing, round-trip model/optimizer state across a stage boundary, and perturb future clean video/audio and local captions across three layers. They also verify stage-1 batch independence and full source/view/tail preservation. Passing these tests does not establish pretrained generation quality, production GPU memory, or distributed readiness.

## Provenance

- Official source snapshot: `MiniMax-AI/MiniMax-H3@d21241f0a4b3acbb34c97dae47fa417b7065e438`. It was downloaded as an archive, so the local Git history begins from a snapshot and does not claim upstream ancestry.
- H3 transformer/Diffusers dependency: `huggingface/diffusers@d30c748f5f5d0925a5af14dc0e6a6de983025e63`; Apache license in `licenses/DIFFUSERS-APACHE-2.0.txt`, source header retained.
- Full original FL2VA weights: `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`, stored outside this Git checkout in `../ckpts/MiniMax-H3/FL2VA/`. All 84 selected files / 144,051,241,571 bytes passed full checksums on 2026-09-09 at 16:59:32 UTC+8.
- WorldGen resolved full-data config SHA256: `a161359868d0412a8b307d5ce616ad3e8d1b949c5fd22c962cdc28116bf35fc5`.
