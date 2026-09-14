# MVH3

Camera-controlled, image-to-video training on MiniMax H3 using the WorldViews
recipe and paired Ours data. The model trains selected original attention
weights; camera encoding adds no parameters or separate AR branch.

## Read the code

Start at `main.py`. It loads a YAML recipe and selects training or inference.

```text
main.py
  train     -> trainer/diffusion.py -> dataset/loader.py -> model/diffusion.py
  inference -> pipeline/inference.py -> the selected sampling pipeline
                                         |
                              h3/modules/model.py
```

| Path | Responsibility |
| --- | --- |
| `trainer/diffusion.py` | Training loop, AdamW groups, gradient clipping, EMA, validation and resume |
| `dataset/loader.py` | Source batches, pixel/camera adaptation, VAE/text encoding and SP queues |
| `dataset/presampled.py` | Paired Ours Parquet sampling, decoding and source augmentation |
| `model/diffusion.py` | Chunk noise, clean-prefix selection, flow loss and resampling forcing |
| `model/chunks.py` | Random latent blocks and their physical frame intervals |
| `model/packing.py` | Token layout, masks, cameras and output-to-latent mapping |
| `pipeline/inference.py` | Image/camera request loading, raw/EMA weights and saved videos |
| `pipeline/chunked_inference.py` | Causal block generation with generated history and KV caches |
| `pipeline/full_sequence_inference.py` | Released H3 sampling and initialization parity |
| `pipeline/i2v_input.py` | Image/prompt/camera inputs without future-video access |
| `h3/modules/model.py` | The actual Transformer architecture and attention configuration |
| `h3/modules/camera.py` | Wrapped decomposed/matrix PRoPE on existing Q/K channels |
| `h3/modules/vae.py` | Native video VAE encoder, decoder and tiling |
| `h3/encoders.py` | Video VAE and native Qwen image/text features |
| `h3/checkpoint.py` | Strict original-weight mapping and loading |
| `h3/distributed/fsdp.py` | FSDP wrapping, activation checkpointing and block compilation |
| `h3/scheduler.py` | Native H3 Euler schedule; optional UniPC lives in `pipeline/unipc.py` |
| `utils/` | Shared configuration, distributed communication, captions, checkpoints, EMA and tracking |

The [code guide](docs/MVH3.md) explains the data fields and numerical contracts.
Experiment outputs and temporary diagnostics stay under ignored `local/`.

## Training

Use the existing WorldViews environment and its compiled FA4 installation.
Required paths are listed in `.env.example`; weights and datasets are external.

```bash
conda activate worldviews
export MVH3_CHECKPOINT=/path/to/MiniMax-H3/FL2VA
export MVH3_VAE=/path/to/converted/video/vae
export MVH3_DATA_ROOT=/path/to/datasets
export MVH3_DATA_ROOT3=/path/to/second/datasets
python main.py -c configs/stage1_compile_buckets.yaml --print-config
torchrun --nproc_per_node=8 main.py -c configs/stage1_compile_buckets.yaml
```

The Stage 1 recipe uses the complete paired SHORT data, resampling forcing off, context noise
0.2/std 0.1, SP8/FSDP8, CPU offload, gradient checkpointing and compilation.
It partitions each sequence into random 3-20-latent blocks before choosing the
clean-prefix cut. Only the noisy suffix contributes loss. Captions retain the
scene and motions whose source-window overlap is strictly greater than 50%.

All original attention layers use wrapped, all-frame decomposed PRoPE. Five
translation frequencies span 0.01-32; PSF and scale conditioning are off.
Every second original attention trains at 1e-5; other original weights stay
frozen. Matrix and alternating modes remain in their named camera configs.
EMA uses fixed decay 0.995. Byted-wandb records configuration, source, scalars,
media and checkpoint manifests through the existing WorldViews integration.

Resume with the saved recipe and topology:

```bash
torchrun --nproc_per_node=8 main.py -c local/run/resolved.yaml \
  resume_ckpt=local/run/ckpt/latest.json
```

Stage 2 additionally sets `h3.stage=2`. Stage transitions, learning rate warmup and resampling forcing warmup
are separate decisions. The managed 64-GPU payload is `scripts/run/train_hr.sh`.

## Inference

A request supplies a prompt, actual FPS and one image/camera description per
view. Camera files are float32 `[frames, 10]` arrays:
`[fx/width, fy/height, cx/width-0.5, cy/height-0.5, rotvec(R_c2w), C_world]`.
Paths are relative to the request JSON.

```json
{
  "prompt": "A street viewed from a moving car.",
  "fps": 24,
  "views": [{"image": "input.png", "camera": "camera.npy", "scale": 1.0}]
}
```

```bash
torchrun --nproc_per_node=8 --module scripts.infer \
  --config configs/init_wrapped.yaml --request request.json \
  --output local/inference --verify-init
```

Base weights use native full-sequence sampling. `--verify-init` requires static
cameras and checks every prediction against the no-camera base computation.
A training checkpoint selects chunked generation automatically; use its saved
configuration and `--checkpoint local/run/ckpt/latest.json`. The `--weights`
option selects raw or EMA weights. Released FL2VA uses distilled CFG 1, shift 12
and 50 Euler sigma points (49 model evaluations). Training retains actual 16 FPS.

## Development

```bash
PYTHONPATH=../python_deps:../diffusers/src python -m pytest tests -q
isort main.py h3 model trainer pipeline dataset utils scripts tests
black main.py h3 model trainer pipeline dataset utils scripts tests
```

Formatting settings are in `pyproject.toml`. Imports use separate standard-library,
third-party and repository groups, sorted by length within each group. Leave a
blank line before comments that introduce a new phase. Spell out business terms
such as resampling forcing; helper names describe their purpose without a leading
underscore. Run CLI modules from the repository root; Python entry points do not
modify `PATH` or `sys.path`. See [single-sequence validation](docs/OVERFIT.md) for the reusable
convergence command. CPU regression tests do not establish GPU convergence,
full-batch memory capacity or multi-node throughput.

## Commit hook and provenance

`origin` is `https://github.com/dendenxu/mvh3`. The WorldViews-style
`scripts/merlin/post-commit` hook mirrors committed source through
`scripts/merlin/push2bd` to `https://code.byted.org/zhenxu.zx/mvh3`; it does not
include local data or uncommitted files. A failed remote push leaves the local
commit intact.

The native implementation derives from MiniMax-H3 snapshot
`d21241f0a4b3acbb34c97dae47fa417b7065e438` and Diffusers revision
`d30c748f5f5d0925a5af14dc0e6a6de983025e63`. Runtime layers are local PyTorch;
Diffusers is used only as a pinned test oracle. Copyrights and licenses are in
`licenses/`. `scripts/convert_video_vae.py` converts the released VAE weights once.
