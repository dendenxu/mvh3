#!/usr/bin/env bash
# Managed HR payload for one 8-node, 64-GPU H3 training world.
set -euo pipefail
umask 077

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source "${MVH3_RUNTIME_ENV:-local/h3_experiment_env.sh}"
mvh3_python="${MVH3_PYTHON:-python}"
# Fresh HR nodes activate WorldViews under /home/tiger before a shm copy exists.
# Record whichever existing interpreter this worker uses.
if ! command -v "$mvh3_python" >/dev/null 2>&1; then
    mvh3_python="$(command -v python)"
fi
mvh3_python="$(command -v "$mvh3_python")"
export PATH="$(dirname "$mvh3_python"):$PATH"
: "${MVH3_CHECKPOINT:?Set the original FL2VA checkpoint}"
: "${MVH3_VAE:?Set the native video VAE}"
: "${MVH3_DATA_ROOT:?Set the paired data root}"
: "${MVH3_DATA_ROOT3:?Set the second data root}"
: "${ARNOLD_WORKER_0_HOST:?Missing master endpoint}"
: "${ARNOLD_WORKER_0_PORT:?Missing master port}"
: "${ARNOLD_MONITOR_TRIAL_ID:?Missing HR trial identity}"
[[ "${ARNOLD_WORKER_NUM:-}" == 8 && "${ARNOLD_WORKER_GPU:-}" == 8 ]]
[[ "${ARNOLD_ID:-}" =~ ^[0-7]$ ]]

export WORLDGEN_TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
unset GIT_DIR GIT_WORK_TREE http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
mvh3_run="${MVH3_RUN_DIR:-local/production64_${ARNOLD_MONITOR_TRIAL_ID}}"
mvh3_config="${MVH3_TRAIN_CONFIG:-configs/diffusion_forcing.yaml}"
"$mvh3_python" -m scripts.environment_report \
  --reference "${MVH3_FA4_REFERENCE:-local/attention_native_reference_v2/fa4_source_receipt.json}" \
  --output "$mvh3_run/runtime_node${ARNOLD_ID}.json"

exec "$mvh3_python" -m torch.distributed.run \
  --nnodes="$ARNOLD_WORKER_NUM" --nproc_per_node="$ARNOLD_WORKER_GPU" \
  --node_rank="$ARNOLD_ID" --master_addr="$ARNOLD_WORKER_0_HOST" \
  --master_port="${ARNOLD_WORKER_0_PORT%%,*}" --rdzv_conf=read_timeout=1800 \
  main.py -c "$mvh3_config" \
  "h3.logdir=$mvh3_run" "h3.compile_cache=$mvh3_run/compile" \
  auto_resume=true resume_ckpt=null vis_init=false \
  save_interval=500 vis_interval=500 "$@"
