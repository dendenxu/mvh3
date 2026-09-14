#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MVH3_CHECKPOINT:?Set the original FL2VA checkpoint directory}"
: "${MVH3_VAE:?Set the converted H3 video VAE directory}"
: "${MVH3_DATA_ROOT:?Set the WorldViews source root}"
: "${MVH3_DATA_ROOT3:?Set the second WorldViews source root}"

overfit_output="${1:-local/overfit_diffusion_forcing}"
overfit_features="${2:-local/overfit_diffusion_forcing/features}"
overfit_python="${MVH3_PYTHON:-python}"
overfit_config="${3:-configs/overfit_diffusion_forcing.yaml}"
mkdir -p "$overfit_output"
exec 9>"$overfit_output/runner.lock"
if ! flock -n 9; then
    echo "An overfit runner already owns $overfit_output" >&2
    exit 1
fi
overfit_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
overfit_log="$overfit_output/run_$overfit_stamp.log"
ln -sfn "$(basename "$overfit_log")" "$overfit_output/latest.log"
exec >>"$overfit_log" 2>&1
trap 'overfit_exit=$?; printf "%s\n" "$overfit_exit" > "$overfit_output/run_$overfit_stamp.exit"' EXIT

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WORLDGEN_TORCH_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MVH3_NCCL_ABORT_ON_EXIT="${MVH3_NCCL_ABORT_ON_EXIT:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
printf 'Host: %s\nStarted: %s\n' "$(hostname)" "$overfit_stamp"
"$overfit_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/overfit.py --config "$overfit_config" --features "$overfit_features" --output "$overfit_output" --stop-after 256
"$overfit_python" -m torch.distributed.run --standalone --nproc_per_node=8 \
    scripts/overfit.py --config "$overfit_config" --features "$overfit_features" --output "$overfit_output"
echo "Overfit training and decoded comparisons complete."
