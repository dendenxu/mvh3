#!/usr/bin/env bash
# Allocate a new WorldViews-style 64-GPU HR job after overfit acceptance.
set -euo pipefail
mvh3_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MVH3_WORLDGEN_ROOT:?Set the existing WorldGen repository path}"
exec mlx job submitv2 --caption '[4d_videorep]mvh3_pre200_64g_ema' \
  --path "$MVH3_WORLDGEN_ROOT/configs/run/8x8.yaml" -- \
  bash "$MVH3_WORLDGEN_ROOT/scripts/run/entry.sh" \
  bash "$mvh3_root/scripts/run/train_hr.sh" "$@"
