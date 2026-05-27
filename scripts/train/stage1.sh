#!/usr/bin/env bash
# Stage 1: IFPruning continued pretraining on SlimPajama chunk pairs.
set -euo pipefail
cd "$(dirname "$0")/../.."

NPROC=${NPROC:-2}
accelerate launch --num_processes "$NPROC" --mixed_precision bf16 \
    -m train.train_stage1_ifprune \
    "$@"
