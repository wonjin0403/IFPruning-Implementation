#!/usr/bin/env bash
# Stage 2: IFPruning SFT on Tulu-v2 + FLAN-V2.
set -euo pipefail
cd "$(dirname "$0")/../.."

NPROC=${NPROC:-2}
accelerate launch --num_processes "$NPROC" --mixed_precision bf16 \
    -m train.train_stage2_sft \
    "$@"
