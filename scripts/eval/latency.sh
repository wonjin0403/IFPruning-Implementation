#!/usr/bin/env bash
# Usage: bash scripts/eval/latency.sh {dense|ifpruned}
set -euo pipefail
cd "$(dirname "$0")/../.."

VARIANT=${1:-dense}
SOURCE_MODEL=${SOURCE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
CKPT=${CKPT:-checkpoints/stage2_sft/step_00060000}
GEN_LEN=${GEN_LEN:-256}

EXTRA_ARGS=()
if [[ "$VARIANT" == "ifpruned" ]]; then
    EXTRA_ARGS+=(--ckpt "$CKPT")
fi

python -m eval.eval_latency \
    --variant "$VARIANT" \
    --source_model "$SOURCE_MODEL" \
    --tokenizer "$SOURCE_MODEL" \
    --gen_len "$GEN_LEN" \
    "${EXTRA_ARGS[@]}"
