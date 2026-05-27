#!/usr/bin/env bash
# Usage: bash scripts/eval/lm_eval_harness.sh {dense|random|static_norm|ifpruned}
set -euo pipefail
cd "$(dirname "$0")/../.."

VARIANT=${1:-dense}
SOURCE_MODEL=${SOURCE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
CKPT=${CKPT:-checkpoints/stage2_sft/step_00060000}
TASKS=${TASKS:-paper_full}
BATCH_SIZE=${BATCH_SIZE:-4}
OUTPUT_DIR=${OUTPUT_DIR:-results/lm_eval/${VARIANT}}

EXTRA_ARGS=()
if [[ "$VARIANT" == "ifpruned" ]]; then
    EXTRA_ARGS+=(--ckpt "$CKPT")
fi

python -m eval.eval_lm_harness \
    --model_type "$VARIANT" \
    --source_model "$SOURCE_MODEL" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
