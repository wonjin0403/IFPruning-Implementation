#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

SOURCE_MODEL=${SOURCE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
CKPT=${CKPT:-checkpoints/stage2_sft/step_00060000}
EVAL_DATASET=${EVAL_DATASET:-processed_datasets/slimpajama_chunk_pairs_eval}

python -m eval.eval_loss \
    --variant ifpruned \
    --source_model "$SOURCE_MODEL" \
    --tokenizer "$SOURCE_MODEL" \
    --ckpt "$CKPT" \
    --eval_dataset "$EVAL_DATASET" \
    "$@"
