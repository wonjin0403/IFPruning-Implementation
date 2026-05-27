#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

SOURCE_MODEL=${SOURCE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
EVAL_DATASET=${EVAL_DATASET:-processed_datasets/slimpajama_chunk_pairs_eval}

python -m eval.eval_loss \
    --variant random \
    --source_model "$SOURCE_MODEL" \
    --tokenizer "$SOURCE_MODEL" \
    --eval_dataset "$EVAL_DATASET" \
    --topk 4096 \
    "$@"
