#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

SOURCE_MODEL=${SOURCE_MODEL:-/nas_data2/LLM_weight/llm/llama3.1/Llama-3.1-8B-Instruct}
EVAL_DATASET=${EVAL_DATASET:-processed_datasets/slimpajama_chunk_pairs_eval}

python -m eval.eval_loss \
    --variant dense \
    --source_model "$SOURCE_MODEL" \
    --tokenizer "$SOURCE_MODEL" \
    --eval_dataset "$EVAL_DATASET" \
    "$@"
