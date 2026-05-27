#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-3B-Instruct}

python -m data.prepare_tulu_sft \
    --tokenizer "$TOKENIZER" \
    --raw_dir raw_datasets/tulu_v2 \
    --out_dir processed_datasets/tulu_v2_sft \
    --predictor_max_len 512 \
    --llm_max_len 4096
