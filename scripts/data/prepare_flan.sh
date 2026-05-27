#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-3B-Instruct}

python -m data.prepare_flan_sft \
    --tokenizer "$TOKENIZER" \
    --raw_dir raw_datasets/flan_v2 \
    --out_dir processed_datasets/flan_v2_sft \
    --predictor_max_len 512 \
    --llm_max_len 4096 \
    --max_samples 800000
