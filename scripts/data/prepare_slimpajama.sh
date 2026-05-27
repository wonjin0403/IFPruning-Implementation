#!/usr/bin/env bash
# Stage 1 (chunk pairs) data prep for SlimPajama.
# Step 1: tokenize with the source-LLM tokenizer for the LLM side.
# Step 2: re-tokenize the predictor side with the predictor backbone's tokenizer
#         (run data.retokenize_predictor; see README for the two-step recipe).
set -euo pipefail
cd "$(dirname "$0")/../.."

TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-3B-Instruct}
NUM_PAIRS=${NUM_PAIRS:-1000000}
OUT=${OUT:-processed_datasets/slimpajama_chunks_qwen3b}

# ---------- Stage 1: (current_chunk, next_chunk) pairs ----------
python -m data.prepare_slimpajama_chunks \
    --tokenizer "$TOKENIZER" \
    --raw_dir raw_datasets/slimpajama \
    --out_dir "$OUT" \
    --predictor_block_size 512 \
    --block_size 2048 \
    --num_pairs "$NUM_PAIRS"

# Held-out eval shard (small, deterministic) — same source stream, different out_dir.
python -m data.prepare_slimpajama_chunks \
    --tokenizer "$TOKENIZER" \
    --raw_dir raw_datasets/slimpajama \
    --out_dir "${OUT}_eval" \
    --predictor_block_size 512 \
    --block_size 2048 \
    --num_pairs 2000
