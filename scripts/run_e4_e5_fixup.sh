#!/usr/bin/env bash
# Re-run only the failed parts: E4 (random/static_norm/ifpruned) + E5 + benchmark figure.
set -uo pipefail
cd /home/wonjin/ifpruning-llama31

PY=/home/wonjin/anaconda3/envs/llama_hint/bin/python
SRC=Qwen/Qwen2.5-3B-Instruct
EXTRACTOR=HuggingFaceTB/SmolLM2-360M
PROMPTS=data/prompts_diverse.json
LOGDIR=logs/master
E6_CKPT=$(ls -d checkpoints/E6/final/step_* 2>/dev/null | sort | tail -1)
echo "[fixup] using E6 ckpt: $E6_CKPT"
TASKS=mmlu,hellaswag,arc_challenge,piqa,winogrande

# E4 — random / static_norm / ifpruned (dense already done)
for VAR in random static_norm ifpruned; do
    LOG="$LOGDIR/E4/${VAR}.log"
    rm -rf results/E4/$VAR 2>/dev/null
    CKPT_ARG=""
    PRED_TOK_ARG=""
    if [[ "$VAR" == "ifpruned" ]]; then
        CKPT_ARG="--ckpt $E6_CKPT"
        PRED_TOK_ARG="--predictor_tokenizer $EXTRACTOR"
    fi
    echo "[fixup] E4 $VAR  --  $(date)"
    CUDA_VISIBLE_DEVICES=0 $PY -m eval.eval_lm_harness \
        --model_type $VAR --source_model $SRC \
        $CKPT_ARG $PRED_TOK_ARG \
        --tasks $TASKS --batch_size 4 --num_fewshot 0 \
        --static_topk 5504 \
        --output_dir results/E4/$VAR \
        > "$LOG" 2>&1
    echo "[fixup] E4 $VAR exit=$?"
done

# E5 — extract masks + analyze
echo "[fixup] E5 extract  --  $(date)"
rm -f masks/E6_predictor.pt 2>/dev/null
mkdir -p masks
CUDA_VISIBLE_DEVICES=0 $PY -m eval.extract_masks \
    --ckpt $E6_CKPT \
    --predictor_tokenizer $EXTRACTOR \
    --prompts_file $PROMPTS \
    --out masks/E6_predictor.pt \
    > "$LOGDIR/E5/extract.log" 2>&1
echo "[fixup] E5 extract exit=$?"

echo "[fixup] E5 analyze  --  $(date)"
rm -rf figures/E5 2>/dev/null
mkdir -p figures/E5
$PY -m analysis.analyze_masks \
    --masks masks/E6_predictor.pt \
    --out_dir figures/E5 \
    > "$LOGDIR/E5/analyze.log" 2>&1
echo "[fixup] E5 analyze exit=$?"

# Figure 5: E4 benchmarks
echo "[fixup] benchmark figure  --  $(date)"
$PY -m analysis.plot_benchmarks \
    --results \
      dense=results/E4/dense/dense_results.json \
      random=results/E4/random/random_results.json \
      static_norm=results/E4/static_norm/static_norm_results.json \
      ifpruned=results/E4/ifpruned/ifpruned_results.json \
    --out figures/E4_benchmark_bars \
    --csv tables/E4_benchmark_results.csv \
    > "$LOGDIR/E4/figure.log" 2>&1

echo "[fixup] === done $(date) ==="
touch $LOGDIR/FIXUP_DONE
