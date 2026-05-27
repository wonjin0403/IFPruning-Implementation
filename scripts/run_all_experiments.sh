#!/usr/bin/env bash
# Master experiment runner for the IFPruning reproduction report.
#
# Order: E1 -> E2 -> E3 -> E6 -> E4 -> E5 -> figures
# Total wallclock: ~29h on 4 H200.
# Logs: logs/master/<experiment>/*.log; metrics.jsonl saved in each output_dir.
# Run inside tmux:  tmux new -s ifprune_master 'bash scripts/run_all_experiments.sh'
set -uo pipefail
cd /home/wonjin/ifpruning-llama31

PY=/home/wonjin/anaconda3/envs/llama_hint/bin/python
ACC=/home/wonjin/anaconda3/envs/llama_hint/bin/accelerate

# Shared paths
SRC=Qwen/Qwen2.5-3B-Instruct
EXTRACTOR=HuggingFaceTB/SmolLM2-360M
DATA=processed_datasets/slimpajama_chunks_qwen3b_smollm
DATA_EVAL=processed_datasets/slimpajama_chunks_qwen3b_eval_smollm
PROMPTS=data/prompts_diverse.json

# Shared training defaults
ACC_LAUNCH="$ACC launch --num_processes 4 --num_machines 1 --dynamo_backend no --mixed_precision bf16"
COMMON_OVR=(
    "train_dataset=$DATA"
    "source_model=$SRC"
    "extractor_init=$EXTRACTOR"
    "tokenizer=$SRC"
    "+predictor_tokenizer=$EXTRACTOR"
    "per_device_train_batch_size=4"
    "gradient_accumulation_steps=4"
    "gradient_checkpointing=false"
    "warmup_ratio=0.05"
    "report_to=none"
    "num_workers=2"
)

LOGDIR=logs/master
mkdir -p $LOGDIR/{E1,E2,E3,E6,E4,E5}

# Helper
run_train() {
    local NAME=$1
    shift
    local LOG=$1
    shift
    echo ""
    echo "=========================================="
    echo "[master] $NAME  --  $(date)"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0,1,2,3 $ACC_LAUNCH \
        -m train.train_stage1_ifprune \
        "${COMMON_OVR[@]}" \
        "$@" \
        > "$LOG" 2>&1
    local rc=$?
    echo "[master] $NAME exit=$rc  --  $(date)"
    return $rc
}

# =============================================================
# E1. Joint LR sweep (500 step × 6 LRs, ~3h)
# =============================================================
for LR in 0 1e-8 1e-7 1e-6 5e-6 1e-5; do
    run_train "E1 lr_llama=$LR" "$LOGDIR/E1/lr_llama_${LR}.log" \
        "output_dir=checkpoints/E1/lr_llama_${LR}" \
        "max_steps=500" "save_steps=500" "logging_steps=10" \
        "lr_llama=$LR" "lr_predictor=1e-4" "lr_scheduler=constant_with_warmup" \
        "pruning.topk_per_layer=5504"
done

# =============================================================
# E2. Frozen Qwen LR sweep (3000 step × 4 LRs, ~6.7h)
# =============================================================
for LR in 5e-5 1e-4 3e-4 1e-3; do
    run_train "E2 lr_pred=$LR" "$LOGDIR/E2/lr_pred_${LR}.log" \
        "output_dir=checkpoints/E2/lr_pred_${LR}" \
        "max_steps=3000" "save_steps=3000" "logging_steps=20" \
        "lr_llama=0" "lr_predictor=$LR" "lr_scheduler=constant_with_warmup" \
        "pruning.topk_per_layer=5504"
done

# =============================================================
# E3. Sparsity sweep (3000 step × 3 top-k, ~5h)
# Best lr_predictor from E2 read here. We default to 1e-4; override after E2 if needed.
# =============================================================
# TODO: replace BEST_LR_PRED with the actual best after E2 (manual edit or analysis script)
BEST_LR_PRED=${BEST_LR_PRED:-1e-4}
for TOPK in 2752 5504 8256; do
    run_train "E3 topk=$TOPK" "$LOGDIR/E3/topk_${TOPK}.log" \
        "output_dir=checkpoints/E3/topk_${TOPK}" \
        "max_steps=3000" "save_steps=3000" "logging_steps=20" \
        "lr_llama=0" "lr_predictor=$BEST_LR_PRED" "lr_scheduler=constant_with_warmup" \
        "pruning.topk_per_layer=$TOPK"
done

# =============================================================
# E6. Long Frozen Qwen final (1 epoch, ~9h)
# =============================================================
run_train "E6 final" "$LOGDIR/E6/final.log" \
    "output_dir=checkpoints/E6/final" \
    "max_steps=15625" "save_steps=3000" "logging_steps=50" \
    "lr_llama=0" "lr_predictor=$BEST_LR_PRED" "lr_scheduler=cosine" \
    "pruning.topk_per_layer=5504"

# =============================================================
# E4. LM-Eval benchmarks on E6 final (~3h)
# =============================================================
E6_CKPT=$(ls -d checkpoints/E6/final/step_* 2>/dev/null | sort | tail -1)
echo "[master] E4 using E6 ckpt: $E6_CKPT"
TASKS=mmlu,hellaswag,arc_challenge,piqa,winogrande
for VAR in dense random static_norm ifpruned; do
    LOG="$LOGDIR/E4/${VAR}.log"
    CKPT_ARG=""
    PRED_TOK_ARG=""
    if [[ "$VAR" == "ifpruned" ]]; then
        CKPT_ARG="--ckpt $E6_CKPT"
        PRED_TOK_ARG="--predictor_tokenizer $EXTRACTOR"
    fi
    echo "[master] E4 $VAR  --  $(date)"
    CUDA_VISIBLE_DEVICES=0 $PY -m eval.eval_lm_harness \
        --model_type $VAR --source_model $SRC \
        $CKPT_ARG $PRED_TOK_ARG \
        --tasks $TASKS --batch_size 4 --num_fewshot 0 \
        --static_topk 5504 \
        --output_dir results/E4/$VAR \
        > "$LOG" 2>&1
    echo "[master] E4 $VAR exit=$?"
done

# =============================================================
# E5. Mask analysis on E6 final (~30min)
# =============================================================
echo "[master] E5 extract masks  --  $(date)"
CUDA_VISIBLE_DEVICES=0 $PY -m eval.extract_masks \
    --ckpt $E6_CKPT \
    --predictor_tokenizer $EXTRACTOR \
    --prompts_file $PROMPTS \
    --out masks/E6_predictor.pt \
    > "$LOGDIR/E5/extract.log" 2>&1

echo "[master] E5 analyze  --  $(date)"
$PY -m analysis.analyze_masks \
    --masks masks/E6_predictor.pt \
    --out_dir figures/E5 \
    > "$LOGDIR/E5/analyze.log" 2>&1

# =============================================================
# Figure generation
# =============================================================
echo ""
echo "=========================================="
echo "[master] Generating figures  --  $(date)"
echo "=========================================="

# Figure 1: E1 joint LR sweep
$PY -m analysis.plot_loss_curves \
    --runs \
      checkpoints/E1/lr_llama_0/metrics.jsonl=lr_llama=0 \
      checkpoints/E1/lr_llama_1e-8/metrics.jsonl=1e-8 \
      checkpoints/E1/lr_llama_1e-7/metrics.jsonl=1e-7 \
      checkpoints/E1/lr_llama_1e-6/metrics.jsonl=1e-6 \
      checkpoints/E1/lr_llama_5e-6/metrics.jsonl=5e-6 \
      checkpoints/E1/lr_llama_1e-5/metrics.jsonl=1e-5 \
    --metric loss --smooth 3 \
    --title "E1: Joint training LR sweep (Qwen2.5-3B, 500 steps)" \
    --out figures/E1_loss_curves

# Figure 1b: gradient norms (Llama)
$PY -m analysis.plot_loss_curves \
    --runs \
      checkpoints/E1/lr_llama_1e-7/metrics.jsonl=1e-7 \
      checkpoints/E1/lr_llama_1e-6/metrics.jsonl=1e-6 \
      checkpoints/E1/lr_llama_5e-6/metrics.jsonl=5e-6 \
      checkpoints/E1/lr_llama_1e-5/metrics.jsonl=1e-5 \
    --metric grad_norm_llama --smooth 3 --ylog \
    --title "E1: Llama gradient norm during joint training" \
    --out figures/E1_grad_norm_llama

# Figure 2: E2 frozen LR sweep
$PY -m analysis.plot_loss_curves \
    --runs \
      checkpoints/E2/lr_pred_5e-5/metrics.jsonl=lr_pred=5e-5 \
      checkpoints/E2/lr_pred_1e-4/metrics.jsonl=1e-4 \
      checkpoints/E2/lr_pred_3e-4/metrics.jsonl=3e-4 \
      checkpoints/E2/lr_pred_1e-3/metrics.jsonl=1e-3 \
    --metric loss --smooth 5 \
    --title "E2: Frozen Qwen predictor LR sweep (3000 steps)" \
    --out figures/E2_loss_curves

# Figure 3: E3 sparsity sweep
$PY -m analysis.plot_loss_curves \
    --runs \
      "checkpoints/E3/topk_2752/metrics.jsonl=top-k=2752 (25%)" \
      "checkpoints/E3/topk_5504/metrics.jsonl=5504 (50%)" \
      "checkpoints/E3/topk_8256/metrics.jsonl=8256 (75%)" \
    --metric loss --smooth 5 \
    --title "E3: Sparsity sweep (frozen Qwen, 3000 steps)" \
    --out figures/E3_topk_sweep

# Figure 4: E6 final long training
$PY -m analysis.plot_loss_curves \
    --runs \
      "checkpoints/E6/final/metrics.jsonl=Frozen-Qwen final" \
    --metric loss --smooth 10 \
    --title "E6: Long frozen Qwen training (15625 steps, 1 epoch over 1M pairs)" \
    --out figures/E6_loss_curve

# Figure 5: E4 benchmarks
$PY -m analysis.plot_benchmarks \
    --results \
      dense=results/E4/dense/dense_results.json \
      random=results/E4/random/random_results.json \
      static_norm=results/E4/static_norm/static_norm_results.json \
      ifpruned=results/E4/ifpruned/ifpruned_results.json \
    --out figures/E4_benchmark_bars \
    --csv tables/E4_benchmark_results.csv

echo ""
echo "[master] === ALL DONE  --  $(date) ==="
touch $LOGDIR/DONE
