# IFPruning on Qwen2.5-3B-Instruct

Reproduction of *Instruction-Following Pruning for Large Language Models* (arXiv:2501.02086) applied to **`Qwen/Qwen2.5-3B-Instruct`**, with **`HuggingFaceTB/SmolLM2-360M`** as the sparsity-predictor backbone (in place of the paper's 302M web-pretrained backbone).

The goal is to plug a small sparsity predictor into Qwen2.5-3B-Instruct and learn an input-conditioned dynamic top-k mask over the FFN intermediate channels (`11008 → 5504` per layer, **50% pruning**), so that **~1.5 B parameters are active** per forward pass. The pipeline is generic (dispatcher in `models/masked_model.py` supports both Llama and Qwen2 source LLMs) — Llama-3.1-8B paths still work, see `--source_model` overrides.

This repository includes:
- a fully runnable, instrumented IFPruning training + evaluation pipeline,
- a 5-experiment suite (`E1`–`E6`) producing all figures and tables for the report,
- the report itself at `report/IFPruning_reproduction_report.md` describing why a small-scale reproduction (~2 B tokens) ends in mask collapse.

## Contents
- [Environment](#environment)
- [Data Preparation](#data-preparation)
- [Training](#training)
  * [Stage 1: IFPruning continued pretraining](#stage-1-ifpruning-continued-pretraining)
  * [Stage 2: IFPruning SFT](#stage-2-ifpruning-sft)
- [Evaluation](#evaluation)
- [The full report-experiment suite](#the-full-report-experiment-suite)
- [Repository layout](#repository-layout)
- [Notes & open questions](#notes--open-questions)
- [References](#references)

## Environment
```bash
conda create -n ifprune python=3.10 -y
conda activate ifprune
pip install -r requirements.txt
```

Default models / paths used in the scripts:
| Variable | Default |
|---|---|
| Source LLM (the model we prune) | `Qwen/Qwen2.5-3B-Instruct` |
| Dense reference at the pruned size | `Qwen/Qwen2.5-1.5B-Instruct` |
| Sparsity-predictor backbone | `HuggingFaceTB/SmolLM2-360M` |
| Continued-pretraining corpus | `MBZUAI-LLM/SlimPajama-627B-DC` |
| SFT corpora | `allenai/tulu-v2-sft-mixture` + `SirNeural/flan_v2` |

Override on a per-shell basis with `SOURCE_MODEL=...` / `TOKENIZER=...` env vars, or per stage by editing the YAML configs in `configs/`.

## Data Preparation

All scripts download raw HF datasets into `raw_datasets/` and write tokenized records into `processed_datasets/`. Raw shards are kept untouched. Both directories are typically symlinks to a NAS path that has lots of disk space.

Two-step prep is recommended because the source LLM (Qwen2.5-3B) and the predictor backbone (SmolLM2) use different tokenizers:

```bash
# Step 1 — Stage 1 continued-pretraining chunk pairs, Qwen2.5 tokenizer on the LLM side
python -m data.prepare_slimpajama_chunks \
    --tokenizer Qwen/Qwen2.5-3B-Instruct \
    --raw_dir raw_datasets/slimpajama \
    --out_dir processed_datasets/slimpajama_chunks_qwen3b \
    --predictor_block_size 512 --block_size 2048 \
    --num_pairs 1000000

# Step 2 — re-tokenize the predictor side (first 512 tokens of each pair) for SmolLM2
python -m data.retokenize_predictor \
    --in_dir processed_datasets/slimpajama_chunks_qwen3b \
    --out_dir processed_datasets/slimpajama_chunks_qwen3b_smollm \
    --old_tokenizer Qwen/Qwen2.5-3B-Instruct \
    --new_tokenizer HuggingFaceTB/SmolLM2-360M \
    --max_len 512 --num_proc 8

# SFT corpora (paper §3 Stage 2)
python -m data.prepare_tulu_sft  --tokenizer Qwen/Qwen2.5-3B-Instruct \
    --raw_dir raw_datasets/tulu_v2 \
    --out_dir processed_datasets/tulu_v2_sft_qwen3b
python -m data.prepare_flan_sft  --tokenizer Qwen/Qwen2.5-3B-Instruct \
    --raw_dir raw_datasets/flan_v2 \
    --out_dir processed_datasets/flan_v2_sft_qwen3b
# Re-tokenize their predictor side with SmolLM2 (same two-step pattern as above)
```

The SFT prep applies Qwen's chat template **per turn** so that `labels` mask everything outside assistant responses to `-100` (paper §9.3: "assistant tokens only, user/system → -100").

## Training

Each stage is launched via `accelerate launch`. Configs live under `configs/`. We use bf16, DDP across 4 GPUs, and `gradient_checkpointing=False` (it interacts badly with the stashed FFN-mask attribute during recomputation).

> **Stage 0 (extractor backbone pretraining)** is omitted from this repo — we use the pre-trained `HuggingFaceTB/SmolLM2-360M` as the extractor backbone, which already covers 4 T tokens of web-crawled data and is closer to the paper's 302M-class predictor than any small model we could pretrain ourselves at this budget. Re-introduce a Stage 0 if you want to fully control the predictor's initialization.

### Stage 1: IFPruning continued pretraining
Loads Qwen2.5-3B-Instruct for the masked LLM, SmolLM2-360M for the predictor backbone, and a randomly initialized 2-layer MLP mask head. Trains all three jointly on `(current_chunk, next_chunk)` pairs.
```bash
bash scripts/train/stage1.sh
```
Equivalent direct CLI:
```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m train.train_stage1_ifprune \
    source_model=Qwen/Qwen2.5-3B-Instruct \
    extractor_init=HuggingFaceTB/SmolLM2-360M \
    tokenizer=Qwen/Qwen2.5-3B-Instruct \
    +predictor_tokenizer=HuggingFaceTB/SmolLM2-360M \
    train_dataset=processed_datasets/slimpajama_chunks_qwen3b_smollm \
    pruning.topk_per_layer=5504 \
    per_device_train_batch_size=4 gradient_accumulation_steps=4 \
    gradient_checkpointing=false \
    lr_llama=1e-6 lr_predictor=1e-4 lr_scheduler=cosine \
    max_steps=15625 output_dir=checkpoints/stage1_qwen3b
```
**Frozen-Qwen variant** (predictor + mask_head only): set `lr_llama=0`. The training code automatically freezes Qwen and skips its optimizer group when `lr_llama <= 0`.

### Stage 2: IFPruning SFT
Initializes from a Stage 1 checkpoint. Trains on Tulu-v2 + FLAN-V2 with Qwen2.5's full chat template; loss is computed only on assistant tokens. The predictor sees the first user message.
```bash
bash scripts/train/stage2.sh
```
Direct CLI (with explicit `init_from`):
```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m train.train_stage2_sft \
    init_from=checkpoints/stage1_qwen3b/step_00015625 \
    train_datasets='[processed_datasets/tulu_v2_sft_qwen3b_smollm,processed_datasets/flan_v2_sft_qwen3b_smollm]' \
    tokenizer=Qwen/Qwen2.5-3B-Instruct \
    +predictor_tokenizer=HuggingFaceTB/SmolLM2-360M \
    per_device_train_batch_size=4 gradient_accumulation_steps=2 \
    gradient_checkpointing=false \
    lr_llama=1e-6 lr_predictor=1e-4 lr_scheduler=cosine \
    max_steps=5000 output_dir=checkpoints/stage2_qwen3b
```

## Evaluation

### Held-out perplexity (4 mask variants)
```bash
# Dense Qwen2.5-3B-Instruct
bash scripts/eval/dense.sh
# Random top-5504 per layer
bash scripts/eval/static_random.sh
# Static W_down norm top-5504 per layer
bash scripts/eval/static_norm.sh
# Trained IFPruning checkpoint
bash scripts/eval/ifpruned.sh
```

### LM-Evaluation-Harness benchmarks
Default tasks: MMLU, HellaSwag, ARC-Challenge, PIQA, WinoGrande (all 0-shot, likelihood-based). The paper's full task list is also available via `--tasks paper_full`.
```bash
bash scripts/eval/lm_eval_harness.sh dense
bash scripts/eval/lm_eval_harness.sh random
bash scripts/eval/lm_eval_harness.sh static_norm
bash scripts/eval/lm_eval_harness.sh ifpruned
```

### Latency benchmark
Measures throughput / first-token latency for dense vs. IFPruning-gathered weights (paper §4.4).
```bash
bash scripts/eval/latency.sh
```

### Mask diagnostic (extract + Jaccard / channel-frequency analysis)
```bash
python -m eval.extract_masks \
    --ckpt checkpoints/stage1_qwen3b/step_00015625 \
    --predictor_tokenizer HuggingFaceTB/SmolLM2-360M \
    --prompts_file data/prompts_diverse.json \
    --out masks/predictor.pt
python -m analysis.analyze_masks --masks masks/predictor.pt --out_dir figures/E5
```

## The full report-experiment suite

The report at `report/IFPruning_reproduction_report.md` is built from six controlled experiments. Reproduce them end-to-end with:
```bash
bash scripts/run_all_experiments.sh
```
This launches `E1` (joint-training LR sweep) → `E2` (frozen-Qwen LR sweep) → `E3` (sparsity sweep) → `E6` (long frozen-Qwen training, 15 625 steps) → `E4` (LM-eval-harness on 4 variants) → `E5` (mask diagnostic), then regenerates every figure under `figures/` and the CSV under `tables/`. Wallclock ≈ 26 h on 4× H200.

Per-experiment logs land in `logs/master/<E?>/<run>.log`; per-run metrics are appended to `<output_dir>/metrics.jsonl` and consumed by the plotting helpers in `analysis/`.

## Repository layout
```
IFPruning-Implementation/
  configs/
    stage1_continued_pretrain.yaml
    stage2_sft.yaml
    eval.yaml
  data/
    prepare_slimpajama_chunks.py     # Stage 1 chunk pairs (LLM-side tokenization)
    prepare_tulu_sft.py
    prepare_flan_sft.py
    retokenize_predictor.py          # swap predictor-side tokenizer (e.g. Qwen → SmolLM2)
    prompts_diverse.json             # 100-prompt mask diagnostic set, 5 categories
    collators.py
  models/
    masked_llama.py                  # Masked Llama (paper Llama-3.1 path, still supported)
    masked_qwen2.py                  # Masked Qwen2 (this work's main path)
    masked_model.py                  # dispatcher (model_type → masked class)
    sparsity_predictor.py            # extractor + 2-layer MLP head, SoftTopK
    topk.py                          # SoftTopK (bisection), ST, random / static-norm baselines
    weight_gather.py                 # inference-time hard-pruned FFN gather
  train/
    train_stage1_ifprune.py          # main training (logs metrics.jsonl)
    train_stage2_sft.py              # SFT continuation (load_stage1_checkpoint)
  eval/
    eval_loss.py                     # held-out chunk-pair perplexity, 4 variants
    eval_lm_harness.py               # MMLU / HellaSwag / ARC / PIQA / WinoGrande
    eval_latency.py                  # tok / s benchmark with weight gather
    extract_masks.py                 # dump masks for diagnostic set
  analysis/
    plot_loss_curves.py              # loss / grad-norm curves from metrics.jsonl
    plot_benchmarks.py               # lm-eval bar charts + CSV
    analyze_masks.py                 # Jaccard heatmap + channel freq + stats
    save_fig.py                      # save both .pdf and .png; report-friendly style
  scripts/
    data/    prepare_*.sh
    train/   stage1.sh, stage2.sh
    eval/    dense.sh, static_*.sh, ifpruned.sh, lm_eval_harness.sh, latency.sh
    run_all_experiments.sh           # end-to-end report-experiment suite
  figures/                           # PDF + PNG, regenerated by analysis/*.py
  tables/                            # CSV outputs
  report/IFPruning_reproduction_report.md
```

## Notes & open questions
The following items in the implementation plan are unspecified in the paper and worth ablating:

- **Mask head bottleneck (`head_hidden_dim`)**: currently `Linear(960, 128) → GELU → Linear(128, 36·11008 = 396288)`. The 128-D bottleneck may be too narrow given the ~400 K-D output. Ablate against 512 / 1024 / 2048.
- **Predictor input truncation (`predictor_max_len`)**: default 512. Long instructions may need 1024 / 2048.
- **Mask granularity at inference**: per-input (default) vs. per-task (paper §4.3). The latter amortizes the materialize cost.
- **Top-k aggressiveness**: at our data scale, 50 % (top-k = 5504) is already on the edge; 25 % does not converge in 3 K steps (see report §6, E3).
- **Joint vs. frozen Qwen for Stage 1**: short-run (E1, 500 steps) shows joint converges; long-run frozen-Qwen (E6, 15 625 steps) is what produced our main checkpoint. The full 4-case matrix (Stage 1 × Stage 2 trainability) is a natural follow-up.

## References
- IFPruning paper: <https://arxiv.org/abs/2501.02086>
- Qwen2.5-3B-Instruct: <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>
- Qwen2.5-1.5B-Instruct (dense reference): <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- SmolLM2-360M (predictor backbone): <https://huggingface.co/HuggingFaceTB/SmolLM2-360M>
- SlimPajama (continued-pretraining corpus): <https://huggingface.co/datasets/MBZUAI-LLM/SlimPajama-627B-DC>
- Tulu-v2 SFT mixture: <https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture>
- FLAN-V2 (SirNeural mirror): <https://huggingface.co/datasets/SirNeural/flan_v2>
- lm-evaluation-harness: <https://github.com/EleutherAI/lm-evaluation-harness>
