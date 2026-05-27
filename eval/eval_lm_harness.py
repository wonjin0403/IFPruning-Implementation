from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.masked_model import masked_from_pretrained
from models.topk import random_topk_mask, static_norm_topk_mask
from train.train_stage2_sft import load_stage1_checkpoint


PAPER_FULL_TASKS = [
    "mmlu",
    "gsm8k",
    "hendrycks_math",
    "humaneval",
    "mbpp",
    "ifeval",
    "alpaca_eval",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "piqa",
]


def build_dense_lm(model_name_or_path: str, batch_size: int, device: str):
    from lm_eval.models.huggingface import HFLM
    return HFLM(pretrained=model_name_or_path, batch_size=batch_size, device=device, dtype="bfloat16")


def build_ifpruning_lm(ckpt: str, source_tokenizer: str, batch_size: int, device: str,
                      mode: str = "ifpruned",
                      static_topk: int = 4096,
                      predictor_tokenizer: str = None,
                      predictor_max_len: int = 512):
    from lm_eval.models.huggingface import HFLM

    class IFPruningHFLM(HFLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._predictor = None
            self._predictor_tokenizer = None
            self._predictor_max_len = predictor_max_len
            self._ifprune_mode = mode
            self._static_mask = None
            self._static_topk = static_topk

        def set_predictor(self, predictor, predictor_tokenizer=None):
            self._predictor = predictor
            self._predictor_tokenizer = predictor_tokenizer

        def set_static_mask(self, mask):
            self._static_mask = mask

        def _build_mask_for_inputs(self, inps: torch.Tensor, attn: torch.Tensor) -> Optional[torch.Tensor]:
            base = self.model
            cfg = base.config
            B = inps.size(0)
            if self._ifprune_mode == "dense":
                return None
            if self._ifprune_mode == "random":
                return random_topk_mask(B, cfg.num_hidden_layers, cfg.intermediate_size,
                                        self._static_topk, device=inps.device)
            if self._ifprune_mode == "static_norm":
                if self._static_mask is None:
                    self._static_mask = static_norm_topk_mask(base, k=self._static_topk).to(inps.device)
                return self._static_mask.unsqueeze(0).expand(B, -1, -1).contiguous()
            if self._ifprune_mode == "ifpruned":
                assert self._predictor is not None
                if self._predictor_tokenizer is not None and self._predictor_tokenizer is not self.tokenizer:
                    texts = self.tokenizer.batch_decode(inps, skip_special_tokens=True)
                    pred_enc = self._predictor_tokenizer(
                        texts,
                        max_length=self._predictor_max_len,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        add_special_tokens=True,
                    )
                    pred_ids = pred_enc["input_ids"].to(inps.device)
                    pred_attn = pred_enc["attention_mask"].to(inps.device)
                else:
                    pred_ids, pred_attn = inps, attn
                with torch.no_grad():
                    mask, _ = self._predictor(input_ids=pred_ids, attention_mask=pred_attn)
                return mask
            raise ValueError(self._ifprune_mode)

        def _model_call(self, inps, attn_mask=None, labels=None):
            base = self.model
            if attn_mask is None:
                attn_mask = torch.ones_like(inps)
            ffn_mask = self._build_mask_for_inputs(inps, attn_mask)
            with torch.no_grad():
                return base(input_ids=inps, attention_mask=attn_mask, ffn_mask=ffn_mask).logits

        def _model_generate(self, context, max_length, stop=None, **gen_kwargs):
            base = self.model
            inps = context if isinstance(context, torch.Tensor) else context["input_ids"]
            attn = torch.ones_like(inps) if not isinstance(context, dict) else context.get("attention_mask", torch.ones_like(inps))
            ffn_mask = self._build_mask_for_inputs(inps, attn)
            base._set_ffn_mask(ffn_mask)
            try:
                return super()._model_generate(context, max_length, stop=stop, **gen_kwargs)
            finally:
                base._set_ffn_mask(None)

    tokenizer = AutoTokenizer.from_pretrained(source_tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mode == "ifpruned":
        llama, predictor = load_stage1_checkpoint(ckpt)
        llama = llama.to(device).eval()
        lm = IFPruningHFLM(
            pretrained=llama,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            dtype="bfloat16",
        )
        pred_tok_obj = None
        if predictor_tokenizer:
            pred_tok_obj = AutoTokenizer.from_pretrained(predictor_tokenizer, use_fast=True)
            if pred_tok_obj.pad_token_id is None:
                pred_tok_obj.pad_token = pred_tok_obj.eos_token
        lm.set_predictor(predictor.to(device).eval(), predictor_tokenizer=pred_tok_obj)
    else:
        llama = masked_from_pretrained(source_tokenizer, torch_dtype=torch.bfloat16).to(device).eval()
        lm = IFPruningHFLM(
            pretrained=llama,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            dtype="bfloat16",
        )
    return lm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["dense", "random", "static_norm", "ifpruned"], default="ifpruned")
    p.add_argument("--ckpt", default=None, help="Required for --model_type ifpruned")
    p.add_argument("--source_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--predictor_tokenizer", default=None,
                   help="Separate tokenizer for the predictor side (e.g., SmolLM2). Default: same as --source_model")
    p.add_argument("--tasks", default="mmlu,gsm8k,arc_easy,arc_challenge,hellaswag,winogrande,piqa,ifeval",
                   help="Comma-separated lm-eval task names. Use 'paper_full' for the full paper list.")
    p.add_argument("--num_fewshot", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_dir", default="results/lm_eval")
    p.add_argument("--static_topk", type=int, default=4096)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import lm_eval

    if args.model_type == "dense":
        lm = build_dense_lm(args.source_model, args.batch_size, args.device)
    else:
        lm = build_ifpruning_lm(
            ckpt=args.ckpt,
            source_tokenizer=args.source_model,
            batch_size=args.batch_size,
            device=args.device,
            mode=args.model_type,
            static_topk=args.static_topk,
            predictor_tokenizer=args.predictor_tokenizer,
        )

    tasks = PAPER_FULL_TASKS if args.tasks == "paper_full" else [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"running lm-eval on {tasks}")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    import json
    out = os.path.join(args.output_dir, f"{args.model_type}_results.json")
    with open(out, "w") as f:
        json.dump(results["results"], f, indent=2)
    print(f"wrote {out}")
    print(json.dumps(results["results"], indent=2))


if __name__ == "__main__":
    main()
