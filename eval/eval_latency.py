from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.weight_gather import build_gathered_llama, mask_to_indices
from train.train_stage2_sft import load_stage1_checkpoint


@contextmanager
def cuda_timer(label: str, sync: bool = True):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if sync:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[time] {label}: {dt*1000:.2f} ms")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["dense", "ifpruned"], required=True)
    p.add_argument("--source_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--prompt", default="Explain dynamic structured pruning of LLMs in three paragraphs.")
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--n_warmup", type=int, default=2)
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if args.variant == "dense":
        model = LlamaForCausalLM.from_pretrained(
            args.source_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(args.device).eval()
    else:
        assert args.ckpt is not None
        masked_llama, predictor = load_stage1_checkpoint(args.ckpt)
        masked_llama = masked_llama.to(args.device).eval()
        predictor = predictor.to(args.device).eval()

        prompt_ids = tok(args.prompt, return_tensors="pt").to(args.device)
        with cuda_timer("predictor (per-task)"):
            with torch.no_grad():
                mask, _ = predictor(input_ids=prompt_ids["input_ids"], attention_mask=prompt_ids["attention_mask"])
            mask = mask[0]

        idx_per_layer = mask_to_indices(mask)
        with cuda_timer("weight gather"):
            model = build_gathered_llama(args.source_model, idx_per_layer, torch_dtype=torch.bfloat16).to(args.device).eval()

    inputs = tok(args.prompt, return_tensors="pt").to(args.device)

    with torch.no_grad():
        for _ in range(args.n_warmup):
            _ = model.generate(**inputs, max_new_tokens=args.gen_len, do_sample=False, pad_token_id=tok.pad_token_id)

    times = []
    with torch.no_grad():
        for _ in range(args.n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(**inputs, max_new_tokens=args.gen_len, do_sample=False, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    tok_per_s = args.gen_len / avg
    print(f"variant={args.variant}  gen_len={args.gen_len}  avg={avg*1000:.2f} ms  ({tok_per_s:.2f} tok/s)")
    print(f"  per-run (s): {['%.3f' % t for t in times]}")


if __name__ == "__main__":
    main()
