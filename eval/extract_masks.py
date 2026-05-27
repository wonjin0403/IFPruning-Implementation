from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_stage2_sft import load_stage1_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Stage 1/2 checkpoint dir")
    p.add_argument("--predictor_tokenizer", required=True,
                   help="Tokenizer for the predictor side (e.g., HuggingFaceTB/SmolLM2-360M)")
    p.add_argument("--prompts_file", required=True, help="JSON: [{prompt, category}, ...]")
    p.add_argument("--out", required=True, help="Output .pt file")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.prompts_file) as f:
        entries = json.load(f)
    prompts = [e["prompt"] for e in entries]
    categories = [e["category"] for e in entries]
    print(f"Loaded {len(prompts)} prompts.")

    tok = AutoTokenizer.from_pretrained(args.predictor_tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"Loading checkpoint {args.ckpt} ...")
    _, predictor = load_stage1_checkpoint(args.ckpt)
    predictor = predictor.to(args.device).eval()

    L = predictor.num_layers
    D = predictor.intermediate_size
    K = predictor.topk

    soft_masks = torch.zeros(len(prompts), L, D, dtype=torch.float32)
    hard_indices = torch.zeros(len(prompts), L, K, dtype=torch.long)

    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            enc = tok(batch, max_length=args.max_len, truncation=True, padding=True, return_tensors="pt").to(args.device)
            mask, _ = predictor(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            soft_masks[i : i + mask.size(0)] = mask.float().cpu()
            idx = mask.topk(K, dim=-1).indices
            hard_indices[i : i + mask.size(0)] = idx.cpu()
            if i % 50 == 0:
                print(f"  {i}/{len(prompts)}")

    payload = {
        "prompts": prompts,
        "categories": categories,
        "soft_masks": soft_masks,
        "hard_indices": hard_indices,
        "meta": {
            "ckpt": args.ckpt,
            "predictor_tokenizer": args.predictor_tokenizer,
            "num_layers": L,
            "intermediate_size": D,
            "topk": K,
        },
    }
    torch.save(payload, args.out)
    print(f"Saved -> {args.out}  (soft_masks {tuple(soft_masks.shape)}, hard_indices {tuple(hard_indices.shape)})")


if __name__ == "__main__":
    main()
