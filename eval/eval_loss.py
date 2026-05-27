from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collators import ChunkPairCollator
from models.masked_model import masked_from_pretrained
from models.topk import random_topk_mask, static_norm_topk_mask
from train.train_stage2_sft import load_stage1_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["dense", "random", "static_norm", "ifpruned"], required=True)
    p.add_argument("--source_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--ckpt", default=None, help="Stage 1/2 checkpoint dir (ifpruned variant)")
    p.add_argument("--eval_dataset", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--topk", type=int, default=4096)
    p.add_argument("--max_batches", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if args.variant == "ifpruned":
        assert args.ckpt is not None
        llama, predictor = load_stage1_checkpoint(args.ckpt)
    else:
        llama = masked_from_pretrained(args.source_model, torch_dtype=torch.bfloat16)
        predictor = None

    static_mask = None
    if args.variant == "static_norm":
        static_mask = static_norm_topk_mask(llama, k=args.topk)
        static_mask = static_mask.to(accelerator.device)

    ds = load_from_disk(args.eval_dataset)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ChunkPairCollator(pad_token_id=tok.pad_token_id),
        num_workers=args.num_workers,
    )
    llama, loader = accelerator.prepare(llama, loader)
    if predictor is not None:
        predictor = accelerator.prepare(predictor)
        predictor.eval()
    llama.eval()

    total_loss = 0.0
    total_tokens = 0
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, disable=not accelerator.is_main_process, desc=args.variant):
            if args.variant == "dense":
                ffn_mask = None
            elif args.variant == "random":
                ffn_mask = random_topk_mask(
                    batch_size=batch["llm_input_ids"].size(0),
                    num_layers=accelerator.unwrap_model(llama).config.num_hidden_layers,
                    intermediate_size=accelerator.unwrap_model(llama).config.intermediate_size,
                    k=args.topk,
                    device=accelerator.device,
                )
            elif args.variant == "static_norm":
                B = batch["llm_input_ids"].size(0)
                ffn_mask = static_mask.unsqueeze(0).expand(B, -1, -1).contiguous()
            elif args.variant == "ifpruned":
                mask, _ = predictor(
                    input_ids=batch["predictor_input_ids"],
                    attention_mask=batch["predictor_attention_mask"],
                )
                ffn_mask = mask
            else:
                raise ValueError(args.variant)

            outputs = llama(
                input_ids=batch["llm_input_ids"],
                attention_mask=batch["llm_attention_mask"],
                labels=batch["labels"],
                ffn_mask=ffn_mask,
            )
            valid = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.float().item() * valid
            total_tokens += valid
            n += 1
            if n >= args.max_batches:
                break

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    if accelerator.is_main_process:
        print(f"variant={args.variant}  loss={avg_loss:.4f}  ppl={ppl:.4f}  tokens={total_tokens}")


if __name__ == "__main__":
    main()
