from __future__ import annotations

import argparse
import os

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="/nas_data2/LLM_weight/llm/llama3.1/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset_name", default="MBZUAI-LLM/SlimPajama-627B-DC")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--raw_dir", default="raw_datasets/slimpajama")
    p.add_argument("--out_dir", default="processed_datasets/slimpajama_chunk_pairs")
    p.add_argument("--predictor_block_size", type=int, default=512)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--num_pairs", type=int, default=200_000)
    p.add_argument("--text_field", default="text")
    p.add_argument("--streaming", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_dir) or ".", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    raw = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        streaming=args.streaming,
        cache_dir=args.raw_dir,
    )

    P = args.predictor_block_size
    B = args.block_size
    pred_in: list[list[int]] = []
    pred_attn: list[list[int]] = []
    llm_in: list[list[int]] = []
    llm_attn: list[list[int]] = []

    pbar = tqdm(total=args.num_pairs, desc="chunk pairs")
    for ex in raw:
        text = ex.get(args.text_field, "")
        if not text:
            continue
        ids = tok(text, add_special_tokens=False).input_ids
        i = 0
        while i + P + B <= len(ids):
            cur = ids[i : i + P]
            nxt = ids[i + P : i + P + B]
            pred_in.append(cur)
            pred_attn.append([1] * P)
            llm_in.append(nxt)
            llm_attn.append([1] * B)
            i += P + B
            pbar.update(1)
            if len(pred_in) >= args.num_pairs:
                break
        if len(pred_in) >= args.num_pairs:
            break
    pbar.close()

    ds = Dataset.from_dict({
        "predictor_input_ids": pred_in,
        "predictor_attention_mask": pred_attn,
        "llm_input_ids": llm_in,
        "llm_attention_mask": llm_attn,
        "labels": llm_in,
    })
    ds.save_to_disk(args.out_dir)
    print(f"wrote {len(pred_in)} (pred={P}, llm={B}) chunk pairs -> {args.out_dir}")


if __name__ == "__main__":
    main()
