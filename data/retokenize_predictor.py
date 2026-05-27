from __future__ import annotations

import argparse
import os

from datasets import load_from_disk
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--old_tokenizer", required=True)
    p.add_argument("--new_tokenizer", required=True)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"loading {args.in_dir}")
    ds = load_from_disk(args.in_dir)
    print(f"  rows: {len(ds)}  features: {list(ds.features)}")

    old_tok = AutoTokenizer.from_pretrained(args.old_tokenizer, use_fast=True)
    new_tok = AutoTokenizer.from_pretrained(args.new_tokenizer, use_fast=True)
    if new_tok.pad_token_id is None:
        new_tok.pad_token = new_tok.eos_token

    max_len = args.max_len

    def reprep(batch):
        texts = old_tok.batch_decode(batch["predictor_input_ids"], skip_special_tokens=True)
        enc = new_tok(
            texts,
            max_length=max_len,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        batch["predictor_input_ids"] = enc["input_ids"]
        batch["predictor_attention_mask"] = enc["attention_mask"]
        return batch

    ds2 = ds.map(
        reprep,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="retokenize predictor",
    )

    os.makedirs(os.path.dirname(args.out_dir) or ".", exist_ok=True)
    ds2.save_to_disk(args.out_dir)
    print(f"wrote {len(ds2)} rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
