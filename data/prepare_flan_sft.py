from __future__ import annotations

import argparse
import os

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from data.prepare_tulu_sft import render_with_assistant_mask, IGNORE_INDEX


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset_name", default="SirNeural/flan_v2",
                   help="Alternative: chiayewken/flan-v2")
    p.add_argument("--split", default="train")
    p.add_argument("--raw_dir", default="raw_datasets/flan_v2")
    p.add_argument("--out_dir", default="processed_datasets/flan_v2_sft")
    p.add_argument("--predictor_max_len", type=int, default=512)
    p.add_argument("--llm_max_len", type=int, default=4096)
    p.add_argument("--max_samples", type=int, default=800_000)
    p.add_argument("--input_field", default="inputs")
    p.add_argument("--target_field", default="targets")
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
        split=args.split,
        streaming=args.streaming,
        cache_dir=args.raw_dir,
    )

    pred_in, pred_attn = [], []
    llm_in, llm_attn, labels = [], [], []

    pbar = tqdm(total=args.max_samples, desc="flan-v2 sft")
    for ex in raw:
        instr = ex.get(args.input_field) or ex.get("instruction") or ""
        resp = ex.get(args.target_field) or ex.get("response") or ""
        if not instr or not resp:
            continue
        msgs = [{"role": "user", "content": instr},
                {"role": "assistant", "content": resp}]

        p = tok(instr, max_length=args.predictor_max_len, truncation=True, add_special_tokens=True)
        pred_in.append(p["input_ids"])
        pred_attn.append(p["attention_mask"])
        ids, attn, lab = render_with_assistant_mask(tok, msgs, args.llm_max_len)
        llm_in.append(ids)
        llm_attn.append(attn)
        labels.append(lab)
        pbar.update(1)
        if len(pred_in) >= args.max_samples:
            break
    pbar.close()

    ds = Dataset.from_dict({
        "predictor_input_ids": pred_in,
        "predictor_attention_mask": pred_attn,
        "llm_input_ids": llm_in,
        "llm_attention_mask": llm_attn,
        "labels": labels,
    })
    ds.save_to_disk(args.out_dir)
    print(f"wrote {len(pred_in)} flan sft samples -> {args.out_dir}")


if __name__ == "__main__":
    main()
