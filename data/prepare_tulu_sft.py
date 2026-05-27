from __future__ import annotations

import argparse
import os
from typing import List, Dict

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

IGNORE_INDEX = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="/nas_data2/LLM_weight/llm/llama3.1/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset_name", default="allenai/tulu-v2-sft-mixture")
    p.add_argument("--split", default="train")
    p.add_argument("--raw_dir", default="raw_datasets/tulu_v2")
    p.add_argument("--out_dir", default="processed_datasets/tulu_v2_sft")
    p.add_argument("--predictor_max_len", type=int, default=512)
    p.add_argument("--llm_max_len", type=int, default=4096)
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def normalize_messages(messages: List[Dict]) -> List[Dict]:
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant", "system"):
            out.append({"role": role, "content": content})
    return out


def first_user_message(messages: List[Dict]) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def render_with_assistant_mask(tokenizer, messages: List[Dict], max_len: int):
    prev_ids: List[int] = []
    full_ids: List[int] = []
    label_mask: List[int] = []

    for i in range(len(messages)):
        rendered = tokenizer.apply_chat_template(
            messages[: i + 1],
            tokenize=True,
            add_generation_prompt=False,
        )
        delta = rendered[len(prev_ids) :]
        full_ids.extend(delta)
        is_assistant = messages[i]["role"] == "assistant"
        label_mask.extend([1 if is_assistant else 0] * len(delta))
        prev_ids = rendered

    full_ids = full_ids[:max_len]
    label_mask = label_mask[:max_len]
    labels = [tid if m else IGNORE_INDEX for tid, m in zip(full_ids, label_mask)]
    return full_ids, [1] * len(full_ids), labels


def main():
    args = parse_args()
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_dir) or ".", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    raw = load_dataset(args.dataset_name, split=args.split, cache_dir=args.raw_dir)
    if args.max_samples is not None:
        raw = raw.select(range(min(args.max_samples, len(raw))))

    pred_in, pred_attn = [], []
    llm_in, llm_attn, labels = [], [], []

    for ex in tqdm(raw, desc="tulu-v2 sft"):
        msgs = normalize_messages(ex["messages"])
        if not any(m["role"] == "assistant" for m in msgs):
            continue
        first_user = first_user_message(msgs)
        if not first_user:
            continue
        p = tok(first_user, max_length=args.predictor_max_len, truncation=True, add_special_tokens=True)
        pred_in.append(p["input_ids"])
        pred_attn.append(p["attention_mask"])
        ids, attn, lab = render_with_assistant_mask(tok, msgs, args.llm_max_len)
        llm_in.append(ids)
        llm_attn.append(attn)
        labels.append(lab)

    ds = Dataset.from_dict({
        "predictor_input_ids": pred_in,
        "predictor_attention_mask": pred_attn,
        "llm_input_ids": llm_in,
        "llm_attention_mask": llm_attn,
        "labels": labels,
    })
    ds.save_to_disk(args.out_dir)
    print(f"wrote {len(pred_in)} sft samples -> {args.out_dir}")


if __name__ == "__main__":
    main()
