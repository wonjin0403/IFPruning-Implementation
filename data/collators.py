from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

IGNORE_INDEX = -100


def _pad(seqs: List[List[int]], pad_id: int, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = min(len(s), max_len)
        out[i, :n] = torch.as_tensor(s[:n], dtype=torch.long)
    return out


@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        ids = _pad([f["input_ids"] for f in features], self.pad_token_id)
        attn = _pad([f["attention_mask"] for f in features], 0, max_len=ids.size(1))
        labels = ids.clone()
        labels[attn == 0] = IGNORE_INDEX
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}


@dataclass
class ChunkPairCollator:
    pad_token_id: int
    predictor_pad_token_id: int | None = None

    def __post_init__(self):
        if self.predictor_pad_token_id is None:
            self.predictor_pad_token_id = self.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pred_ids = _pad([f["predictor_input_ids"] for f in features], self.predictor_pad_token_id)
        pred_attn = _pad([f["predictor_attention_mask"] for f in features], 0, max_len=pred_ids.size(1))
        llm_ids = _pad([f["llm_input_ids"] for f in features], self.pad_token_id)
        llm_attn = _pad([f["llm_attention_mask"] for f in features], 0, max_len=llm_ids.size(1))
        labels = llm_ids.clone()
        labels[llm_attn == 0] = IGNORE_INDEX
        return {
            "predictor_input_ids": pred_ids,
            "predictor_attention_mask": pred_attn,
            "llm_input_ids": llm_ids,
            "llm_attention_mask": llm_attn,
            "labels": labels,
        }


@dataclass
class SFTCollator:
    pad_token_id: int
    predictor_pad_token_id: int | None = None

    def __post_init__(self):
        if self.predictor_pad_token_id is None:
            self.predictor_pad_token_id = self.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        pred_ids = _pad([f["predictor_input_ids"] for f in features], self.predictor_pad_token_id)
        pred_attn = _pad([f["predictor_attention_mask"] for f in features], 0, max_len=pred_ids.size(1))
        llm_ids = _pad([f["llm_input_ids"] for f in features], self.pad_token_id)
        llm_attn = _pad([f["llm_attention_mask"] for f in features], 0, max_len=llm_ids.size(1))
        labels = _pad([f["labels"] for f in features], IGNORE_INDEX, max_len=llm_ids.size(1))
        return {
            "predictor_input_ids": pred_ids,
            "predictor_attention_mask": pred_attn,
            "llm_input_ids": llm_ids,
            "llm_attention_mask": llm_attn,
            "labels": labels,
        }
