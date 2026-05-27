from __future__ import annotations

import torch
from transformers import AutoConfig

from .masked_llama import MaskedLlamaForCausalLM, masked_llama_from_pretrained
from .masked_qwen2 import MaskedQwen2ForCausalLM, masked_qwen2_from_pretrained


def masked_from_pretrained(
    model_name_or_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
):
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    mt = getattr(cfg, "model_type", None)
    if mt == "llama":
        return masked_llama_from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation,
        )
    elif mt == "qwen2":
        return masked_qwen2_from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation,
        )
    else:
        raise ValueError(f"masked_from_pretrained: unsupported model_type={mt!r} at {model_name_or_path}")


def masked_class_for(model_type: str):
    """Return the right MaskedXxxForCausalLM class for a given model_type string."""
    if model_type == "llama":
        return MaskedLlamaForCausalLM
    if model_type == "qwen2":
        return MaskedQwen2ForCausalLM
    raise ValueError(f"masked_class_for: unsupported model_type={model_type!r}")
