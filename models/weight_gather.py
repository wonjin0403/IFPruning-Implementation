from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP


def _make_pruned_mlp(orig_mlp: LlamaMLP, idx: torch.Tensor, hidden_size: int) -> LlamaMLP:
    k = idx.numel()
    cfg = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=k,
        hidden_act="silu",
    )
    new_mlp = LlamaMLP(cfg)
    with torch.no_grad():
        new_mlp.gate_proj.weight.copy_(orig_mlp.gate_proj.weight[idx, :])
        new_mlp.up_proj.weight.copy_(orig_mlp.up_proj.weight[idx, :])
        new_mlp.down_proj.weight.copy_(orig_mlp.down_proj.weight[:, idx])
        if getattr(orig_mlp.gate_proj, "bias", None) is not None:
            new_mlp.gate_proj.bias.copy_(orig_mlp.gate_proj.bias[idx])
        if getattr(orig_mlp.up_proj, "bias", None) is not None:
            new_mlp.up_proj.bias.copy_(orig_mlp.up_proj.bias[idx])
        if getattr(orig_mlp.down_proj, "bias", None) is not None:
            new_mlp.down_proj.bias.copy_(orig_mlp.down_proj.bias)
    new_mlp.to(dtype=orig_mlp.gate_proj.weight.dtype, device=orig_mlp.gate_proj.weight.device)
    return new_mlp


def gather_pruned_mlp(
    model: LlamaForCausalLM,
    topk_indices_per_layer: Sequence[torch.Tensor],
) -> LlamaForCausalLM:
    assert len(topk_indices_per_layer) == len(model.model.layers)
    hidden_size = model.config.hidden_size
    for layer, idx in zip(model.model.layers, topk_indices_per_layer):
        layer.mlp = _make_pruned_mlp(layer.mlp, idx.to(layer.mlp.gate_proj.weight.device), hidden_size)
    return model


def build_gathered_llama(
    model_name_or_path: str,
    topk_indices_per_layer: Sequence[torch.Tensor],
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    return gather_pruned_mlp(model, topk_indices_per_layer)


def mask_to_indices(ffn_mask: torch.Tensor) -> List[torch.Tensor]:
    assert ffn_mask.dim() == 2
    return [m.nonzero(as_tuple=False).squeeze(-1) for m in ffn_mask]
