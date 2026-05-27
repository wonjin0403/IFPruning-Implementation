from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


class MaskedQwen2MLP(Qwen2MLP):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self._mask: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        if self._mask is not None:
            hidden = hidden * self._mask.unsqueeze(1).to(hidden.dtype)
        return self.down_proj(hidden)


class MaskedQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self._swap_mlps()

    def _swap_mlps(self):
        for layer in self.model.layers:
            new_mlp = MaskedQwen2MLP(self.config)
            new_mlp.load_state_dict(layer.mlp.state_dict())
            layer.mlp = new_mlp

    def _set_ffn_mask(self, ffn_mask: Optional[torch.Tensor]):
        if ffn_mask is None:
            for layer in self.model.layers:
                layer.mlp._mask = None
            return
        assert ffn_mask.dim() == 3, f"ffn_mask must be [B, L, D], got {ffn_mask.shape}"
        assert ffn_mask.size(1) == self.config.num_hidden_layers
        assert ffn_mask.size(2) == self.config.intermediate_size
        for i, layer in enumerate(self.model.layers):
            layer.mlp._mask = ffn_mask[:, i, :]

    def forward(self, *args, ffn_mask: Optional[torch.Tensor] = None, **kwargs):
        try:
            self._set_ffn_mask(ffn_mask)
            return super().forward(*args, **kwargs)
        finally:
            self._set_ffn_mask(None)


def masked_qwen2_from_pretrained(
    model_name_or_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> MaskedQwen2ForCausalLM:
    base = Qwen2ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    model = MaskedQwen2ForCausalLM(base.config)
    model.load_state_dict(base.state_dict(), strict=True)
    model.to(dtype=torch_dtype)
    del base
    return model
