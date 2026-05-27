from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel

from .topk import straight_through_topk, soft_topk


def build_extractor_config(
    vocab_size: int = 128256,
    hidden_size: int = 768,
    intermediate_size: int = 3072,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 8192,
    rope_theta: float = 500000.0,
    rms_norm_eps: float = 1e-5,
) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_act="silu",
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        tie_word_embeddings=False,
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=128001,
    )


class LlamaExtractorSparsityPredictor(nn.Module):
    def __init__(
        self,
        extractor_body: LlamaModel,
        num_layers: int = 32,
        intermediate_size: int = 14336,
        topk: int = 4096,
        head_hidden_dim: int = 128,
        mask_method: str = "st",
    ):
        super().__init__()
        self.extractor = extractor_body
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.topk = topk
        self.mask_method = mask_method

        ext_hidden = extractor_body.config.hidden_size
        self.mask_head = nn.Sequential(
            nn.Linear(ext_hidden, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, num_layers * intermediate_size),
        )

    @staticmethod
    def _last_nonpad_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.long().sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, last_idx]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        outputs = self.extractor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        h = self._last_nonpad_hidden(last_hidden, attention_mask)
        scores = self.mask_head(h).view(input_ids.size(0), self.num_layers, self.intermediate_size)

        if self.mask_method == "st":
            mask = straight_through_topk(scores, k=self.topk)
        elif self.mask_method == "soft":
            mask = soft_topk(scores, k=self.topk)
        elif self.mask_method == "none":
            mask = None
        else:
            raise ValueError(f"Unknown mask_method {self.mask_method!r}")
        return mask, scores


def build_predictor_from_extractor(
    extractor_ckpt_or_config,
    num_layers: int = 32,
    intermediate_size: int = 14336,
    topk: int = 4096,
    head_hidden_dim: int = 128,
    mask_method: str = "st",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> LlamaExtractorSparsityPredictor:
    if isinstance(extractor_ckpt_or_config, str):
        lm = LlamaForCausalLM.from_pretrained(extractor_ckpt_or_config, torch_dtype=torch_dtype)
        body = lm.model
    elif isinstance(extractor_ckpt_or_config, LlamaConfig):
        lm = LlamaForCausalLM(extractor_ckpt_or_config).to(dtype=torch_dtype)
        body = lm.model
    else:
        raise TypeError(type(extractor_ckpt_or_config))

    predictor = LlamaExtractorSparsityPredictor(
        extractor_body=body,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        topk=topk,
        head_hidden_dim=head_hidden_dim,
        mask_method=mask_method,
    )
    predictor.to(dtype=torch_dtype)
    return predictor
