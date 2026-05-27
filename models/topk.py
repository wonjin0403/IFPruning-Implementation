from __future__ import annotations

import torch


def straight_through_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    probs = torch.sigmoid(scores)
    idx = probs.topk(k, dim=-1).indices
    hard = torch.zeros_like(probs)
    hard.scatter_(-1, idx, 1.0)
    return hard + probs - probs.detach()


def soft_topk(scores: torch.Tensor, k: int, n_iter: int = 30) -> torch.Tensor:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    d = scores.size(-1)
    if k > d:
        raise ValueError(f"k={k} exceeds last-dim size d={d}")
    if k == d:
        return torch.ones_like(scores)

    orig_dtype = scores.dtype
    s = scores.float()

    lo = s.min(dim=-1, keepdim=True).values - 30.0
    hi = s.max(dim=-1, keepdim=True).values + 30.0

    target = float(k)
    for _ in range(n_iter):
        mid = (lo + hi) * 0.5
        cur_sum = torch.sigmoid(s - mid).sum(dim=-1, keepdim=True)
        too_many = cur_sum > target
        lo = torch.where(too_many, mid, lo)
        hi = torch.where(too_many, hi, mid)

    tau = (lo + hi) * 0.5
    lam = torch.sigmoid(s - tau)

    _, idx = s.topk(k, dim=-1)
    indicator = torch.zeros_like(lam)
    indicator.scatter_(-1, idx, 1.0)

    return (lam * indicator).to(orig_dtype)


def random_topk_mask(
    batch_size: int,
    num_layers: int,
    intermediate_size: int,
    k: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random top-k baseline. Picks k random channels per (sample, layer)."""
    scores = torch.rand(batch_size, num_layers, intermediate_size, device=device, generator=generator)
    idx = scores.topk(k, dim=-1).indices
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, idx, 1.0)
    return mask


@torch.no_grad()
def static_norm_topk_mask(
    model,
    k: int,
    norm: str = "l2",
) -> torch.Tensor:
    """Static input-independent baseline. Selects channels with the largest
    column norm of `down_proj.weight` (the FFN output projection): a channel
    that contributes more to the residual stream is kept.

    Returns a [num_hidden_layers, intermediate_size] {0,1} mask. Broadcast to
    [B, L, D] by the caller.
    """
    layers = model.model.layers
    masks = []
    for layer in layers:
        W = layer.mlp.down_proj.weight
        if norm == "l2":
            scores = W.float().pow(2).sum(dim=0).sqrt()
        elif norm == "l1":
            scores = W.float().abs().sum(dim=0)
        else:
            raise ValueError(f"Unknown norm {norm!r}")
        idx = scores.topk(k).indices
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, idx, 1.0)
        masks.append(mask)
    return torch.stack(masks, dim=0)
