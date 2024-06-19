import math
from typing import Tuple

import torch
from torch.nn import functional as F


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    return_attn=False,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Uses naive PyTorch sdpa implementation if we need to return_attn. Otherwise use the optimized version.

    The naive implementation will be optimized later.
    """
    if not return_attn:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        ), None
    
    print("Spose we won't be here?")
    B, L, S = query.size(0), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    needs_masking = is_causal or attn_mask is not None
    if needs_masking:
        assert not (
            attn_mask is not None and is_causal
        ), "Should only be passing in attn_mask or is_causal=True."
        attn_bias = torch.zeros(B, 1, L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            attn_mask = (
                torch.ones(L, S, dtype=torch.bool, device=query.device)
                .tril(diagonal=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
        attn_weight += attn_bias.to(attn_weight.device)

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_weight
