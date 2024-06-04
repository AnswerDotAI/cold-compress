from abc import ABC, abstractmethod

import torch
import torch.nn as nn


LARGE_INTEGER = int(1e9)  # This is used to assign high priority ids


class KVCache(ABC, nn.Module):
    # Define which hyperparameters are relevant for the cache.
    # Override as needed for sub-classes.
    relevant_kwargs = ["max_cache_length"]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        head_specific=False,
        **kwargs,
    ):
        super().__init__()

        # Assign each kwarg as an attribute of the class
        for key, value in kwargs.items():
            setattr(self, key, value)

        cache_shape = (max_batch_size, n_heads, self.max_cache_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))
        # This is used to keep track of the order in which the cache is filled.
        # We use n_heads as an optional second dimension to allow for head-specific evictions.
        self.register_buffer(
            "pos",
            torch.full(
                (
                    max_batch_size,
                    n_heads if head_specific else 1,
                    self.max_cache_length,
                ),
                -1,
                dtype=torch.int,
            ),
        )

        self.updates = 0
        self.insertions = 0

    def is_prefill(self):
        # If we are in the prefill stage, we have updated the cache at most once (self.updates <=1)
        # Prefill --> full self-attention (no KV-cache needed).
        # Otherwise --> query the KV-cache.
        return self.updates == 0

    def reset(self):
        """
        If needed, this will reset the cache, although it is likely not necessary for most cache types.
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.pos.fill_(-1)
        self.insertions = 0
        self.updates = 0

    def requires_attn(self):
        """
        Returns whether the cache requires attention weights for cache management.
        """
        return False

    def update(self, input_pos, k_val, v_val):
        """
        Updates the cache with the given input positions, keys, and values.

        Parameters:
            input_pos (torch.Tensor): A tensor of input positions.
            k_val (torch.Tensor): A tensor of keys.
            v_val (torch.Tensor): A tensor of values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated cache of keys and values,
            both truncated to the minimum of the current insertions and the maximum cache length.
        """

        self._update(input_pos, k_val, v_val)

        # Update counters
        self.updates += 1
        self.insertions += input_pos.shape[0]

        # Truncate the unfilled part of the cache
        # Since we always fill in-order it will be at the end
        truncate_idx = min(self.insertions, self.max_cache_length)
        return self.k_cache[:, :, :truncate_idx, :], self.v_cache[
            :, :, :truncate_idx, :
        ]

    @abstractmethod
    def _update(self, input_pos, k_val, v_val):
        """
        Cache-specific update logic.
        Takes in the input positions and the corresponding k and v values.
        Modifies self.pos, self.k_cache, self.v_cache place.
        """
        pass

    def fill(self, fill_indices, input_pos, k_val, v_val):
        self.k_cache[:, :, fill_indices] = k_val
        self.v_cache[:, :, fill_indices] = v_val
        self.pos[:, :, fill_indices] = input_pos.int()


class KVCacheFull(KVCache):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        self.fill(fill_indices=input_pos, input_pos=input_pos, k_val=k_val, v_val=v_val)


class KVCacheWindow(KVCache):
    relevant_kwargs = ["max_cache_length", "global_tokens"]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

        # This turns True when the global tokens are fully filled
        self.global_filled = self.global_tokens == 0

    def mark_global_tokens(self) -> bool:
        """
        Update POS tensor to give global tokens highest priority.

        Return a boolean indicating whether or not all global tokens were filled.

        If it returns True, this function won't be called again to save computation.
        """
        # We put max priority on leading "global" tokens
        global_mask = torch.logical_and(self.pos < self.global_tokens, self.pos >= 0)
        # Give self.score an arbitrary high value for global tokens so that they are not replaced
        self.pos.masked_fill_(global_mask, LARGE_INTEGER)
        return (global_mask.sum() == self.global_tokens).item()

    def _update(self, input_pos, k_val, v_val):
        # Prefill case: If prompt > window, then we need to chop off early positions
        window = self.k_cache.shape[2]
        if input_pos.shape[0] > window:
            # [global; ...; window - global] --> [global; window - global]
            # Indices for first global_tokens tokens and last (window - global_tokens) tokens
            keep_idxs = list(range(self.global_tokens)) + list(
                range(
                    input_pos.shape[0] - window + self.global_tokens, input_pos.shape[0]
                )
            )
            input_pos = input_pos[keep_idxs]
            k_val = k_val[:, :, keep_idxs]
            v_val = v_val[:, :, keep_idxs]

        # Identify the lowest positions in the cache that are not filled
        pos = self.pos[:, 0, :].squeeze(1)
        _, min_k_indices = pos.topk(input_pos.shape[0], largest=False)
        min_k_indices = min_k_indices.squeeze(0)

        self.fill(
            fill_indices=min_k_indices, input_pos=input_pos, k_val=k_val, v_val=v_val
        )

        # This is a potentially costly operation which doesn't need to be repeated once we've filled the global tokens
        self.global_filled = self.global_filled or self.mark_global_tokens()


def get_cache_constructor(cache_strategy):
    if cache_strategy == "full":
        return KVCacheFull
    elif cache_strategy == "window":
        return KVCacheWindow
    else:
        raise ValueError(f"Invalid cache strategy: {cache_strategy}")
