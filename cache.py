from abc import ABC, abstractmethod

import torch
import torch.nn as nn


LARGE_INTEGER = int(1e9)  # This is used to assign high priority ids


class KVCache(ABC, nn.Module):
    # Define which hyperparameters are relevant for the cache.
    # Override as needed for sub-classes.
    relevant_kwargs = ["max_cache_length"]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__()

        # Assign each kwarg as an attribute of the class
        for key, value in kwargs.items():
            setattr(self, key, value)

        cache_shape = (max_batch_size, n_heads, self.max_cache_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

        self.updates = 0
        self.insertions = 0

    def is_prefill(self):
        # If we are in the prefill stage, we have never updated the cache
        # Prefill --> full self-attention (no KV-cache needed).
        # Otherwise --> query the KV-cache.
        return self.updates == 0
    
    @abstractmethod
    def return_attention(self) -> bool:
        """
        A Cache specific method to determine if we need to return the attention weights (requires overhead).
        """
        pass

    def reset(self):
        """
        If needed, this will reset the cache, although it is likely not necessary for most cache types.
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.updates = 0
        self.insertions = 0

    @abstractmethod
    def _update(self, input_pos, k_val, v_val):
        """
        Cache-specific update logic.
        Takes in the input positions and the corresponding k and v values.
        Modifies self.k_cache, self.v_cache in place.
        """
        pass

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
        self.updates += 1
        self.insertions += input_pos.shape[0]

        self._update(input_pos, k_val, v_val)

        # Truncate the unfilled part of the cache
        # Since we always fill in-order it will be at the end
        truncate_idx = min(self.insertions, self.max_cache_length)
        return self.k_cache[:, :, :truncate_idx, :], self.v_cache[:, :, :truncate_idx, :]

    def fill(self, fill_indices, k_val, v_val):
        """
        Modifies the cache in-place with key-value pairs at given fill_indices. 

        Args:
            fill_indices (torch.Tensor): The indices specifying the positions to fill in the cache.
            k_val (torch.Tensor): The key values to fill in the fill_indices slots.
            v_val (torch.Tensor): The value values to fill in the fill_indices slots.

        Returns:
            None
        """
        assert fill_indices.shape[0] == k_val.shape[2] == v_val.shape[2]

        self.k_cache[:, :, fill_indices, :] = k_val
        self.v_cache[:, :, fill_indices, :] = v_val

    def update_attn(self, input_pos, attn):
        if self.return_attention() and not hasattr(self, "_update_attn"):
            raise Exception("If your KVCache subclass requests attention weights, it must implement an _update_attn function.")

        self._update_attn(input_pos, attn)


class KVCacheFull(KVCache):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _update(self, input_pos, k_val, v_val):
        self.fill(fill_indices=input_pos, k_val=k_val, v_val=v_val)

    def return_attention(self):
        return False


class KVCacheWindow(KVCache):
    relevant_kwargs = ["max_cache_length", "global_tokens"]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)
        self.register_buffer(
            "pos",
            torch.full((max_batch_size, self.max_cache_length), -1, dtype=torch.int),
        )

        # This turns True when the global tokens are fully filled
        self.global_filled = self.global_tokens == 0

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
        _, min_k_indices = self.pos.topk(input_pos.shape[0], largest=False)

        # Sort the indices in ascending order
        min_k_indices, _ = min_k_indices.squeeze(0).sort()

        self.fill(fill_indices=min_k_indices, k_val=k_val, v_val=v_val)

        # Todo: bug in GPT-Fast where pos is sometimes int and sometimes long
        # Solution for now: convert to input_pos.int() to avoid errors.
        self.pos[:, min_k_indices] = input_pos.int()

        # This is a potentially costly operation which doesn't need to be repeated once we've filled the global tokens
        if not self.global_filled:
            # We put max priority on leading "global" tokens
            global_mask = torch.logical_and(
                self.pos < self.global_tokens, self.pos >= 0
            )
            # Give self.pos an arbitrary high value for global tokens so that they are not replaced
            self.pos.masked_fill_(global_mask, LARGE_INTEGER)
            self.global_filled = global_mask.sum() == self.global_tokens

    def return_attention(self):
        return False


class KVCacheHeavyHitters(KVCache):
    relevant_kwargs = ["max_cache_length", "global_tokens", "history"]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)
        self.register_buffer(
            "pos",
            torch.full((max_batch_size, self.max_cache_length), -1, dtype=torch.int),
        )

        # This turns True when the global tokens are fully filled
        self.global_filled = self.global_tokens == 0

    def _update(self, input_pos, k_val, v_val):
        pass

    def _update_attn(self, input_pos, attn):
        # TODO
        pass

    def return_attention(self):
        return True


def get_cache_constructor(cache_strategy):
    if cache_strategy == "full":
        return KVCacheFull
    elif cache_strategy == "window":
        return KVCacheWindow
    else:
        raise ValueError(f"Invalid cache strategy: {cache_strategy}")
