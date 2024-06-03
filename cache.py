from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np


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
        pos_shape = (max_batch_size, n_heads, self.max_cache_length)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("pos", torch.full(pos_shape, -1, dtype=torch.int))

        self.updates = 0
        self.insertions = 0

    def is_prefill(self):
        # If we are in the prefill stage, we have never updated the cache
        # Prefill --> full self-attention (no KV-cache needed).
        # Otherwise --> query the KV-cache.
        return self.updates == 0

    def is_full(self):
        return self.insertions >= self.max_cache_length
    
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

        self._update(input_pos, k_val, v_val)

        # Update counters
        self.updates += 1
        self.insertions += input_pos.shape[0]

        # Truncate the unfilled part of the cache
        # Since we always fill in-order it will be at the end
        truncate_idx = min(self.insertions, self.max_cache_length)
        return self.k_cache[:, :, :truncate_idx, :], self.v_cache[:, :, :truncate_idx, :]

    def fill(self, fill_indices, input_pos, k_val, v_val, head_idx=None):
        """
        Modifies the cache in-place with key-value pairs at given fill_indices.

        Args:
            fill_indices (torch.Tensor): The indices specifying the positions to fill in the cache.
            input_pos (torch.Tensor): The input positions corresponding to the fill_indices.
            k_val (torch.Tensor): The key values to fill in the fill_indices slots.
            v_val (torch.Tensor): The value values to fill in the fill_indices slots.
            head_idx (int, optional): The head index to fill. If None, fill all heads. If not None, k_val.shape[1] (head dim) must be 1.

        Returns:
            None
        """
        # fill_indices [seq_len]
        # input_pos [seq_len]
        # k_val [batch_size, n_heads or 1, seq_len, head_dim]
        # head_idx int or None: int iff k_val.shape[1] == 1
        assert len(fill_indices) == len(input_pos) == k_val.shape[2] == v_val.shape[2]

        head_idx = head_idx or slice(None)  # slice(None) is equivalent to ":"
        self.pos[:, head_idx, fill_indices] = input_pos.int()
        self.k_cache[:, head_idx, fill_indices, :] = k_val
        self.v_cache[:, head_idx, fill_indices, :] = v_val

    def update_attn(self, attn: torch.Tensor, mask: torch.Tensor = None):
        if self.return_attention() and not hasattr(self, "_update_attn"):
            raise Exception("If your KVCache subclass requests attention weights, it must implement an _update_attn function.")

        self._update_attn(attn, mask)


class KVCacheFull(KVCache):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _update(self, input_pos, k_val, v_val):
        # The full cache is filled in the same order as the original input positions
        self.fill(fill_indices=input_pos, input_pos=input_pos, k_val=k_val, v_val=v_val)

    def return_attention(self):
        return False


class KVCacheWindow(KVCache):
    relevant_kwargs = ["max_cache_length", "global_tokens"]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

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
        # For window, all heads are the same so let's just use the first head for "pos"
        pos = self.pos[:, 0, :].squeeze(1)
        _, min_k_indices = pos.topk(input_pos.shape[0], largest=False)

        # Sort the indices in ascending order
        min_k_indices, _ = min_k_indices.squeeze(0).sort()

        self.fill(fill_indices=min_k_indices, input_pos=input_pos, k_val=k_val, v_val=v_val)

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
    relevant_kwargs = ["max_cache_length", "global_tokens", "history_att_window"]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

        # TODO Should we use static tensors for this?
        self.att_history = [[[] for _ in range(n_heads)] for _ in range(self.max_cache_length)]

        # This turns True when the global tokens are fully filled
        self.global_filled = self.global_tokens == 0

    def _update(self, input_pos, k_val, v_val):
        # Prefill case: If prompt > window, we don't have a strategy yet to chop off early positions
        if input_pos.shape[0] > self.max_cache_length:
            raise Exception("Heavy Hitters cache does not yet support prefilling with a prompt longer than the max_cache_length.")
        
        num_in_cache = min(self.insertions, self.max_cache_length)

        cand_new_num = num_in_cache + input_pos.shape[0]

        # If the cache is unfilled, we can just insert the new keys and values
        if cand_new_num <= self.max_cache_length:
            # Insert the new keys and values in the first unfilled slots
            fill_indices = list(range(self.insertions, self.insertions + input_pos.shape[0]))
            self.fill(fill_indices=fill_indices, input_pos=input_pos, k_val=k_val, v_val=v_val)
        # If the cache would be overfilled, we need to evict the historically least attended to token
        elif cand_new_num == self.max_cache_length + 1:
            # Where n is the number of keys in the cache at the time the attention was computed
            # Evictions are specific to the head, so we need to compute this for each head
            for head_idx in range(k_val.shape[1]):
                # Compute the number of times each token in the cache was attended to with a probability less than 1/n
                unimportances = [
                    len([att for att in self.att_history[i][head_idx] if att[0] < 1 / att[1]]) for i in range(num_in_cache)
                ]

                # Find the index of the least attended to token
                fill_indices = list(sorted(np.argsort(unimportances)[-input_pos.shape[0]:]))
                # Evict the least attended to token
                self.fill(fill_indices=fill_indices, input_pos=input_pos, k_val=k_val[:, head_idx:head_idx+1, :], v_val=v_val[:, head_idx:head_idx+1, :], head_idx=head_idx)
            unimportances = [
                len([att for att in self.att_history[i] if att[0] < 1 / att[1]]) for i in range(num_in_cache)
            ]

            fill_indices = list(sorted(np.argsort(unimportances)[-input_pos.shape[0]:]))
            self.fill(fill_indices=fill_indices, input_pos=input_pos, k_val=k_val, v_val=v_val)
        else:
            # TODO better handling of this ... in general we input_pos.shape[0] is always 1 after initial prefill
            raise Exception("Cache is overfilled. This should not happen since input_pos.shape[0] should always be 1.")

    def _update_attn(self, attn: torch.tensor, mask: torch.Tensor = None):
        _, _, qn, kn = attn.shape

        # We should only be computing attention over the non-empty part of the cache
        assert kn == min(self.max_cache_length, self.insertions)
        for qi in range(qn):
            num_attended = int(mask[0, 0, qi, :].sum().item()) if mask is not None else kn
            for ki in range(num_attended):
                attns = attn[0, :, qi, ki]
                for head_idx, prob in enumerate(attns):
                    # Add the attention prob as well as the denom of the softmax used to compute the prob
                    self.att_history[ki][head_idx].append((prob.item(), num_attended))
                    # If it's beyond the historical window, remove the oldest value
                    if len(self.att_history[ki][head_idx]) > self.history_att_window:
                        self.att_history[ki][head_idx].pop(0)
    
    def fill(self, fill_indices, input_pos, k_val, v_val):
        super().fill(fill_indices, input_pos, k_val, v_val)

        num_heads = k_val.shape[1]

        # Reset the history at these slots
        for idx in fill_indices:
            self.att_history[idx] = [[] for _ in range(num_heads)]

    def return_attention(self):
        return True


def get_cache_constructor(cache_strategy):
    if cache_strategy == "full":
        return KVCacheFull
    elif cache_strategy == "window":
        return KVCacheWindow
    elif cache_strategy == "heavy_hitters":
        return KVCacheHeavyHitters
    else:
        raise ValueError(f"Invalid cache strategy: {cache_strategy}")
