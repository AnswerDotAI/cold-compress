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

    def reset(self):
        """
        If needed, this will reset the cache, although it is likely not necessary for most cache types.
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.pos.fill_(-1)
        self.insertions = 0
        self.updates = 0

    def return_attn(self):
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

        if self.updates > 1:
            # Truncate the unfilled part of the cache
            # Since we always fill in-order it will be at the end
            truncate_idx = min(self.insertions, self.max_cache_length)
            return self.k_cache[:, :, :truncate_idx, :], self.v_cache[
                :, :, :truncate_idx, :
            ]
        else:
            # We are in the prefill stage, so just return the original k and v values
            return k_val, v_val

    @abstractmethod
    def _update(self, input_pos, k_val, v_val):
        """
        Cache-specific update logic.
        Takes in the input positions and the corresponding k and v values.
        Modifies self.pos, self.k_cache, self.v_cache place.
        """
        pass

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
        # k_val, v_val [batch_size, n_heads or 1, seq_len, head_dim]
        # head_idx int or None: int iff k_val.shape[1] == 1
        assert len(fill_indices) == len(input_pos) == k_val.shape[2] == v_val.shape[2]

        head_idx = slice(None) if head_idx is None else head_idx  # slice(None) is equivalent to ":"
        self.pos[:, head_idx, fill_indices] = input_pos.int()
        self.k_cache[:, head_idx, fill_indices, :] = k_val
        self.v_cache[:, head_idx, fill_indices, :] = v_val

    def update_attn_history(self, attn):
        raise Exception(f"{self.__class__.__name__} requested return_attn=True but has not yet implemented a update_attn_history function.")


class KVCacheFull(KVCache):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, head_specific=False, **kwargs)

    def _update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        self.fill(fill_indices=input_pos, input_pos=input_pos, k_val=k_val, v_val=v_val)


class KVCacheWindow(KVCache):
    relevant_kwargs = ["max_cache_length", "global_tokens"]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, head_specific=False, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, head_specific, **kwargs)

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
        # Give self.pos an highest possible position value for global tokens so that they are not replaced
        self.pos.masked_fill_(global_mask, self.max_cache_length)
        return (global_mask.sum() == self.global_tokens).item()

    def fill_prefill(self, input_pos, k_val, v_val):
        """
        Fill the cache during the prefill (prompt encoding + new tokens until cache is full).
        This occurs just once so better to separate out the logic from #fill
        """
        max_cache_length = self.k_cache.shape[2]
        num_tokens = input_pos.shape[0]

        # If the prompt is longer than the max cache length
        if num_tokens > max_cache_length:
            # [global; ...; window - global] --> [global; window - global]
            # Indices for first global_tokens tokens and last (window - global_tokens) tokens
            keep_idxs = list(range(self.global_tokens)) + list(
                range(
                    input_pos.shape[0] - max_cache_length + self.global_tokens, num_tokens
                )
            )
            input_pos = input_pos[keep_idxs]
            k_val = k_val[:, :, keep_idxs]
            v_val = v_val[:, :, keep_idxs]

        self.fill(fill_indices=input_pos, input_pos=input_pos, k_val=k_val, v_val=v_val)
        self.global_filled = self.mark_global_tokens()

    def _update(self, input_pos, k_val, v_val):
        if self.insertions < self.max_cache_length:  # Insert into first unfilled spots
            self.fill_prefill(input_pos, k_val, v_val)
            return

        # Identify the lowest positions in the cache that are not filled
        pos = self.pos[:, 0, :].squeeze(1)
        # NB: Torch.topk does not guarantee sorted order
        _, fill_indices = pos.topk(input_pos.shape[0], largest=False)
        fill_indices = fill_indices.squeeze(0)

        self.fill(fill_indices=fill_indices, input_pos=input_pos, k_val=k_val, v_val=v_val)

        # This is a potentially costly operation which doesn't need to be repeated once we've filled the global tokens
        self.global_filled = self.global_filled or self.mark_global_tokens()


class KVCacheScissorhands(KVCacheWindow):
    relevant_kwargs = ["max_cache_length", "global_tokens", "history_window_size", "drop_amount", "recent_window", "normalize_history_attn"]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, head_specific=True, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, head_specific, **kwargs)

        # Initialize a buffer for the attention histories
        history_shape = (max_batch_size, n_heads, self.max_cache_length, self.history_window_size)
        self.register_buffer("attn_history", torch.empty(history_shape, dtype=torch.bool))
        self.attn_counter = 0

        # Different eviction queue for each attention head
        self.eviction_queue = [[] for _ in range(n_heads)]

        # We set global tokens to a really high number in the self.pos so this will break that
        if self.normalize_history_attn: assert self.global_tokens == 0

    def return_attn(self) -> bool:
        """
        Whether or not we need to return attention weights for cache management.

        We return attention weights if 2 conditions are met:
        1) The cache is not in the prefill stage
        2) The number of tokens left in the eviction queue < attention history window.

        The number of tokens in eviction queue specifies how many turns before we need to re-calculate importance.
        We only need to start recording once the number of steps until recomputation is equal to the recent window.
        """
        return not self.is_prefill() and len(self.eviction_queue[0]) <= self.history_window_size

    def update_attn_history(self, attn: torch.Tensor):
        """
        Insert the most recent attention into the history buffer.

        Rather than raw probability, insert a binary indicator of whether the attention is "low".
        """
        attn = attn.squeeze()
        keys = attn.shape[1]
        attn_is_low = (attn < 1 / keys).int()
        self.attn_history[:, :, :keys, self.attn_counter % self.history_window_size] = attn_is_low
        self.attn_counter += 1

    def refill_eviction_queue(self, input_pos: int):
        # Identify the tokens with the most "low" attentions
        unimportant_cts = self.attn_history.sum(dim=-1)

        if self.normalize_history_attn:
            # TODO: Consider taking the sqrt / log of the denominator
            unimportant_cts = unimportant_cts.float() / (input_pos - self.pos)

        # Save the recent tokens from being evicted
        unimportant_cts.masked_fill_(self.pos >= input_pos - self.recent_window, -1)

        _, unimportant_toks = unimportant_cts.topk(self.drop_amount, dim=-1, sorted=True, largest=True)
        self.eviction_queue = unimportant_toks.squeeze(0).tolist()

    def _update(self, input_pos, k_val, v_val):
        if self.insertions < self.max_cache_length:  # Insert into first unfilled spots
            self.fill_prefill(input_pos, k_val, v_val)
            return

        # This is a potentially costly operation which doesn't need to be repeated once we've filled the global tokens
        self.global_filled = self.global_filled or self.mark_global_tokens()

        # Refill the eviction queue only if it is empty (potentially expensive operation)
        len(self.eviction_queue[0]) > 0 or self.refill_eviction_queue(input_pos.item())

        fill_indices = [self.eviction_queue[head_idx].pop(0) for head_idx in range(len(self.eviction_queue))]

        for head_idx, fi in enumerate(fill_indices):
            self.fill(fill_indices=[fi], input_pos=input_pos, k_val=k_val[:, head_idx:head_idx+1], v_val=v_val[:, head_idx:head_idx+1], head_idx=head_idx)
            self.attn_history[:, head_idx, fi, :] = 0


def get_cache_constructor(cache_strategy):
    if cache_strategy == "full":
        return KVCacheFull
    elif cache_strategy == "window":
        return KVCacheWindow
    elif cache_strategy == "scissor":
        return KVCacheScissorhands
    else:
        raise ValueError(f"Invalid cache strategy: {cache_strategy}")
