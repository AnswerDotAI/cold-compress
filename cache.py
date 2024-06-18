import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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

        # Can we evict different tokens for different heads?
        # If the answer is yes, we need to store self.pos for each head.
        self.head_specific = head_specific
        self.register_buffer(
            "pos",  # Track pos to keep track of the original positions of each item in cache.
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

    def is_prefill(self):
        """
        Returns whether the cache is in the prefill stage.
        """
        return self.updates == 0

    def compress_prompt(self, input_pos, k_val, v_val, attn):
        # If the prompt is longer than the cache, we need to compress it to fit cache and then store (update).
        assert (
            input_pos.shape[0] > self.max_cache_length
        ), "You called compress_prompt in prefill stage yet prompt is not longer than max_cache_length."
        # Can we make head specific evictions (--> use SnapKV) else just take the global_tokens + most recent tokens
        compressor = (
            self._compress_prompt_w_snapkv
            if self.head_specific
            else self._compress_prompt_w_recent_global
        )
        input_pos, k_val, v_val, attn = compressor(input_pos, k_val, v_val, attn)
        self.update(input_pos, k_val, v_val)
        if attn is not None:
            self.update_attn_history(attn)

    def _compress_prompt_w_recent_global(self, input_pos, k_val, v_val, attn):
        # [global; ...; window - global] --> [global; window - global]
        # Indices for first global_tokens tokens and last (window - global_tokens) tokens
        # Making this a tensor seems to give a speedup, but I haven't fully benchmarked
        keep_idxs = torch.tensor(
            list(range(self.global_tokens))
            + list(
                range(
                    input_pos.shape[0] - self.max_cache_length + self.global_tokens,
                    input_pos.shape[0],
                )
            ),
            dtype=torch.long,
            device=k_val.device,
        )
        assert len(keep_idxs) == self.max_cache_length
        k_val = k_val[:, :, keep_idxs]
        v_val = v_val[:, :, keep_idxs]
        return input_pos[: self.max_cache_length], k_val, v_val, None

    def _compress_prompt_w_snapkv(
        self, input_pos, k_val, v_val, attn, observation_len=16, kernel_size=5
    ):
        """
        Use SnapKV to compress the prompt
        Inspired by the pseudo code on Page 7 of https://arxiv.org/abs/2404.14469
        """

        pool = torch.nn.AvgPool1d(
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            ceil_mode=False,
            count_include_pad=False,
        )
        priority = attn[:, :, -observation_len:, :].mean(dim=2)
        prev_shape = priority.shape

        # We'll be returning the attention history so we need to keep a copy before it's modified
        attn_history = priority.clone()
        priority = pool(priority)
        assert (
            priority.shape == prev_shape
        ), f"Pooling operation should not change the dimension: {prev_shape} -> {priority.shape}"
        priority[:, :, -observation_len:] = (
            1.0  # Ensure the observation window is selected
        )
        keep_idxs = (
            priority.topk(self.max_cache_length, dim=-1).indices.sort(dim=-1).values
        )

        attn_history = attn_history.gather(2, keep_idxs)

        keep_idxs_rep = keep_idxs.unsqueeze(-1).expand(-1, -1, -1, k_val.shape[-1])
        k_val_compressed = k_val.gather(2, keep_idxs_rep)
        v_val_compressed = v_val.gather(2, keep_idxs_rep)

        return keep_idxs.squeeze(0), k_val_compressed, v_val_compressed, attn_history

    def update(self, input_pos, k_val, v_val):
        """
        Updates the cache with the given input positions, keys, and  values.

        Parameters:
            input_pos (torch.Tensor): A tensor of input positions.
            k_val (torch.Tensor): A tensor of keys.
            v_val (torch.Tensor): A tensor of values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, bool]: A tuple containing the updated cache of keys and values,
            both truncated to the minimum of the current insertions and the maximum cache length. The last value
            is a boolean return_attn indicating whether the cache requires attention weights. If True, the model
            will call self.update_attn_history with the attention weights.
        """
        is_prefill = self.is_prefill()
        num_tokens = input_pos.shape[-1]
        prompt_overflow = num_tokens > self.max_cache_length

        # k_val: [B, H, S, D] -> S is > 1 for prefill, 1 for new tokens
        if is_prefill:
            assert num_tokens > 1
        else:
            assert num_tokens == 1

        # If the prompt is longer than the cache, we need to call compress_prompt
        # We return it as a callback since we don't have the attention weights yet
        if prompt_overflow:
            return k_val, v_val, self.compress_prompt

        # If the cache requires attention weights to manage evictions, we need to pass self.update_attn_history as a callback
        # We wrap it as a lambda so that it has the same function signature as the compress_prompt callback despite only needing attn
        attn_history_callback = (
            (lambda input_pos, k_val, v_val, attn: self.update_attn_history(attn))
            if self.return_attn()
            else None
        )

        self._update(input_pos, k_val, v_val)

        # Update counters
        self.updates += 1
        self.insertions += num_tokens

        # Truncate the cache based on number of insertions. It will be at the end since we prefill in-order
        k = (
            self.k_cache[:, :, : self.insertions, :]
            if self.insertions < self.max_cache_length
            else self.k_cache
        )
        v = (
            self.v_cache[:, :, : self.insertions, :]
            if self.insertions < self.max_cache_length
            else self.v_cache
        )
        return k, v, attn_history_callback

    @abstractmethod
    def _update(self, input_pos, k_val, v_val):
        """
        Cache-specific update logic.
        Takes in the input positions and the corresponding k and v values.
        Modifies self.pos, self.k_cache, self.v_cache place.
        """
        pass

    def fill_contiguous(self, input_pos, k_val, v_val, start=None, end=None):
        """
        A simple utility to fill the cache and pos.
        If start and end are provided, only fill the cache between those indices.
        Otherwise, treat start as self.insertions and end as self.insertions + num_new_insertions.
        Will also mark the global_tokens as they are updated.
        """
        num_new_insertions = k_val.shape[2]
        if start is None:
            assert end is None
            start, end = self.insertions, self.insertions + num_new_insertions

        # Assert all positions are unfilled - remove for speed
        # if self.insertions < self.max_cache_length:
        #     slice = self.pos[:, :, start:end]
        #     assert (
        #         torch.min(slice) == -1 and torch.max(slice) == -1
        #     ), "Trying to fill already filled positions during prefill."
        self.pos[:, :, start:end] = input_pos.int()
        self.k_cache[:, :, start:end, :] = k_val
        self.v_cache[:, :, start:end, :] = v_val

        if hasattr(
            self, "global_tokens"
        ):  # If we have global tokens we need to mark them in self.pos
            # Update global tokens to the prompt size if set to -1
            resize_global_tokens = self.global_tokens == -1
            if resize_global_tokens:
                self.global_tokens = num_new_insertions
            self.global_filled = self.global_filled or self.mark_global_tokens(
                num_new_insertions + self.insertions
            )

    def fill_headwise(self, fill_indices, input_pos, k_val, v_val):
        """
        Modifies the cache in-place with key-value pairs at given fill_indices.

        Args:
            fill_indices (torch.Tensor): The indices specifying the positions to fill in the cache.
            input_pos (torch.Tensor): The input positions corresponding to the fill_indices.
            k_val (torch.Tensor): The key values to fill in the fill_indices slots.
            v_val (torch.Tensor): The value values to fill in the fill_indices slots.

        Returns:
            None
        """
        # fill_indices [num_heads] or [1]
        # input_pos [seq_len] or [num_heads, seq_len]
        # k_val, v_val [batch_size, n_heads, seq_len, head_dim]
        assert input_pos.shape[-1] == k_val.shape[2] == v_val.shape[2]

        # input_pos is either [seq_len] or [num_heads, seq_len]
        pos_fill_indices = fill_indices.view(1, -1, 1)
        cache_fill_indices = fill_indices.view(1, len(fill_indices), 1, 1).expand(
            1, k_val.shape[1], 1, k_val.shape[-1]
        )
        input_pos = input_pos.view(1, -1, 1).expand(1, k_val.shape[1], 1).int()
        self.pos.scatter_(2, pos_fill_indices, input_pos.int())
        self.k_cache.scatter_(2, cache_fill_indices, k_val)
        self.v_cache.scatter_(2, cache_fill_indices, v_val)

    def update_attn_history(self, attn):
        """
        Update the attention history with the most recent attention weights.
        """
        raise Exception(
            f"{self.__class__.__name__} requested return_attn=True but has not yet implemented a update_attn_history function."
        )

    def mark_global_tokens(self, num_total_insertions: int) -> bool:
        """
        Update POS tensor to give global tokens highest priority.

        num_total_insertions: The total number of tokens inserted so far. The sum of self.insertions and num_new_insertions.

        Return a boolean indicating whether or not all global tokens were filled.

        If it returns True, this function won't be called again to save computation.
        """
        assert hasattr(
            self, "global_tokens"
        ), "This cache does not have global tokens so we cannot mark them."
        # Give self.pos an highest possible position value for global tokens so that they are not replaced
        num_to_mark = min(self.global_tokens, num_total_insertions)
        self.pos[:, :, :num_to_mark] = self.max_cache_length
        return num_to_mark == self.global_tokens


class KVCacheFull(KVCache):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific=False, **kwargs
        )

    def _update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        return self.fill_contiguous(input_pos, k_val, v_val)


class KVCacheRandom(KVCache):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific=False, **kwargs
        )
        self.global_tokens = 0
        self.global_filled = True

    def _update(self, input_pos, k_val, v_val):
        start = end = None  # We will fill the cache in order if start and end are None

        need_to_evict = self.insertions >= self.max_cache_length
        if need_to_evict:  # Select a spot at random
            start = torch.randint(low=0, high=self.max_cache_length, size=(1,)).item()
            end = start + 1

        # Specify specific start and end indices
        self.fill_contiguous(input_pos, k_val, v_val, start=start, end=end)


class KVCacheWindow(KVCache):
    relevant_kwargs = ["max_cache_length", "global_tokens"]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        head_specific=False,
        **kwargs,
    ):
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific, **kwargs
        )

        # This turns True when the global tokens are fully filled
        self.global_filled = self.global_tokens == 0

    def _update(self, input_pos, k_val, v_val):
        start = end = None  # We will fill the cache in order if start and end are None

        need_to_evict = self.insertions >= self.max_cache_length
        if need_to_evict:  # Identify the least recent spot
            start = torch.argmin(self.pos).item()
            assert (
                input_pos.shape[0] == 1
            ), "Should only be passing 1 new token at a time after cache is filled!"
            end = start + 1

        # Specify specific start and end indices
        self.fill_contiguous(input_pos, k_val, v_val, start=start, end=end)


class KVCacheScissorhands(KVCacheWindow):
    relevant_kwargs = [
        "max_cache_length",
        "global_tokens",
        "history_window_size",
        "drop_amount",
        "recent_window",
        "attn_thresholding",
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        head_specific=True,
        **kwargs,
    ):
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific, **kwargs
        )

        # Initialize a buffer for the attention histories
        history_num_shape = (
            max_batch_size,
            n_heads,
            self.max_cache_length,
            self.history_window_size,
        )
        history_denom_shape = (
            max_batch_size,
            n_heads,
            self.max_cache_length,
        )
        self.register_buffer(
            "attn_history_num",
            torch.zeros(
                history_num_shape, dtype=torch.bool if self.attn_thresholding else dtype
            ),
        )

        # Ideally, we could use the self.pos to track the number of times a token has been attended to
        # But any change to cache management or how self.pos is stored would break this.
        self.register_buffer(
            "attn_history_denom", torch.zeros(history_denom_shape, dtype=torch.int32)
        )

        self.attn_counter = 0

        # Different eviction queue for each attention head
        eviction_queue_shape = (
            max_batch_size,
            n_heads,
            self.drop_amount,
        )
        self.register_buffer(
            "eviction_queue", torch.zeros(eviction_queue_shape, dtype=torch.int32)
        )
        # Start with an "empty queue"
        self.eviction_idx = self.drop_amount

    def queue_len(self):
        return self.drop_amount - self.eviction_idx

    def return_attn(self) -> bool:
        """
        Whether or not we need to return attention weights for cache management.

        We return attention weights if 2 conditions are met:
        1) The cache is not in the prefill stage
        2) The number of tokens left in the eviction queue < attention history window.

        The number of tokens in eviction queue specifies how many turns before we need to re-calculate importance.
        We only need to start recording once the number of steps until recomputation is equal to the recent window.
        """

        return not self.is_prefill() and self.queue_len() <= self.history_window_size

    def update_attn_history(self, attn: torch.Tensor):
        """
        Insert the most recent attention into the history buffer.

        Rather than raw probability, insert a binary indicator of whether the attention is "low".
        """
        attn = attn.squeeze()
        keys = attn.shape[1]
        self.attn_history_num[
            :, :, :keys, self.attn_counter % self.history_window_size
        ] = (attn < 1 / keys).int() if self.attn_thresholding else attn
        self.attn_history_denom[:, :, :keys] += 1
        self.attn_counter += 1

    def refill_eviction_queue(self, input_pos: int):
        # Identify the tokens with consistently "low" attentions
        numerator = self.attn_history_num.sum(dim=-1).float()
        # The denominator is the number of times this token's history has been recorded
        # We only record most self.history_window_size recent scores so need to clamp it
        denominator = self.attn_history_denom.clamp_max(self.history_window_size)

        avg_unimportant_cts = numerator / denominator

        assert (
            torch.max(avg_unimportant_cts) <= 1
        ), "Average unimportant counts should be between 0 and 1."

        # Save the global & most recent tokens from being evicted
        avg_unimportant_cts.masked_fill_(self.pos >= input_pos - self.recent_window, -1)

        _, toks_to_evict = avg_unimportant_cts.topk(
            self.drop_amount, dim=-1, sorted=True, largest=True
        )

        # The eviction queue will be empty so just re-assign it
        self.eviction_queue = toks_to_evict
        self.eviction_idx = 0

    def _update(self, input_pos, k_val, v_val):
        if self.insertions < self.max_cache_length:  # Insert into first unfilled spots
            self.fill_contiguous(input_pos, k_val, v_val)
            return

        assert (
            self.global_filled
        ), "Global tokens should be all marked as filled when cache is filled."

        # Refill the eviction queue only if it is empty (potentially expensive operation)
        self.queue_len() > 0 or self.refill_eviction_queue(input_pos.item())

        # Evict the next token in the queue (self.eviction_idx) and increment it
        fill_indices = self.eviction_queue[0, :, self.eviction_idx]
        self.eviction_idx += 1

        self.fill_headwise(fill_indices, input_pos, k_val, v_val)
        num_fill = fill_indices.view(1, -1, 1, 1).expand(
            1, -1, 1, self.attn_history_num.shape[-1]
        )
        denom_fill = fill_indices.view(1, -1, 1)
        self.attn_history_num.scatter_(
            2, num_fill, torch.zeros_like(num_fill, dtype=self.attn_history_num.dtype)
        )
        self.attn_history_denom.scatter_(
            2, denom_fill, torch.zeros_like(denom_fill, dtype=torch.int32)
        )


def get_cache_constructor(cache_strategy):
    if cache_strategy == "full":
        return KVCacheFull
    elif cache_strategy == "random":
        return KVCacheRandom
    elif cache_strategy == "window":
        return KVCacheWindow
    elif cache_strategy == "scissor":
        return KVCacheScissorhands
    else:
        raise ValueError(f"Invalid cache strategy: {cache_strategy}")
