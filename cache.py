from abc import ABC, abstractmethod
from typing import Tuple, Callable
import torch
import torch.nn as nn
from prompt_compression import prompt_compressor_constructor
import argparse


def add_cache_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("cache_args")
    # KV-Cache Kwargs
    group.add_argument(
        "--max_cache_length",
        type=float,
        default=[1.0],
        nargs="+",
        help="Cache size per layer. If len < n layers, the values are tiled. Must have len divisible by n layers. \
        If 0 < x <= 1, it is percent of |prompt| + max new tokens. Otherwise, if > 1, its the maximum size.",
    )
    group.add_argument(
        "--cache_strategy",
        default="full",
        choices=["full", "random", "window", "scissor", "l2"],
    )

    group.add_argument(
        "--prompt_compression_strategy",
        default="recent_global",
        choices=["recent_global", "snapkv", "l2"],
        help="If |prompt| exceeds max_cache_length, we need to specify a strategy for compressing it to max_cache_length.",
    )

    # Optional Cache Kwargs depending on cache_strategy
    group.add_argument(
        "--global_tokens",
        default=4,
        type=int,
        help="The number of initial tokens to always include in the KV-Cache.  \
        If using window strategy, the actual window becomes max_cache_length - global_tokens.",
    )

    # Locality
    group.add_argument(
        "--recent_window",  # NB: for KVCacheWindow, recent_window is implicitly set to self.max_cache_length - self.global_tokens.
        default=10,  # 10 is default specified in ScissorHands paper ("r" in Algorithm 2).
        type=int,
        help="The number of recently generated tokens to always spare from eviction.",
    )

    # Scissorhands-specific Hyperparameters (--cache_strategy == "scissor")
    ## See Algorithm 1 & 2 in arxiv.org/abs/2305.17118
    group.add_argument(
        "--history_window_size",  # Equivalent to "m" in Algorithm 2.
        default=400,  # 400 is default specified in paper.
        type=int,
        help="The number of past tokens to consider when computing 'Heavy Hitters' in the KV-Cache.",
    )
    group.add_argument(
        "--drop_amount",  # Equivalent to "m" in Algorithm 2.
        default=0.5,  # 0.4 is default specified in paper.
        type=float,
        help="The number of tokens to evict KV-Cache reaches capacity (max_cache_length). Expressed as a fraction of max_cache_length.",
    )
    group.add_argument(
        "-attn_thresholding",
        default=False,
        action="store_true",
        help="Whether to accumulate number of times a token was unimportant (binary) versus raw un-normalized probabilities. If true, more memory efficient.",
    )

    group.add_argument(
        "--attn_record_freq",
        default=10,
        type=int,
        help="How often to record attention weights for the ScissorHands cache..",
    )


def cache_compatibility(args):
    if args.cache_strategy == "full":
        # Full implies no compression, which means --max_cache_length = [1.0] (same size as prompt + max_new_tokens)
        assert all(
            [l == 1.0 for l in args.max_cache_length]
        ), "Full cache strategy only supports max_cache_length=1.0."

    # Attention-based eviction policies must use an attention-based prompt compressor
    if args.cache_strategy in {"scissor"}:
        assert (
            args.prompt_compression_strategy == "snapkv"
        ), 'Scissor requires "snapkv" prompt compression strategy'

    print("The cache argument values you provided appear compatible with each other!")


class KVCache(ABC, nn.Module):
    # Define which hyperparameters are relevant for the cache.
    # Override as needed for sub-classes.
    relevant_kwargs = ["max_cache_length", "global_tokens"]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        head_specific=False,
        variable_length=False,
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
        self.register_buffer(
            "cache_cts",
            torch.zeros((n_heads if variable_length else 1), dtype=torch.int),
        )

        # Incase the |prompt| > max_cache_length, we need to compress the prompt which requires a separate strategy
        self.prompt_compressor = (
            None
            if self.prompt_compression_strategy is None
            else prompt_compressor_constructor(self.prompt_compression_strategy)(
                head_specific=self.head_specific, **kwargs
            )
        )

        # This turns True when the global tokens are fully filled
        self.global_filled = self.global_tokens == 0

    def reset(self):
        """
        If needed, this will reset the cache, although it is likely not necessary for most cache types.
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_cts.zero_()
        self.pos.fill_(-1)

    def return_attn(self):
        """
        Returns whether the cache requires attention weights for cache management.
        """
        return False

    def is_prefill(self):
        """
        Returns whether the cache is in the prefill stage.
        """
        # self.cache_cts is either tensor scalar or tensor of shape [num_heads] for variable-length caches.
        return self.cache_cts.max() == 0

    def compress_prompt_w_attn(self, input_pos, k_val, v_val, attn) -> None:
        # If the prompt is longer than the cache, we need to compress it to fit cache and then store (update).
        assert (
            input_pos.shape[-1] > self.max_cache_length
        ), "You called compress_prompt in prefill stage yet prompt is not longer than max_cache_length."
        input_pos, k_val, v_val, attn = self.prompt_compressor(
            input_pos, k_val, v_val, attn
        )
        self.update(input_pos, k_val, v_val)
        self.update_attn_history(attn)

    def compress_prompt(
        self, input_pos, k_val, v_val
    ) -> Tuple[torch.Tensor, torch.Tensor, Callable | None]:
        # We need to compress the prompt to fit the cache and then store (update).
        _, _, callback = self.update(*self.prompt_compressor(input_pos, k_val, v_val))
        # Yet we return the un-compressed KV since during pre-fill we compute full causal attention.
        return k_val, v_val, callback

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

        prompt_compressor_requires_attn = (
            self.prompt_compressor is not None
            and self.prompt_compressor.requires_attn()
        )

        # If the prompt is longer than the cache, we need to call compress_prompt
        # We return it as a callback since we don't have the attention weights yet
        if prompt_overflow and prompt_compressor_requires_attn:
            return k_val, v_val, self.compress_prompt_w_attn

        # If prompt is too long but we don't need attention to compress, we do it now and return the compressed vals
        if prompt_overflow:
            return self.compress_prompt(input_pos, k_val, v_val)

        # If the cache requires attention weights to manage evictions, we need to pass self.update_attn_history as a callback
        # We wrap it as a lambda so that it has the same function signature as the compress_prompt callback despite only needing attn
        attn_history_callback = (
            (lambda input_pos, k_val, v_val, attn: self.update_attn_history(attn))
            if self.return_attn()
            else None
        )

        self.cache_cts += self._update(input_pos, k_val, v_val)

        # Truncate the cache based on number of insertions. It will be at the end since we prefill in-order
        k = (
            self.k_cache[:, :, : self.cache_cts, :]
            if self.cache_cts < self.max_cache_length
            else self.k_cache
        )
        v = (
            self.v_cache[:, :, : self.cache_cts, :]
            if self.cache_cts < self.max_cache_length
            else self.v_cache
        )
        return k, v, attn_history_callback

    @abstractmethod
    def _update(self, input_pos, k_val, v_val) -> torch.Tensor | None:
        """
        Cache-specific update logic.
        Takes in the input positions and the corresponding k and v values.
        Modifies self.pos, self.k_cache, self.v_cache place.

        Returns a tensor indicating the number of tokens inserted - number of tokens evicted.
        None is equivalent to 0.
        """
        pass

    def fill_contiguous(self, input_pos, k_val, v_val, start=None, end=None):
        """
        A simple utility to fill the cache and pos.
        If start and end are provided, only fill the cache between those indices.
        Otherwise, treat start as self.cache_cts and end as self.cache_cts + num_new_insertions.
        Will also mark the global_tokens as they are updated.
        """
        num_insertions = self.cache_cts.squeeze()
        num_new_insertions = k_val.shape[2]
        if start is None:
            assert end is None
            start = num_insertions
            end = start + num_new_insertions

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
                num_insertions + num_new_insertions
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

        num_total_insertions: The total number of tokens inserted so far. The sum of cache_cts and num_new_insertions.

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
        # Never any prompt compression for full cache
        self.prompt_compression_strategy = None
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific=False, **kwargs
        )

    def _update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        self.fill_contiguous(input_pos, k_val, v_val)
        return input_pos.shape[-1]


class KVCacheRandom(KVCache):
    relevant_kwargs = [
        "max_cache_length",
        "global_tokens",
        "prompt_compression_strategy",
    ]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific=False, **kwargs
        )

    def _update(self, input_pos, k_val, v_val):
        start = end = None  # We will fill the cache in order if start and end are None

        need_to_evict = self.cache_cts >= self.max_cache_length
        if need_to_evict:  # Select a spot at random
            start = torch.randint(low=0, high=self.max_cache_length, size=(1,)).item()
            end = start + 1

        # Specify specific start and end indices
        self.fill_contiguous(input_pos, k_val, v_val, start=start, end=end)
        return input_pos.shape[-1]


class KVCacheWindow(KVCache):
    relevant_kwargs = [
        "max_cache_length",
        "global_tokens",
        "prompt_compression_strategy",
        # NB: "recent_window" is ignored as a relevant kwarg. It is fixed to self.max_cache_length - self.global_tokens.
    ]

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

    def _update(self, input_pos, k_val, v_val):
        start = end = None  # We will fill the cache in order if start and end are None

        need_to_evict = self.cache_cts >= self.max_cache_length
        if need_to_evict:  # Identify the least recent spot
            start = torch.argmin(self.pos).item()
            assert (
                input_pos.shape[-1] == 1
            ), "Should only be passing 1 new token at a time after cache is filled!"
            end = start + 1

        # Specify specific start and end indices
        self.fill_contiguous(input_pos, k_val, v_val, start=start, end=end)

        return input_pos.shape[-1]


class KVCacheL2(KVCacheWindow):
    relevant_kwargs = [
        "max_cache_length",
        "global_tokens",
        "recent_window",
        "prompt_compression_strategy",
    ]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific=True, **kwargs
        )

        key_norm_shape = (max_batch_size, n_heads, self.max_cache_length)
        self.register_buffer("key_norm", torch.zeros(key_norm_shape, dtype=dtype))

    def _update(self, input_pos, k_val, v_val):
        key_norm = torch.linalg.vector_norm(k_val, ord=2, dim=-1)

        need_to_evict = self.cache_cts >= self.max_cache_length

        if need_to_evict:
            # Set global and recent tokens to have lowest possible eviction score (-inf)
            eviction_score = self.key_norm.masked_fill(
                self.pos >= input_pos - self.recent_window, float("-inf")
            )
            fill_indices = torch.argmax(eviction_score, dim=-1).squeeze(0)
            self.fill_headwise(fill_indices, input_pos, k_val, v_val)

            # Do a scatter update to update the key norms
            fill_indices = fill_indices.view(1, -1, 1)
            self.key_norm.scatter_(2, fill_indices, key_norm)
        else:  # Insert into first unfilled spots
            self.fill_contiguous(input_pos, k_val, v_val)
            start, end = self.cache_cts, self.cache_cts + k_val.shape[2]
            self.key_norm[:, :, start:end] = key_norm

        return input_pos.shape[-1]

    def update_attn_history(self, attn):
        """
        This will be called if |prompt| > max_cache_length and SnapKV prompt compression is used.
        Because L2 cache does not require attention weights, this function is a no-op.
        """
        pass


class KVCacheScissorhands(KVCacheWindow):
    relevant_kwargs = [
        "max_cache_length",
        "global_tokens",
        "history_window_size",
        "drop_amount",
        "recent_window",
        "attn_thresholding",
        "prompt_compression_strategy",
        "attn_record_freq",
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

        self.register_buffer("attn_counter", torch.zeros((), dtype=torch.int64))

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
        self.register_buffer("eviction_idx", torch.tensor(self.drop_amount))

        assert self.recent_window >= self.attn_record_freq, (
            f"Since recent window ({self.recent_window}) < attention record frequency ({self.attn_record_freq}), you will get nan scores when "
            "deciding which tokens to evict because >0 non-local tokens will have no attention history."
        )

    def queue_len(self):
        return self.drop_amount - self.eviction_idx

    def return_attn(self) -> bool:
        """
        Whether or not we need to return attention weights for cache management.

        We return attention weights if 3 conditions are met:
        1) The cache is not in the prefill stage.
        2) The number of tokens left in the eviction queue // the frequency with which we record attention < attention history window.
        3) The number of insertions is a multiple of the frequency with which we record attention.

        The number of tokens in eviction queue specifies how many turns before we need to re-calculate importance.
        We only need to start recording once the number of steps until recomputation is equal to the recent window.
        """

        return (
            not self.is_prefill()
            and self.queue_len() // self.attn_record_freq <= self.history_window_size
            and self.cache_cts.squeeze() % self.attn_record_freq == 0
        )

    def update_attn_history(self, attn: torch.Tensor):
        """
        Insert the most recent attention into the history buffer.

        Rather than raw probability, insert a binary indicator of whether the attention is "low".
        """
        attn = attn.squeeze()
        keys = attn.shape[1]
        self.attn_history_num[
            :, :, :keys, self.attn_counter % self.history_window_size
        ] = (attn >= 1 / keys).int() if self.attn_thresholding else attn
        self.attn_history_denom[:, :, :keys] += 1
        self.attn_counter += 1

    def refill_eviction_queue(self, input_pos: int):
        # Identify the tokens with consistently "low" attentions
        numerator = self.attn_history_num.sum(dim=-1).float()
        # The denominator is the number of times this token's history has been recorded
        # We only record most self.history_window_size recent scores so need to clamp it
        denominator = self.attn_history_denom.clamp_max(self.history_window_size)

        attn_scores = numerator / denominator

        # Save the global & most recent tokens from being evicted
        attn_scores.masked_fill_(self.pos >= input_pos - self.recent_window, 1.0)

        assert torch.max(attn_scores) <= 1, "Average scores should be between 0 and 1."

        _, toks_to_evict = attn_scores.topk(
            self.drop_amount, dim=-1, sorted=True, largest=False
        )

        # The eviction queue will be empty so just re-assign it
        self.eviction_queue = toks_to_evict
        self.eviction_idx.zero_()

    def _update(self, input_pos, k_val, v_val):
        num_new_tokens = input_pos.shape[-1]

        need_to_evict = self.cache_cts >= self.max_cache_length

        if not need_to_evict:  # Insert into first unfilled spots
            self.fill_contiguous(input_pos, k_val, v_val)
            return num_new_tokens

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

        return num_new_tokens


def get_cache_constructor(cache_strategy):
    if cache_strategy == "full":
        return KVCacheFull
    elif cache_strategy == "l2":
        return KVCacheL2
    elif cache_strategy == "random":
        return KVCacheRandom
    elif cache_strategy == "window":
        return KVCacheWindow
    elif cache_strategy == "scissor":
        return KVCacheScissorhands
    else:
        raise ValueError(f"Invalid cache strategy: {cache_strategy}")
