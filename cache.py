import regex as re
from abc import ABC, abstractmethod
from typing import Tuple, Callable

import math
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

    # ScissorHands (https://arxiv.org/abs/2305.17118) recommends smaller caches at higher levels --> pyramid
    group.add_argument(
        "--cache_length_pattern",
        default="tile",
        choices=["tile", "repeat", "funnel", "pyramid"],
    )

    strategies = ["full", "random", "window", "scissor", "l2", "fastgen"]
    debug_strategies = [f"debug_{strategy}" for strategy in strategies]
    strategies.extend(debug_strategies)

    group.add_argument(
        "--cache_strategy",
        default=["full"],
        nargs="+",
        choices=strategies,
    )

    group.add_argument(
        "--cache_strategy_pattern",
        default="tile",
        choices=["tile", "repeat"],
        help="How to apply the cache_strategy across layers.",
    )

    # Dealing with Long Prompts
    parser.add_argument(
        "--feed_long_prompts",
        default=False,
        action="store_true",
        help="If True and |prompt| > max_cache_length, prefill with prompt[:max_cache_length], and feed prompt[max_cache_length:] sequentially.",
    )
    group.add_argument(
        "--prompt_compression_strategy",  # This doesn't matter if args.feed_long_prompts is True
        default=["recent_global"],
        nargs="+",
        choices=["recent_global", "snapkv", "l2", "random"],
        help="If |prompt| exceeds max_cache_length, we need to specify a strategy for compressing it to max_cache_length.",
    )

    # Optional Cache Kwargs depending on cache_strategy
    group.add_argument(
        "--global_tokens",
        default=1,
        type=int,
        help="The number of initial tokens to always include in the KV-Cache.  \
        If using window strategy, the actual window becomes max_cache_length - global_tokens.",
    )

    # Locality
    group.add_argument(
        "--recent_window",  # NB: for KVCacheWindow, recent_window is implicitly set to self.max_cache_length - self.global_tokens.
        default=10,  # 10 is default specified in ScissorHands paper ("r" in Algorithm 2).
        type=float,  # If < 1, it is a fraction of max_cache_length.
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
        default=0.0,  # 0 means we re-calculate eviction token every time. 0.4 is default specified in paper.
        type=float,
        help="The number of tokens to evict KV-Cache reaches capacity (max_cache_length). Expressed as a fraction of max_cache_length.",
    )
    group.add_argument(
        "--attn_thresholding",
        default=False,
        action="store_true",
        help="Whether to accumulate number of times a token was unimportant (binary) versus raw un-normalized probabilities. If true, more memory efficient.",
    )
    group.add_argument(
        "--attn_record_freq",
        default=1,
        type=int,
        help="How often to record attention weights for the ScissorHands cache.",
    )

    # FastGen-specific Hyperparameters (--cache_strategy == "fastgen")
    parser.add_argument(
        "--heavy_hitter_frac",
        default=0.3,
        type=float,
        help="Fraction of max_cache_length to consider as heavy hitters in the KV-cache.",
    )

    parser.add_argument(
        "--min_recovery_frac",
        default=0.9,
        type=float,
        help="Mininum fraction of recovered attentions (|compressed_attn - uncompressed_attn| < epsilon). The lower the value, the higher the compression.",
    )


def cache_compatibility(args):
    for length, cache_strat, prompt_strat in zip(
        args.max_cache_length, args.cache_strategy, args.prompt_compression_strategy
    ):
        if cache_strat == "full":
            assert (
                length == 1.0
            ), "Full cache strategy only supports max_cache_length=1.0."
        if cache_strat == "scissor":
            assert (
                prompt_strat == "snapkv"
            ), f'Scissor requires "snapkv" prompt compression strategy, not {prompt_strat}'

    print("The cache argument values you provided appear compatible with each other!")


def create_window_attention_mask(seq_len, window_size, device):
    # Initialize the mask tensor with zeros
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        mask[i, max(0, i - window_size) : i] = True
    return mask


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
        k_cache = torch.zeros(cache_shape, dtype=dtype)
        v_cache = torch.zeros(cache_shape, dtype=dtype)
        self.register_buffer("k_cache", k_cache)
        self.register_buffer("v_cache", v_cache)

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
        self.always_keep_prompt = self.global_tokens == -1

        # KVCacheFastGen requires profiling attention heads during prefill. This must be handled with separate callback.
        self.prefill_attn_callback = None

    def reset(self):
        """
        Resets the cache to its initial state for a new example.

        NB: For more performance, don't reset k_cache and v_cache since we overwrite them in update.
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_cts.zero_()
        self.pos.fill_(-1)
        if self.always_keep_prompt:
            self.global_tokens = (
                -1
            )  # -1 means we will resize it to the prompt size during prefill
        self.global_filled = self.global_tokens == 0

    def return_attn(self):
        """
        Returns whether the cache requires attention weights for cache management.
        """
        return False

    def compute_statistics(self, seq_len):
        """
        Computes statistics about the cache.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The cache size, the number of tokens inserted, and the compression ratio.
        """
        return {
            "compression_ratio": self.compression_ratio(seq_len).item(),
        }

    def compression_ratio(self, seq_len):
        """
        Returns the compression ratio of the cache.
        """
        # Final token isn't passed to cache so must -1 from seq_len
        n = seq_len - 1
        return ((n - torch.clamp_max(self.cache_cts, self.max_cache_length)) / n).mean()

    def return_kv_cache(self):
        # Truncate the cache based on number of insertions. It will be at the end since we prefill in-order.
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

        # Since we truncate there's no mask
        mask = None
        return k, v, mask

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
        # If you need input_ids you will have to pass them to prompt_compressor and have prompt_compressor return them in proper order.
        # Only FastGen uses them for now and it has its own prefill callback called "profile_and_update".
        self.update(input_pos, k_val, v_val, input_ids=None)
        self.update_attn_history(attn)

    def compress_prompt(
        self, input_pos, k_val, v_val
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable | None]:
        mask = None  # We will be performing causal attention on full inputs (mask won't be used)
        if self.prompt_compressor.requires_attn():
            compress_callback = {
                "func": (
                    lambda input_pos,
                    input_ids,
                    k_val,
                    v_val,
                    attn: self.compress_prompt_w_attn(input_pos, k_val, v_val, attn)
                )
            }
            return k_val, v_val, mask, compress_callback

        # If we can compress without attention, we don't need to pass it as a callback. We can call update now and see if there's a different callback.
        _, _, _, new_callback = self.update(
            *self.prompt_compressor(input_pos, k_val, v_val)
        )
        # Yet we return the un-compressed KV since during pre-fill we compute full causal attention.
        return k_val, v_val, mask, new_callback

    def attn_history_callback(self) -> Callable | None:
        """
        Returns a callback to update the attention history.

        Returns None if attention is not needed
        """
        return (
            {
                "func": lambda input_pos,
                input_ids,
                k_val,
                v_val,
                attn: self.update_attn_history(attn)
            }
            if self.return_attn()
            else None
        )

    def update(self, input_pos, k_val, v_val, input_ids=None):
        """
        Updates the cache with the given input positions, keys, and  values.

        Parameters:
            input_pos (torch.Tensor): A tensor of input positions.
            k_val (torch.Tensor): A tensor of keys.
            v_val (torch.Tensor): A tensor of values.
            input_ids (torch.Tensor): A tensor of input ids.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, bool]: A tuple containing the updated cache of keys and values,
            both truncated to the minimum of the current insertions and the maximum cache length. The last value
            is a boolean return_attn indicating whether the cache requires attention weights. If True, the model
            will call self.update_attn_history with the attention weights.
        """
        is_prefill = self.is_prefill()
        num_tokens = input_pos.shape[-1]

        # FastGen requires a special callback for prefill that profiles attention heads and updates the cache.
        prefill_callback = None if not self.is_prefill() else self.prefill_attn_callback
        if prefill_callback is not None:
            mask = None
            return k_val, v_val, mask, prefill_callback

        # If the prompt is longer than the cache, we need to compress it to fit cache and then store (update).
        prompt_overflow = num_tokens > self.max_cache_length
        if prompt_overflow:
            return self.compress_prompt(input_pos, k_val, v_val)

        # k_val: [B, H, S, D] -> S is > 1 for prefill, 1 for new tokens
        if is_prefill:
            assert num_tokens > 1
        else:
            assert num_tokens == 1

        attn_history_callback = (
            {
                "func": lambda input_pos,
                input_ids,
                k_val,
                v_val,
                attn: self.update_attn_history(attn)
            }
            if self.return_attn()
            else None
        )

        self.cache_cts += self._update(input_pos, k_val, v_val, input_ids=input_ids)

        k, v, mask = self.return_kv_cache()
        return k, v, mask, attn_history_callback

    @abstractmethod
    def _update(self, input_pos, k_val, v_val, input_ids=None):
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
        num_insertions = self.cache_cts[
            0
        ]  # If we are calling this function, self.cache_cts should be uniform across all heads
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
        self.pos[:, :, :num_to_mark] = int(1e9)
        return num_to_mark == self.global_tokens


class KVCacheFull(KVCache):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        # Never any prompt compression for full cache
        self.prompt_compression_strategy = None
        self.global_tokens = 0  # No global tokens for full cache (they are all global)
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific=False, **kwargs
        )

    def _update(self, input_pos, k_val, v_val, input_ids=None):
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

    def _update(self, input_pos, k_val, v_val, input_ids=None):
        start = end = None  # We will fill the cache in order if start and end are None

        need_to_evict = self.cache_cts >= self.max_cache_length
        if need_to_evict:  # Select a spot at random
            start = torch.randint(low=0, high=self.max_cache_length, size=(1,))
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
        variable_length=False,
        **kwargs,
    ):
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            head_specific,
            variable_length,
            **kwargs,
        )

    def _update(self, input_pos, k_val, v_val, input_ids=None):
        start = end = None  # We will fill the cache in order if start and end are None

        need_to_evict = self.cache_cts >= self.max_cache_length
        if need_to_evict:  # Identify the least recent spot
            start = torch.argmin(self.pos)
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

    def _update(self, input_pos, k_val, v_val, input_ids=None):
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
        requires_eviction_queue=True,
        variable_length=False,
        **kwargs,
    ):
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            head_specific,
            variable_length,
            **kwargs,
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

        assert self.recent_window >= self.attn_record_freq, (
            f"Since recent window ({self.recent_window}) < attention record frequency ({self.attn_record_freq}), you will get nan scores when "
            "deciding which tokens to evict because >0 non-local tokens will have no attention history."
        )

        if requires_eviction_queue:
            # Different eviction queue for each attention head
            eviction_queue_shape = (
                max_batch_size,
                n_heads,
                self.drop_amount,
            )
            self.register_buffer(
                "eviction_queue", torch.zeros(eviction_queue_shape, dtype=torch.int32)
            )
            # Start with an "empty queue" so that we can fill it up.
            self.register_buffer("eviction_idx", torch.tensor(self.drop_amount))

            assert self.queue_len() == 0

    def reset(self):
        super().reset()
        self.attn_history_num.zero_()
        self.attn_history_denom.zero_()
        self.attn_counter.zero_()
        if hasattr(self, "eviction_queue"):
            self.eviction_queue.zero_()
            # Start with an "empty queue" so that we can fill it up
            self.eviction_idx.fill_(self.drop_amount)
            assert self.queue_len() == 0

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
            and (
                self.cache_cts.squeeze() % self.attn_record_freq == 0
                or self.attn_counter == 0
            )
        )

    def update_attn_history(self, attn: torch.Tensor):
        """
        Insert the most recent attention into the history buffer.

        If self.attn_thresholding = True, insert a binary indicator of whether the attention >= uniform attention.
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

        avg_attn = numerator / denominator

        # Save the global & most recent tokens from being evicted
        avg_attn.masked_fill_(self.pos >= input_pos - self.recent_window, 1.0)

        _, toks_to_evict = avg_attn.topk(
            self.drop_amount, dim=-1, sorted=True, largest=False
        )

        # The eviction queue will be empty so just re-assign it
        self.eviction_queue = toks_to_evict
        self.eviction_idx.zero_()

    def _update(self, input_pos, k_val, v_val, input_ids=None):
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


class KVCacheFastGen(KVCacheScissorhands):
    relevant_kwargs = [
        "max_cache_length",
        "history_window_size",
        "recent_window",
        "attn_thresholding",
        "token_ids",
        "prompt_compression_strategy",
        "min_recovery_frac",
        "heavy_hitter_frac",
    ]

    strategies = [
        "special",
        "special_punc",
        "special_punc_heavy",
        "special_punc_heavy_local",
        "full",
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
        self.global_tokens = 0  # No global tokens for FastGen
        self.attn_record_freq = 1  # We record attention every step for FastGen
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            head_specific,
            requires_eviction_queue=False,
            variable_length=True,
            **kwargs,
        )

        special_ids = [torch.tensor(ids) for ids in kwargs["token_ids"]["special"]]
        self.register_buffer("special_ids", torch.nested.nested_tensor(special_ids))

        # Store the punctuation vocabulary ids
        punc_ids = torch.Tensor(kwargs["token_ids"]["punctuation"])
        self.register_buffer("punc_ids", punc_ids)
        # As well as a mask showing where punctuation ids are in the KV cache
        # We store this to avoid re-computing the mask every time and having to store input_ids

        mask_shape = (max_batch_size, n_heads, self.max_cache_length)
        self.register_buffer("special_mask", torch.zeros(mask_shape, dtype=torch.bool))
        self.register_buffer("punc_mask", torch.zeros(mask_shape, dtype=torch.bool))

        # We need to use a mask since not all heads have same number of tokens. We can't simply truncate.
        # 1 dimension stands for query dimension, which will always be 1 (next token) for KV cache attention.
        kv_mask_shape = (max_batch_size, n_heads, 1, self.max_cache_length)
        self.register_buffer("mask", torch.zeros(kv_mask_shape, dtype=torch.bool))

        # NB: Kwargs are sdpa attention kwargs, not the kwargs for the "func"
        self.prefill_attn_callback = {
            "func": self.profile_and_update,
            "kwargs": {"return_attn_logits": False},
        }

    def return_attn(self):
        # We use a special callback for FastGen to profile the attention heads during prefill.
        return not self.is_prefill() and self.requires_heavy_check

    def return_kv_cache(self):
        return self.k_cache, self.v_cache, self.mask

    def eviction_idx_for_head(self, head_idx, input_pos, apply_window=False):
        numerator = (
            self.attn_history_num[:, head_idx, : self.cache_cts[head_idx]]
            .sum(dim=-1)
            .float()
        )
        # The denominator is the number of times this token's history has been recorded
        # We only record most self.history_window_size recent scores so need to clamp it
        denominator = self.attn_history_denom[
            :, head_idx, : self.cache_cts[head_idx]
        ].clamp_max(self.history_window_size)
        avg_attn = numerator / denominator

        # Save the special & punctuation tokens from being evicted
        save_mask = torch.logical_or(
            self.special_mask[:, head_idx, : self.cache_cts[head_idx]],
            self.punc_mask[:, head_idx, : self.cache_cts[head_idx]],
        )
        if apply_window:
            save_mask = torch.logical_or(
                save_mask,
                self.pos[:, head_idx, : self.cache_cts[head_idx]]
                >= input_pos - self.recent_window,
            )

        avg_attn.masked_fill_(save_mask, 1)
        fill_idx = avg_attn.argmin(dim=-1)

        return fill_idx

    def select_fill_idx(self, strategy, head_idx, input_pos, is_punc: bool = False):
        fill_idx = None
        eviction_required = False

        def _end_idx():
            # We need to clone because self.cache_cts will be incremented later and we don't want to have fill_idx as a mutable reference
            return min(self.max_cache_length - 1, self.cache_cts[head_idx].clone())

        if strategy == KVCacheFastGen.strategies.index("special"):
            pass  # We are assuming we don't generate special tokens
        elif strategy == KVCacheFastGen.strategies.index("special_punc"):
            if is_punc:
                fill_idx = _end_idx()
        elif strategy == KVCacheFastGen.strategies.index(
            "special_punc_heavy"
        ) or strategy == KVCacheFastGen.strategies.index("special_punc_heavy_local"):
            apply_window = strategy == KVCacheFastGen.strategies.index(
                "special_punc_heavy_local"
            )
            # If there's still room in the cache, just fill it in the next open slot
            budget = (
                self.num_special
                + self.num_punc
                + (self.heavy_hitter_frac * self.max_cache_length)
            )
            if apply_window:  # We are also allowed budget for the recent tokens
                budget += self.recent_window

            eviction_required = self.cache_cts[head_idx] >= budget
            if eviction_required:
                # Figure out which token to evict -- make sure we don't evict special or punc
                fill_idx = self.eviction_idx_for_head(
                    head_idx, input_pos, apply_window=apply_window
                )
                self.attn_history_num[:, head_idx, fill_idx, :].fill_(0)
                self.attn_history_denom[:, head_idx, fill_idx].fill_(0)
            else:
                # We can fit it in the cache
                fill_idx = _end_idx()
        elif strategy == KVCacheFastGen.strategies.index("full"):
            fill_idx = _end_idx()
        else:
            raise ValueError(f"Unrecognized strategy index {strategy}.")

        return fill_idx, eviction_required

    def reset(self):
        super().reset()
        self.num_special = 0
        self.num_punc = 0
        self.mask.zero_()
        self.special_mask.zero_()
        self.punc_mask.zero_()
        self.cache_strategies = None
        self.requires_heavy_check = True

    def _update(self, input_pos, k_val, v_val, input_ids=None):
        n_heads = k_val.shape[1]

        is_punc = torch.isin(input_ids, self.punc_ids)

        # If fill idx is None we place value at the back (which is truncated for attention calculation anyway)
        fill_indices = torch.full(
            (n_heads,),
            self.max_cache_length - 1,
            dtype=torch.int64,
            device=k_val.device,
        )

        cache_ct_incr = torch.zeros_like(fill_indices)

        for head_idx, strategy in enumerate(self.cache_strategies):
            fill_idx, eviction_required = self.select_fill_idx(
                strategy, head_idx, input_pos, is_punc=is_punc
            )

            if fill_idx is None:
                continue

            cache_ct_incr[head_idx] = 1

            fill_indices[head_idx] = fill_idx
            if not eviction_required:
                # We can't use all fill indices to bulk assign mask because some fill_indices are dummies (self.max_cache_length - 1)
                self.mask[:, head_idx, :, fill_idx] = True

        self.fill_headwise(fill_indices, input_pos, k_val, v_val)
        self.punc_mask.scatter_(
            2, fill_indices.view(1, -1, 1), is_punc.view(1, 1, 1).expand(1, n_heads, 1)
        )

        # Only update global self.num_punc once (not once per head)
        # If a head keeps punc tokens, each head will have same number of punc tokens (no punc evictions)
        if is_punc:
            self.num_punc += 1

        return cache_ct_incr

    def build_special_ids_mask(self, input_ids):
        seq_len = input_ids.shape[-1]
        special_ids_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for special_id in self.special_ids:
            # Iterate over input_ids to check for the exact sub-sequence
            id_len = len(special_id)
            if id_len == 1:
                special_ids_mask[torch.where(input_ids == special_id)[0]] = True
            else:
                for i in range(seq_len - id_len + 1):
                    if torch.equal(input_ids[i : i + id_len], special_id):
                        special_ids_mask[i : i + id_len] = True
        return special_ids_mask

    def build_punc_ids_mask(self, input_ids):
        # TODO should be on same device as model with register_buffer
        if self.punc_ids.device != input_ids.device:
            self.punc_ids = self.punc_ids.to(input_ids.device)
        punc_ids_mask = torch.isin(input_ids, self.punc_ids)
        return punc_ids_mask

    def profile_attn_heads(self, input_pos, input_ids, attn):
        input_ids = input_ids.squeeze(0)
        seq_len = input_ids.shape[-1]
        n_heads = attn.shape[1]

        special_mask = self.build_special_ids_mask(input_ids)
        special_mask_exp = special_mask.view(1, 1, -1).expand(n_heads, seq_len, seq_len)
        # Store number of special tokens for later use
        self.num_special = special_mask.sum()

        punc_mask = self.build_punc_ids_mask(input_ids)
        self.num_punc = punc_mask.sum()

        punc_mask_exp = punc_mask.view(1, 1, -1).expand(n_heads, seq_len, seq_len)
        window_mask = create_window_attention_mask(
            seq_len, self.recent_window, input_ids.device
        )

        # Average of cumulative attention probs (use input_pos to normalize)
        cum_attn = torch.softmax(attn, dim=-1).squeeze(0).sum(dim=1) / (
            seq_len - input_pos
        )
        heavy_hitters = (
            cum_attn.topk(
                # Can calculate heavy hitters based on seq_len
                math.ceil(min(self.heavy_hitter_frac * seq_len, seq_len)),
                dim=1,
                largest=True,
            )
            .indices.unsqueeze(1)
            .expand(-1, seq_len, -1)
        )

        # Hybrid Strategies:
        # - special
        # - special + punc
        # - special + punc + frequent / heavy
        # - special + punc + frequent / heavy + local
        # - full
        special_punc = torch.logical_or(special_mask_exp, punc_mask_exp)
        special_punc_heavy = special_punc.scatter(2, heavy_hitters, True)
        special_punc_heavy_local = torch.logical_or(special_punc_heavy, window_mask)

        masks = torch.stack(
            [
                special_mask_exp,
                special_punc,
                special_punc_heavy,
                special_punc_heavy_local,
                torch.ones_like(special_mask_exp),
            ]
        )

        attn_rep = attn.expand(masks.shape[0], -1, -1, -1)

        compressed_scores = attn_rep.masked_fill(~masks, 0).sum(dim=-1).mean(dim=-1)

        # For each column, return the first row which has cost >= min_recovery_frac
        cache_strategies = (
            (compressed_scores >= self.min_recovery_frac).int().argmax(dim=0)
        )

        # Take the last query's mask as the initial KV-Cache fill mask
        masks_all = masks[:, :, -1, :].transpose(1, 0)
        # Select mask based on self.cache_strategies
        mask_optimal = masks_all.gather(
            1, cache_strategies.view(-1, 1, 1).expand(-1, -1, seq_len)
        ).squeeze(1)

        return cache_strategies, special_mask, punc_mask, mask_optimal, cum_attn

    def profile_and_update(self, input_pos, input_ids, k_val, v_val, attn):
        """
        Profile the attention heads to determine the optimal KV-cache allocation.
        """
        assert self.is_prefill(), "Should only be profiling during prefill stage."

        input_ids = input_ids.squeeze(0)
        seq_len = input_ids.shape[-1]
        n_heads = attn.shape[1]
        dim = k_val.shape[-1]

        # Profile cache attention heads to define strategy for each head
        self.cache_strategies, special_mask, punc_mask, mask_optimal, cum_attn = (
            self.profile_attn_heads(input_pos, input_ids, attn)
        )

        # Uncomment to show which strategies are selected
        # print([self.strategies[i] for i in self.cache_strategies.tolist()])

        # If none of the heads selected a heavy hitter strategy, we don't need to track attention weights
        self.requires_heavy_check = any(
            ["heavy" in KVCacheFastGen.strategies[i] for i in self.cache_strategies]
        )

        # Put the selected items (true values from mask) to the front. Re-arrange attentions as well.
        order = mask_optimal.int().argsort(dim=1, descending=True)
        order_exp = order.view(1, n_heads, seq_len, 1).expand(-1, -1, -1, dim)

        k_val = k_val.gather(2, order_exp)
        v_val = v_val.gather(2, order_exp)
        input_pos = input_pos.unsqueeze(0).expand(n_heads, -1).gather(1, order)
        self.fill_contiguous(input_pos, k_val, v_val)

        # Record number of tokens to be inserted into the cache
        self.cache_cts = mask_optimal.sum(dim=1)

        # We will need to remove special tokens and punctuation from heavy hitter eviction so need to their positions.
        special_mask = special_mask.view(1, -1).expand(n_heads, -1).gather(1, order)
        punc_mask = punc_mask.view(1, -1).expand(n_heads, -1).gather(1, order)
        self.special_mask[:, :, :seq_len] = special_mask
        self.punc_mask[:, :, :seq_len] = punc_mask

        # Update mask to reflect how many items have been inserted into each head
        range_mask = (
            torch.arange(seq_len, device=self.mask.device)
            .view(1, -1)
            .expand(n_heads, -1)
        )
        self.mask[:, :, :, :seq_len] = (
            range_mask < self.cache_cts.view(-1, 1).expand(-1, seq_len)
        ).view(-1, n_heads, 1, seq_len)

        if self.requires_heavy_check:
            # Update attention mask to indicate which we attentions are allowed.
            cum_attn = cum_attn.gather(1, order)
            self.update_attn_history(cum_attn)


class KVCacheAnalysis(KVCache):
    relevant_kwargs = [
        "max_cache_length",
        "history_window_size",
        "recent_window",
        "attn_thresholding",
        "token_ids",
        "prompt_compression_strategy",
        "min_recovery_frac",
        "heavy_hitter_frac",
        "global_tokens",
        "drop_amount",
        "prompt_compression_strategy",
        "attn_record_freq",
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        cache_strategy="scissor",
        **kwargs,
    ):
        # Never any prompt compression for full cache
        full_kwargs = {
            "prompt_compression_strategy": None,
            "global_tokens": 0,
            "max_cache_length": kwargs["max_cache_length"],
        }
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific=False, **full_kwargs
        )

        # Initialize the compressed cache we want to analyze.
        self.compressed = get_cache_constructor(cache_strategy=cache_strategy)[0](
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            **kwargs,
        )

        self.register_buffer(
            "attention_losses",
            torch.full((self.max_cache_length,), fill_value=-1, dtype=dtype),
        )

    def return_attn(self):
        return self.compressed.return_attn()

    def update(self, input_pos, k_val, v_val, input_ids=None):
        k, v, mask, _ = super().update(input_pos, k_val, v_val, input_ids=input_ids)
        _, _, _, attn_callback = self.compressed.update(
            input_pos, k_val, v_val, input_ids=input_ids
        )

        if attn_callback is not None and input_pos.shape[-1] == 1:
            # This is hairy but we need to re-write callback to call this class's update_attn_history not the compressed
            # This is because we need to filter the attention weights to only the tokens in the compressed cache first.
            attn_callback = self.attn_history_callback()
            assert attn_callback is not None

        return k, v, mask, attn_callback

    def _update(self, input_pos, k_val, v_val, input_ids=None):
        # input_pos: [S], k_val: [B, H, S, D]
        self.fill_contiguous(input_pos, k_val, v_val)
        return input_pos.shape[-1]

    def reset(self):
        super().reset()
        self.compressed.reset()
        self.attention_losses.fill_(-1)

    def update_attn_history(self, attn: torch.Tensor):
        indices = self.compressed.pos.clone().long()

        # Global tokens will have been set to max seq length
        # We need to set them back to actual global tokens
        indices[:, :, : self.compressed.global_tokens] = (
            torch.arange(self.compressed.global_tokens, device=indices.device)
            .view(1, 1, -1)
            .expand(1, indices.shape[1], -1)
        )
        cutoff = min(indices.shape[-1], attn.shape[-1])
        assert torch.min(indices[:, :, :cutoff]) > -1
        assert torch.all(torch.eq(indices[:, :, cutoff:], -1))
        indices = indices[:, :, :cutoff]
        attn_compressed = attn.squeeze(2).gather(2, indices).unsqueeze(2)
        self.compressed.update_attn_history(attn_compressed)

        attn_loss = (1 - attn_compressed.sum(dim=-1)).mean()
        insert_idx = torch.where(self.attention_losses == -1)[0][0]
        self.attention_losses[insert_idx] = attn_loss

    def compute_statistics(self, seq_len):
        """
        Computes statistics about the cache.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The cache size, the number of tokens inserted, and the compression ratio.
        """
        stats = super().compute_statistics(seq_len)
        cutoff = torch.where(self.attention_losses == -1)[0]
        if len(cutoff) > 0:
            cutoff = cutoff[0]
        else:
            cutoff = len(self.attention_losses)
        stats["attention_loss"] = (self.attention_losses[:cutoff].sum() / cutoff).item()
        return stats


def get_cache_constructor(cache_strategy):
    relevant_kwargs = None
    if cache_strategy == "full":
        cls = KVCacheFull
    elif cache_strategy == "l2":
        cls = KVCacheL2
    elif cache_strategy == "random":
        cls = KVCacheRandom
    elif cache_strategy == "window":
        cls = KVCacheWindow
    elif cache_strategy == "scissor":
        cls = KVCacheScissorhands
    elif cache_strategy == "fastgen":
        cls = KVCacheFastGen
    elif cache_strategy.startswith("debug"):
        cache_strategy = re.sub(r"debug_+", "", cache_strategy).strip()
        relevant_kwargs = get_cache_constructor(cache_strategy)[1]
        cls = (
            lambda max_batch_size, n_heads, head_dim, dtype, **kwargs: KVCacheAnalysis(
                max_batch_size,
                n_heads,
                head_dim,
                dtype,
                cache_strategy=cache_strategy,
                **kwargs,
            )
        )
    else:
        raise ValueError(f"Invalid cache strategy: {cache_strategy}")

    return cls, relevant_kwargs or cls.relevant_kwargs
