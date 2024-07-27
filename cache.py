import math
import regex as re
from abc import ABC, abstractmethod
from collections import Counter
from prompt_compression import get_prompt_compressor_constructor

import argparse
import torch
import torch.nn as nn


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

    # ScissorHands (https://arxiv.org/abs/2305.17118) recommends large caches at higher levels --> funnel
    # Yet PyramidKV (https://arxiv.org/abs/2406.02069) recommends the opposite --> pyramid shaped
    group.add_argument(
        "--cache_length_pattern",
        default="tile",
        choices=["tile", "repeat", "funnel", "pyramid"],
    )

    strategies = [
        "full",
        "random",
        "recent_global",
        "heavy_hitter",
        "l2",
        "hybrid",
        "keep_it_odd",
    ]
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
        help="If |prompt| exceeds max_cache_length, we need to specify a strategy for compressing it to max_cache_length.",
    )

    # Optional Cache Kwargs depending on cache_strategy
    group.add_argument(
        "--global_tokens",
        default=1,
        type=int,
        help="The number of initial tokens to always include in the KV-Cache.  \
        If using recent_global strategy, the actual window size becomes max_cache_length - global_tokens.",
    )

    # Locality
    group.add_argument(
        "--recent_window",  # NB: for KVCacheRecentGlobal, recent_window is implicitly set to self.max_cache_length - self.global_tokens.
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
        "--attn_thresholding",
        default=False,
        action="store_true",
        help="Whether to accumulate number of times a token was unimportant (binary) versus raw un-normalized probabilities. If true, more memory efficient.",
    )

    # Hybrid, e.g., FastGen -specific Hyperparameters (--cache_strategy == "hybrid")
    parser.add_argument(
        "--min_recovery_frac",
        default=0.9,
        type=float,
        help="Mininum fraction of recovered attentions (|compressed_attn - uncompressed_attn| < epsilon). The lower the value, the higher the compression.",
    )


def cache_compatibility(args):
    for (
        length,
        cache_strat,
    ) in zip(args.max_cache_length, args.cache_strategy):
        if cache_strat == "hybrid":
            assert (
                not args.compile
            ), "Hybrid cache strategy is currently not supported with compile=True."

        if cache_strat in {"full", "hybrid"}:
            assert (
                length == 1.0
            ), f"{cache_strat} cache strategy only supports max_cache_length=1.0."

    print("The cache argument values you provided appear compatible with each other!")


def create_window_attention_mask(seq_len, window_size, device, global_tokens: int = 4):
    # Initialize the mask tensor with zeros
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    # Add global tokens
    mask[:, :global_tokens] = True
    for i in range(seq_len):
        mask[i, max(0, i + 1 - window_size) : i + 1] = True
    return mask


class KVCache(ABC, nn.Module):
    # Define which hyperparameters are relevant for the cache.
    # Override as needed for sub-classes.
    relevant_kwargs = ["max_cache_length", "global_tokens", "max_seq_length"]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        head_specific=False,  # IFF True, heads can contain different tokens, e.g., cache evictions are "head_specific".
        variable_length=False,  # IFF True, the number of tokens inserted can vary across heads. Only true for KVCacheHybrid.
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
        self.n_heads = n_heads
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

        # We need to use a mask since not all heads have same number of tokens. We can't simply truncate.
        # 1 dimension stands for query dimension, which will always be 1 (next token) for KV cache attention.
        kv_mask_shape = (max_batch_size, n_heads, 1, self.max_cache_length)
        self.register_buffer("mask", torch.zeros(kv_mask_shape, dtype=torch.bool))

    def reset(self):
        """
        Resets the cache to its initial state for a new example.

        NB: For more performance, don't reset k_cache and v_cache since we overwrite them in update.
        """
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.mask.zero_()
        self.cache_cts.zero_()
        self.pos.fill_(-1)

    def return_attn(self):
        """
        Returns whether the cache requires attention weights for cache management.
        """
        return False

    def memory_usage(self):
        tensors = []
        for obj in vars(self).values():
            if torch.is_tensor(obj):
                tensors.append(obj)
            elif isinstance(obj, dict):
                for vv in obj.values():
                    if torch.is_tensor(vv):
                        tensors.append(vv)

        return sum([t.element_size() * t.numel() for t in tensors]) / (1024**3)

    def compute_statistics(self, seq_len):
        """
        Computes statistics about the cache.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The cache size, the number of tokens inserted, and the compression ratio.
        """
        return {
            "compression_ratio": self.compression_ratio(seq_len).item(),
            "cache_memory_gb": self.memory_usage(),
        }

    def compression_ratio(self, seq_len):
        """
        Returns the compression ratio of the cache.
        """
        # Final token isn't passed to cache so must -1 from seq_len
        n = seq_len - 1
        assert torch.all(self.cache_cts <= self.max_cache_length)
        return ((n - self.cache_cts) / n).mean()

    def return_kv_cache(self):
        return self.k_cache, self.v_cache, self.mask

    def update_kv(self, input_pos, k_val, v_val, is_prefill, **kwargs):
        """
        Cache-specific update logic.
        Takes in the input positions and the corresponding k and v values.
        Modifies self.pos, self.k_cache, self.v_cache place.

        Returns a tensor indicating the number of tokens inserted - number of tokens evicted.
        None is equivalent to 0.
        """
        if is_prefill:
            num_insertions = self._prefill_update(input_pos, k_val, v_val, **kwargs)
        else:
            num_insertions = self._decoding_update(input_pos, k_val, v_val, **kwargs)
        self.cache_cts += num_insertions[: len(self.cache_cts)]

        # [Optional] Update any internal model state
        k, v, mask = (
            self.return_kv_cache()
        )  # By default, just returns self.k_cache, self.v_cache, self.mask
        return k, v, mask

    def update_state(self, *args, **kwargs):
        """
        Optional method to update cache-specific internal state (excludes self.k_cache, self.v_cache, and self.pos).
        """
        pass

    def _decoding_update(self, input_pos, k_val, v_val, **kwargs):
        """
        Decoding logic for the cache.
        """
        eviction_idx = self._eviction_idx(input_pos)

        # Num insertions means we inserted into an unfilled slot (previous pos == -1)
        # They should be all the same unless variable_length = True
        num_insertions = (
            (self.pos.gather(2, eviction_idx.view(1, -1, 1)).squeeze() == -1)
            .int()
            .view(-1)
        )

        self._fill(input_pos, k_val, v_val, fill_idxs=eviction_idx)

        return num_insertions

    def _eviction_idx(self, input_pos):
        scores = self._token_importances(input_pos)

        if scores.ndim == 1:
            scores = scores.unsqueeze(0)

        # Protect global tokens
        scores[:, : self.global_tokens] = float("inf")

        # Evict unfilled slots (pos == -1)
        scores.masked_fill_(self.pos.view(scores.shape) == -1, float("-inf"))

        # Evict least important token
        return torch.argmin(scores, dim=-1)

    def _prefill_update(self, input_pos, k_val, v_val, **kwargs):
        input_pos = input_pos.int()
        fill_idxs = torch.arange(input_pos.shape[-1], device=input_pos.device)
        self._fill_contiguous(input_pos, k_val, v_val, fill_idxs=fill_idxs)
        # Saves a fraction of time to return as a tensor rather than integer
        return torch.tensor(
            [input_pos.shape[-1]], dtype=torch.int, device=input_pos.device
        )

    def _fill_contiguous(
        self, input_pos, k_val, v_val, fill_idxs: torch.Tensor | int, **kwargs
    ):
        """
        A simple utility to fill the cache and pos.
        """
        self.pos[:, :, fill_idxs] = input_pos
        self.k_cache[:, :, fill_idxs, :] = k_val
        self.v_cache[:, :, fill_idxs, :] = v_val
        update_mask = kwargs.get("update_mask", True)
        if update_mask:
            self.mask[:, :, :, fill_idxs] = True

    @abstractmethod
    def _fill(self, input_pos, k_val, v_val, fill_idxs: torch.Tensor | int, **kwargs):
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
        raise NotImplementedError

    def update_attn_history(self, attn):
        """
        Update the attention history with the most recent attention weights.
        """
        raise Exception(
            f"{self.__class__.__name__} requested return_attn=True but has not yet implemented a update_attn_history function."
        )


class KVCacheHeadConstant(KVCache):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(
            max_batch_size, n_heads, head_dim, dtype, head_specific=False, **kwargs
        )

    def _fill(self, input_pos, k_val, v_val, fill_idxs: torch.Tensor | int, **kwargs):
        return self._fill_contiguous(input_pos, k_val, v_val, fill_idxs, **kwargs)


class KVCacheHeadSpecific(KVCache):
    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        variable_length=False,
        **kwargs,
    ):
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            head_specific=True,
            variable_length=variable_length,
            **kwargs,
        )

    def _fill(self, input_pos, k_val, v_val, fill_idxs: torch.Tensor | int, **kwargs):
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
        pos_fill_indices = fill_idxs.view(1, -1, 1)
        cache_fill_indices = fill_idxs.view(1, len(fill_idxs), 1, 1).expand(
            1, k_val.shape[1], 1, k_val.shape[-1]
        )
        input_pos = input_pos.view(1, -1, 1).expand(1, k_val.shape[1], 1).int()
        self.pos.scatter_(2, pos_fill_indices, input_pos.int())
        self.k_cache.scatter_(2, cache_fill_indices, k_val)
        self.v_cache.scatter_(2, cache_fill_indices, v_val)

        update_mask = kwargs.get("update_mask", True)
        if update_mask:
            self.mask.scatter_(3, fill_idxs.view(1, -1, 1, 1), True)


class KVCacheFull(KVCacheHeadConstant):
    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        self.global_tokens = 0  # No global tokens for full cache (they are all global)
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _eviction_idx(self, input_pos):
        # Select the first unfilled slot
        return self.pos[0, 0].argmin().view(1)


class KVCacheRandom(KVCacheHeadConstant):
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "global_tokens",
        "recent_window",
    ]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

    def _token_importances(self, input_pos):
        # Assign random importance
        scores = torch.rand(self.max_cache_length, device=input_pos.device)
        # Protect Recent Tokens
        scores[self.pos[0, 0] >= input_pos - self.recent_window] = float("inf")
        return scores


class KVCacheRecentGlobal(KVCacheHeadConstant):
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "global_tokens",
        # NB: "recent_window" is ignored as a relevant kwarg. It is fixed to self.max_cache_length - self.global_tokens.
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            **kwargs,
        )

    def _eviction_idx(self, input_pos):
        return (
            torch.argmin(self.pos[:, :, self.global_tokens :], dim=-1)
            + self.global_tokens
        ).view(1)


class KVCacheL2(KVCacheHeadSpecific):
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "global_tokens",
        "recent_window",
    ]

    def __init__(
        self, max_batch_size, n_heads, head_dim, dtype=torch.bfloat16, **kwargs
    ):
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **kwargs)

        key_norm_shape = (max_batch_size, n_heads, self.max_cache_length)
        self.register_buffer("key_norm", torch.zeros(key_norm_shape, dtype=dtype))

    def _token_importances(self, input_pos):
        # 1. Lowest l2 norms have high importance (- self.key_norm)
        # 2. Lowest score needs to be > -1 : we evict unfilled tokens first (+ max value such that min score is 0)
        # 3. Save Recent Window (+ inf)
        return (
            (self.key_norm.max() - self.key_norm)
            .masked_fill(self.pos >= input_pos - self.recent_window, float("inf"))
            .squeeze(0)
        )

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        if is_prefill:
            self.key_norm = torch.linalg.vector_norm(self.k_cache, ord=2, dim=-1)
        else:
            # Find where the just insserted input_pos is in the cache and update its key norm
            fill_indices = (self.pos == input_pos).nonzero()[:, -1]
            keys = self.k_cache.gather(
                2,
                fill_indices.view(1, -1, 1, 1).expand(1, -1, 1, self.k_cache.shape[-1]),
            )
            key_norm = torch.linalg.vector_norm(keys, ord=2, dim=-1)
            self.key_norm.scatter_(2, fill_indices.view(1, -1, 1), key_norm)


class KVCacheHeavyHitter(KVCacheHeadSpecific):
    # This class mostly follows the logic in ScissorHands (https://arxiv.org/abs/2305.17118)
    # But it is very similar to other Heavy Hitter methods (H20, PyramidKV, etc.)
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "global_tokens",
        "history_window_size",
        "recent_window",
        "attn_thresholding",
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        variable_length=False,
        **kwargs,
    ):
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
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

        self.register_buffer("attn_counter", torch.zeros((1,), dtype=torch.int64))

    def reset(self):
        super().reset()
        self.attn_history_num.zero_()
        self.attn_history_denom.zero_()
        self.attn_counter.zero_()

    def return_attn(self) -> bool:
        return True

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        """
        Insert the most recent attention into the history buffer.

        If self.attn_thresholding = True, insert a binary indicator of whether the attention >= uniform attention.
        """

        # Resize attn to be max cache length with zero padding if need be
        seq_len = attn.shape[-1]

        if (
            is_prefill and attn.ndim == 4
        ):  # Prefill, we may receive the full attention map and have to average across queries
            # Normalize using input_pos to only count non-zero attentions bc/ of causal mask
            attn = attn.squeeze(0).sum(dim=1) / (seq_len - input_pos)

        attn = attn.view(1, self.n_heads, -1, 1)
        attn = (attn >= 1 / self.cache_cts).int() if self.attn_thresholding else attn

        # Torch.compile doesn't support dyanmic slicing so we need to zero-pad to full dimension
        padding = max(self.max_cache_length - seq_len, 0)
        pad_attn = torch.zeros(
            1, self.n_heads, padding, 1, dtype=attn.dtype, device=attn.device
        )
        attn = torch.cat([attn, pad_attn], dim=2)

        history_idx = self.attn_counter % self.history_window_size
        self.attn_history_num[:, :, :, history_idx] = attn
        self.attn_history_denom += 1
        self.attn_counter += 1

    def _eviction_idx(self, input_pos):
        # Identify the tokens with consistently "low" attentions
        numerator = self.attn_history_num.sum(dim=-1).float()
        # The denominator is the number of times this token's history has been recorded
        # We only record most self.history_window_size recent scores so need to clamp it
        denominator = self.attn_history_denom.clamp(1, self.history_window_size)

        avg_attn = numerator / denominator

        # Save the global & most recent tokens from being evicted
        avg_attn.masked_fill_(
            torch.logical_or(
                self.pos < self.global_tokens,
                self.pos >= input_pos - self.recent_window,
            ),
            1.0,
        )

        avg_attn.masked_fill_(self.pos == -1, 0.0)

        fill_idxs = avg_attn.argmin(dim=-1).squeeze()

        # Zero-out the attention history for these newly inserted slots
        num_fill = fill_idxs.view(1, -1, 1, 1).expand(
            1, -1, 1, self.attn_history_num.shape[-1]
        )
        denom_fill = fill_idxs.view(1, -1, 1)
        self.attn_history_num.scatter_(
            2, num_fill, torch.zeros_like(num_fill, dtype=self.attn_history_num.dtype)
        )
        self.attn_history_denom.scatter_(
            2, denom_fill, torch.zeros_like(denom_fill, dtype=torch.int32)
        )

        return fill_idxs


class KVCacheHybrid(KVCacheHeavyHitter):
    # This class mostly follows the logic in FastGen (https://arxiv.org/abs/2310.01801)
    # Yet, it allows for a wider set of hybrid strategies to be considered during profiling.
    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "global_tokens",
        "token_ids",
        "min_recovery_frac",
        "hybrid_strategies",
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        self.attn_thresholding = False
        self.history_window_size = 400  # Default value for ScissorHands
        self.recent_window = (
            None  # Dummy value: Recent windows are defined per attention head
        )
        super().__init__(
            max_batch_size,
            n_heads,
            head_dim,
            dtype,
            variable_length=True,
            **kwargs,
        )

        self.requires_special = any(
            ["special" in strat["strategy"] for strat in self.hybrid_strategies]
        )
        mask_shape = (max_batch_size, n_heads, self.max_cache_length)
        if self.requires_special:
            special_ids = [torch.tensor(ids) for ids in kwargs["token_ids"]["special"]]
            self.register_buffer("special_ids", torch.nested.nested_tensor(special_ids))
            # As well as a mask showing where special ids are in the KV cache
            # We store this to avoid re-computing the mask every time and having to store all input_ids
            self.register_buffer(
                "special_mask", torch.zeros(mask_shape, dtype=torch.bool)
            )
            self.register_buffer("num_special", torch.zeros((1,), dtype=torch.int))

        self.requires_punc = any(
            ["punc" in strat["strategy"] for strat in self.hybrid_strategies]
        )
        if self.requires_punc:
            # Store the punctuation vocabulary ids
            punc_ids = torch.Tensor(kwargs["token_ids"]["punctuation"])
            self.register_buffer("punc_ids", punc_ids)
            # As well as a mask showing where punctuation ids are in the KV cache
            # We store this to avoid re-computing the mask every time and having to store input_ids
            self.register_buffer("punc_mask", torch.zeros(mask_shape, dtype=torch.bool))
            self.register_buffer("num_punc", torch.zeros((1,), dtype=torch.int))

        self.requires_heavy_hitter = any(
            ["heavy_hitter" in strat["strategy"] for strat in self.hybrid_strategies]
        )

        # We need to use a mask since not all heads have same number of tokens. We can't simply truncate.
        # 1 dimension stands for query dimension, which will always be 1 (next token) for KV cache attention.
        kv_mask_shape = (max_batch_size, n_heads, 1, self.max_cache_length)
        self.register_buffer("mask", torch.zeros(kv_mask_shape, dtype=torch.bool))

    def return_attn(self):
        return self.requires_heavy_hitter

    def _eviction_idx_for_head(
        self,
        head_idx,
        input_pos,
        recent_window,
        apply_window=False,
        apply_special=False,
        apply_punc=False,
    ):
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

        save_mask = torch.zeros_like(avg_attn, dtype=torch.bool)
        save_mask[:, : self.global_tokens] = 1

        if apply_special:
            save_mask = self.special_mask[:, head_idx, : self.cache_cts[head_idx]]

        if apply_punc:
            punc_mask = self.punc_mask[:, head_idx, : self.cache_cts[head_idx]]
            save_mask |= punc_mask

        if apply_window:
            window_mask = (
                self.pos[:, head_idx, : self.cache_cts[head_idx]]
                >= input_pos - recent_window
            )
            save_mask |= window_mask

        avg_attn.masked_fill_(save_mask, 1)
        fill_idx = avg_attn.argmin(dim=-1)

        return fill_idx

    def _select_fill_idx(self, strategy, head_idx, input_pos, is_punc: bool = False):
        def _end_idx():
            # We need to clone because self.cache_cts will be incremented later and we don't want to have fill_idx as a mutable reference
            return min(self.max_cache_length - 1, self.cache_cts[head_idx].clone())

        strategy = self.hybrid_strategies[strategy]
        name = strategy["strategy"]

        # If is punctuation token and we are preserving, we always add it to the end
        if "punc" in name and is_punc:
            return _end_idx(), False

        if name == "full":
            return _end_idx(), False

        # Every strategy has a budget for global tokens
        budget = torch.tensor(
            [self.global_tokens], dtype=torch.int, device=input_pos.device
        )
        if "special" in name:
            budget += self.num_special

        if "punc" in name:
            budget += self.num_punc

        if "window" in name:
            budget += round(strategy["recent_window"] * self.max_cache_length)

        if "heavy_hitter" in name:
            budget += round(strategy["heavy_hitter_frac"] * self.max_cache_length)

        eviction_required = self.cache_cts[head_idx] >= budget

        if not eviction_required:
            return _end_idx(), False

        if name == "window":
            fill_idx = (
                torch.argmin(
                    self.pos[:, head_idx, self.global_tokens : self.cache_cts[head_idx]]
                )
                + self.global_tokens
            )
            return fill_idx, True  # Eviction Required

        if "heavy_hitter" in name:
            recent_window = round(
                strategy.get("recent_window", 0) * self.max_cache_length
            )
            fill_idx = self._eviction_idx_for_head(
                head_idx,
                input_pos,
                recent_window=recent_window,
                apply_window="window" in name,
                apply_punc="punc" in name,
                apply_special="special" in name,
            )

            return fill_idx, True  # Eviction Required

        # If we reach here, we have a hybrid strategy that is not window, heavy hitter, or full
        assert "punc" in name or "special" in name, f"Invalid hybrid strategy {name}"
        return None, False

    def reset(self):
        super().reset()
        self.cache_strategies.fill = None  # Free up memory temporarily

        if hasattr(self, "special_mask"):
            self.special_mask.zero_()
            self.num_special.zero_()

        if hasattr(self, "punc_mask"):
            self.punc_mask.zero_()
            self.num_punc.zero_()

    def _decoding_update(self, input_pos, k_val, v_val, **kwargs):
        input_ids = kwargs.get("input_ids")
        n_heads = k_val.shape[1]

        is_punc = (
            torch.isin(input_ids, self.punc_ids) if hasattr(self, "punc_ids") else False
        )

        # If fill idx is None we place value at the back (which is truncated for attention calculation anyway)
        fill_indices = torch.full(
            (n_heads,),
            self.max_cache_length - 1,
            dtype=torch.int64,
            device=k_val.device,
        )

        cache_ct_incr = torch.zeros_like(fill_indices)

        for head_idx, strategy in enumerate(self.cache_strategies):
            fill_idx, eviction_required = self._select_fill_idx(
                strategy, head_idx, input_pos, is_punc=is_punc
            )

            if fill_idx is None:
                continue

            fill_indices[head_idx] = fill_idx
            if eviction_required:
                # Reset attention history since we've inserted a new token
                self.attn_history_num[:, head_idx, fill_idx, :].fill_(0)
                self.attn_history_denom[:, head_idx, fill_idx].fill_(0)
            else:
                # Increment cache_ct_incr for heads that have grown (no eviction)
                cache_ct_incr[head_idx] = 1
                # We can't use all fill indices to bulk assign mask because some fill_indices are dummies (self.max_cache_length - 1)
                self.mask[:, head_idx, :, fill_idx] = True

        # We have to insert all new tokens into the cache to be be able to bulk insert
        # But some aren't actually being inserted (fill_idx = self.max_cache_length - 1)
        # Thus we only flip the mask to True for the tokens that are actually being inserted (done above)
        kwargs = {"update_mask": False}
        self._fill(input_pos, k_val, v_val, fill_indices, **kwargs)

        # Only update global self.num_punc once (not once per head)
        # If a head keeps punc tokens, each head will have same number of punc tokens (no punc evictions)
        if is_punc and hasattr(self, "num_punc"):
            self.punc_mask.scatter_(
                2,
                fill_indices.view(1, -1, 1),
                is_punc.view(1, 1, 1).expand(1, n_heads, 1),
            )
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

    def compute_statistics(self, seq_len):
        stats = super().compute_statistics(seq_len)

        # Compute counts of usage for hybrid strategies
        cts = Counter(
            [
                self.hybrid_strategies[i]["strategy"]
                for i in self.cache_strategies.tolist()
            ]
        )
        stats["avg_strategy_idx"] = sum(self.cache_strategies.tolist()) / len(
            self.cache_strategies
        )
        stats.update(
            {
                strategy: cts.get(strategy, 0) / len(self.cache_strategies)
                for strategy in sorted(
                    list(set([x["strategy"] for x in self.hybrid_strategies]))
                )
            }
        )
        return stats

    def build_masks(self, cum_attn, special_mask, punc_mask, total_len):
        device = cum_attn.device
        n_heads, seq_len = cum_attn.shape
        masks = []
        for s in self.hybrid_strategies:
            strat_mask = torch.zeros(
                n_heads, seq_len, seq_len, dtype=torch.bool, device=device
            )
            # All strategies have global tokens
            strat_mask[:, :, : self.global_tokens] = True

            name = s["strategy"]
            if "special" in name:
                strat_mask |= special_mask.view(1, 1, -1).expand(
                    n_heads, seq_len, seq_len
                )

            if "punc" in name:
                strat_mask |= punc_mask.view(1, 1, -1).expand(n_heads, seq_len, seq_len)

            if "window" in name:
                assert (
                    "recent_window" in s and s["recent_window"] <= 1
                ), "Window strategy should have recent_window expressed as a fraction <= 1."
                strat_mask |= (
                    create_window_attention_mask(
                        seq_len,
                        max(1, int(s["recent_window"] * total_len)),
                        device,
                        global_tokens=self.global_tokens,
                    )
                    .unsqueeze(0)
                    .expand(n_heads, seq_len, seq_len)
                )

            if "heavy_hitter" in name:
                # Compute heavy hitters over tokens which are still masked
                avail_idxs = torch.where(~strat_mask[0, -1, :])[0]

                attn_slice = cum_attn.gather(
                    1, avail_idxs.unsqueeze(0).expand(n_heads, -1)
                )

                num_hh = math.ceil(
                    min(
                        s["heavy_hitter_frac"] * total_len,
                        len(avail_idxs),
                    )
                )

                heavy_hitters = (
                    attn_slice.topk(num_hh, dim=1, largest=True)
                    .indices.sort(dim=1)
                    .values
                )

                heavy_hitters_idx = (
                    avail_idxs.view(1, -1).expand(n_heads, -1).gather(1, heavy_hitters)
                )
                strat_mask.scatter_(
                    2,
                    heavy_hitters_idx.view(n_heads, 1, num_hh).expand(
                        n_heads, seq_len, num_hh
                    ),
                    True,
                )
            if name == "full":
                strat_mask.fill_(True)

            masks.append(strat_mask)
        return torch.stack(masks)

    def profile_attn_heads(self, input_pos, attn, **kwargs):
        input_ids = kwargs["input_ids"]
        input_ids = input_ids.squeeze(0)
        seq_len = input_ids.shape[-1]

        # Only build masks as needed
        special_mask = punc_mask = None
        if self.requires_special:
            special_mask = self.build_special_ids_mask(input_ids)
            self.num_special = special_mask.sum()

        if self.requires_punc:
            punc_mask = self.build_punc_ids_mask(input_ids)
            self.num_punc = punc_mask.sum()

        cum_attn = (
            None  # Only aggregate attention if its needed by one of the strategies
        )
        if any(["heavy_hitter" in s["strategy"] for s in self.hybrid_strategies]):
            # Average of cumulative attention probs (use input_pos to normalize)
            cum_attn = attn.squeeze(0).sum(dim=1) / (seq_len - input_pos)

        masks_for_scoring = self.build_masks(
            cum_attn, special_mask, punc_mask, total_len=seq_len
        )

        # Compute optimal strategies for each head based on prompt proportions
        attn_rep = attn.expand(masks_for_scoring.shape[0], -1, -1, -1)
        compressed_scores = (
            attn_rep.masked_fill(~masks_for_scoring, 0).sum(dim=-1).mean(dim=-1)
        )

        # For each column, return the first row which has cost >= min_recovery_frac
        cache_strategies = (
            (compressed_scores >= self.min_recovery_frac).int().argmax(dim=0)
        )

        # Base insertions on the optimal strategy across full sequence length
        assert self.max_cache_length >= seq_len
        masks_for_filling = self.build_masks(
            cum_attn, special_mask, punc_mask, total_len=self.max_cache_length
        )
        # Take the last query's mask as the initial KV-Cache fill mask
        masks_all = masks_for_filling[:, :, -1, :].transpose(1, 0)
        # Select mask based on self.cache_strategies
        mask_optimal = masks_all.gather(
            1, cache_strategies.view(-1, 1, 1).expand(-1, -1, seq_len)
        ).squeeze(1)

        return cache_strategies, special_mask, punc_mask, mask_optimal, cum_attn

    def profile_and_update(self, input_pos, k_val, v_val, attn, **kwargs):
        """
        Profile the attention heads to determine the optimal KV-cache allocation.
        """
        input_ids = kwargs["input_ids"]
        input_ids = input_ids.squeeze(0)
        seq_len = input_ids.shape[-1]
        n_heads = attn.shape[1]
        dim = k_val.shape[-1]

        # Profile cache attention heads to define strategy for each head
        self.cache_strategies, special_mask, punc_mask, mask_optimal, cum_attn = (
            self.profile_attn_heads(input_pos, attn, **kwargs)
        )

        # Uncomment to show which strategies are selected
        # print([self.hybrid_strategies[i] for i in self.cache_strategies.tolist()])

        # If none of the heads selected a heavy hitter strategy, we don't need to track attention weights
        # Same for punctuation and special tokens
        self.requires_heavy_hitter = any(
            [
                "heavy_hitter" in self.hybrid_strategies[i]["strategy"]
                for i in self.cache_strategies
            ]
        )
        self.requires_punc = any(
            [
                "punc" in self.hybrid_strategies[i]["strategy"]
                for i in self.cache_strategies
            ]
        )
        self.requires_special = any(
            [
                "special" in self.hybrid_strategies[i]["strategy"]
                for i in self.cache_strategies
            ]
        )

        # Put the selected items (true values from mask) to the front. Re-arrange attentions as well.
        order = mask_optimal.int().argsort(dim=1, descending=True)
        order_exp = order.view(1, n_heads, seq_len, 1).expand(-1, -1, -1, dim)

        # We dump all the KV pairs into the cache yet order them based on the optimal strategy
        k_val = k_val.gather(2, order_exp)
        v_val = v_val.gather(2, order_exp)
        input_pos = input_pos.unsqueeze(0).expand(n_heads, -1).gather(1, order).int()
        fill_idxs = torch.arange(seq_len, device=input_pos.device)
        self._fill_contiguous(input_pos, k_val, v_val, fill_idxs)

        # Record number of tokens to be inserted into the cache
        self.cache_cts = mask_optimal.sum(dim=1)

        # Can remove for speed: doesn't change code but makes it easier to debug and see what's actually in the cache
        for head_idx in range(n_heads):
            self.pos[:, head_idx, self.cache_cts[head_idx] :].fill_(-1)
            self.k_cache[:, head_idx, self.cache_cts[head_idx] :].fill_(0)
            self.v_cache[:, head_idx, self.cache_cts[head_idx] :].fill_(0)

        if hasattr(self, "special_mask"):
            # We will need to remove special tokens and punctuation from heavy hitter eviction so need to their positions.
            special_mask = special_mask.view(1, -1).expand(n_heads, -1).gather(1, order)
            self.special_mask[:, :, :seq_len] = special_mask

        if hasattr(self, "punc_mask"):
            punc_mask = punc_mask.view(1, -1).expand(n_heads, -1).gather(1, order)
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

        if self.requires_heavy_hitter:
            # Update attention mask to indicate which we attentions are allowed.
            cum_attn = cum_attn.gather(1, order).unsqueeze(0)
            super().update_state(
                input_pos, k_val, v_val, is_prefill=True, attn=cum_attn, **kwargs
            )

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        """
        Insert the most recent attention into the history buffer.

        If self.attn_thresholding = True, insert a binary indicator of whether the attention >= uniform attention.
        """
        # We handle state updating during prefill for Hybrid as part of the profile and update stage
        if is_prefill:
            self.profile_and_update(input_pos, k_val, v_val, attn, **kwargs)
        elif (
            self.requires_heavy_hitter
        ):  # If none of the heads require attention, there's no state to update
            super().update_state(input_pos, k_val, v_val, is_prefill, attn, **kwargs)
        else:
            assert attn is None, "Attn should be None if no attention is required."


class KVCacheAnalysis(KVCacheFull):
    """
    This cache is triggered by prepending `debug_` to an existing cache strategy.

    It will analyze the attention loss incurred from compressing with that cache strategy.
    """

    relevant_kwargs = [
        "max_cache_length",
        "max_seq_length",
        "history_window_size",
        "recent_window",
        "attn_thresholding",
        "global_tokens",
        "prompt_compression_strategy",
    ]

    def __init__(
        self,
        max_batch_size,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
        cache_strategy="heavy_hitter",
        **kwargs,
    ):
        # Never any prompt compression for full cache
        full_kwargs = {
            "global_tokens": 0,  # Every token gets saved (no explicit global tokens)
            "max_cache_length": kwargs["max_seq_length"],
            "prompt_compression_strategy": kwargs["prompt_compression_strategy"],
        }
        super().__init__(max_batch_size, n_heads, head_dim, dtype, **full_kwargs)

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

        self.register_buffer(
            "attention_loss_ctr",
            torch.zeros((1,), dtype=torch.int),
        )

        self.prompt_compressor = get_prompt_compressor_constructor(
            self.prompt_compression_strategy
        )(head_specific=self.compressed.head_specific, **kwargs)

        # Necessary for compatability check with prompt compression strategy
        self.head_specific = self.compressed.head_specific

    def return_attn(self):
        return self.compressed.return_attn()

    def update_kv(self, input_pos, k_val, v_val, is_prefill, **kwargs):
        k, v, mask = super().update_kv(input_pos, k_val, v_val, is_prefill, **kwargs)
        # Conditionally update the compressed cache if prompt < max_cache_length

        # If prompt is too long for compressed cache we will need to compress it in update_state before inserting.
        # We need to wait for update_state because we might need attention for compression
        can_update_compressed = input_pos.shape[-1] < self.compressed.max_cache_length
        if can_update_compressed:
            _, _, _ = self.compressed.update_kv(
                input_pos, k_val, v_val, is_prefill, **kwargs
            )

        return k, v, mask

    def reset(self):
        super().reset()
        self.compressed.reset()
        self.attention_losses.fill_(-1)
        self.attention_loss_ctr.zero_()

    def update_state(self, input_pos, k_val, v_val, is_prefill, attn, **kwargs):
        # We might need to compress the prompt and update the compressed cache
        needs_prompt_compression = (
            is_prefill and input_pos.shape[-1] > self.compressed.max_cache_length
        )
        if needs_prompt_compression:
            kwargs = {"attn": attn}
            input_pos, k_val, v_val, attn = self.prompt_compressor(
                input_pos, k_val, v_val, **kwargs
            )
            _, _, _ = self.compressed.update_kv(input_pos, k_val, v_val, is_prefill)
            self.compressed.update_state(input_pos, k_val, v_val, is_prefill, attn)
        elif is_prefill:
            # Don't record attention loss in prefill since compressed and non-compressed prefill attentions are the same
            # Just update the state for the compressed cache and return
            self.compressed.update_state(input_pos, k_val, v_val, is_prefill, attn)
        else:
            assert not is_prefill
            indices = self.compressed.pos.clone().long()
            # Avoid scatter issue we need to assign unfilled indices to last attention value (which will also be 0)
            indices[indices == -1] = attn.shape[-1] - 1
            attn_compressed = attn.squeeze(2).gather(2, indices)
            self.compressed.update_state(
                input_pos, k_val, v_val, is_prefill, attn_compressed
            )

            # Compute attention loss as the sum of the attention probabilities for evicted tokens
            # Equivalently, 1 - the sum of the attention probabilities for the tokens in the compressed cache
            attn_loss = (1 - attn_compressed.sum(dim=-1)).mean()
            self.attention_losses[self.attention_loss_ctr] = attn_loss
            self.attention_loss_ctr += 1

    def compute_statistics(self, seq_len):
        """
        Computes statistics about the cache.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The cache size, the number of tokens inserted, and the compression ratio.
        """
        stats = super().compute_statistics(seq_len)
        losses = self.attention_losses[: self.attention_loss_ctr]
        assert not torch.any(losses == -1)
        for k in range(500, len(losses), 500):
            stats[f"attention_loss@{k}"] = losses[:k].mean().item()
        stats["attention_loss"] = losses.mean().item()
        return stats


def get_cache_constructor(cache_strategy):
    relevant_kwargs = None
    if cache_strategy == "full":
        cls = KVCacheFull
    elif cache_strategy == "l2":
        cls = KVCacheL2
    elif cache_strategy == "random":
        cls = KVCacheRandom
    elif cache_strategy == "recent_global":
        cls = KVCacheRecentGlobal
    elif cache_strategy == "heavy_hitter":
        cls = KVCacheHeavyHitter
    elif cache_strategy == "hybrid":
        cls = KVCacheHybrid
    elif cache_strategy.startswith("debug"):
        cache_strategy = re.sub(r"debug_+", "", cache_strategy).strip()
        relevant_kwargs = get_cache_constructor(cache_strategy)[1] + [
            "prompt_compression_strategy"
        ]
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
