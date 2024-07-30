import torch
from abc import ABC, abstractmethod
from typing import Tuple


class PromptCompressor(ABC):
    def __init__(self, head_specific, **kwargs) -> None:
        # Assign each kwarg as an attribute of the class
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.head_specific = head_specific
        assert self.is_compatible(), f"Prompt compressor ({self.__class__.__name__}) is not compatible with the chosen cache strategy."

    @abstractmethod
    def requires_attn(self) -> bool:
        pass

    @abstractmethod
    def __call__(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def is_compatible(self) -> bool:
        pass


class PromptCompressorFull(PromptCompressor):
    """
    This is a dummy (pass through) method which returns its inputs
    """

    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        # Can be used with any cache
        return True

    def requires_attn(self) -> bool:
        return False

    def __call__(self, input_pos, k_val, v_val, **kwargs):
        return input_pos, k_val, v_val, None


class PromptCompressorRandom(PromptCompressor):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        # Can be used with any cache
        return True

    def requires_attn(self) -> bool:
        return False

    def __call__(self, input_pos, k_val, v_val, **kwargs):
        seq_len = input_pos.shape[0]
        global_idxs = torch.arange(self.global_tokens, device=input_pos.device)
        full_middle_n = seq_len - self.global_tokens - self.recent_window
        filt_middle_n = self.max_cache_length - self.global_tokens - self.recent_window
        rand_middle_idxs = (
            (
                self.global_tokens
                + torch.randperm(full_middle_n, device=input_pos.device)[:filt_middle_n]
            )
            .sort()
            .values
        )
        recent_idxs = torch.arange(
            seq_len - self.recent_window, seq_len, device=input_pos.device
        )
        keep_idxs = torch.cat([global_idxs, rand_middle_idxs, recent_idxs], dim=0)
        assert len(keep_idxs) == self.max_cache_length
        k_val = k_val[:, :, keep_idxs]
        v_val = v_val[:, :, keep_idxs]
        return keep_idxs, k_val, v_val, None


class PromptCompressorRecentGlobal(PromptCompressor):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        # Can be used with any cache
        return True

    def requires_attn(self) -> bool:
        return False

    def __call__(self, input_pos, k_val, v_val, **kwargs):
        # [global; ...; window - global] --> [global; window - global]
        # Indices for first global_tokens tokens and last (window - global_tokens) tokens
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
        k_val = k_val[:, :, keep_idxs]
        v_val = v_val[:, :, keep_idxs]
        return keep_idxs, k_val, v_val, None


class PromptCompressorHeavyHitter(PromptCompressor):
    """
    Use SnapKV to compress the prompt
    Based on the pseudo code on Page 7 of https://arxiv.org/abs/2404.14469
    """

    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

        self.kernel_size = 5
        self.observation_len = 16

        # Pooling layer to smooth out the attention distribution
        # Feel free to remove this or optimize the kernel size
        self.pool = torch.nn.AvgPool1d(
            self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            ceil_mode=False,
            count_include_pad=False,
        )

    def is_compatible(self) -> bool:
        # Can only be used with head-specific KV-caches
        return self.head_specific

    def requires_attn(self) -> bool:
        return True

    def __call__(self, input_pos, k_val, v_val, **kwargs):
        attn = kwargs.pop("attn")

        seq_len = input_pos.shape[0]
        obs_len = min(self.observation_len, seq_len)

        priority = attn[:, :, -obs_len:, :].mean(dim=2)
        prev_shape = priority.shape

        # We'll be returning the attention history so we need to keep a copy before it's modified
        priority = self.pool(priority)
        assert (
            priority.shape == prev_shape
        ), f"Pooling operation should not change the dimension: {prev_shape} -> {priority.shape}"
        priority[:, :, -obs_len:] = 1.0  # Ensure the observation window is selected
        priority[:, :, : self.global_tokens] = (
            1.0  # Ensure the global tokens are selected
        )
        keep_idxs = (
            priority.topk(self.max_cache_length, dim=-1).indices.sort(dim=-1).values
        )

        # Return average attention across prompt to insert into KV Cache's attention history tracker
        cum_attn = attn.sum(dim=2) / (seq_len - input_pos)
        cum_attn = cum_attn.gather(2, keep_idxs)

        keep_idxs_rep = keep_idxs.unsqueeze(-1).expand(-1, -1, -1, k_val.shape[-1])
        k_val_compressed = k_val.gather(2, keep_idxs_rep)
        v_val_compressed = v_val.gather(2, keep_idxs_rep)

        return keep_idxs.squeeze(0), k_val_compressed, v_val_compressed, cum_attn


class PromptCompressorL2(PromptCompressor):
    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        return self.head_specific

    def requires_attn(self) -> bool:
        return False

    def __call__(self, input_pos, k_val, v_val, **kwargs):
        key_norm = torch.linalg.vector_norm(k_val, ord=2, dim=-1)

        # Give low score to global and recent tokens
        locality_mask = torch.logical_or(
            input_pos < self.global_tokens,
            input_pos >= input_pos.shape[0] - self.recent_window,
        ).view(1, 1, -1)

        eviction_scores = key_norm.masked_fill(locality_mask, float("-inf"))

        keep_idxs = (
            eviction_scores.topk(self.max_cache_length, dim=-1, largest=False)
            .indices.sort(dim=-1)
            .values
        )

        keep_idxs_rep = keep_idxs.unsqueeze(-1).expand(-1, -1, -1, k_val.shape[-1])
        k_val_compressed = k_val.gather(2, keep_idxs_rep)
        v_val_compressed = v_val.gather(2, keep_idxs_rep)

        return keep_idxs, k_val_compressed, v_val_compressed, None


class PromptCompressorKeepItOdd(PromptCompressor):
    """
    A toy example of a prompt compressor that keeps the odd indices of the prompt
    """

    def __init__(self, head_specific, **kwargs) -> None:
        super().__init__(head_specific, **kwargs)

    def is_compatible(self) -> bool:
        return True

    def requires_attn(self) -> bool:
        return False

    def __call__(self, input_pos, k_val, v_val, **kwargs):
        # Compute odd indices from keep_idxs to input_pos.shape[0] - window
        odd_idxs = [
            i
            for i in range(self.global_tokens, input_pos.shape[0] - self.recent_window)
            if i % 2 == 1
        ]
        keep_n = max(0, self.max_cache_length - self.global_tokens - self.recent_window)
        odd_idxs = odd_idxs[-keep_n:]

        keep_idxs = torch.tensor(
            list(range(self.global_tokens))
            + odd_idxs
            + list(
                range(
                    input_pos.shape[0] - self.recent_window,
                    input_pos.shape[0],
                )
            ),
            dtype=torch.long,
            device=k_val.device,
        )
        k_val = k_val[:, :, keep_idxs]
        v_val = v_val[:, :, keep_idxs]
        return keep_idxs, k_val, v_val, None


def get_prompt_compressor_constructor(strategy):
    if strategy == "full":
        return PromptCompressorFull
    if strategy == "recent_global":
        return PromptCompressorRecentGlobal
    elif strategy == "heavy_hitter":
        return PromptCompressorHeavyHitter
    elif strategy == "l2":
        return PromptCompressorL2
    elif strategy == "random":
        return PromptCompressorRandom
    elif strategy == "keep_it_odd":
        return PromptCompressorKeepItOdd
    else:
        raise ValueError(f"Unknown prompt compression strategy: {strategy}")
