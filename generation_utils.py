import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config

from model import Transformer, find_multiple


default_device = "cuda" if torch.cuda.is_available() else "cpu"


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    causal_mask = (
        torch.tril(torch.ones(len(input_pos), len(input_pos), dtype=torch.bool))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(x.device)
    )
    logits = model(x, input_pos, mask=causal_mask)
    return sample(logits, **sampling_kwargs)


def decode_one_token(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    terminator_ids: Optional[list] = None,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )

            if terminator_ids and next_token in terminator_ids:
                break

            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs,
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor(
        [input_pos], dtype=torch.int64, device=cur_token.device
    )
    draft_tokens, draft_probs = decode_n_tokens(
        draft_model,
        cur_token.view(1, -1),
        orig_input_pos.clone(),
        speculate_k,
        **sampling_kwargs,
    )

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device),
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
    rejected_locations = (
        torch.rand_like(accept_draft_prob) > accept_draft_prob
    ).nonzero()

    if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])


def normalize_cache_length(
    max_cache_length: float, max_seq_length: int, multiple_of: int = 8
) -> int:
    """
    Computes the absolute cache length given the max_cache_length and max_seq_length.
    """
    if 0 < max_cache_length <= 1:
        max_cache_length = round(max_seq_length * max_cache_length)
    else:
        assert int(max_cache_length) == max_cache_length
        max_cache_length = int(max_cache_length)
        if max_cache_length > max_seq_length:
            print(
                f"Warning: max_cache_length ({max_cache_length}) is greater than max_seq_length ({max_seq_length}). Setting to {max_seq_length}"
            )
            max_cache_length = max_seq_length
    return min(find_multiple(max_cache_length, multiple_of), max_seq_length)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback=lambda x: x,
    terminator_ids: Optional[list] = None,
    cache_kwargs: dict = None,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    max_seq_length = min(T + max_new_tokens, model.config.block_size)
    if interactive:
        max_seq_length = 350
    print(f"Maximum context length of {max_seq_length} tokens.")

    max_new_tokens = max_seq_length - T

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = (
        max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    )

    # Normalize max_cache_length to absolute cache length if provided as a fraction of the max seq sequence length
    cache_kwargs["max_cache_length"] = list(
        map(
            lambda l: normalize_cache_length(l, max_seq_length),
            cache_kwargs["max_cache_length"],
        )
    )
    assert (
        model.config.n_layer % len(cache_kwargs["max_cache_length"]) == 0
    ), f'max_cache_length ({len(cache_kwargs["max_cache_length"])}) must be a factor of {model.config.n_layer} layers.'

    tile_size = model.config.n_layer // len(cache_kwargs["max_cache_length"])
    cache_kwargs["max_cache_length"] = [
        item for item in cache_kwargs["max_cache_length"] for _ in range(tile_size)
    ]

    # Gets called twice when model is wrapped in torch.compile which causes an error without the if statement
    if type(cache_kwargs["drop_amount"]) != list:
        cache_kwargs["drop_amount"] = [
            max(int(cache_kwargs["drop_amount"] * l), 1)
            for l in cache_kwargs["max_cache_length"]
        ]

    assert cache_kwargs["global_tokens"] <= min(
        cache_kwargs["max_cache_length"]
    ), "Global tokens must be less than max_cache_length."

    with torch.device(device):
        model.setup_caches(max_batch_size=1, **cache_kwargs)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=1, **cache_kwargs)

    # create an empty tensor (all -1) of the expected final shape and fill in the current tokens
    # GPT-Fast had this as empty but the values of empty are non-deterministic
    seq = torch.full((max_seq_length,), -1, dtype=dtype, device=device)
    seq[:T] = prompt
    input_pos = torch.arange(0, T, device=device)

    ret = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    next_token = ret[0].clone()
    next_tok_probs = ret[1].clone()
    if is_speculative:
        prefill(draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < max_seq_length - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(max_seq_length - input_pos - 1, len(next_tokens))
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[:num_added]
            for i in next_tokens[:num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, generated_tok_probs = decode_n_tokens(
            model,
            next_token.view(1, -1),
            input_pos,
            max_new_tokens - 1,
            callback=callback,
            terminator_ids=terminator_ids,
            **sampling_kwargs,
        )
        if len(generated_tokens) > 0:
            seq[T + 1 : T + 1 + len(generated_tokens)] = torch.cat(generated_tokens)

    # Truncate seq to first instance of -1 if -1 is present
    if -1 in seq:
        seq = seq[: torch.where(seq == -1)[0][0]]

    generate_stats = {"accept_counts": accept_counts}
    return seq, generate_stats, [next_tok_probs] + generated_tok_probs


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = "cuda" in device
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)
    if use_tp:
        from tp import apply_tp

        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()
