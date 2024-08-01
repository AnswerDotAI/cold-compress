import itertools
import time
from typing import Optional, Tuple
from pathlib import Path

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention import SDPBackend, sdpa_kernel

import argparse
import yaml
from model import Transformer, find_multiple
from tokenizer import TokenizerInterface

default_device = "cuda" if torch.cuda.is_available() else "cpu"


def snake_to_capitalized(s):
    return " ".join(word.capitalize() for word in s.split("_"))


def print_stats(stats_dict):
    # Separate the stats into layered and non-layered
    layered_stats = {}
    non_layered_stats = {}

    for key, value in stats_dict.items():
        parts = key.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            stat = snake_to_capitalized(parts[0])
            layer = int(parts[1])
            if stat not in layered_stats:
                layered_stats[stat] = []
            layered_stats[stat].append((layer, value))
        else:
            non_layered_stats[snake_to_capitalized(key)] = value

    # Print non-layered stats
    for key, value in non_layered_stats.items():
        print(f"{key}: {value:.02f}")

    # Print layered stats
    for stat in sorted(layered_stats.keys()):
        layers_list = sorted(layered_stats[stat])
        layers_str = ", ".join(f"{layer}={value:.02f}" for layer, value in layers_list)
        print(f"{stat} By Layer: {layers_str}")


def add_generation_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("generation_args")
    # Generation hparams
    group.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/model.pth",
        help="Model checkpoint path.",
    )

    group.add_argument("--profile", type=Path, default=None, help="Profile path.")

    group.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )

    group.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )

    group.add_argument(
        "--attn_top_k",
        type=float,
        default=1.0,
        help="Fraction of top-K attentions over which to compute values. 1.0 means all V are used regardless of attention weight (QK).",
    )


def merge_cache_config(args):
    if not args.cache_config:
        return args
    # Get parent directory of current file
    if not args.cache_config.endswith(".yaml"):
        args.cache_config = args.cache_config + ".yaml"
    yaml_fn = Path(__file__).parent / "cache_configs" / args.cache_config
    assert yaml_fn.exists(), f"Cache config file {yaml_fn} does not exist."
    with open(yaml_fn, "r") as f:
        cache_kwargs = yaml.safe_load(f)
        # Over-write args with cache_kwargs
        args = argparse.Namespace(**{**vars(args), **cache_kwargs})
    return args


def compute_max_seq_length(
    model, prompt_lens: list[int], target_lens: list[int], max_new_tokens: int
) -> int:
    max_prompt_length = max(len(prompt_lens[i]) for i in range(len(prompt_lens)))
    # Should either pass target_lens or max_new_tokens
    max_target_lens = (
        0
        if target_lens is None
        else max(len(target_lens[i]) for i in range(len(target_lens)))
    )
    max_new_tokens = max(max_new_tokens, max_target_lens)
    max_seq_length = max_prompt_length + max_new_tokens
    if max_seq_length > model.config.block_size:
        print(
            f"Warning: The longest prompt puts the desired max_seq_length at {max_seq_length}, which is greater than models max of {model.config.block_size}."
        )
        print(f"Setting to model's max_seq_length of {model.config.block_size}.")
        max_seq_length = model.config.block_size
    print(f"Maximum context length of {max_seq_length} tokens.")
    return max_prompt_length, max_seq_length


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def greedy(logits, next_token):
    probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
    if next_token is None:
        idx_next = torch.argmax(probs, keepdim=True).to(dtype=torch.int)
    else:
        idx_next = next_token
    return idx_next, probs


def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    next_token: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    causal_mask = (
        torch.tril(torch.ones(len(input_pos), len(input_pos), dtype=torch.bool))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(x.device)
    )
    logits = model(x, input_pos, mask=causal_mask, is_prefill=True)
    return greedy(logits, next_token)


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    next_token: torch.Tensor = None,
    attn_top_k: float = 1,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    logits = model(
        x,
        input_pos,
        is_prefill=False,
        attn_top_k=attn_top_k,
    )
    return greedy(logits, next_token=next_token)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    decode_one_token: callable,
    num_new_tokens: int,
    terminator_ids: Optional[list] = None,
    attn_top_k: float = 1,
    prefix: Optional[torch.Tensor] = None,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with sdpa_kernel(
            [SDPBackend.MATH]
        ):  # Actually better for Inductor to codegen attention here
            teacher_force = prefix is not None and i < len(prefix)
            next_token = prefix[i].view(1) if teacher_force else None
            next_token, next_prob = decode_one_token(
                model,
                cur_token,
                input_pos,
                next_token=next_token,
                attn_top_k=attn_top_k,
                **sampling_kwargs,
            )

            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())

            if terminator_ids and next_token in terminator_ids and not teacher_force:
                break

            input_pos += 1
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


def apply_pattern(
    pattern: list[str | int],
    out_size: int,
    extension_strategy: str = "tile",
    max_seq_length: int = None,
):
    """
    Extend a given pattern across n_layers of the model.
    """
    assert extension_strategy in {
        "tile",
        "repeat",
        "pyramid",
        "funnel",
    }, "extension_strategy must be one of 'tile', 'repeat', 'pyramid', or 'funnel'."
    assert (
        out_size % len(pattern) == 0
    ), f"{len(pattern)} must be a divisible factor of the number of layers ({out_size})."
    factor = out_size // len(pattern)

    if extension_strategy in {"funnel", "pyramid"}:
        assert (
            len(pattern) == 1
        ), "Funnel and pyramid patterns must have a single element."
        return apply_pyramid_pattern(
            pattern[0],
            max_seq_length,
            out_size,
            decreasing=extension_strategy == "pyramid",
        )
    elif extension_strategy == "tile":
        return [item for item in pattern for _ in range(factor)]
    else:  # Repeat
        return pattern * factor


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
                f"FYI: max_cache_length ({max_cache_length}) is greater than max_seq_length ({max_seq_length}). Setting to {max_seq_length}"
            )
            max_cache_length = max_seq_length
    return min(find_multiple(max_cache_length, multiple_of), max_seq_length)


def apply_pyramid_pattern(
    max_cache_length: int,
    max_seq_length: int,
    model_n_layer: int,
    decreasing: bool = True,
    min_cache_length: int = 256,
):
    # Implements https://arxiv.org/abs/2406.02069
    # Paper finds best beta of 14
    beta = 14
    min_allowable = min(min_cache_length, max_cache_length)
    total_len = max_cache_length * model_n_layer
    min_cache_length = total_len / (model_n_layer * beta)
    max_cache_length = 2 * total_len / model_n_layer
    diff = (max_cache_length - min_cache_length) / model_n_layer
    cache_lens = [min_cache_length]
    for l in range(1, model_n_layer - 1):
        cache_lens.append(min_cache_length + diff * l)
    cache_lens.append(max_cache_length)
    cache_lens = [normalize_cache_length(int(l), max_seq_length) for l in cache_lens]

    overflow = 0
    num_overflow = 0
    for i in range(len(cache_lens)):
        if cache_lens[i] < min_allowable:
            overflow += min_allowable - cache_lens[i]
            cache_lens[i] = min_allowable
            num_overflow += 1

    if num_overflow < len(cache_lens):
        decr_amount = overflow // (len(cache_lens) - num_overflow)
        for i in range(len(cache_lens)):
            if cache_lens[i] > min_allowable:
                # This will change the overall cache length slightly if min_allowable threshold is hit but should be very minor
                cache_lens[i] = max(min_allowable, cache_lens[i] - decr_amount)

    if decreasing:
        cache_lens = cache_lens[::-1]
        assert cache_lens[-1] < cache_lens[0], "Cache lengths should be decreasing."
    else:
        assert cache_lens[0] < cache_lens[-1], "Cache lengths should be increasing."

    return cache_lens


def setup_caches(
    model: Transformer,
    tokenizer: TokenizerInterface,
    device: torch.device,
    max_seq_length: int,
    cache_kwargs: dict = None,
) -> dict:
    # Normalize max_cache_length to absolute cache length if provided as a fraction of the max seq sequence length
    cache_kwargs["max_seq_length"] = max_seq_length
    cache_kwargs["max_cache_length"] = list(
        map(
            lambda l: normalize_cache_length(l, max_seq_length),
            cache_kwargs["max_cache_length"],
        )
    )

    cache_kwargs["max_cache_length"] = apply_pattern(
        pattern=cache_kwargs["max_cache_length"],
        out_size=model.config.n_layer,
        extension_strategy=cache_kwargs["cache_length_pattern"],
        max_seq_length=max_seq_length,
    )

    assert len(cache_kwargs["cache_strategy"]) == len(
        cache_kwargs["prompt_compression_strategy"]
    ), "You must specify a prompt_compression_strategy for each cache_strategy."

    cache_kwargs["cache_strategy"] = apply_pattern(
        pattern=cache_kwargs["cache_strategy"],
        out_size=model.config.n_layer,
        extension_strategy=cache_kwargs["cache_strategy_pattern"],
    )
    cache_kwargs["prompt_compression_strategy"] = apply_pattern(
        pattern=cache_kwargs["prompt_compression_strategy"],
        out_size=model.config.n_layer,
        extension_strategy=cache_kwargs["cache_strategy_pattern"],
    )

    if type(cache_kwargs["recent_window"]) != list:
        if cache_kwargs["recent_window"] <= 1:
            cache_kwargs["recent_window"] = [
                max(1, int(cache_kwargs["recent_window"] * l))
                for l in cache_kwargs["max_cache_length"]
            ]
        else:
            cache_kwargs["recent_window"] = [
                max(1, min(cache_kwargs["recent_window"], l))
                for l in cache_kwargs["max_cache_length"]
            ]

    assert cache_kwargs["global_tokens"] <= min(
        cache_kwargs["max_cache_length"]
    ), "Global tokens must be less than max_cache_length."

    if cache_kwargs["cache_strategy"][0] == "hybrid":
        # We need to pass the special and punctuation token ids to the cache via cache_kwargs
        cache_kwargs["token_ids"] = {
            "special": tokenizer.special_ids(),
            "punctuation": tokenizer.punctuation_ids(),
        }

    with torch.device(device):
        model.setup_caches(max_batch_size=1, **cache_kwargs)

    return cache_kwargs


def reset_caches(model: Transformer):
    model.reset_caches()


def get_cache_stats(model: Transformer, prompt_len: int, gen_len: int):
    return model.get_cache_stats(prompt_len, gen_len)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    prefill: callable,
    decode_one_token: callable,
    max_new_tokens: int,
    next_tokens: Optional[torch.Tensor] = None,
    terminator_ids: Optional[list] = None,
    feed_long_prompts: bool = False,
    decode_first_token: bool = False,
    attn_top_k: float = 1,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    prompt_length = prompt.size(0)

    device, dtype = prompt.device, prompt.dtype

    min_cache_length = model.min_cache_length()
    # Subtract 1 in case we need one generation step over which to compute attention, etc.
    max_prompt_len = min_cache_length - 1
    prefix = None
    # If we asked to have prompt truncated and fed, we need to do split prompt into prompt and prefix
    # We also define a rare yet important edge case: if |prompt| is exactly cache length
    # We might have to start evictions before having had a change to record any state (attentions).
    # In this scenario let's decrement prompt by 1 and start "generating" on the prefix
    if (
        feed_long_prompts and prompt_length > max_prompt_len
    ) or prompt_length == min_cache_length:
        prompt, prefix = prompt[:max_prompt_len], prompt[max_prompt_len:]
        max_new_tokens += len(prefix)
        prompt_length = max_prompt_len

    if decode_first_token:
        prompt, prefix = prompt[:-1], prompt[-1:]
        max_new_tokens += 1
        prompt_length -= 1

    # create an empty tensor (all -1) of the expected final shape and fill in the current tokens
    # GPT-Fast had this as empty but the values of empty are non-deterministic
    seq = torch.full((prompt_length + max_new_tokens,), -1, dtype=dtype, device=device)
    seq[:prompt_length] = prompt
    input_pos = torch.arange(0, prompt_length, device=device)

    if next_tokens is not None:  # We are in teacher forcing mode for Perplexity task
        max_new_tokens = len(next_tokens)
        next_token = next_tokens[0].view(1)
        prefix = next_tokens[1:]
    elif prefix is not None:  # We are in partial teacher forcing due to a long prompt
        next_token = prefix[0].view(1)
        prefix = prefix[1:]
    else:
        next_token = prefix = None  # We are in normal generation mode

    # create an empty tensor (all -1) of the expected final shape and fill in the current tokens
    # GPT-Fast had this as empty but the values of empty are non-deterministic
    seq = torch.full((prompt_length + max_new_tokens,), -1, dtype=dtype, device=device)
    seq[:prompt_length] = prompt
    input_pos = torch.arange(0, prompt_length, device=device)

    t0 = time.perf_counter()

    ret = prefill(
        model,
        prompt.view(1, -1),
        input_pos,
        next_token=next_token,
        **sampling_kwargs,
    )

    t1 = time.perf_counter()

    prefill_seconds = t1 - t0

    next_token = ret[0].clone()
    next_tok_probs = ret[1].clone()
    seq[prompt_length] = next_token

    input_pos = torch.tensor([prompt_length], device=device, dtype=torch.int)
    generated_tokens, generated_tok_probs = decode_n_tokens(
        model,
        next_token.view(1, -1),
        input_pos,
        decode_one_token,
        max_new_tokens - 1,
        terminator_ids=terminator_ids,
        prefix=prefix,
        attn_top_k=attn_top_k,
        **sampling_kwargs,
    )

    t2 = time.perf_counter()
    decode_seconds = t2 - t1

    total_seconds = t2 - t0

    prefill_tokens = prompt_length
    decode_tokens = (
        len(generated_tokens) + 1
    )  # +1 because we generate 1 token from prefill

    decode_toks_per_sec = decode_tokens / decode_seconds
    prefill_toks_per_sec = prefill_tokens / prefill_seconds
    total_toks_per_sec = decode_tokens / total_seconds

    perf_stats = {
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "prefill_toks_per_sec": prefill_toks_per_sec,
        "decode_toks_per_sec": decode_toks_per_sec,
        "total_toks_per_sec": total_toks_per_sec,
        "total_seconds": total_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_seconds_frac_of_total": decode_seconds / total_seconds,
        "memory_used_gb": torch.cuda.max_memory_reserved() / 1e9,
    }

    if len(generated_tokens) > 0:
        seq[prompt_length + 1 : prompt_length + 1 + len(generated_tokens)] = torch.cat(
            generated_tokens
        )

    # Truncate seq to first instance of -1 if -1 is present
    if -1 in seq:
        seq = seq[: torch.where(seq == -1)[0][0]]

    return seq, [next_tok_probs] + generated_tok_probs, perf_stats


def load_model(checkpoint_path, device, precision, use_tp):
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


def get_model_size(model):
    model_size = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            for p in itertools.chain(child.parameters(), child.buffers()):
                model_size += p.numel() * p.dtype.itemsize
    return model_size


def compile_funcs(compile=True):
    if compile:
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token,
            fullgraph=True,
            # dynamic=True,
            mode="reduce-overhead",
            # options={"trace.graph_diagram": True, "trace.enabled": True}
        )
        prefill = torch.compile(
            prefill,
            fullgraph=True,
            dynamic=True,
            # options={"trace.graph_diagram": True, "trace.enabled": True}
        )
    return prefill, decode_one_token
