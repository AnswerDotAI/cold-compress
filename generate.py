# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from tokenizer import get_tokenizer
from generation_utils import generate, encode_tokens, _load_model


def _get_model_size(model):
    model_size = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size


B_INST, E_INST = "[INST]", "[/INST]"


def main(
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    device=default_device,
    cache_kwargs: dict = {},
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        # If there's no tokenizer.model, try to load the tokenizer from the parent directory
        # NOTE: We assume the tokenizer in the parent directory is compatible with huggingface transformers
        tokenizer_path = checkpoint_path.parent

    global print
    from tp import maybe_init_dist

    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = (
        "chat" in str(checkpoint_path).lower()
        or "instruct" in str(checkpoint_path).lower()
    )

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path, is_chat=is_chat)

    if is_chat:
        tokens = tokenizer.encode_prompt(prompt)
        encoded = torch.tensor(tokens, dtype=torch.int, device=device)
    else:
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    prompt_length = encoded.size(0)

    if args.no_terminators:
        terminator_ids = None
    else:
        terminator_ids = tokenizer.get_terminator_ids()

    torch.manual_seed(1234)
    model_size = _get_model_size(model)
    print(f"{model_size / 1e9:.02f} billion parameters in model.")

    if compile:
        if is_speculative and use_tp:  # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = (
                False  # Bug with cudagraph trees in this case
            )

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(
                model_forward, mode="reduce-overhead", fullgraph=True
            )

        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        "tokens_per_sec": [],
        "accept_counts": [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(device=device)  # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                tokens = tokenizer.encode_prompt(prompt)
                encoded = torch.tensor(tokens, dtype=torch.int, device=device)
            else:
                encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode(".")[0]
            done_generating = False

            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x: x
        t0 = time.perf_counter()
        import contextlib

        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics, _ = generate(
                model,
                encoded,
                max_new_tokens,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
                terminator_ids=terminator_ids,
                cache_kwargs=cache_kwargs,
            )
            aggregate_metrics["accept_counts"].append(metrics["accept_counts"])
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device)  # MKG
        t = time.perf_counter() - t0

        if not interactive:
            print(tokenizer.decode(y.tolist()))
        else:
            print()
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        print(f"Tokens generated: {tokens_generated}")
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    print("==========")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics["accept_counts"])]
        acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(
            f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}"
        )

    print(
        f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
    )
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--prompt",
        type=str,
        default="long_prompt_short_output.txt",
        help="Input prompt. If it ends in .txt, we will load the prompt from the ./prompts dir.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to launch in interactive mode",
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "-no_terminators",
        default=False,
        action="store_true",
        help="If you want the model to generate the full max tokens. Useful for profiling memory.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path(__file__).resolve().parent
        / "checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/model.pth",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--compile_prefill",
        action="store_true",
        help="Whether to compile the prefill (improves prefill perf, but higher compile times)",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--speculate_k", type=int, default=5, help="Speculative execution depth."
    )
    parser.add_argument(
        "--draft_checkpoint_path",
        type=Path,
        default=None,
        help="Draft checkpoint path.",
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )

    # KV-Cache Kwargs
    parser.add_argument(
        "--max_cache_length",
        type=float,
        default=[1.0],
        nargs="+",
        help="Cache size per layer. If len < n layers, the values are tiled. Must have len divisible by n layers. \
        If 0 < x <= 1, it is percent of |prompt| + max new tokens. Otherwise, if > 1, its the maximum size.",
    )
    parser.add_argument(
        "--cache_strategy",
        default="full",
        choices=["full", "random", "window", "scissor"],
    )
    # Optional Cache Kwargs depending on cache_strategy
    parser.add_argument(
        "--global_tokens",
        default=4,
        type=int,
        help="The number of initial tokens to always include in the KV-Cache.  \
        If using window strategy, the actual window becomes max_cache_length - global_tokens.",
    )

    # Scissorhands-specific Hyperparameters (--cache_strategy == "scissor")
    ## See Algorithm 1 & 2 in arxiv.org/abs/2305.17118
    parser.add_argument(
        "--history_window_size",  # Equivalent to "m" in Algorithm 2.
        default=400,  # 400 is default specified in paper.
        type=int,
        help="The number of past tokens to consider when computing 'Heavy Hitters' in the KV-Cache.",
    )
    parser.add_argument(
        "--drop_amount",  # Equivalent to "m" in Algorithm 2.
        default=0.5,  # 0.4 is default specified in paper.
        type=float,
        help="The number of tokens to evict KV-Cache reaches capacity (max_cache_length). Expressed as a fraction of max_cache_length.",
    )
    parser.add_argument(
        "--recent_window",  # Equivalent to "r" in Algorithm 2.
        default=10,  # 10 is default specified in paper.
        type=int,
        help="The number of recently generated tokens to always save when evicting tokens from the ScissorHands KV-Cache.",
    )
    parser.add_argument(
        "-attn_thresholding",
        default=False,
        action="store_true",
        help="Whether to accumulate number of times a token was unimportant (binary) versus raw un-normalized probabilities. If true, less precise yet more space efficient.",
    )

    args = parser.parse_args()

    if args.prompt.endswith(".txt"):
        prompt_fn = Path(__file__).resolve().parent / "prompts" / args.prompt
        with open(prompt_fn) as fd:
            args.prompt = fd.read().strip()

    if args.cache_strategy == "full":
        # Full implies no compression, which means --max_cache_length = [1.0] (same size as prompt + max_new_tokens)
        assert all(
            [l == 1.0 for l in args.max_cache_length]
        ), "Full cache strategy only supports max_cache_length=1.0."

    cache_kwargs = {
        "cache_strategy": args.cache_strategy,
        "max_cache_length": args.max_cache_length,
        "global_tokens": args.global_tokens,
        "history_window_size": args.history_window_size,
        "drop_amount": args.drop_amount,
        "recent_window": args.recent_window,
        "attn_thresholding": args.attn_thresholding,
    }

    main(
        args.prompt,
        args.interactive,
        args.num_samples,
        args.max_new_tokens,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.compile,
        args.compile_prefill,
        args.profile,
        args.draft_checkpoint_path,
        args.speculate_k,
        args.device,
        cache_kwargs,
    )
