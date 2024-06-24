# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
import contextlib
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo.config
import torch._inductor.config

from cache import add_cache_arguments
from generation_utils import (
    decode_one_token,
    prefill,
    device_sync,
    compute_max_seq_length,
)
from tokenizer import encode

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from tokenizer import get_tokenizer, encode
from generation_utils import generate, load_model, get_model_size, setup_caches
from cache import add_cache_arguments, cache_compatibility


def main(
    prompt: str = "Hello, my name is",
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    compile: bool = True,
    profile: Optional[Path] = None,
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
    is_chat = (
        "chat" in str(checkpoint_path).lower()
        or "instruct" in str(checkpoint_path).lower()
    )

    print("Loading model ...")
    t0 = time.time()
    model = load_model(checkpoint_path, device, precision, use_tp)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path, is_chat=is_chat)

    inputs = [encode(tokenizer, prompt, device=device, is_chat=is_chat)]

    terminator_ids = tokenizer.get_terminator_ids()

    torch.manual_seed(1234)
    model_size = get_model_size(model)
    print(f"{model_size / 1e9:.02f} billion parameters in model.")

    if compile:
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )
        prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        "tokens_per_sec": [],
    }

    device_sync(device=device)  # MKG

    max_prompt_length, max_seq_length = compute_max_seq_length(
        model, inputs, max_new_tokens
    )
    max_new_tokens = min(max_new_tokens, max_seq_length - max_prompt_length)
    setup_caches(model, tokenizer, inputs[0].device, max_seq_length, cache_kwargs)
    t0 = time.perf_counter()

    if (not profile) or (use_tp and rank != 0):
        prof = contextlib.nullcontext()
    else:
        torch.profiler._utils._init_for_cuda_graphs()
        prof = torch.profiler.profile()

    with prof:
        y, _ = generate(
            model,
            inputs[0],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            terminator_ids=terminator_ids,
        )
    print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
    if hasattr(prof, "export_chrome_trace"):
        if use_tp:
            prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
        else:
            prof.export_chrome_trace(f"{profile}.json")
    device_sync(device=device)  # MKG
    t = time.perf_counter() - t0

    print(tokenizer.decode(y.tolist()))
    tokens_generated = y.size(0) - max_prompt_length
    tokens_sec = tokens_generated / t
    aggregate_metrics["tokens_per_sec"].append(tokens_sec)
    print(f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    print("==========")

    print(
        f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
    )
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Simple Single Prompt Generation (for development and debugging purposes)."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="long_prompt_short_output.txt",
        help="Input prompt. If it ends in .txt, we will load the prompt from the ./prompts dir.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
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
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )

    add_cache_arguments(parser)

    args = parser.parse_args()

    if args.prompt.endswith(".txt"):
        prompt_fn = Path(__file__).resolve().parent / "prompts" / args.prompt
        with open(prompt_fn) as fd:
            args.prompt = fd.read().strip()

    cache_compatibility(args)

    main(
        args.prompt,
        args.max_new_tokens,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.compile,
        args.profile,
        args.device,
        cache_kwargs=vars(args),
    )
