# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, List
from collections import defaultdict
from tqdm.auto import tqdm

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

from model import Transformer, find_multiple
from tokenizer import get_tokenizer
from generate import _load_model, generate, encode_tokens
from task import TASK_MAPPING, AutoTask


def main(
    tasks: List[str],
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
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
    model = _load_model(checkpoint_path, device, precision, use_tp)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path, is_chat=is_chat)

    terminator_ids = tokenizer.get_terminator_ids()

    torch.manual_seed(1234)

    eval_tasks = {task: AutoTask.from_name(task) for task in tasks}

    task_metrics = defaultdict(dict)
    for task_name, task in eval_tasks.items():
        print(f"Evaluating task: {task}")
        aggregate_metrics = {
            "tokens_per_sec": [],
            "accept_counts": [],
        }
        start = 0
        predictions = []
        for prompt in tqdm(task.get_eval_prompts()):
            if is_chat:
                tokens = tokenizer.encode_prompt(prompt)
                encoded = torch.tensor(tokens, dtype=torch.int, device=device)
            else:
                encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
            prompt_length = encoded.size(0)

            device_sync(device=device)  # MKG
            callback = lambda x: x
            t0 = time.perf_counter()
            import contextlib

            if not profile or (use_tp and rank != 0):
                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()
            with prof:
                y, metrics = generate(
                    model,
                    encoded,
                    max_new_tokens=task.max_tokens,
                    callback=callback,
                    temperature=temperature,
                    top_k=top_k,
                    terminator_ids=terminator_ids,
                    cache_kwargs=cache_kwargs.copy(),
                    interactive=False,
                    draft_model=None,
                )
                aggregate_metrics["accept_counts"].append(metrics["accept_counts"])
            if hasattr(prof, "export_chrome_trace"):
                if use_tp:
                    prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
                else:
                    prof.export_chrome_trace(f"{profile}.json")
            device_sync(device=device)  # MKG
            t = time.perf_counter() - t0

            pred = tokenizer.decode(y.tolist()).split(
                tokenizer.decode(encoded.tolist())
            )[1]
            predictions.append(pred)
            tokens_generated = y.size(0) - prompt_length
            tokens_sec = tokens_generated / t
            aggregate_metrics["tokens_per_sec"].append(tokens_sec)

        print(
            f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
        )
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        task_metrics[task_name]["tokens_per_sec"] = torch.mean(
            torch.tensor(aggregate_metrics["tokens_per_sec"])
        ).item()
        task_metrics[task_name]["task_metrics"] = task.compute_metrics(predictions)
        print(task_metrics[task_name]["task_metrics"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["squality"],
        choices=list(TASK_MAPPING.keys()),
        help="List of tasks to be evaluated.",
    )

    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")

    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature for sampling."
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
        / "checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth",
        help="Model checkpoint path.",
    )

    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")

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
        default=0,  # 0.4 is default specified in paper.
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
        args.tasks,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.profile,
        args.device,
        cache_kwargs,
    )
