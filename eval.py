# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
import yaml
import contextlib
from pathlib import Path
from typing import Optional, List
from collections import defaultdict
from tqdm.auto import tqdm

import torch
import torch._dynamo.config
import torch._inductor.config


from cache import add_cache_arguments, cache_compatibility, setup_caches, reset_caches
from generation_utils import device_sync, compute_max_seq_length
from tokenizer import encode


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from tokenizer import get_tokenizer
from generation_utils import load_model, generate
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
    compile=True,
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

    if cache_kwargs["cache_strategy"] == "fastgen":
        # We need to pass the special and punctuation token ids to the cache via cache_kwargs
        cache_kwargs["token_ids"] = {
            "special": tokenizer.special_ids(),
            "punctuation": tokenizer.punctuation_ids(),
        }

    terminator_ids = tokenizer.get_terminator_ids()

    torch.manual_seed(1234)

    if compile:
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )
        prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    eval_tasks = {task: AutoTask.from_name(task) for task in tasks}

    task_metrics = defaultdict(dict)
    for task_name, task in eval_tasks.items():
        print(f"Evaluating task: {task}")
        aggregate_metrics = {
            "tokens_per_sec": [],
        }
        predictions = []
        all_probs = []

        # Make a copy of the cache kwargs for this task in case of any in-place modifications
        task_cache_kwargs = cache_kwargs.copy()

        inputs = [
            encode(tokenizer, row["prompt"], device="cpu", is_chat=is_chat)
            for row in tqdm(task.get_test(), desc="Encoding Prompts")
        ]

        _, max_seq_length = compute_max_seq_length(model, inputs, task.max_tokens)

        print(f"Maximum context length of {max_seq_length} tokens.")
        setup_caches(
            model, tokenizer, inputs[0].device, max_seq_length, task_cache_kwargs
        )

        for i in tqdm(range(len(inputs))):
            inputs[i] = inputs[i].to(device)
            prompt_length = inputs[i].size(0)

            max_new_tokens = min(task.max_tokens, max_seq_length - prompt_length)
            assert max_new_tokens > 0, f"Prompt too long for model: {prompt_length}"

            device_sync(device=device)  # MKG
            t0 = time.perf_counter()

            if not profile or (use_tp and rank != 0):
                prof = contextlib.nullcontext()
            else:
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()
            with prof:
                y, probs = generate(
                    model,
                    inputs[i],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    terminator_ids=terminator_ids,
                    cache_kwargs=cache_kwargs.copy(),
                )
            if hasattr(prof, "export_chrome_trace"):
                if use_tp:
                    prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
                else:
                    prof.export_chrome_trace(f"{profile}.json")
            device_sync(device=device)  # MKG
            t = time.perf_counter() - t0

            pred = tokenizer.decode(y.tolist()).split(
                tokenizer.decode(inputs[i].tolist())
            )[1]
            predictions.append(pred)
            if task.requires_logits:
                all_probs.append(
                    {k: v for k, v in zip(tokenizer.get_vocab(), probs[0].tolist())}
                )
            tokens_generated = y.size(0) - prompt_length
            tokens_sec = tokens_generated / t
            aggregate_metrics["tokens_per_sec"].append(tokens_sec)

            # Reset Counters for KV Cache
            reset_caches(model)

        print(
            f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
        )
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        task_metrics[task_name]["tokens_per_sec"] = torch.mean(
            torch.tensor(aggregate_metrics["tokens_per_sec"])
        ).item()
        if task.requires_logits:
            task_metrics[task_name]["task_metrics"] = task.test_metrics(all_probs)
        else:
            task_metrics[task_name]["task_metrics"] = task.test_metrics(predictions)
        print(task_metrics[task_name]["task_metrics"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluation script for different KV-Cache Compression Algorithms."
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["truthfulqa"],
        choices=list(TASK_MAPPING.keys()),
        help="List of tasks to be evaluated.",
    )

    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")

    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature for sampling."
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

    parser.add_argument(
        "--cache_config",
        type=str,
        default=None,
        help="Name of YAML file in ./cache_configs.",
    )

    add_cache_arguments(parser)

    args = parser.parse_args()

    if args.cache_config:
        # Get parent directory of current file
        if not args.cache_config.endswith(".yaml"):
            args.cache_config = args.cache_config + ".yaml"
        yaml_fn = Path(__file__).parent / "cache_configs" / args.cache_config
        assert yaml_fn.exists(), f"Cache config file {yaml_fn} does not exist."
        with open(yaml_fn, "r") as f:
            cache_args = yaml.safe_load(f)
            # Over-write args with cache_args
            args = argparse.Namespace(**{**vars(args), **cache_args})

    cache_compatibility(args)

    main(
        args.tasks,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.profile,
        args.device,
        cache_kwargs=vars(args),
    )
