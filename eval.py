# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
import yaml
import json
import contextlib
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional, List
from collections import defaultdict
from tqdm.auto import tqdm

import torch
import torch._dynamo.config
import torch._inductor.config


from cache import add_cache_arguments, cache_compatibility, get_cache_constructor
from model import Transformer
from generation_utils import (
    add_generation_arguments,
    compute_max_seq_length,
    decode_one_token,
    device_sync,
    get_cache_stats,
    prefill,
    reset_caches,
    setup_caches,
)
from tokenizer import encode, TokenizerInterface


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


def run_task(
    task: AutoTask,
    model: Transformer,
    tokenizer: TokenizerInterface,
    is_chat: bool = False,
    profile: Optional[Path] = None,
    feed_long_prompts=False,
    device=default_device,
    cache_kwargs: dict = {},
    use_tp: bool = False,
    rank: int = None,
    terminator_ids: List[int] = None,
):
    aggregate_metrics = defaultdict(list)
    predictions = []
    all_probs = []
    task_metrics = {}

    test = task.get_test()
    prompts = test["prompt"]

    inputs = [
        encode(tokenizer, prompt, device="cpu", is_chat=is_chat)
        for prompt in tqdm(prompts, desc="Encoding Prompts")
    ]

    _, max_seq_length = compute_max_seq_length(model, inputs, task.max_tokens)

    setup_caches(model, tokenizer, device, max_seq_length, cache_kwargs.copy())

    for i in tqdm(range(len(inputs))):
        input = inputs[i].to(device)
        prompt_length = input.size(0)

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
                input,
                max_new_tokens=max_new_tokens,
                terminator_ids=terminator_ids,
                feed_long_prompts=feed_long_prompts,
            )
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device)  # MKG
        t = time.perf_counter() - t0

        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        aggregate_metrics["num_toks"].append(tokens_generated)

        # Reset Counters for KV Cache
        cache_stats = get_cache_stats(model, prompt_length, tokens_generated)
        for k, v in cache_stats.items():
            aggregate_metrics[k].append(v)

        # Decode: remove EoT and prompt
        end = y.size(0)
        if y[-1] in terminator_ids:
            end = -1
        pred = tokenizer.decode(y[prompt_length:end].tolist())

        if args.debug:
            print(f"Prediction: {pred}")

        predictions.append(pred)
        if task.requires_logits:
            all_probs.append(
                {k: v for k, v in zip(tokenizer.get_vocab(), probs[0].tolist())}
            )

        reset_caches(model)

    print(
        f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
    )
    max_mem_gb = torch.cuda.max_memory_reserved() / 1e9
    print(f"Memory used: {max_mem_gb} GB")
    task_metrics["max_memory_gb"] = max_mem_gb

    for k, v in aggregate_metrics.items():
        task_metrics[k] = sum(v) / len(v)

    if task.requires_logits:
        metrics = task.test_metrics(all_probs)
    else:
        metrics = task.test_metrics(predictions)

    pred_df = pd.DataFrame({"prompt": prompts, "prediction": predictions})

    for k, v in metrics.items():
        if type(v) == dict:
            for kk, vv in v.items():
                task_metrics[f"{k}_{kk}"] = vv
        else:
            task_metrics[k] = v
    return task_metrics, pred_df


def main(
    tasks: List[str],
    debug: bool = False,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    profile: Optional[Path] = None,
    compile=True,
    feed_long_prompts=False,
    device=default_device,
    cache_kwargs: dict = {},
    out_dir: Path = None,
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

    task_kwargs = {"model_max_length": model.config.max_length, "debug": args.debug}
    eval_tasks = {task: AutoTask.from_name(task, **task_kwargs) for task in tasks}

    task_metrics = defaultdict(dict)
    args_fn = out_dir / "args.json"
    all_out_fn = out_dir / "all_metrics.json"
    for task_name, task in eval_tasks.items():
        print(f"Running task {task_name} ...")
        task_out_fn = out_dir / f"{task_name}_metrics.json"
        pred_out_fn = out_dir / f"{task_name}_predictions.csv"
        if task_out_fn.exists() and not cache_kwargs["overwrite"]:
            print(f"Task {task_name} already evaluated. Skipping.")
            with open(task_out_fn, "r") as fd:
                task_metrics[task_name] = json.load(fd)
        else:
            task_metrics[task_name], predictions = run_task(
                task,
                model,
                tokenizer,
                is_chat,
                profile,
                feed_long_prompts,
                device,
                cache_kwargs,
                use_tp,
                rank,
                terminator_ids,
            )

            predictions.to_csv(pred_out_fn, index=False)

            if debug:
                print(f"Results for {task_name}:")
                print(task_metrics[task_name])

            with open(task_out_fn, "w") as fd:
                print(f"Saving results for {task_name} to {task_out_fn}")
                json.dump(task_metrics[task_name], fd, indent=2)

        if not args_fn.exists():
            # Only save args once and only save if we've gotten through a full eval and are ready to dump metrics
            with open(args_fn, "w") as fd:
                # Convert Path objects to strings
                cache_kwargs_json = {
                    k: str(v) if type(v) == Path else v for k, v in cache_kwargs.items()
                }
                json.dump(cache_kwargs_json, fd, indent=2)

    with open(all_out_fn, "w") as fd:
        json.dump(task_metrics, fd, indent=2)


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

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode uses first 10 examples in dataset.",
    )

    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Whether to over-write existing results if they exist.",
    )

    parser.add_argument(
        "--cache_config",
        type=str,
        default=None,
        help="Name of YAML file in ./cache_configs.",
    )

    add_generation_arguments(parser)
    add_cache_arguments(parser)

    args = parser.parse_args()

    RELEVANT_CACHE_KWARGS = get_cache_constructor(args.cache_strategy).relevant_kwargs

    def args_to_str(args):
        def process_num(n):
            # Return integer floats as "1" not 1.0
            # Otherwise, no op
            if type(n) == float and int(n) == n:
                return int(n)
            return n

        return "__".join(
            sorted(
                [
                    f"{k}=" + ",".join([str(process_num(m)) for m in v])
                    if type(v) == list
                    else f"{k}={process_num(v)}"
                    for k, v in vars(args).items()
                    if k in RELEVANT_CACHE_KWARGS
                ]
            )
        )

    out_dir = (
        Path(__file__).parent / "results" / args.cache_strategy / args_to_str(args)
    )

    print(f"Saving to {out_dir}")
    # Make out_dir and don't err out if it already exists
    if out_dir.exists():
        print(f"Output directory {out_dir} already exists.")
        if args.overwrite:
            print(f"Removing {out_dir}.")
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_config:
        # Get parent directory of current file
        if not args.cache_config.endswith(".yaml"):
            args.cache_config = args.cache_config + ".yaml"
        yaml_fn = Path(__file__).parent / "cache_configs" / args.cache_config
        assert yaml_fn.exists(), f"Cache config file {yaml_fn} does not exist."
        with open(yaml_fn, "r") as f:
            cache_kwargs = yaml.safe_load(f)
            # Over-write args with cache_kwargs
            args = argparse.Namespace(**{**vars(args), **cache_kwargs})

    cache_compatibility(args)

    for k, v in vars(args).items():
        print(f"{k} -> {v}")

    main(
        args.tasks,
        args.debug,
        args.checkpoint_path,
        args.profile,
        args.compile,
        args.feed_long_prompts,
        args.device,
        cache_kwargs=vars(args),
        out_dir=out_dir,
    )
