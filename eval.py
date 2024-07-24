# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
import argparse
import json
import regex as re
import contextlib
import shutil
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from collections import defaultdict, Counter
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
    merge_cache_config,
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


def flatten_dict(in_dict: dict) -> dict:
    out_dict = {}
    for k, v in in_dict.items():
        if type(v) == dict:
            for kk, vv in v.items():
                out_dict[f"{k}_{kk}"] = vv
        else:
            out_dict[k] = v
    return out_dict


def compress_list(l):
    if len(l) < 3:
        return l
    else:
        counter = Counter(l)
        return [f"{k}x{v}" for k, v in counter.items()]


def args_to_str(args):
    if "debug" in args.cache_strategy[0]:
        debug_suffix = "__debug"
        cache_strategy = [
            re.sub(r"debug_+", "", cs).strip() for cs in args.cache_strategy
        ]
    else:
        cache_strategy = args.cache_strategy
        debug_suffix = ""
    RELEVANT_CACHE_KWARGS = list(
        sorted(
            set(
                itertools.chain(
                    *[get_cache_constructor(cs)[1] for cs in cache_strategy]
                )
            )
        )
    )

    def process_num(n):
        # Return integer floats as "1" not 1.0
        # Otherwise, no op
        if type(n) == float and int(n) == n:
            return int(n)
        return n

    RELEVANT_CACHE_KWARGS.append("cache_length_pattern")
    RELEVANT_CACHE_KWARGS.append("cache_strategy_pattern")
    if hasattr(args, "attn_top_k") and args.attn_top_k != 1.0:
        RELEVANT_CACHE_KWARGS.append("attn_top_k")

    args_dict = vars(args).copy()

    # Hybrid Strategies will be too long to save in a file name so just need to pick the strategy
    if "hybrid_strategies" in args_dict:
        args_dict["hybrid_strategies"] = [
            x["strategy"] for x in args_dict["hybrid_strategies"]
        ]

    return (
        "__".join(
            sorted(
                [
                    f"{k}=" + ",".join(compress_list([str(process_num(m)) for m in v]))
                    if type(v) == list
                    else f"{k}={process_num(v)}"
                    for k, v in args_dict.items()
                    if k in RELEVANT_CACHE_KWARGS
                ]
            )
        )
        + debug_suffix
    )


def run_task(
    args: argparse.Namespace,
    task: AutoTask,
    model: Transformer,
    tokenizer: TokenizerInterface,
    is_chat: bool = False,
    profile: Optional[Path] = None,
    feed_long_prompts=False,
    decode_first_token=False,
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

    if len(test) == 0:
        print(
            f"No test data found for {task.__class__.__name__}. Skipping. Possibly all filtered out by tokenizer for being too long."
        )
        return None, None, None

    prompts = test["prompt"]

    inputs = [
        encode(tokenizer, prompt, device="cpu", is_chat=is_chat)
        for prompt in tqdm(prompts, desc="Encoding Prompts")
    ]

    if task.requires_perplexity:
        assert (
            len(test["labels"][0]) == 1
        ), "Only one label supported for perplexity tasks"
        label_ids = [
            encode(tokenizer, label[0], device="cpu", is_chat=False, bos=False)
            for label in tqdm(test["labels"], desc="Encoding Labels")
        ]
        _, max_seq_length = compute_max_seq_length(model, inputs, label_ids, 0)
    else:
        label_ids = None
        _, max_seq_length = compute_max_seq_length(model, inputs, None, task.max_tokens)

    # Estimate median sequence length
    median_seq_length = int(np.median([len(i) for i in inputs]) + task.max_tokens / 2)

    target_length = (
        max_seq_length
        if any([x in {"full", "hybrid"} or "debug" in x for x in args.cache_strategy])
        else median_seq_length
    )

    task_cache_kwargs = setup_caches(
        model, tokenizer, device, target_length, cache_kwargs.copy()
    )

    for i in tqdm(range(len(inputs))):
        input = inputs[i].to(device)
        next_tokens = None if label_ids is None else label_ids[i].to(device)
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
                next_tokens=next_tokens,
                terminator_ids=terminator_ids if next_tokens is None else None,
                attn_top_k=args.attn_top_k,
                feed_long_prompts=feed_long_prompts,
                decode_first_token=decode_first_token,
            )

        if next_tokens is not None:
            nll = -torch.tensor(
                [
                    torch.log(probs[j][next_tokens[j]])
                    for j in range(next_tokens.size(0))
                ]
            )
            for k in range(500, len(nll), 500):
                aggregate_metrics[f"ppl@{k}"].append(
                    float(torch.exp(torch.mean(nll[:k])).item())
                )
            aggregate_metrics["ppl"].append(float(torch.exp(torch.mean(nll)).item()))

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

        if (
            not task.requires_perplexity
        ):  # Perplexity tasks don't decode from model so don't save predictions
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
                    {k: v for k, v in zip(tokenizer.get_vocab(), probs[-1].tolist())}
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

    if task.requires_perplexity:
        pred_df = None
    else:
        pred_units = all_probs if task.requires_logits else predictions
        task_metrics.update(flatten_dict(task.test_metrics(pred_units)))
        pred_df = pd.DataFrame({"prompt": prompts, "prediction": predictions})

    return task_metrics, pred_df, task_cache_kwargs


def main(
    args: argparse.Namespace,
    tasks: List[str],
    debug: bool = False,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    profile: Optional[Path] = None,
    compile=True,
    feed_long_prompts=False,
    decode_first_token=False,
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

    if cache_kwargs["cache_strategy"] == "hybrid":
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

    task_kwargs = {
        "model_max_length": model.config.max_length,
        "num_samples": args.num_samples,
        "tokenizer": tokenizer.encode_prompt if is_chat else tokenizer.encode,
    }
    if tasks == ["all"]:
        # Evaluate all tasks
        tasks = list(TASK_MAPPING.keys())
    eval_tasks = {task: AutoTask.from_name(task, **task_kwargs) for task in tasks}

    task_metrics = defaultdict(dict)
    args_fn = out_dir / "args.json"
    all_out_fn = out_dir / "all_metrics.json"
    for task_name, task in eval_tasks.items():
        print(f"Running task {task_name} ...")
        task_out_fn = out_dir / f"{task_name}_metrics.json"
        task_args_out_fn = out_dir / f"{task_name}_args.json"
        pred_out_fn = out_dir / f"{task_name}_predictions.csv"
        if task_out_fn.exists() and not cache_kwargs["overwrite"]:
            print(f"Task {task_name} already evaluated. Skipping.")
            with open(task_out_fn, "r") as fd:
                task_metrics[task_name] = json.load(fd)
        else:
            task_metrics[task_name], predictions, task_args = run_task(
                args,
                task,
                model,
                tokenizer,
                is_chat,
                profile,
                feed_long_prompts,
                decode_first_token,
                device,
                cache_kwargs,
                use_tp,
                rank,
                terminator_ids,
            )

            if task_metrics[task_name] is None:
                continue

            if predictions is not None:
                predictions.to_csv(pred_out_fn, index=False)

            if debug:
                print(f"Results for {task_name}:")
                print(task_metrics[task_name])

            with open(task_out_fn, "w") as fd:
                print(f"Saving results for {task_name} to {task_out_fn}")
                json.dump(task_metrics[task_name], fd, indent=4)

            with open(task_args_out_fn, "w") as fd:
                print(f"Saving dynamic args for {task_name} to {task_args_out_fn}")
                # Convert Path objects to strings
                task_args_json = {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in task_args.items()
                }
                json.dump(task_args_json, fd, indent=4)

        if not args_fn.exists():
            # Only save args once and only save if we've gotten through a full eval and are ready to dump metrics
            with open(args_fn, "w") as fd:
                # Convert Path objects to strings
                cache_kwargs_json = {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in cache_kwargs.items()
                }
                json.dump(cache_kwargs_json, fd, indent=4)

    with open(all_out_fn, "w") as fd:
        json.dump(task_metrics, fd, indent=4)


def setup(args) -> Path:
    out_dir = (
        Path(__file__).parent
        / "results"
        / args.checkpoint_path.parent.name
        / "__".join(compress_list(args.cache_strategy))
        / args_to_str(args)
    )

    print(f"Saving to {out_dir}")
    # Make out_dir and don't err out if it already exists
    if out_dir.exists():
        print(f"Output directory {out_dir} already exists.")
        if args.overwrite:
            print(f"Removing {out_dir}.")
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_compatibility(args)

    for k, v in vars(args).items():
        print(f"{k} -> {v}")

    return out_dir


def add_eval_args(parser):
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["truthfulqa"],
        choices=list(TASK_MAPPING.keys()) + ["all"],
        help="List of tasks to be evaluated.",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode uses first 10 examples in dataset.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of examples to sample for evaluation. Defaults to None, which uses the full dataset.",
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

    parser.add_argument(
        "--decode_first_token",
        default=False,
        action="store_true",
        help="If True will truncate cache after prefill and then decode the first token.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script for different KV-Cache Compression Algorithms."
    )

    add_eval_args(parser)
    add_generation_arguments(parser)
    add_cache_arguments(parser)

    args = merge_cache_config(parser.parse_args())

    if args.tasks[0] == "all":
        args.tasks = list(TASK_MAPPING.keys())
        print(f"Running all tasks: {', '.join(args.tasks)}")

    out_dir = setup(args)

    main(
        args,
        args.tasks,
        args.debug,
        args.checkpoint_path,
        args.profile,
        args.compile,
        args.feed_long_prompts,
        args.decode_first_token,
        args.device,
        cache_kwargs=vars(args),
        out_dir=out_dir,
    )
