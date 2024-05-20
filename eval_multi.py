# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import argparse
from pathlib import Path

import torch
import torch._dynamo.config
import torch._inductor.config


from cache import add_cache_arguments
from generation_utils import add_generation_arguments

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from eval import add_eval_args, setup, merge_cache_config, main as eval_main


HPARAMS = {
    "max_cache_length": [[8192], [4096], [2048], [1024], [512], [256], [128]],
    "min_recovery_frac": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep a hyper-parameter for a KV-Cache Compression Algorithms."
    )

    parser.add_argument(
        "--hparam",
        default="max_cache_length",
        help="The hyper-parameter to sweep.",
    )

    add_eval_args(parser)
    add_generation_arguments(parser)
    add_cache_arguments(parser)

    args = merge_cache_config(parser.parse_args())

    assert args.hparam in HPARAMS, f"Set {args.hparam} in HPARAMS dictionary first."

    for v in HPARAMS[args.hparam]:
        # Copy the args object to avoid modifying the original
        exp_args = argparse.Namespace(**vars(args))
        print(f"Setting {args.hparam} to {v}")
        setattr(exp_args, args.hparam, v)

        out_dir = setup(exp_args)

        eval_main(
            args,
            args.tasks,
            args.debug,
            args.checkpoint_path,
            args.profile,
            args.compile,
            args.feed_long_prompts,
            args.device,
            cache_kwargs=vars(exp_args),
            out_dir=out_dir,
        )
