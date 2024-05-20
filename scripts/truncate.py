import os
from pathlib import Path
import re
import shutil

import argparse
import torch


def keep_weight(key, trunc_layers):
    if "layers" in key:
        layer_num = int(re.search(r'layers\.(\d+)', key).group(1))
        return layer_num < trunc_layers
    else:
        return True


def main():
    parser = argparse.ArgumentParser(description='Script to remove all but the first K layers of model. For Debugging.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("checkpoints/meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument('--trunc_layers', type=int, default=4)
    
    # Add more arguments if needed
    args = parser.parse_args()

    trunc_dir = args.checkpoint_dir.with_name(args.checkpoint_dir.name + f"-{args.trunc_layers}-Layers")
    os.makedirs(trunc_dir, exist_ok=True)

    # If trunc_dir has tokenizer.model file, exit without error
    if (trunc_dir / "tokenizer.model").exists():
        print(f"Truncated model already exists at {trunc_dir}. Exiting without error.")
        exit(0)

    # Copy tokenizer.model file to trunc_dir
    shutil.copy(args.checkpoint_dir / "tokenizer.model", trunc_dir / "tokenizer.model")

    weights = torch.load(args.checkpoint_dir / "model.pth", map_location="cpu")
    new_weights = dict({
        key: value for key, value in weights.items() if keep_weight(key, args.trunc_layers)
    })
    torch.save(new_weights, trunc_dir / "model.pth")

    orig_size = sum([
        value.numel() for value in weights.values()
    ])

    new_size = sum([
        value.numel() for value in new_weights.values()
    ])

    print(f"Reduced number of parameters from {orig_size} to {new_size} by truncating to first {args.trunc_layers} layers.")


if __name__ == '__main__':
    main()