#!/bin/bash

set -e

# Use env vars if they exist, otherwise set defaults
: "${HF:=Qwen/Qwen2-1.5B-Instruct}"

# Export the variables
export HF

python scripts/download.py --repo_id $HF
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$HF
