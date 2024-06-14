#!/bin/bash

set -e

# Use env vars if they exist, otherwise set defaults
: "${HF:=meta-llama/Meta-Llama-3-8B-Instruct}"
: "${TRUNC_LAYERS:=4}"

# Export the variables
export HF
export TRUNC_LAYERS


python scripts/download.py --repo_id $HF
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$HF
python scripts/truncate.py --checkpoint_dir checkpoints/$HF --trunc_layers $TRUNC_LAYERS
