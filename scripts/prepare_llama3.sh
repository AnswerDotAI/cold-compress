#!/bin/bash

set -e

HF=meta-llama/Meta-Llama-3-8B-Instruct
NAME=Llama-3-8B
TRUNC_LAYERS=4


python scripts/download.py --repo_id $HF
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$HF
python scripts/truncate.py --checkpoint_dir checkpoints/$HF --trunc_layers $TRUNC_LAYERS
