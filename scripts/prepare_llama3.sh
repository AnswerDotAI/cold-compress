#!/bin/bash

set -e

HF=meta-llama/Meta-Llama-3-8B-Instruct
NAME=Llama-3-8B

python scripts/download.py --repo_id $HF
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$HF
