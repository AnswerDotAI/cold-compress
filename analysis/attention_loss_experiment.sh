#!/bin/bash

set -e

DIR=$(dirname $(dirname "$0"))
export CKPT=$DIR/checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth
RATIOS=(0.25 0.5 0.75)

# For loop over ratios
for RATIO in $RATIOS
do
    echo "Running Attention Loss Experiments for ${CKPT} at Ratio ${RATIO}"
    python eval.py --compile --tasks pg19 --checkpoint_path $CKPT --cache_strategy scissor --prompt_compression_strategy snapkv --max_cache_length $RATIO
    python eval.py --compile --tasks pg19 --checkpoint_path $CKPT --cache_strategy debug_scissor --prompt_compression_strategy snapkv --max_cache_length $RATIO
done
