#!/bin/bash

set -e

DIR=$(dirname $(dirname "$0"))
export CKPT=$DIR/checkpoints/Qwen/Qwen2-1.5B-Instruct/model.pth
RATIOS=(0.25 0.5 0.75)
GLOBAL_TOKENS=4

# For loop over ratios
for RATIO in $RATIOS
do
    echo "Running Attention Loss Experiments for ${CKPT} at Ratio ${RATIO}"
    python eval.py --compile --tasks pg19 --global_tokens $GLOBAL_TOKENS --checkpoint_path $CKPT --cache_strategy heavy_hitter --prompt_compression_strategy heavy_hitter --max_cache_length $RATIO
    python eval.py --compile --tasks pg19 --global_tokens $GLOBAL_TOKENS --checkpoint_path $CKPT --cache_strategy debug_heavy_hitter --prompt_compression_strategy heavy_hitter --max_cache_length $RATIO
done
