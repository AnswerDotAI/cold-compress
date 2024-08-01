#!/bin/bash

set -e

DIR=$(dirname $(dirname "$0"))
export CKPT=$DIR/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/model.pth
NUM_SAMPLES=500
GLOBAL_TOKENS=4
CACHE_STRATEGY="heavy_hitter"
PROMPT_STRATEGY="heavy_hitter"
TASKS="rulerniah musique dolomites"

SHARED_ARGS="--compile --tasks ${TASKS} --global_tokens ${GLOBAL_TOKENS} --checkpoint_path ${CKPT} --num_samples ${NUM_SAMPLES}"

MAX_CACHE_LENGTHS=(0.1 0.25 0.5)

for MAX_CACHE_LENGTH in $MAX_CACHE_LENGTHS
do
    echo "Starting experiments with Max Cache Length=${MAX_CACHE_LENGTH}."
    python eval.py $SHARED_ARGS --max_cache_length $MAX_CACHE_LENGTH --cache_length_pattern pyramid
    python eval.py $SHARED_ARGS --max_cache_length $MAX_CACHE_LENGTH --cache_length_pattern repeat
    python eval.py $SHARED_ARGS --max_cache_length $MAX_CACHE_LENGTH --cache_length_pattern tile
done
