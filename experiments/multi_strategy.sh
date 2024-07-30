#!/bin/bash

set -e

DIR=$(dirname $(dirname "$0"))
export CKPT=$DIR/checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct/model.pth
NUM_SAMPLES=500
GLOBAL_TOKENS=4
TASKS="rulerniah musique dolomites"

SHARED_ARGS="--compile --tasks ${TASKS} --global_tokens ${GLOBAL_TOKENS} --checkpoint_path ${CKPT} --num_samples ${NUM_SAMPLES}"

MAX_CACHE_LENGTHS=(0.25 0.5 0.75)  # fraction of full cache to set for "local" layers

for MAX_CACHE_LENGTH in $MAX_CACHE_LENGTHS
do
    # Select "local", e.g., compressed, strategy
    COMPRESS_STRAT="window"
    COMPRESS_PROMPT_STRAT="recent_global"

    LOCAL2GLOBAL_ARGS="--cache_strategy ${COMPRESS_STRAT} full \
    --prompt_compression_strategy ${COMPRESS_PROMPT_STRAT} recent_global \
    --max_cache_length ${MAX_CACHE_LENGTH} 1.0"

    GLOBAL2LOCAL_ARGS="--cache_strategy full ${COMPRESS_STRAT} \
    --prompt_compression_strategy recent_global ${COMPRESS_PROMPT_STRAT} \
    --max_cache_length 1.0 ${MAX_CACHE_LENGTH}"

    ALTERNATING_ARGS="--cache_length_pattern repeat --cache_strategy_pattern repeat"
    REPEATING_ARGS="--cache_length_pattern tile --cache_strategy_pattern tile"

    A="${SHARED_ARGS} ${LOCAL2GLOBAL_ARGS} ${ALTERNATING_ARGS}"
    B="${SHARED_ARGS} ${LOCAL2GLOBAL_ARGS} ${REPEATING_ARGS}"
    C="${SHARED_ARGS} ${GLOBAL2LOCAL_ARGS} ${ALTERNATING_ARGS}"
    D="${SHARED_ARGS} ${GLOBAL2LOCAL_ARGS} ${REPEATING_ARGS}"

    echo $A
    python eval.py $A

    echo $B
    python eval.py $B

    echo $C
    python eval.py $C

    echo $D
    python eval.py $D
done
