#!/bin/bash

set -e

MODELS=(
	"checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct/model.pth"
	"checkpoints/Qwen/Qwen2-7B-Instruct/model.pth"
)
NUM_SAMPLES=500
CACHE_SIZES="0.75 0.5 0.25 0.1 0.05"
TASKS="truthfulqa rulerqa rulerniah rulervt rulercwe scrollsquality musique squality dolomites qmsum repobench"
CACHE_CONFIGS="random l2 heavy_hitter recent_global"

for MODEL in ${MODELS[@]}; do
	echo "Starting evals for ${MODEL}"
	python parallelize_evals.py \
		--checkpoint_path $MODEL \
		--config_names $CACHE_CONFIGS \
		--tasks $TASKS \
		--cache_sizes $CACHE_SIZES \
		--num_samples $NUM_SAMPLES \
		--add_full
done
